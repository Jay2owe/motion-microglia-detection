"""Build a provenance sidecar for an accepted run that predates the sidecar.

From a fresh run onwards `stage_m22_accepted_history` saves
`out/{stem}_provenance.tif` itself, from arrays it already holds. Runs made
before that change still have every stage's reconstruction record on disk, so
this reads them back and writes the same sidecar into its own immutable run
folder. Nothing is written into the run being described.

The one thing this refuses to do is guess. A sidecar is only as good as its
alignment: joining one run's reconstruction record onto another run's labels
would mark real observations as reconstructed. So the labels are hashed, the
hash is compared against what the accepted run recorded, and every stage stack
is named in the manifest with the run it came from and its own hash.

    python code/build_provenance_sidecar.py \
        --run r01_95_B4 --stem 95_B4 --accepted-run r05_accepted_95_B4

Stages that the manifest chain cannot reach - a run resumed after M5 records no
link back to the M4 that fed it - are supplied explicitly:

    --stage-run m4_events=r05_accepted

An override is not a patch over one stage. It is the only route to everything
above that stage, so each supplied run is followed upstream in turn: naming the
M4 run is also how the detection record under M2 is found.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Config, Run, load_stack, read_json, save_stack, sha256
from pipeline import (_INFERRED_STACKS, _align_review_stack, pack_provenance,
                      provenance_metrics)

ACCEPTED_STAGE = "m22_accepted_history"

# The base record every accepted run rests on, and the only stack that carries
# foreground the tracker never resolved.
_BASE_STACKS = (
    ("m4_events", "{stem}_inferred_pixels.tif", "inferred"),
    ("m4_events", "{stem}_unresolved_pixels.tif", "unresolved"),
)


def _walk(stage: str, run: str, chain: dict[str, str]) -> None:
    """Follow run.json upstream links from one stage, filling `chain`."""
    while stage and stage not in chain:
        manifest_path = ROOT / stage / run / "run.json"
        if not manifest_path.is_file():
            return
        chain[stage] = run
        upstream = read_json(manifest_path).get("upstream")
        if not upstream or "/" not in upstream:
            return
        stage, run = upstream.split("/", 1)


def resolve_chain(accepted_run: str, overrides: dict[str, str]) -> dict[str, str]:
    """Map each stage to the run that produced it, by following run.json.

    Each override is walked upstream as well, because the link that was broken
    is exactly the one hiding everything above it.
    """
    chain: dict[str, str] = {}
    _walk(ACCEPTED_STAGE, accepted_run, chain)
    for stage, run in overrides.items():
        chain.pop(stage, None)
        _walk(stage, run, chain)
        chain.setdefault(stage, run)
    return chain


def gather(stem: str, chain: dict[str, str], frames: int,
           ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Union every reachable reconstruction record, aligned to `frames`."""
    used: list[dict] = []

    def read(stage: str, template: str, align: bool) -> np.ndarray | None:
        run = chain.get(stage)
        if run is None:
            return None
        path = ROOT / stage / run / "mid" / template.format(stem=stem)
        if not path.is_file():
            return None
        stack = load_stack(path).astype(bool)
        if not align and len(stack) != frames:
            raise ValueError(
                f"{path} has {len(stack)} frames but the labels have {frames}; "
                "this stack is recorded in accepted-frame space and must match "
                "exactly, so the run pairing is wrong")
        used.append({"stage": stage, "run": run, "path": str(path),
                     "sha256": sha256(path), "frames": int(len(stack)),
                     "aligned": align})
        return _align_review_stack(stack, frames) if align else stack

    base = {name: read(stage, template, True)
            for stage, template, name in _BASE_STACKS}
    if base["inferred"] is None or base["unresolved"] is None:
        missing = [name for name, value in base.items() if value is None]
        raise FileNotFoundError(
            "the M4 base record is missing " + ", ".join(missing) +
            "; supply the run that produced it with --stage-run "
            "m4_events=<run>, because a sidecar without it would mark "
            "reconstructed pixels as observed")
    inferred = base["inferred"].copy()
    for stage, template, _required, align in _INFERRED_STACKS:
        stack = read(stage, template, align)
        if stack is not None:
            inferred |= stack
    return inferred, base["unresolved"], used


def detection(stem: str, chain: dict[str, str], frames: int,
              used: list[dict]) -> np.ndarray:
    """The segmentation's own foreground, before any identity was assigned.

    This is what makes bit 2 possible: an outline pixel with nothing here is
    one the tracker supplied, not one the microscope showed.
    """
    run = chain.get("m2_anchor")
    path = (ROOT / "m2_anchor" / run / "out" / f"{stem}_observations.tif"
            if run else None)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"the detection record m2_anchor/<run>/out/{stem}_observations.tif "
            "is missing; supply the run that produced it with --stage-run "
            "m2_anchor=<run>. Without it every outline pixel would be recorded "
            "as one the microscope showed, and the sidecar would state the "
            "opposite of the truth about the pixels the tracker supplied")
    stack = load_stack(path)
    used.append({"stage": "m2_anchor", "run": run, "path": str(path),
                 "sha256": sha256(path), "frames": int(len(stack)),
                 "aligned": True})
    return _align_review_stack(stack, frames) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill a provenance sidecar for an accepted run")
    parser.add_argument("--run", required=True,
                        help="new immutable run name for the sidecar")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--accepted-run", required=True,
                        help=f"the {ACCEPTED_STAGE} run holding the labels")
    parser.add_argument("--labels", type=Path,
                        help="labels the sidecar must align to "
                             "(default: the accepted run's out/<stem>.tif)")
    parser.add_argument("--stage-run", action="append", default=[],
                        metavar="STAGE=RUN",
                        help="supply a stage the manifest chain cannot reach")
    parser.add_argument("--expect-labels-sha256",
                        help="refuse to write unless the labels hash to this")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    accepted_dir = ROOT / ACCEPTED_STAGE / args.accepted_run
    labels_path = args.labels or accepted_dir / "out" / f"{args.stem}.tif"
    if not labels_path.is_file():
        raise FileNotFoundError(f"accepted labels not found: {labels_path}")
    labels_sha = sha256(labels_path)
    if args.expect_labels_sha256 and labels_sha != args.expect_labels_sha256:
        raise ValueError(
            f"labels {labels_path} hash to {labels_sha}, not "
            f"{args.expect_labels_sha256}; a sidecar built against the wrong "
            "stack is worse than no sidecar")
    recorded = (read_json(accepted_dir / "run.json").get("summary") or {}
                ).get("labels_sha256")

    overrides = dict(item.split("=", 1) for item in args.stage_run)
    chain = resolve_chain(args.accepted_run, overrides)

    labels = load_stack(labels_path)
    inferred, unresolved, used = gather(args.stem, chain, len(labels))
    detected = detection(args.stem, chain, len(labels), used)
    provenance = pack_provenance(inferred, unresolved, labels, detected)

    run = Run("m23_provenance", args.run, {
        "stem": args.stem,
        "mode": "backfill",
        "accepted_run": args.accepted_run,
        "labels": str(labels_path),
        "labels_sha256": labels_sha,
        "labels_sha256_recorded_by_accepted_run": recorded,
        "labels_match_accepted_run": recorded is None or recorded == labels_sha,
        "resolved_chain": chain,
        "explicit_overrides": overrides,
        "encoding": {"bit0": "owning identity decided by reconstruction",
                     "bit1": "foreground the tracker left unresolved",
                     "bit2": "outline pixel the detection never saw"},
    }, upstream=f"{ACCEPTED_STAGE}/{args.accepted_run}")
    try:
        output = run.dir / "out" / f"{args.stem}_provenance.tif"
        save_stack(output, provenance, cfg.values["frame_interval_min"])
        run.record("provenance", output)
        summary = {"contributing_stacks": used,
                   "stacks_used": len(used),
                   **provenance_metrics(provenance, labels)}
        run.finish(summary)
    except BaseException as error:
        run.fail(error)
        raise
    print(f"DONE {args.stem} {args.run}: {output}")
    for record in used:
        print(f"  {record['stage']:30} {record['run']}")
    for key, value in provenance_metrics(provenance, labels).items():
        print(f"  {key:44} {value}")


if __name__ == "__main__":
    main()
