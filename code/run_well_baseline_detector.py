"""Create a per-well A000 disruption and persistence baseline.

This is a thin adapter around the accepted A3 detector/scorer chain. The legacy
scorer expects its candidate TIFF to be named ``95_A3.tif``; the snapshot stage
creates that internal alias while every manifest and review retains the real
recording stem.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

from well_workflow import find_motion_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted_evidence(run: Path, stem: str) -> dict[str, Path]:
    history = run / "mid/accepted_history"
    paths = {
        "labels": run / f"out/{stem}.tif",
        "unclaimed": run / f"out/{stem}_unclaimed_original_ids.tif",
        "raw": run.parents[1] / f"registered inputs/{stem}.tif",
        "points": history / "26_latent_body_tracks/out/latent_track_points.csv",
        "episodes": history / "26_latent_body_tracks/out/encounter_episodes.csv",
        "frames": history / "26_latent_body_tracks/out/encounter_frames.csv",
        "reconnections": (
            history / "26_latent_body_tracks/out/reconnection_hypotheses.csv"),
        "application_audit": (
            history / "46_transient_misownership_recovery/out/application_audit.csv"),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing accepted evidence:\n" + "\n".join(missing))
    return paths


def semantic_links(tuning: Path, sources: dict[str, Path]) -> dict[str, Path]:
    links = tuning / "inputs/source_links"
    links.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, source in sources.items():
        suffix = "".join(source.suffixes)
        target = links / f"{name}{suffix}"
        if target.exists():
            if not os.path.samefile(target, source) and sha256(target) != sha256(source):
                raise RuntimeError(f"Frozen input drift at {target}")
        else:
            try:
                os.link(source, target)
            except OSError:
                if source.stat().st_size > 8 << 20:
                    os.symlink(source, target)
                else:
                    shutil.copy2(source, target)
        result[name] = target
    return result


def frozen_path(tuning: Path, link: Path) -> Path:
    copied = tuning / "inputs/src" / link.name
    return copied if copied.is_file() else link


def snapshot_stage(_upstream: Path | None, params: dict, out) -> dict:
    labels = out.out / "95_A3.tif"
    unclaimed = out.out / "95_A3_unclaimed_original_ids.tif"
    application_audit = out.out / "application_audit.csv"
    conflicted_audit = out.out / "conflicted_lineage_audit.csv"
    shutil.copy2(params["labels_path"], labels)
    shutil.copy2(params["unclaimed_path"], unclaimed)
    shutil.copy2(params["application_audit_path"], application_audit)
    conflicted_audit.write_text(
        "proposal_id,physical_track,assigned_identity,outcome,reason,changed_pixels,changed_frames\n",
        encoding="utf-8")
    return {
        "outputs": {"detector_labels_alias": labels,
                    "detector_unclaimed_alias": unclaimed,
                    "application_audit": application_audit,
                    "conflicted_lineage_audit": conflicted_audit},
        "summary": {
            "recording_stem": params["recording_stem"],
            "biological_change_pixels": 0,
            "targeting_mode": "field_wide_discovery",
            "identity_targets": 0,
            "track_targets": 0,
            "frame_targets": 0,
            "coordinate_targets": 0,
            "event_targets": 0,
            "region_targets": 0,
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render_reviews(
    root: Path,
    issue: Path,
    stem: str,
    labels: Path,
    raw: Path,
    motion: Path,
    events: Path,
    members: Path,
    issue_number: int,
) -> dict[str, Path]:
    review = issue / (
        f"reviews/I{issue_number:03d}-R01-A000-BASELINE/full_dataset")
    digest = sha256(labels)
    common = [
        "--baseline-labels", str(labels), "--candidate-labels", str(labels),
        "--raw", str(raw), "--motion", str(motion),
        "--output-dir", str(review), "--stem", stem,
        "--baseline-title", "accepted A000",
        "--candidate-title", "accepted A000",
        "--source-frame-offset", "2", "--motion-frame-offset", "3",
        "--expected-baseline-sha256", digest,
        "--expected-candidate-sha256", digest,
    ]
    command = [sys.executable, str(root / "code/build_review_tiff.py"), *common]
    run_checked([*command, "--dry-run"])
    run_checked(command)

    event_dir = review / "event_diagnostics"
    event_common = [
        "--labels", str(labels), "--raw", str(raw),
        "--baseline-events", str(events), "--candidate-events", str(events),
        "--event-members", str(members), "--output-dir", str(event_dir),
        "--stem", stem, "--source-frame-offset", "2",
        "--expected-labels-sha256", digest,
    ]
    event_command = [
        sys.executable, str(root / "code/build_event_review_tiff.py"), *event_common]
    run_checked([*event_command, "--dry-run"])
    run_checked(event_command)
    return {"review": review, "event_review": event_dir}


def copy_accepted_stage(tuning: Path, stage: str, run: str) -> None:
    source = tuning / stage / run
    target = tuning / "accepted" / stage
    if target.exists():
        return
    target.mkdir(parents=True)
    for name in ("out", "qc"):
        item = source / name
        if item.exists():
            shutil.copytree(item, target / name)
    shutil.copy2(source / "run.json", target / "run.json")


def append_attempt(path: Path, row: dict) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        existing = list(csv.DictReader(handle))
        fields = list(existing[0]) if existing else list(row)
    if any(item["attempt_id"] == row["attempt_id"] for item in existing):
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    root = find_motion_root(Path(__file__))
    issue = (root / args.issue).resolve()
    tuning = issue / "analysis" / args.tuning_name
    accepted_run = (root / args.accepted_run).resolve()
    motion = (root / args.motion).resolve()
    if not issue.is_relative_to(root / "issues"):
        raise ValueError("issue folder must remain under Motion/issues")
    if not motion.is_file():
        raise FileNotFoundError(motion)

    evidence = accepted_evidence(accepted_run, args.stem)
    evidence.update({"motion": motion, "cases": issue / "review_cases.csv"})
    links = semantic_links(tuning, evidence)

    skill = (root.parents[2] / "Macros and Scripts/Claude/shared-skills"
             / "analysis-tuner")
    sys.path.insert(0, str(skill / "scripts"))
    from tuner import Tuner  # type: ignore  # noqa: E402

    tuner = Tuner(tuning)
    if not (tuning / "inputs/inputs.csv").is_file():
        tuner.freeze_inputs(links.values())
    source = {name: frozen_path(tuning, link) for name, link in links.items()}
    source["window"] = source["labels"]

    scorer_root = (root / "issues/01_95_A3"
                   / "ISSUE-092-duplicate-soma-owner-exclusivity"
                   / "analysis/duplicate_soma_exclusivity_tuning")
    scorer_driver = load_module(
        "accepted_a3_scorer_driver_for_wells", scorer_root / "scripts/run_round01.py")
    scorer = load_module(
        "accepted_a3_disruption_scorer_for_wells", scorer_root / "code/m3_disruption_score.py")
    persistence = load_module(
        "accepted_a3_persistence_scorer_for_wells", scorer_root / "code/m4_persistence_score.py")

    snapshot = tuner.stage_run(
        "m1_accepted_snapshot", "r01c_a000_baseline",
        params={"recording_stem": args.stem,
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "application_audit_path": str(source["application_audit"]),
                "targeting_mode": "field_wide_discovery"},
        fn=snapshot_stage, code=[Path(__file__)],
        notes="Byte-identical accepted parent; legacy 95_A3 filename is an internal detector adapter only.")
    score = tuner.stage_run(
        "m2_disruption_score", "r01c_a000_baseline",
        upstream=f"m1_accepted_snapshot/{snapshot['run']}",
        params=scorer_driver.score_params(source), fn=scorer.run,
        code=[scorer_root / "code/m3_disruption_score.py",
              scorer_root / "code/m4_persistence_score.py"],
        notes="Current accepted target-free disruption detector, run fresh on this well.")
    persistence_run = tuner.stage_run(
        "m3_persistence_score", "r01c_a000_baseline",
        upstream=f"m1_accepted_snapshot/{snapshot['run']}",
        params={"baseline_labels_path": str(source["labels"]),
                "window_reference_labels_path": str(source["window"])},
        fn=persistence.run,
        code=[scorer_root / "code/m4_persistence_score.py",
              root / "code/continuity_confidence.py"],
        notes="Fixed-window persistence baseline for this well.")

    score_out = tuning / score["dir"] / "out"
    events_path = score_out / "candidate_disruptive_events.csv"
    members_path = score_out / "candidate_event_members.csv"
    events = pd.read_csv(events_path).sort_values(
        ["disruption_score", "impact", "first_frame"], ascending=[False, False, True])
    top = events.head(20).copy()
    round_dir = tuning / "rounds" / args.round_name
    round_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(round_dir / "top_events.csv", index=False)
    issue_match = re.match(r"ISSUE-(\d+)-", issue.name)
    if issue_match is None:
        raise ValueError(
            "issue folder must begin with ISSUE-NNN- so review files remain canonical")
    issue_number = int(issue_match.group(1))
    reviews = render_reviews(
        root, issue, args.stem, evidence["labels"], evidence["raw"], motion,
        events_path, members_path, issue_number)

    for stage, manifest in (
        ("m1_accepted_snapshot", snapshot),
        ("m2_disruption_score", score),
        ("m3_persistence_score", persistence_run),
    ):
        copy_accepted_stage(tuning, stage, manifest["run"])

    top_event = top.iloc[0].to_dict() if len(top) else None
    summary = {
        "schema": "motion.well-baseline.v1",
        "status": "baselined",
        "well": args.stem,
        "accepted_parent": str(accepted_run.relative_to(root)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frames_removed": [1, 2],
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events", "regions")},
        "inputs_csv": str((tuning / "inputs/inputs.csv").relative_to(root)),
        "labels_sha256": sha256(evidence["labels"]),
        "unclaimed_sha256": sha256(evidence["unclaimed"]),
        "disruption": score["summary"]["candidate"],
        "persistence": persistence_run["summary"],
        "top_event": top_event,
        "top_events_csv": str((round_dir / "top_events.csv").relative_to(root)),
        "stage_runs": [snapshot["dir"], score["dir"], persistence_run["dir"]],
        "review": str(reviews["review"].relative_to(root)),
        "event_review": str(reviews["event_review"].relative_to(root)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(tuning / "accepted_base.json", summary)
    write_json(issue.parent / "WELL_BASELINE.json", summary)

    append_attempt(issue / "attempts.csv", {
        "attempt_id": "R01-A000", "date": date.today().isoformat(),
        "approaches": f"accepted {accepted_run.name} unchanged",
        "stage_runs": "|".join(summary["stage_runs"]),
        "outputs_sha256": sha256(events_path),
        "primary": f"{len(events)} freshly detected events",
        "controls": "1/1 neutral baseline control",
        "regressions": "not applicable; A000 accepted parent",
        "guardrails": "zero biological change; zero targets",
        "elapsed_s": summary["elapsed_seconds"], "identical_to": "accepted parent",
        "review_status": "complete baseline review",
        "decision": "baseline",
        "notes": f"Fresh {args.stem} catalogue; no prior event list reused.",
    })
    print(json.dumps(summary, indent=2, default=str))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stem", required=True)
    result.add_argument("--accepted-run", required=True)
    result.add_argument("--issue", required=True)
    result.add_argument("--motion", required=True)
    result.add_argument(
        "--tuning-name", default="highest_disruption_event_tuning",
        help="Directory name beneath the issue's analysis folder.")
    result.add_argument(
        "--round-name", default="R01-rank-current-disruptions",
        help="Directory name beneath the tuning folder's rounds directory.")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
