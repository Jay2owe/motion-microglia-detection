"""Rebuild pending Issue 027/028 reviews without changing analysis attempts.

The original active review files are copied into an immutable render-history folder
before replacement. Candidate labels, score tables, and approval state are untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import tifffile

import continuity_review


ROOT = Path(__file__).resolve().parents[1]
A3_ISSUES = ROOT / "issues" / "01_95_A3"
FULL_NAME = "95_A3_full_field_before_after_change_motion_NUMBERED_DISPLAY_ONLY.tif"
OUTLINE_NAME = "95_A3_candidate_outlines_NUMBERED_DISPLAY_ONLY.tif"
REVISION = "R02_established_style_red_change"


SPECS = [
    {
        "issue": "ISSUE-027-short-blank-detection-dropouts",
        "review": "I027-R01-A003-P90",
        "attempt": "I027-R01-A003-P90",
        "candidate_title": "ISSUE 027 A003 SHORT-BLANK RECOVERY",
        "candidate": "analysis/short_blank_recovery_tuning/m1_raw_only_recovery/r02_a003_p90_gap10/out/95_A3.tif",
        "score": "analysis/short_blank_recovery_tuning/m2_score/r02_a003_p90_gap10/out",
        "old_full_sha256": "a42919a1527f8674926644e42cc2c793d3cd54c500e1c2a7097bc81632cf47fe",
        "old_outline_sha256": "ca8fda3f448fa390dd8b3623fc7147f96f0173771355b1052c90aff49c542c70",
    },
    {
        "issue": "ISSUE-028-competitor-occupied-identity-gaps",
        "review": "I028-R01-A003-ALL-GAPS",
        "attempt": "I028-R01-A003-ALL-GAPS",
        "candidate_title": "ISSUE 028 A003 HOST-SURPLUS RECOVERY",
        "candidate": "analysis/host_surplus_recovery_tuning/m1_host_only_recovery/r02_a003_all_internal_gaps/out/95_A3.tif",
        "score": "analysis/host_surplus_recovery_tuning/m2_score/r02_a003_all_internal_gaps/out",
        "old_full_sha256": "63ccc8ffe23dbf5f0db9e4ce97673e732ac12117ee6a2384ffefcab1ac31c4d0",
        "old_outline_sha256": "621ef60d596f7736450f08d783abf088cb1db0ce183122b8038e498794796664",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def validate_change_panel(full_path: Path, baseline_path: Path,
                          candidate_path: Path) -> dict:
    full = tifffile.imread(full_path)
    baseline = tifffile.imread(baseline_path)
    candidate = tifffile.imread(candidate_path)
    expected_shape = (99, baseline.shape[1] + 24, baseline.shape[2] * 4, 3)
    if full.shape != expected_shape:
        raise AssertionError(f"review shape {full.shape} != {expected_shape}")
    width = baseline.shape[2]
    change_body = full[:, 24:, 2 * width:3 * width]
    red = np.array([255, 45, 45], np.uint8)
    rendered_red = np.all(change_body == red, axis=-1)
    changed = baseline != candidate
    if not np.array_equal(rendered_red, changed):
        raise AssertionError("red change pixels do not exactly match label differences")
    non_red = change_body[~rendered_red]
    if not np.all((non_red[:, 0] == non_red[:, 1])
                  & (non_red[:, 1] == non_red[:, 2])):
        raise AssertionError("change panel contains coloured pixels outside changes")
    return {
        "shape": list(map(int, full.shape)),
        "changed_pixels": int(changed.sum()),
        "red_pixels": int(rendered_red.sum()),
        "non_red_pixels_all_greyscale": True,
        "change_panel_numbers": False,
    }


def rebuild(spec: dict) -> dict:
    issue = A3_ISSUES / spec["issue"]
    review = issue / "reviews" / spec["review"]
    full_dataset = review / "full_dataset"
    active_tiffs = full_dataset / "tiffs"
    old_full = active_tiffs / FULL_NAME
    old_outline = active_tiffs / OUTLINE_NAME
    if sha256(old_full) != spec["old_full_sha256"]:
        raise RuntimeError(f"refusing to replace changed review: {old_full}")
    if sha256(old_outline) != spec["old_outline_sha256"]:
        raise RuntimeError(f"refusing to replace changed review: {old_outline}")

    revision_root = full_dataset / "render_revisions" / REVISION
    if revision_root.exists():
        raise FileExistsError(f"immutable render revision already exists: {revision_root}")
    out = SimpleNamespace(
        out=revision_root / "out", mid=revision_root / "mid",
        qc=revision_root / "qc")
    for directory in (out.out, out.mid, out.qc):
        directory.mkdir(parents=True, exist_ok=False)

    candidate_path = issue / spec["candidate"]
    score_path = issue / spec["score"]
    baseline_path = ROOT / "registered inputs/accepted detection results/labels/95_A3.tif"
    params = {
        "attempt_id": spec["attempt"],
        "baseline_labels_path": str(baseline_path),
        "candidate_labels_path": str(candidate_path),
        "raw_path": str(ROOT.parent / "Longitudinal Cell Tracker/registered_inputs/r02_grossfix/out/95_A3.tif"),
        "motion_path": str(ROOT.parent / "Longitudinal Cell Tracker/motion_inputs/r05_grossfix/qc/95_A3_motion_evidence.tif"),
        "review_cases_path": str(issue / "review_cases.csv"),
        "source_frame_offset": 2,
        "candidate_title": spec["candidate_title"],
    }
    continuity_review.run(score_path, params, out)
    new_full = out.qc / FULL_NAME
    new_outline = out.qc / OUTLINE_NAME
    validation = validate_change_panel(new_full, baseline_path, candidate_path)

    archive = full_dataset / "render_history" / "R01_initial_renderer"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(old_full, archive / FULL_NAME)
    shutil.copy2(old_outline, archive / OUTLINE_NAME)
    shutil.copy2(full_dataset / "review_summary.json",
                 archive / "review_summary.json")
    for preview in active_tiffs.glob("preview_review_frame_50*.png"):
        shutil.copy2(preview, archive / preview.name)

    shutil.copy2(new_full, old_full)
    shutil.copy2(new_outline, old_outline)
    shutil.copy2(out.out / "review_summary.json",
                 full_dataset / "review_summary.json")
    shutil.copy2(out.out / "reviewer_labels.csv",
                 full_dataset / "reviewer_labels.csv")

    record = {
        "render_revision": REVISION,
        "scientific_attempt_changed": False,
        "candidate_labels": {
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": sha256(candidate_path),
        },
        "score_metrics": {
            "path": str((score_path / "metrics.json").relative_to(ROOT)),
            "sha256": sha256(score_path / "metrics.json"),
        },
        "previous_render": {
            "archive": str(archive.relative_to(ROOT)),
            "full_sha256": spec["old_full_sha256"],
            "outline_sha256": spec["old_outline_sha256"],
        },
        "active_render": {
            "full_path": str(old_full.relative_to(ROOT)),
            "full_sha256": sha256(old_full),
            "outline_path": str(old_outline.relative_to(ROOT)),
            "outline_sha256": sha256(old_outline),
        },
        "style": {
            "base_candidate_motion": "established issue-review palette, annotations, contrast, and 24-pixel header",
            "change": "no numbers; grey candidate outlines; all and only changed pixels red",
        },
        "validation": validation,
        "producer_code": [
            code_record(ROOT / "code/continuity_review.py"),
            code_record(ROOT / "code/review_rendering.py"),
            code_record(ROOT / "code/common.py"),
            code_record(Path(__file__).resolve()),
        ],
    }
    record_path = review / "render_revision.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    records = [rebuild(spec) for spec in SPECS]
    # The base panel must be identical in both active reviews.
    panels = []
    for spec in SPECS:
        path = (A3_ISSUES / spec["issue"] / "reviews" / spec["review"] /
                "full_dataset/tiffs" / FULL_NAME)
        stack = tifffile.imread(path)
        panels.append(stack[:, :, :425])
    if not np.array_equal(panels[0], panels[1]):
        raise AssertionError("accepted-base panels differ between the two reviews")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
