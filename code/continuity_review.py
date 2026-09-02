"""Self-contained complete-field review renderer for continuity candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from common import change_counts, change_overlay, save_rgb_stack
from review_rendering import header, unique_outline


def _align(stack: np.ndarray, frames: int, offset: int) -> np.ndarray:
    if len(stack) == frames:
        return stack
    if len(stack) >= frames:
        indices = np.minimum(offset + np.arange(frames), len(stack) - 1)
        return stack[indices]
    raise ValueError(f"cannot align {len(stack)} frames to {frames} at offset {offset}")


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("a completed score run is required")
    score = json.loads((upstream_dir / "metrics.json").read_text(encoding="utf-8"))
    eligible = bool(
        score["pooled_confidence_delta"] > 0
        and score["group_mean_confidence_delta"] > 0
        and score["identity_set_exact"]
        and score["per_frame_existing_identity_preserved"]
        and score["assigned_unclaimed_disjoint"]
        and score["zero_signal_additions"] == 0)
    if not eligible:
        raise AssertionError("only an automatically eligible leader may be reviewed")
    baseline = tifffile.imread(params["baseline_labels_path"])
    candidate = tifffile.imread(params["candidate_labels_path"])
    if baseline.shape != candidate.shape or len(candidate) != 99:
        raise ValueError("review requires matching complete 99-frame labels")
    offset = int(params.get("source_frame_offset", 2))
    raw = _align(tifffile.imread(params["raw_path"]), len(candidate), offset)
    motion = _align(tifffile.imread(params["motion_path"]), len(candidate), offset)
    before = unique_outline(raw, baseline)
    after = unique_outline(raw, candidate)
    difference = change_overlay(raw, baseline, candidate)
    motion_view = unique_outline(raw, candidate, motion)
    counts = change_counts(baseline, candidate)
    notes = [f"{pixels} px changed" if pixels else "unchanged"
             for pixels in counts["per_frame"]]
    candidate_title = str(params["candidate_title"])
    full = np.concatenate([
        header(before, "CURRENT ACCEPTED BASE", offset),
        header(after, candidate_title, offset),
        header(difference, f"CHANGE: RED = {counts['pixels']} px MOVIE",
               offset, notes),
        header(motion_view, f"{candidate_title} + MOTION", offset),
    ], axis=2)
    full_path = out.qc / (
        "95_A3_full_field_before_after_change_motion_NUMBERED_DISPLAY_ONLY.tif")
    outline_path = out.qc / "95_A3_candidate_outlines_NUMBERED_DISPLAY_ONLY.tif"
    save_rgb_stack(full_path, full)
    save_rgb_stack(outline_path, header(after, candidate_title, offset))
    reviewer = pd.read_csv(params["review_cases_path"])
    reviewer["candidate_human_verdict"] = ""
    reviewer["candidate_notes"] = ""
    reviewer_path = out.out / "reviewer_labels.csv"
    reviewer.to_csv(reviewer_path, index=False)
    summary = {
        "attempt_id": params["attempt_id"],
        "frames": 99,
        "complete_field": True,
        "native_resolution_per_panel": [int(candidate.shape[1]),
                                         int(candidate.shape[2])],
        "fixed_movie_contrast": True,
        "identity_numbers": {"base": True, "candidate": True,
                             "change": False, "motion": True},
        "render_style": "established issue review style",
        "change_style": "grey candidate outlines; all changed pixels red; no numbers",
        "panels": ["accepted raw + outlines + numbers",
                   "candidate raw + outlines + numbers",
                   "red changed pixels + grey candidate outlines; no numbers",
                   "candidate outlines + numbers over aligned motion"],
        "changed_pixels": counts["pixels"],
        "changed_frames": counts["frames"],
        "automatic_review_eligibility": eligible,
        "human_review": "pending Jamie approval",
    }
    summary_path = out.out / "review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {
        "full_field_review": full_path, "candidate_outlines": outline_path,
        "reviewer_labels": reviewer_path, "review_summary": summary_path,
    }, "summary": summary}
