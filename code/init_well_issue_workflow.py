"""Scaffold the next numbered well and its A000 disruption issue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from well_workflow import find_motion_root, well_directory


ATTEMPT_HEADER = (
    "attempt_id,date,approaches,stage_runs,outputs_sha256,primary,controls,"
    "regressions,guardrails,elapsed_s,identical_to,review_status,decision,notes\n")
CASE_HEADER = (
    "case_id,case_type,mechanism,review_frame_start,review_frame_end,"
    "source_imagej_frame_start,source_imagej_frame_end,track_ref,identities,"
    "expected_identity,notes\n")


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def create_well(
    root: Path,
    *,
    order: int,
    stem: str,
    accepted_run: str,
    question: str,
) -> Path:
    root = Path(root).resolve()
    well = root / "issues" / well_directory(order, stem)
    if well.exists():
        raise FileExistsError(f"well already exists: {well}")
    issue = well / "ISSUE-001-highest-disruption-event"
    tuning = issue / "analysis/highest_disruption_event_tuning"
    round_dir = tuning / "rounds/R01-rank-current-disruptions"
    round_dir.mkdir(parents=True)

    write_new(well / "README.md", f"""# {stem} issue workflow

Accepted parent: `{accepted_run}`. The disruption catalogue is generated fresh
from {stem}; no earlier well's event list is reused.
""")
    write_new(well / "ISSUES.md", f"""# {stem} issue ledger

Accepted parent: `{accepted_run}`.

| ID | Visible problem | Status | Current reference | Next action |
|---|---|---|---|---|
| [ISSUE-001](ISSUE-001-highest-disruption-event/issue.md) | Highest disruption in the freshly scored accepted result. | Baselining | `analysis/highest_disruption_event_tuning` | Render and classify A000. |
""")
    write_new(issue / "issue.md", f"""# ISSUE-001 - Highest current disruption

- **Status:** Baselining
- **First seen in:** pending fresh {stem} detector run
- **Accepted baseline run:** `{accepted_run}`

## Reviewer's description

> {question}

## Acceptance

Correct the visible physical error with field-wide discovery and zero supplied
identities, tracks, frames, coordinates, events, regions or review cases. Protect
all earlier wells under `WORKFLOW_CONTRACT.md`.
""")
    write_new(issue / "attempts.csv", ATTEMPT_HEADER)
    write_new(issue / "review_cases.csv", CASE_HEADER + (
        f"{stem}-BASELINE-CONTROL,control,baseline_catalogue,1,1,3,3,,,,"
        "Neutral A000 baseline row; expand before testing a candidate.\n"))
    write_new(round_dir / "question.md", f"# Question\n\n> {question}\n")
    write_new(round_dir / "plan.md", """# Plan

Run A000 through `m1_accepted_snapshot`, `m2_disruption_score` and
`m3_persistence_score`; render the canonical label and event TIFFs; visually
classify the leading event; then freeze failures and controls before A001.
""")
    write_new(well / "workflow.json", json.dumps({
        "schema": "motion.well-workflow.v1",
        "well_order": order,
        "well": stem,
        "accepted_parent": accepted_run,
        "first_issue": str(issue.relative_to(root)),
        "status": "awaiting_a000",
    }, indent=2) + "\n")
    return well


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--order", type=int, required=True)
    result.add_argument("--stem", required=True)
    result.add_argument("--accepted-run", required=True)
    result.add_argument("--question", required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    project = find_motion_root(Path(__file__))
    created = create_well(
        project, order=arguments.order, stem=arguments.stem,
        accepted_run=arguments.accepted_run, question=arguments.question)
    print(created)
