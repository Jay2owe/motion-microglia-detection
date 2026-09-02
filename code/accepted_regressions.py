"""Dataset-local acceptance checks kept outside label production.

The production pipeline must not import this module. Named identities, approved arrays,
review totals, and historic approval records belong in a validation contract supplied
to this scorer, never in label-calculation parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from common import sha256
from global_soma_ledger import substantial_identity_conflicts
from model_review import score_review_gold_standard


def review_counts(score: pd.DataFrame, gold: pd.DataFrame) -> dict[str, int]:
    """Return the four historic review totals without embedding expected values."""
    under = score[score.truth == "undersegmentation"]
    two_ranks = set(gold.loc[gold.true_cell_count == 2, "review_rank"])
    return {
        "reviewed_two_cell_separated": int(
            under[under.review_rank.isin(two_ranks)].cores_separated.sum()),
        "undersegmentation_separated": int(under.cores_separated.sum()),
        "single_cell_gate_passed": int(
            score[score.truth == "single_cell"].gate_passed.sum()),
        "uncertain_separated": int(
            score[score.truth == "uncertain"].cores_separated.sum()),
    }


def _row(check: str, observed, expected) -> dict:
    return {
        "check": check,
        "observed": str(observed),
        "expected": str(expected),
        "passed": bool(observed == expected),
    }


def review_count_checks(score: pd.DataFrame, gold: pd.DataFrame,
                        expected: dict[str, int]) -> list[dict]:
    """Compare calculated review totals to a caller-owned validation contract."""
    observed = review_counts(score, gold)
    return [_row(name, observed[name], int(value))
            for name, value in expected.items()]


def array_contract_checks(labels: np.ndarray, baseline: np.ndarray,
                          approved: np.ndarray) -> list[dict]:
    """Check approved-array equality and foreground preservation."""
    return [
        _row("candidate_array_exact", np.array_equal(labels, approved), True),
        _row("foreground_support_exact",
             np.array_equal(labels > 0, baseline > 0), True),
    ]


def identity_contract_checks(labels: np.ndarray, spec: dict) -> list[dict]:
    """Score caller-supplied identity counts, windows, and exclusions."""
    identities = set(map(int, np.unique(labels))) - {0}
    rows = [_row("active_identity_count", len(identities),
                 int(spec["expected_active_identities"]))]
    for window in spec.get("required_identity_windows", []):
        first = int(window["first_frame"])
        last = int(window["last_frame"])
        identity = int(window["identity"])
        observed = int(np.count_nonzero(np.any(
            labels[first - 1:last] == identity, axis=(1, 2))))
        rows.append(_row(
            f"identity_{identity}_present_frames_{first}_{last}",
            observed, last - first + 1))
    for identity_value in spec.get("forbidden_identities", []):
        identity = int(identity_value)
        rows.append(_row(f"identity_{identity}_present",
                         bool(np.any(labels == identity)), False))
    return rows


def _paths(project: Path, spec: dict, producer_out: Path | None,
           producer_mid: Path | None) -> tuple[Path, Path]:
    return (
        producer_out or project / spec["producer_out"],
        producer_mid or project / spec["producer_mid"],
    )


def score_stage(project: Path, stage: str, spec: dict,
                producer_out: Path | None = None,
                producer_mid: Path | None = None) -> dict:
    """Score one producer output using only its external validation contract."""
    out_dir, mid_dir = _paths(project, spec, producer_out, producer_mid)
    labels_path = out_dir / "95_A3.tif"
    labels = tifffile.imread(labels_path)
    baseline = tifffile.imread(project / spec["baseline_labels"])
    approved_path = project / spec["approved_candidate"]
    approved = tifffile.imread(approved_path)
    rows = [
        _row("approved_candidate_sha256", sha256(approved_path),
             spec["approved_candidate_sha256"]),
        _row("candidate_file_sha256", sha256(labels_path),
             spec["approved_candidate_sha256"]),
    ]
    rows.extend(array_contract_checks(labels, baseline, approved))

    gold = pd.read_csv(project / spec["review_benchmark"])
    review_score = score_review_gold_standard(labels, gold)
    rows.extend(review_count_checks(
        review_score, gold, spec["expected_review_counts"]))
    conflicts = substantial_identity_conflicts(
        labels, int(spec["minimum_substantial_core_px"]))
    rows.append(_row("substantial_identity_conflicts", len(conflicts), 0))
    rows.extend(identity_contract_checks(labels, spec))

    if stage == "layered_lineage":
        reviewer = pd.read_csv(project / spec["approved_review_labels"])
        verdicts = set(reviewer.verdict.astype(str).str.lower())
        reviewer_pass = (len(reviewer) == int(spec["reviewer_rows"])
                         and verdicts == {spec["reviewer_verdict"]})
        rows.append(_row("approved_reviewer_rows_and_verdict",
                         reviewer_pass, True))
        for check, actual_name, expected_key in (
                ("shape_source_array_exact", "95_A3_shape_source.tif", "expected_shape_source"),
                ("comparison_array_exact", "95_A3_attempt005_comparison.tif", "expected_comparison"),
                ("stable_intermediate_array_exact", "95_A3_red_led_stable.tif", "expected_stable_intermediate")):
            actual = tifffile.imread(mid_dir / actual_name)
            expected = tifffile.imread(project / spec[expected_key])
            rows.append(_row(check, np.array_equal(actual, expected), True))
        reviewer_output = reviewer
    elif stage == "oscillatory_lineage":
        approval = (project / spec["approval_record"]).read_text(encoding="utf-8")
        rows.append(_row("approval_record_phrases",
                         all(phrase in approval
                             for phrase in spec["approval_phrases"]), True))
        reviewer = pd.read_csv(project / spec["approved_review_labels"])
        reviewer_output = reviewer[
            reviewer.variant.astype(str).eq(spec["approved_variant"])
            & reviewer.verdict.astype(str).str.lower().eq("approved")]
        rows.append(_row("one_approved_variant_row", len(reviewer_output), 1))
        for filename, expected in spec["expected_decision_rows"].items():
            rows.append(_row(f"{filename}_rows",
                             len(pd.read_csv(out_dir / filename)), int(expected)))
    else:
        raise ValueError(f"unknown acceptance stage: {stage}")

    checks = pd.DataFrame(rows)
    checks.insert(0, "stage", stage)
    return {
        "checks": checks,
        "review_score": review_score,
        "reviewer_verdicts": reviewer_output,
        "conflicts": conflicts,
        "labels_sha256": sha256(labels_path),
    }


def score_accepted_lineage_outputs(
        project_root: Path, contract_path: Path, output_root: Path,
        producer_paths: dict[str, dict[str, Path]] | None = None) -> dict:
    """Score both accepted lineage stages and write relocated validation artifacts."""
    project = Path(project_root)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    producer_paths = producer_paths or {}
    results = {}
    tables = []
    for stage in ("layered_lineage", "oscillatory_lineage"):
        overrides = producer_paths.get(stage, {})
        result = score_stage(
            project, stage, contract[stage],
            overrides.get("out"), overrides.get("mid"))
        stage_out = output_root / stage
        stage_out.mkdir(parents=True, exist_ok=True)
        result["checks"].to_csv(stage_out / "validation_checks.csv", index=False)
        result["review_score"].to_csv(
            stage_out / "review_gold_standard_score.csv", index=False)
        reviewer_name = ("approved_reviewer_verdicts.csv"
                         if stage == "layered_lineage"
                         else "approved_reviewer_verdict.csv")
        result["reviewer_verdicts"].to_csv(stage_out / reviewer_name, index=False)
        result["conflicts"].to_csv(
            stage_out / "co_present_soma_identity_conflicts.csv", index=False)
        tables.append(result["checks"])
        results[stage] = result
    checks = pd.concat(tables, ignore_index=True)
    checks.to_csv(output_root / "validation_checks.csv", index=False)
    failed = checks.loc[~checks.passed]
    if len(failed):
        raise AssertionError(
            "acceptance regression failed: " + ", ".join(failed.check))
    return {
        "checks": checks,
        "validation_checks": int(len(checks)),
        "all_validation_checks_pass": True,
        "layered_labels_sha256": results["layered_lineage"]["labels_sha256"],
        "oscillatory_labels_sha256": results["oscillatory_lineage"]["labels_sha256"],
    }
