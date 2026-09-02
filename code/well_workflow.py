"""Shared path and earlier-well gates for per-recording issue workflows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


WELL_DIRECTORY = re.compile(r"^(?P<order>\d{2})_(?P<stem>[A-Za-z0-9][A-Za-z0-9_-]*)$")


@dataclass(frozen=True)
class ReferenceImpact:
    classification: str
    changed_fraction_of_named_pixels: float
    arithmetic_mean_persistence_delta: float
    pooled_persistence_delta: float
    disruption_burden_fraction_delta: float
    high_disruption_event_delta: int
    internal_gap_frames_delta: int
    strong_benefit: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def find_motion_root(start: Path) -> Path:
    """Find this project without depending on issue-folder depth."""
    path = Path(start).resolve()
    candidates = (path, *path.parents) if path.is_dir() else path.parents
    for candidate in candidates:
        if (candidate / "WORKFLOW_CONTRACT.md").is_file():
            return candidate
    raise FileNotFoundError(f"No Motion WORKFLOW_CONTRACT.md above {start}")


def well_directory(order: int, stem: str) -> str:
    if not 0 < order < 100:
        raise ValueError("well order must be between 1 and 99")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stem):
        raise ValueError(f"invalid recording stem: {stem!r}")
    return f"{order:02d}_{stem}"


def parse_well_directory(name: str) -> tuple[int, str]:
    match = WELL_DIRECTORY.fullmatch(name)
    if not match:
        raise ValueError(f"invalid well directory: {name!r}")
    return int(match.group("order")), match.group("stem")


def relocate_legacy_issue_path(path: str | Path) -> Path:
    """Map the pre-well A3 issue path to its canonical relocated path."""
    text = str(path)
    text = text.replace("issues/ISSUE-", "issues/01_95_A3/ISSUE-")
    text = text.replace("issues\\ISSUE-", "issues\\01_95_A3\\ISSUE-")
    return Path(text)


def classify_reference_impact(
    *,
    changed_fraction_of_named_pixels: float,
    arithmetic_mean_persistence_delta: float,
    pooled_persistence_delta: float,
    disruption_burden_fraction_delta: float,
    high_disruption_event_delta: int,
    internal_gap_frames_delta: int,
    byte_identical: bool = False,
) -> ReferenceImpact:
    """Classify a later-well rule's impact on a protected earlier well."""
    values = (
        changed_fraction_of_named_pixels,
        arithmetic_mean_persistence_delta,
        pooled_persistence_delta,
        disruption_burden_fraction_delta,
    )
    if any(not isinstance(value, (int, float)) for value in values):
        raise TypeError("reference-impact metrics must be numeric")
    if not 0.0 <= changed_fraction_of_named_pixels <= 1.0:
        raise ValueError("changed named-pixel fraction must be between 0 and 1")

    if byte_identical:
        return ReferenceImpact(
            "unchanged", changed_fraction_of_named_pixels,
            arithmetic_mean_persistence_delta, pooled_persistence_delta,
            disruption_burden_fraction_delta, high_disruption_event_delta,
            internal_gap_frames_delta, False, ("labels and ledger are byte-identical",))

    failures: list[str] = []
    if arithmetic_mean_persistence_delta < -0.002:
        failures.append("arithmetic-mean persistence fell by more than 0.002")
    if pooled_persistence_delta < -0.002:
        failures.append("pooled persistence fell by more than 0.002")
    if disruption_burden_fraction_delta > 0.01:
        failures.append("disruption burden rose by more than 1%")
    if high_disruption_event_delta > 0:
        failures.append("high-disruption event count increased")
    if internal_gap_frames_delta > 1:
        failures.append("internal gap frames increased by more than one")

    strong_benefit = (
        arithmetic_mean_persistence_delta >= 0.005
        or pooled_persistence_delta >= 0.005
        or disruption_burden_fraction_delta <= -0.05
    )
    if failures:
        classification = "blocked"
    elif changed_fraction_of_named_pixels <= 0.02:
        classification = "limited_pass"
    elif strong_benefit:
        classification = "material_benefit_review"
    else:
        classification = "blocked"
        failures.append("material A3 change has no strong measured benefit")

    return ReferenceImpact(
        classification, changed_fraction_of_named_pixels,
        arithmetic_mean_persistence_delta, pooled_persistence_delta,
        disruption_burden_fraction_delta, high_disruption_event_delta,
        internal_gap_frames_delta, strong_benefit, tuple(failures),
    )
