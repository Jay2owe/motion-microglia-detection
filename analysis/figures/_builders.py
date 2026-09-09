"""Locate canonical figure builders without importing plotting libraries."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review"
COMPATIBILITY_MARKER = "REVIEW_COMPATIBILITY_LAUNCHER"


def builder_files() -> list[Path]:
    """All result and review builders, ordered by their figure number.

    Root-level numbered files carrying ``COMPATIBILITY_MARKER`` preserve old
    commands after a diagnostic builder moves into ``review/``; they are
    launchers, not a second copy of the figure.
    """
    candidates = [*HERE.glob("[0-9][0-9]_*.py"),
                  *REVIEW.glob("[0-9][0-9]_*.py")]
    canonical = [path for path in candidates
                 if COMPATIBILITY_MARKER not in path.read_text(encoding="utf-8")]
    return sorted(canonical, key=lambda path: (int(path.name[:2]), path.as_posix()))


def is_review_builder(path: Path) -> bool:
    """Whether a canonical builder belongs to the audit/review module."""
    return path.resolve().parent == REVIEW.resolve()
