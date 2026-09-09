"""Compatibility launcher for the audit/review territory-anchoring page."""

from pathlib import Path
import runpy


REVIEW_COMPATIBILITY_LAUNCHER = True


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "review" / Path(__file__).name),
        run_name="__main__",
    )
