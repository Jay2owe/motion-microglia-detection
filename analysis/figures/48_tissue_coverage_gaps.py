"""Observed occupancy history and combined coverage within the ever-occupied tissue."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=48, slug="tissue-coverage-gaps",
    title="Gaps in tissue coverage over time",
    summary="Observed occupancy history and combined coverage within the ever-occupied tissue",
    grammar="vacancy age maps and coverage trace",
    reads=READS,
    panels=(Panel("spatial", spatial.field_map, min_width_inches=6., min_height_inches=6.),),
    options=options_for("gaps"),
)
def build(ctx):
    return build_spatial(ctx, "gaps")


if __name__ == "__main__":
    run_figure("tissue-coverage-gaps")

