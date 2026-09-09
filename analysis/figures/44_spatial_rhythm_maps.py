"""Broad-search period estimates and eligible peak timing mapped to measured cell positions."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=44, slug="spatial-rhythm-maps",
    title="Rhythm periods and peak timing across the tissue",
    summary="Broad-search period estimates and eligible peak timing mapped to measured cell positions",
    grammar="spatial rhythm parameter maps",
    reads=READS,
    panels=(Panel("spatial", spatial.cell_map, min_width_inches=6., min_height_inches=6.),),
    options=options_for("period"),
)
def build(ctx):
    return build_spatial(ctx, "period")


if __name__ == "__main__":
    run_figure("spatial-rhythm-maps")

