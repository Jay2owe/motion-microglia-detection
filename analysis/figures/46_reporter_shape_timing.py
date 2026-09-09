"""Relative peak timing between two metrics for cells with compatible supported rhythms."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=46, slug="reporter-shape-timing",
    title="Reporter-to-shape timing across the tissue",
    summary="Relative peak timing between two metrics for cells with compatible supported rhythms",
    grammar="paired metric timing map",
    reads=READS,
    panels=(Panel("spatial", spatial.cell_map, min_width_inches=6., min_height_inches=6.),),
    options=options_for("timing"),
)
def build(ctx):
    return build_spatial(ctx, "timing")


if __name__ == "__main__":
    run_figure("reporter-shape-timing")

