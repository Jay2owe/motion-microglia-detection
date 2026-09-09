"""Measured temporal patterns ordered by physical cell position rather than fitted phase."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=45, slug="spatial-rhythm-progression",
    title="Rhythm progression across space",
    summary="Measured temporal patterns ordered by physical cell position rather than fitted phase",
    grammar="spatially ordered cell time matrix",
    reads=READS,
    panels=(Panel("spatial", spatial.spatial_matrix, min_width_inches=6., min_height_inches=6.),),
    options=options_for("progression"),
)
def build(ctx):
    return build_spatial(ctx, "progression")


if __name__ == "__main__":
    run_figure("spatial-rhythm-progression")

