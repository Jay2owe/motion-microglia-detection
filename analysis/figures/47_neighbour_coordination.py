"""Within-recording association between cell proximity and simultaneous metric changes."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=47, slug="neighbour-coordination",
    title="Neighbour coordination across the tissue",
    summary="Within-recording association between cell proximity and simultaneous metric changes",
    grammar="spatial connections and permutation distribution",
    reads=READS,
    panels=(Panel("spatial", spatial.connection_map, min_width_inches=6., min_height_inches=6.),),
    options=options_for("neighbours"),
)
def build(ctx):
    return build_spatial(ctx, "neighbours")


if __name__ == "__main__":
    run_figure("neighbour-coordination")

