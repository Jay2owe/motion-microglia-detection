"""The spatial distribution of within-cell metric changes across actual recording frames."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from _schema import Panel, figure, run_figure
from _spatial import READS, options_for, build as build_spatial
from panels import spatial


@figure(
    number=43, slug="tissue-expansion-sequence",
    title="Within-cell changes across the tissue",
    summary="The spatial distribution of within-cell metric changes across actual recording frames",
    grammar="small multiples spatial maps",
    reads=READS,
    panels=(Panel("spatial", spatial.field_map, min_width_inches=6., min_height_inches=6.),),
    options=options_for("expansion"),
)
def build(ctx):
    return build_spatial(ctx, "expansion")


if __name__ == "__main__":
    run_figure("tissue-expansion-sequence")
