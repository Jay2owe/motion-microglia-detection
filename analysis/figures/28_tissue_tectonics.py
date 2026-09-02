"""Figure 28: first ownership and contested tissue pixels.

Which cell reached each pixel first, and how many cells ever claimed it. A
pixel claimed by three identities over a recording is either a shared boundary
or a tracking failure, and the map is where the difference shows.

    python analysis/figures/28_tissue_tectonics.py <run>
    ... --min-contested 3        raise the bar for a contested pixel
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tifffile

from _derive import _scaled_field
from _schema import FigureContext, FigureResult, Input, Option, Panel, Stack, figure, run_figure

from panels import common
from panels import territory as territory_panels


@figure(
    number=28,
    slug="tissue-tectonics",
    summary="first ownership and contested tissue pixels",
    title="First ownership and contested tissue pixels",
    reads=(Stack("first_owner.tif", module="territory"),
           Stack("owner_count.tif", module="territory"),
           Input("labels")),
    panels=(
        Panel("ownership", territory_panels.ownership_map, title="First occupant"),
        Panel("contested", title="Cells per pixel"),
        Panel("handoffs", title="Identity handoffs"),
    ),
    options=(
        Option("min_contested", default=2),
    ),
    grammar="ownership and contested maps",
)
def build(ctx: FigureContext) -> FigureResult:
    first_owner = ctx.stack("first_owner.tif")
    owner_count = ctx.stack("owner_count.tif")
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("tissue-tectonics needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    identities = sorted(int(value) for value in np.unique(labels) if value)
    unions = {identity: np.any(labels == identity, axis=0) for identity in identities}
    rows, pairs = [], []
    for identity in identities:
        owned = first_owner == identity
        contested = unions[identity] & (owner_count > 1)
        partners = 0
        for other in identities:
            if other <= identity:
                continue
            overlap = int(np.count_nonzero(unions[identity] & unions[other]))
            if overlap:
                pairs.append({"identity_a": identity, "identity_b": other,
                              "shared_claimed_px": overlap})
                partners += 1
        partners += sum(1 for row in pairs if row["identity_b"] == identity)
        rows.append({"identity": identity, "owned_px": int(owned.sum()),
                     "exclusive_px": int(np.count_nonzero(unions[identity] & (owner_count == 1))),
                     "contested_px": int(contested.sum()),
                     "contested_share": float(contested.sum() / max(unions[identity].sum(), 1)),
                     "partners": partners})
    data = pd.DataFrame(rows)
    pair_data = pd.DataFrame(pairs, columns=["identity_a", "identity_b", "shared_claimed_px"])
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    # Field maps read best beside each other rather than as narrow stacked strips.
    fig.set_size_inches(*ctx.theme.canvas(13.8, 7.2), forward=True)
    gap = 0.045
    width = (0.84 - gap * (len(panels) - 1)) / len(panels)
    for position, name in enumerate(panels.keys):
        axes[name].set_position([0.10 + position * (width + gap), 0.20, width, 0.56])
    if "ownership" in axes:
        territory_panels.ownership_map(axes["ownership"], first_owner, owner_count,
                                       ctx.theme, field=_scaled_field(ctx))
        axes["ownership"]._semantic_legend_handled = True
    if "contested" in axes:
        minimum = max(2, int(ctx.option("min_contested")))
        contested = np.where(owner_count >= minimum, owner_count, np.nan)
        contested_handle = axes["contested"].imshow(
            contested, cmap=ctx.theme["sequential_cmap"], vmin=minimum,
            vmax=max(float(np.nanmax(contested)), float(minimum)),
            extent=(0, _scaled_field(ctx)["width"], _scaled_field(ctx)["height"], 0),
        )
        axes["contested"].set_aspect("equal")
        common.inset_colour_bar(axes["contested"], contested_handle, ctx.theme,
                                label="Cells that occupied this pixel")
    if "handoffs" in axes:
        axes["handoffs"].text(0.5, 0.5, "No recorded handoff map\nin the analysis manifest",
                              ha="center", va="center", transform=axes["handoffs"].transAxes)
        axes["handoffs"].set_axis_off()
    if "ownership" in axes:
        fig.legend(
            handles=[
                Patch(facecolor=ctx.theme.colour("morphology")),
                Patch(facecolor=ctx.theme.colour("highlight")),
                Patch(facecolor=ctx.theme.colour("missing")),
            ],
            labels=[
                "Hue separates first occupants (not an identity key)",
                "Occupied by more than one cell",
                "Never occupied",
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3,
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle="Recorded handoffs are not present in this image-only analysis manifest; the comparison is left explicit.",
        auxiliary={"contested_pairs.csv": pair_data},
    )


if __name__ == "__main__":
    run_figure("tissue-tectonics")
