"""Figure 33: shared behavioural programmes across cell sequences.

Cells whose state sequences match after time alignment are grouped. Warping is
capped, so two cells are only called similar if they did the same things in
roughly the same order at roughly the same rate.

    python analysis/figures/33_regime_programmes.py <run>
The distances come from ``sequence_distance.csv`` rather than being worked out
here. How much warping the alignment allows is a setting of that module, not of
this page, and is read back out of the run so the caption cannot describe a
window somebody has since changed.

    python analysis/figures/33_regime_programmes.py <run>
    ... --clusters 6             how many groups are cut from the tree
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd

from _derive import _regime_labels, _regime_rows
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import coupling as coupling_panels
from panels import morphology as morphology_panels


@figure(
    number=33,
    slug="regime-programmes",
    summary="shared behavioural programmes across cell sequences",
    title="Shared behavioural programmes across cell sequences",
    reads=(Table("cell_frame.csv", module="regimes"),
           Table("regime_profiles.csv", module="regimes"),
           Table("sequence_distance.csv", module="sequence_distance")),
    panels=(
        Panel("similarity", coupling_panels.similarity_matrix,
              title="Sequence mismatch between cells"),
        Panel("tree", common.dendrogram, title="Programme grouping"),
        Panel("groups", morphology_panels.regime_ribbon,
              title="State sequences grouped by programme"),
    ),
    options=(
        # How many groups to cut the tree into is a drawing choice and stays
        # here. How much warping the alignment allows is a method setting and
        # belongs to the `sequence_distance` module, which is where it went.
        Option("clusters", default=4),
        Option("hour_ticks", default=24.0),
    ),
    grammar="sequence distance matrix tree and grouped regime ribbon",
)
def build(ctx: FigureContext) -> FigureResult:
    regimes = _regime_rows(ctx.table("cell_frame.csv"))
    regime_names = _regime_labels(ctx.table("regime_profiles.csv"))
    pairs = ctx.table("sequence_distance.csv")
    window_hours = float(ctx.module_params("sequence_distance")["window_hours"])
    identities = sorted(set(pairs["identity_a"].astype(int))
                        | set(pairs["identity_b"].astype(int)))
    index = {identity: position for position, identity in enumerate(identities)}
    expected = len(identities) * (len(identities) - 1) // 2
    if len(pairs) != expected:
        # A missing pair would read as a distance of zero, which the tree would
        # take for two identical cells. That is the one way this page can draw a
        # confident wrong answer, so it refuses instead.
        raise SystemExit(
            f"sequence_distance.csv holds {len(pairs)} pairs for {len(identities)} cells, "
            f"not the {expected} every pair would make. The run filtered pairs out - "
            f"check min_overlap_steps in the sequence_distance settings - and a pair "
            f"this page cannot see would be drawn as two cells at no distance at all."
        )
    distances = np.zeros((len(identities), len(identities)), dtype=float)
    for row in pairs.itertuples():
        left_index, right_index = index[int(row.identity_a)], index[int(row.identity_b)]
        distances[left_index, right_index] = distances[right_index, left_index] = row.dtw_distance
    data = pairs[["identity_a", "identity_b", "dtw_distance",
                  "alignment_offset_hours", "alignment_overlap_steps"]].copy()
    if len(identities) > 1:
        tree = linkage(squareform(distances, checks=False), method="average")
        cluster_count = max(1, min(int(ctx.option("clusters")), len(identities)))
        groups = fcluster(tree, cluster_count, criterion="maxclust")
        leaf_order = leaves_list(tree)
    else:
        tree = np.empty((0, 4)); groups = np.ones(len(identities), dtype=int); leaf_order = np.arange(len(identities))
    cluster_map = dict(zip(identities, groups.astype(int)))
    if not data.empty:
        data["cluster_a"] = data["identity_a"].map(cluster_map)
        data["cluster_b"] = data["identity_b"].map(cluster_map)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    display_order = sorted(range(len(identities)), key=lambda index: (groups[index],
                                                                      list(leaf_order).index(index)))
    if "similarity" in axes:
        similarity_handle = coupling_panels.similarity_matrix(
            axes["similarity"], distances, ctx.theme,
            labels=["" for _ in identities], order=display_order,
            cmap=ctx.theme["sequential_cmap"], diagonal_mask=True,
            x_label="Cell regime sequence", y_label="Cells",
        ).extra["handle"]
        common.inset_colour_bar(
            axes["similarity"], similarity_handle, ctx.theme,
            label="Sequence mismatch after time alignment",
        )
    if "tree" in axes and len(tree):
        common.dendrogram(axes["tree"], tree, ctx.theme, labels=None)
        axes["tree"].set_ylabel("Sequence mismatch")
    if "groups" in axes:
        pivot = regimes.pivot(index="identity", columns="hours", values="regime").reindex(identities)
        morphology_panels.regime_ribbon(axes["groups"], pivot.to_numpy(float), pivot.columns,
                                        ctx.theme, order=display_order, hour_ticks=ctx.hour_ticks,
                                        y_label="Cells", legend=False,
                                        regime_labels=regime_names)
        handles = [Patch(facecolor=ctx.theme.colour(morphology_panels.REGIME_ROLES[value]))
                   for value in sorted(regime_names)]
        axes["groups"]._semantic_legend_handled = True
        rect = tuple(axes["groups"].get_position().bounds)
        fig.legend(
            handles=handles,
            labels=[regime_names[value] for value in sorted(regime_names)],
            loc="center", bbox_to_anchor=(0.60, rect[1] + rect[3] + 0.055),
            ncol=2, frameon=False, fontsize=ctx.theme.size("caption"),
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=f"{len(identities)} regime sequences; warping is capped at {window_hours:g} h.",
        auxiliary={"sequence_distance_matrix.csv": pd.DataFrame(distances, index=identities, columns=identities)},
    )


if __name__ == "__main__":
    run_figure("regime-programmes")
