"""Figure 33: cells grouped by their per-frame size-and-movement states.

Each cell contributes a timeline of the nine low/medium/high size crossed with
low/medium/high movement states. Cells whose timelines match after limited
time alignment are grouped. The alignment is capped, so two cells are only
called similar if they occupied the same states in roughly the same order at
roughly the same rate.

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
    summary="cells grouped by aligned size-and-movement timelines",
    title="Cells grouped by similarity of their size-and-movement timelines",
    reads=(Table("cell_frame.csv", module="regimes"),
           Table("regime_profiles.csv", module="regimes"),
           Table("sequence_distance.csv", module="sequence_distance")),
    panels=(
        Panel("similarity", coupling_panels.similarity_matrix,
              min_width_inches=7.5, min_height_inches=7.5,
              title="Fraction of aligned steps assigned different states"),
        Panel("tree", common.dendrogram,
              min_width_inches=11.0, min_height_inches=7.0,
              title="Grouping tree from the different-state fraction"),
        Panel("groups", morphology_panels.regime_ribbon,
              min_width_inches=11.0, min_height_inches=8.0,
              title="Per-frame states, ordered by similarity group"),
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
                  "alignment_offset_hours", "alignment_overlap_steps"]].copy().rename(
        columns={
            "dtw_distance": "different_state_fraction",
            "alignment_offset_hours": "mean_alignment_offset_hours",
            "alignment_overlap_steps": "aligned_steps_with_both_cells_observed",
        }
    )
    if len(identities) > 1:
        tree = linkage(squareform(distances, checks=False), method="average")
        cluster_count = max(1, min(int(ctx.option("clusters")), len(identities)))
        groups = fcluster(tree, cluster_count, criterion="maxclust")
        leaf_order = leaves_list(tree)
    else:
        tree = np.empty((0, 4)); groups = np.ones(len(identities), dtype=int); leaf_order = np.arange(len(identities))
    cluster_map = dict(zip(identities, groups.astype(int)))
    if not data.empty:
        data["similarity_group_a"] = data["identity_a"].map(cluster_map)
        data["similarity_group_b"] = data["identity_b"].map(cluster_map)
    panels = ctx.panels()
    fig, axes = ctx.layout(
        panels, top_inches=2.15, bottom_inches=2.2, right_inches=5.0,
        gap_inches=2.1,
    )
    display_order = sorted(range(len(identities)), key=lambda index: (groups[index],
                                                                      list(leaf_order).index(index)))
    ordered_identities = [identities[position] for position in display_order]
    ordered_groups = groups[np.asarray(display_order, dtype=int)]
    if "similarity" in axes:
        similarity_handle = coupling_panels.similarity_matrix(
            axes["similarity"], distances, ctx.theme,
            labels=[str(identity) for identity in identities], order=display_order,
            cmap=ctx.theme["sequential_cmap"], diagonal_mask=False,
            x_label="Cell identity", y_label="Cell identity",
        ).extra["handle"]
        similarity_handle.set_clim(0, 1)
        tick_step = max(1, int(np.ceil(len(identities) / 9)))
        tick_positions = np.arange(0, len(identities), tick_step)
        tick_labels = [str(ordered_identities[position]) for position in tick_positions]
        axes["similarity"].set_xticks(tick_positions, labels=tick_labels, rotation=0)
        axes["similarity"].set_yticks(tick_positions, labels=tick_labels)
        axes["similarity"].set_aspect("equal")
        common.inset_colour_bar(
            axes["similarity"], similarity_handle, ctx.theme,
            label="Different aligned states (fraction)",
        )
    if "tree" in axes and len(tree):
        common.dendrogram(
            axes["tree"], tree, ctx.theme, labels=None, orientation="right",
        )
        axes["tree"].set_xlabel("Different aligned states (fraction)")
        axes["tree"].set_ylabel("Cells")
        axes["tree"].set_xlim(left=0)
    if "groups" in axes:
        pivot = regimes.pivot(index="identity", columns="hours", values="regime").reindex(identities)
        morphology_panels.regime_ribbon(axes["groups"], pivot.to_numpy(float), pivot.columns,
                                        ctx.theme, order=display_order, hour_ticks=ctx.hour_ticks,
                                        y_label="Similarity group", legend=False,
                                        regime_labels=regime_names)
        group_centres = []
        group_labels = []
        for group in dict.fromkeys(int(value) for value in ordered_groups):
            positions = np.flatnonzero(ordered_groups == group)
            group_centres.append(float(positions.mean()))
            group_labels.append(f"Group {group} (n={len(positions)})")
            if positions[-1] < len(ordered_groups) - 1:
                axes["groups"].axhline(
                    float(positions[-1]) + 0.5,
                    color=ctx.theme.colour("page"),
                    linewidth=ctx.theme.stroke("emphasis"),
                )
        axes["groups"].set_yticks(group_centres, labels=group_labels)
        handles = [Patch(facecolor=morphology_panels.regime_colour(value, ctx.theme))
                   for value in sorted(regime_names)]
        common.semantic_legend(
            axes["groups"], ctx.theme,
            handles=handles,
            labels=[regime_names[value] for value in sorted(regime_names)],
            location="right", columns=1,
        )
    group_assignments = pd.DataFrame({
        "identity": identities,
        "similarity_group": groups.astype(int),
    })
    group_assignments["display_order"] = group_assignments["identity"].map(
        {identity: position for position, identity in enumerate(ordered_identities)}
    )
    distance_matrix = pd.DataFrame(distances, index=identities, columns=identities)
    distance_matrix.index.name = "cell_identity"
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=(
            f"{len(identities)} cell timelines arranged into {len(set(groups))} descriptive "
            f"groups; pairwise alignment may move by at most {window_hours:g} h."
        ),
        footnote=(
            "A per-frame state is current size (low, medium or high) crossed with movement "
            "into that frame (low, medium or high). Different-state fraction is the lowest "
            "fraction of aligned steps with unequal states after limited timeline sliding "
            "or stretching; 0 = identical and 1 = different throughout. White means no "
            "classified observation."
        ),
        auxiliary={
            "sequence_distance_matrix.csv": distance_matrix,
            "cell_similarity_groups.csv": group_assignments,
        },
    )


if __name__ == "__main__":
    run_figure("regime-programmes")
