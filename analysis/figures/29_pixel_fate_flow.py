"""Figure 29: pixel fate across recording stages.

Every pixel is classified at each stage - core, transient, never visited - and
the flows between classes are drawn. The sensitivity panel repeats the whole
thing at a second pair of cut-offs, because where the boundaries sit is a
choice and the figure should show what it costs.

    python analysis/figures/29_pixel_fate_flow.py <run>
    ... --stages 12              more time slices
    ... --thresholds 0.6,0.3     core and transient cut-offs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from _derive import _selected_identities
from _schema import FigureContext, FigureResult, Input, Option, Panel, figure, run_figure

from panels import common
from panels import territory as territory_panels


@figure(
    number=29,
    slug="pixel-fate-flow",
    summary="pixel fate across recording stages",
    title="Pixel fate across recording stages",
    reads=(Input("labels"),),
    panels=(
        Panel("flow", territory_panels.fate_flow, block=True,
              title="Pooled pixel fate"),
        Panel("shares", common.stacked_area,
              title="Pixel-fate composition through time"),
        Panel("sensitivity", territory_panels.fate_flow, block=True,
              title="Threshold-sensitivity comparison"),
    ),
    options=(
        Option("stages", default=8),
        Option("thresholds", default=[0.8, 0.2]),
        Option("cells", default="4", cast=str),
        Option("hour_ticks", default=24.0),
    ),
    grammar="alluvial flows and stacked shares",
)
def build(ctx: FigureContext) -> FigureResult:
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("pixel-fate-flow needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    n_stages = max(2, int(ctx.option("stages")))
    thresholds = ctx.option("thresholds")
    if len(thresholds) not in {2, 4}:
        raise SystemExit("--thresholds needs core,transient or primary_core,primary_transient,alternate_core,alternate_transient")
    primary = thresholds[:2]
    alternate = thresholds[2:] if len(thresholds) == 4 else [0.7, 0.3]
    for pair in (primary, alternate):
        if not (0 <= pair[1] < pair[0] <= 1):
            raise SystemExit("each --thresholds pair must satisfy 0 <= transient < core <= 1")
    bins = [np.asarray(values, dtype=int) for values in np.array_split(np.arange(labels.shape[0]), n_stages)]
    stage_hours = np.asarray([values.mean() * ctx.interval / 60 for values in bins])
    support = pd.DataFrame({"identity": [int(value) for value in np.unique(labels) if value]})
    support["observed_frames"] = support["identity"].map(lambda identity: int(np.count_nonzero(np.any(labels == identity, axis=(1, 2)))))
    selected = _selected_identities(support, ctx.option("cells"))
    class_labels = list(territory_panels.FATE_ROLES)
    class_index = {name: index for index, name in enumerate(class_labels)}

    def classify(identity: int, cuts: list[float]) -> np.ndarray:
        masks = labels > 0 if identity == 0 else labels == identity
        union = masks.any(axis=0)
        coordinates = np.flatnonzero(union)
        result = np.full((len(coordinates), n_stages), class_index["vacated"], dtype=int)
        previous_owner = np.zeros(len(coordinates), dtype=int)
        ever = np.zeros(len(coordinates), dtype=bool)
        for stage, indices in enumerate(bins):
            frequency = masks[indices].mean(axis=0).ravel()[coordinates]
            current = frequency > 0
            result[current & (frequency >= cuts[0]), stage] = class_index["core"]
            result[current & (frequency > cuts[1]) & (frequency < cuts[0]), stage] = class_index["fringe"]
            result[current & (frequency <= cuts[1]), stage] = class_index["transient"]
            result[~current & ~ever, stage] = class_index["vacated"]
            if identity == 0:
                owners = np.zeros(len(coordinates), dtype=int)
                flat = labels[indices].reshape(len(indices), -1)[:, coordinates]
                for pixel in range(flat.shape[1]):
                    nonzero = flat[:, pixel][flat[:, pixel] > 0]
                    if len(nonzero):
                        values, counts = np.unique(nonzero, return_counts=True)
                        owners[pixel] = int(values[np.argmax(counts)])
                transferred = current & (previous_owner > 0) & (owners > 0) & (owners != previous_owner)
                result[transferred, stage] = class_index["transferred"]
                previous_owner = np.where(owners > 0, owners, previous_owner)
            ever |= current
        return result

    primary_matrices = {identity: classify(identity, primary) for identity in [0, *selected]}
    alternate_matrix = classify(0, alternate)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    flow_tables = []
    if "flow" in axes:
        rect = tuple(axes["flow"].get_position().bounds); axes["flow"].remove()
        territory_panels.fate_flow(fig, rect, primary_matrices[0], ctx.theme,
                                   stage_labels=[f"{hour:.1f} h" for hour in stage_hours])
        axes["flow"] = fig.axes[-1]
        axes["flow"]._semantic_legend_handled = True
    for identity, matrix in primary_matrices.items():
        table = common.flow(fig, (0, 0, 0.001, 0.001), matrix, ctx.theme,
                            class_labels=class_labels,
                            class_roles=[territory_panels.FATE_ROLES[name] for name in class_labels],
                            stage_labels=[str(value) for value in range(n_stages)]).data
        plt.delaxes(fig.axes[-1])
        table.insert(0, "identity", identity)
        table["from_class"] = table["from_class"].map(dict(enumerate(class_labels)))
        table["to_class"] = table["to_class"].map(dict(enumerate(class_labels)))
        table = table.rename(columns={"count": "pixels"})
        flow_tables.append(table)
    figure_data = pd.concat(flow_tables, ignore_index=True)
    if "shares" in axes:
        matrix = primary_matrices[0]
        shares = pd.DataFrame({name: np.count_nonzero(matrix == code, axis=0)
                               for name, code in class_index.items()})
        common.stacked_area(
            axes["shares"], stage_hours, class_labels, shares, ctx.theme,
            looks={name: common.Look(colour=ctx.theme.colour(territory_panels.FATE_ROLES[name]))
                   for name in class_labels},
            labels=[territory_panels.FATE_LABELS[name] for name in class_labels],
            x_label="Hours from start of recording", y_label="Pooled pixels",
        )
        axes["shares"].set_xticks(ctx.theme.hour_ticks(float(stage_hours.min()), float(stage_hours.max()), ctx.hour_ticks))
        common.semantic_legend(axes["shares"], ctx.theme, location="inside")
        legend = axes["shares"].get_legend()
        if legend is not None:
            legend.remove()
        axes["shares"]._semantic_legend_handled = True
    sensitivity_data = pd.DataFrame()
    if "sensitivity" in axes:
        rect = tuple(axes["sensitivity"].get_position().bounds); axes["sensitivity"].remove()
        sensitivity_data = territory_panels.fate_flow(
            fig, rect, alternate_matrix, ctx.theme,
            stage_labels=[f"{hour:.1f} h" for hour in stage_hours],
        ).data
        axes["sensitivity"] = fig.axes[-1]
        axes["sensitivity"]._semantic_legend_handled = True
    if "flow" in axes or "sensitivity" in axes:
        fig.legend(
            handles=[Patch(facecolor=ctx.theme.colour(territory_panels.FATE_ROLES[name]))
                     for name in class_labels],
            labels=[territory_panels.FATE_LABELS[name] for name in class_labels],
            loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=len(class_labels),
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Pooled flow plus {len(selected)} identities across {n_stages} stages; threshold pairs "
                       f"{primary[0]:g}/{primary[1]:g} and {alternate[0]:g}/{alternate[1]:g}.",
        auxiliary={"sensitivity_flow.csv": sensitivity_data},
    )


if __name__ == "__main__":
    run_figure("pixel-fate-flow")
