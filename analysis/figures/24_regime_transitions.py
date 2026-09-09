"""Figure 24: observed behavioural regime transitions against shuffled time.

Which state follows which, beside the same table with time shuffled. The
shuffle is what says whether the observed structure is more than the states'
own frequencies.

    python analysis/figures/24_regime_transitions.py <run>
    ... --shuffles 1000          replicates behind the comparison
    ... --normalise none         raw counts rather than row shares
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _derive import _regime_labels
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import morphology as morphology_panels


@figure(
    number=24,
    slug="regime-transitions",
    summary="observed behavioural regime transitions against shuffled time",
    title="Observed behavioural regime transitions against shuffled time",
    reads=(Table("regime_transitions.csv", module="regimes"),
           Table("regime_profiles.csv", module="regimes")),
    panels=(
        Panel("matrix", morphology_panels.regime_transitions,
              min_width_inches=5.5, min_height_inches=5.5,
              title="Observed next state"),
        Panel("null", common.matrix, min_width_inches=5.5, min_height_inches=5.5,
              title="Time-shuffled comparison"),
        Panel("dwell", morphology_panels.dwell_times,
              title="Time spent in each state"),
    ),
    options=(
        Option("shuffles", default=200),
        Option("normalise", default="row", cast=str),
    ),
    grammar="transition matrices and dwell histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    transitions = ctx.table("regime_transitions.csv")
    regime_names = _regime_labels(ctx.table("regime_profiles.csv"))
    regimes = sorted(set(transitions["from_regime"]) | set(transitions["to_regime"]))
    n = max(regimes) + 1 if regimes else 0
    counts = np.zeros((n, n), dtype=int)
    for _, row in transitions.iterrows():
        counts[int(row["from_regime"]), int(row["to_regime"])] += 1
    shuffles = int(ctx.option("shuffles"))
    generator = np.random.default_rng(20260825)
    null = np.zeros((shuffles, n, n), dtype=float)
    source = transitions["from_regime"].to_numpy(int)
    target = transitions["to_regime"].to_numpy(int)
    for replicate in range(shuffles):
        shuffled = generator.permutation(target)
        for left, right in zip(source, shuffled):
            null[replicate, left, right] += 1
        totals = null[replicate].sum(axis=1, keepdims=True)
        null[replicate] = np.divide(null[replicate], totals, out=np.zeros_like(null[replicate]), where=totals != 0)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    null_mean = null.mean(axis=0) if shuffles else np.zeros_like(fractions)
    null_lo = np.percentile(null, 2.5, axis=0) if shuffles else np.zeros_like(fractions)
    null_hi = np.percentile(null, 97.5, axis=0) if shuffles else np.zeros_like(fractions)
    rows = []
    for left in range(n):
        for right in range(n):
            rows.append({"from_regime": left, "to_regime": right, "count": int(counts[left, right]),
                         "row_fraction": fractions[left, right], "null_fraction": null_mean[left, right],
                         "null_lo": null_lo[left, right], "null_hi": null_hi[left, right],
                         "excess": fractions[left, right] - null_mean[left, right]})
    figure_data = pd.DataFrame(rows)
    panels = ctx.panels()
    normalise = ctx.option("normalise")
    mask_diagonal = not (panels.keys == ["matrix"] and normalise == "none")
    if panels.keys == ["matrix", "null", "dwell"]:
        fig, axes = ctx.grid_layout(
            panels,
            {
                "matrix": (0, 0, 1, 1),
                "null": (0, 1, 1, 1),
                "dwell": (1, 0, 1, 2),
            },
        )
    else:
        fig, axes = ctx.layout(panels)
    labels = [regime_names.get(index, f"State {index}") for index in range(n)]
    column_labels = [str(index) for index in range(n)]
    if "matrix" in axes:
        morphology_panels.regime_transitions(axes["matrix"], counts, ctx.theme,
                                              labels=labels, column_labels=column_labels,
                                              normalise=normalise,
                                              diagonal_mask=mask_diagonal)
    if "null" in axes:
        common.matrix(axes["null"], null_mean, ctx.theme,
                      row_labels=[str(index) for index in range(n)],
                      column_labels=column_labels, cmap=ctx.theme["sequential_cmap"],
                      x_label="Shuffled to state", y_label="Shuffled from state",
                      diagonal_mask=mask_diagonal)
    if "dwell" in axes:
        dwell = transitions.drop_duplicates("dwell_id") if "dwell_id" in transitions else transitions
        morphology_panels.dwell_times(axes["dwell"], dwell, ctx.theme,
                                      regime_ids=regimes, minutes_per_frame=ctx.interval,
                                      regime_labels=regime_names)
        common.semantic_legend(axes["dwell"], ctx.theme, location="inside", columns=2)
        axes["dwell"].set_title("", loc="left")
    matrix_handles = [axes[name].images[0] for name in ("matrix", "null")
                      if name in axes and axes[name].images]
    if matrix_handles:
        scale_max = max(float(np.nanmax(fractions)), float(np.nanmax(null_mean)), 1e-12)
        for handle in matrix_handles:
            handle.set_clim(0, scale_max)
        key_axis = axes["null"] if "null" in axes else axes["matrix"]
        common.inset_colour_bar(
            key_axis, matrix_handles[-1], ctx.theme,
            label="Share of transitions" if normalise != "none" else "Transitions",
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Observed consecutive transitions against {shuffles} within-table target shuffles.",
        auxiliary={"dwell_runs.csv": transitions.drop_duplicates("dwell_id")
                                  if "dwell_id" in transitions else transitions},
    )


if __name__ == "__main__":
    run_figure("regime-transitions")
