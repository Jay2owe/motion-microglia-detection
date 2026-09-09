"""Figure 32: when cells return to a previous measured state.

Every observation time is compared with every other observation of the same
cell. Nearby multimetric state vectors count as a return; diagonal runs show
that a returned state continued into a repeated sequence.

The numbers are read from ``recurrence.csv`` and ``recurrence_quantification.csv``
rather than worked out here. One thing is still computed on this page and only
one: the wall itself, which is a 99 x 99 distance matrix per cell and exists to
be looked at. Writing that to a table would be writing a picture into a CSV.

Everything about *how* recurrence is measured - the state vector, the threshold,
the surrogate count - is a setting of the ``recurrence`` module and is read back
out of the run, so the wall is built with exactly what the table was measured
with. It cannot draw a wall made one way beside a determinism ranking made
another.

    python analysis/figures/32_recurrence_wall.py <run>
    ... --cells 9                how many walls are drawn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.spatial.distance import cdist
import numpy as np

from _derive import _selected_identities
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from analysis.modules.recurrence import DEFAULTS as RECURRENCE_DEFAULTS, state_matrix

from panels import common
from panels import coupling as coupling_panels


@figure(
    number=32,
    slug="recurrence-wall",
    summary="when cells return to a previous measured state",
    title="When cells return to a previous measured state",
    reads=(Table("cell_frame.csv", module="coupling"),
           Table("recurrence.csv", module="recurrence"),
           Table("recurrence_quantification.csv", module="recurrence")),
    panels=(
        Panel("wall", coupling_panels.recurrence,
              item_width_inches=3.0, item_height_inches=3.0,
              title="Every observation time compared with every other time"),
        Panel("rate", common.trace, min_width_inches=11.0, min_height_inches=5.0,
              title="How often a state returns after each time separation"),
        Panel("quantified", common.dumbbell,
              min_width_inches=11.0, min_height_inches=7.0,
              title="Returns that continue into repeated sequences"),
    ),
    options=(
        # Both are drawing choices. The method settings that used to live here -
        # metrics, threshold, max_lag, shuffles - belong to the module now, and
        # are read back from the run rather than repeated.
        Option("cells", default="9", cast=str),
        Option("hour_ticks", default=3.0),
    ),
    grammar="time-by-time state-distance maps, return-rate traces, and noise comparisons",
)
def build(ctx: FigureContext) -> FigureResult:
    rates = ctx.table("recurrence.csv")
    quantified = ctx.table("recurrence_quantification.csv")
    frame = ctx.table("cell_frame.csv")
    settings = {**RECURRENCE_DEFAULTS, **ctx.module_params("recurrence")}
    if "detrend" in quantified and quantified["detrend"].notna().any():
        settings["detrend"] = quantified["detrend"].dropna().mode().iloc[0]
    if ("detrend_window_hours" in quantified
            and quantified["detrend_window_hours"].notna().any()):
        settings["detrend_window_hours"] = float(
            quantified["detrend_window_hours"].dropna().mode().iloc[0])
    metrics = [metric for metric in settings["metrics"] if metric in frame]
    if not metrics:
        raise SystemExit("none of the recurrence state vector is in cell_frame.csv")
    quantile = float(settings["threshold_quantile"])
    complete = frame.dropna(subset=metrics)
    measured = set(quantified["identity"].astype(int))
    cells = [identity for identity in _selected_identities(complete, ctx.option("cells"))
             if identity in measured]

    # The threshold comes out of the table rather than being worked out again,
    # so the line drawn on the wall is the line the determinism beside it was
    # counted against.
    thresholds = quantified.set_index("identity")["recurrence_threshold_distance"]
    walls = {}
    for identity in cells:
        group = complete[complete["identity"] == identity][
            ["frame_index", "hours", *metrics]].sort_values("frame_index")
        hours = group["hours"].to_numpy(float)
        values = state_matrix(
            hours,
            group[metrics].to_numpy(float),
            detrend=str(settings["detrend"]),
            window_hours=float(settings["detrend_window_hours"]),
        )
        finite_state = np.isfinite(values).all(axis=1)
        walls[int(identity)] = (
            cdist(values[finite_state], values[finite_state]),
            hours[finite_state],
            float(thresholds[identity]),
        )

    # The drawn cells only, and this figure's own columns: the run stamps every
    # table with stem, condition and subject, which belong in a run folder and
    # not in a figure's audit table beside the numbers it plotted.
    drawn = rates["identity"].isin(cells)
    figure_data = rates.loc[
        drawn,
        ["identity", "lag_frames", "lag_hours", "recurrence_rate",
         "recurrence_surrogate_mean", "recurrence_surrogate_lo",
         "recurrence_surrogate_hi", "recurrence_pairs"],
    ].copy().rename(columns={
        "lag_frames": "time_separation_frames",
        "lag_hours": "time_separation_hours",
        "recurrence_rate": "same_state_pair_fraction",
        "recurrence_surrogate_mean": "matched_noise_mean",
        "recurrence_surrogate_lo": "matched_noise_lower_95",
        "recurrence_surrogate_hi": "matched_noise_upper_95",
        "recurrence_pairs": "observation_pairs_compared",
    })
    quantification = quantified[quantified["identity"].isin(cells)].drop(
        columns=[c for c in ("stem", "condition", "subject") if c in quantified.columns]
    ).rename(columns={
        "recurrence_threshold_distance": "same_state_distance_cutoff",
        "determinism": "repeated_sequence_fraction",
        "laminarity": "stationary_run_fraction",
        "determinism_surrogate": "matched_noise_repeated_sequence_fraction",
        "laminarity_surrogate": "matched_noise_stationary_run_fraction",
        "recurrence_observations": "observations",
        "recurrence_metrics_used": "measurements_used",
    })
    panels = ctx.panels()
    shown = list(walls)
    columns = min(4, max(1, int(np.ceil(np.sqrt(len(shown))))))
    rows = max(1, int(np.ceil(len(shown) / columns)))
    wall_panel = ctx.spec.panel("wall")
    minimum_sizes = ({"wall": wall_panel.grid_minimum(len(shown), columns)}
                     if "wall" in panels else None)
    fig, axes = ctx.layout(
        panels, minimum_sizes=minimum_sizes, top_inches=2.15,
        gap_inches=1.8,
    )
    if "wall" in axes:
        rect = tuple(axes["wall"].get_position().bounds); axes["wall"].remove()
        scale = ctx.theme.canvas_scale
        gap_x = wall_panel.item_gap_inches * scale / fig.get_figwidth()
        gap_y = wall_panel.item_gap_inches * scale / fig.get_figheight()
        width = (rect[2] - gap_x * (columns - 1)) / columns
        height = (rect[3] - gap_y * (rows - 1)) / rows
        made, wall_handles = [], []
        wall_max = max(float(np.nanmax(values[0])) for values in walls.values()) if walls else 1.0
        for position, identity in enumerate(shown):
            row, column = divmod(position, columns)
            ax = fig.add_axes([
                rect[0] + column * (width + gap_x),
                rect[1] + (rows - 1 - row) * (height + gap_y), width, height])
            distances, hours, cutoff = walls[identity]
            wall_handle = coupling_panels.recurrence(
                ax, distances, hours, ctx.theme,
                threshold=cutoff, hour_ticks=ctx.hour_ticks,
            ).extra["handle"]
            wall_handle.set_clim(0, wall_max)
            ax.set_xlabel("")
            ax.set_ylabel("")
            endpoints = [float(hours[0]), float(hours[-1])]
            if row == rows - 1:
                ax.set_xticks(endpoints, labels=[f"{value:g}" for value in endpoints])
                ax.tick_params(axis="x", labelsize=ctx.theme.size("caption"))
            else:
                ax.set_xticks([])
            if column == 0:
                ax.set_yticks(endpoints, labels=[f"{value:g}" for value in endpoints])
                ax.tick_params(axis="y", labelsize=ctx.theme.size("caption"))
            else:
                ax.set_yticks([])
            ax.text(0.03, 0.97, f"Cell {identity}", transform=ax.transAxes, va="top",
                    fontsize=ctx.theme.size("annotation"), color="white",
                    bbox={"facecolor": ctx.theme.colour("ink"), "alpha": 0.62,
                          "edgecolor": "none", "pad": 1.0})
            made.append(ax); wall_handles.append(wall_handle)
        axes["wall"] = made[0] if made else fig.add_axes(rect)
        if made:
            fig.text(rect[0], rect[1] + rect[3] + 0.014, ctx.spec.panel("wall").heading(),
                     fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
            fig.text(
                rect[0] + rect[2] / 2, rect[1] - 0.018,
                "Time of first observation (h)", ha="center", va="top",
                fontsize=ctx.theme.size("panel"),
            )
            fig.text(
                rect[0] - 0.055, rect[1] + rect[3] / 2,
                "Time of second observation (h)", ha="center", va="center",
                rotation=90, fontsize=ctx.theme.size("panel"),
            )
            common.inset_colour_bar(
                made[-1], wall_handles[-1], ctx.theme,
                label="State difference (0 = identical)",
            )
    if "rate" in axes and not figure_data.empty:
        for cell_index, (_, group) in enumerate(figure_data.groupby("identity")):
            axes["rate"].plot(group["time_separation_hours"], group["same_state_pair_fraction"],
                              color=ctx.theme.colour("morphology"), alpha=0.18,
                              linewidth=ctx.theme.stroke("guide"),
                              label="Individual cells" if cell_index == 0 else None)
        aggregate = figure_data.groupby("time_separation_hours", as_index=False).agg(
            same_state_pair_fraction=("same_state_pair_fraction", "median"),
            matched_noise_mean=("matched_noise_mean", "mean"),
            matched_noise_lower_95=("matched_noise_lower_95", "mean"),
            matched_noise_upper_95=("matched_noise_upper_95", "mean"))
        axes["rate"].fill_between(
            aggregate["time_separation_hours"], aggregate["matched_noise_lower_95"],
            aggregate["matched_noise_upper_95"],
                                  color=ctx.theme.colour("reference"), alpha=0.25, linewidth=0,
                                  label="Matched non-rhythmic noise, 95% interval")
        common.trace(
            axes["rate"], aggregate["time_separation_hours"],
            aggregate["same_state_pair_fraction"], ctx.theme,
            role="morphology", label="Median across cells", hour_ticks=ctx.hour_ticks,
            y_label="Observation pairs in the\nsame measured state (fraction)",
            x_label="Time separating the two observations (h)",
        )
        common.semantic_legend(axes["rate"], ctx.theme, location="above", columns=3)
    if "quantified" in axes and not quantification.empty:
        ordered = quantification.sort_values("repeated_sequence_fraction", ascending=True)
        common.dumbbell(
            axes["quantified"], ordered["matched_noise_repeated_sequence_fraction"],
            ordered["repeated_sequence_fraction"], ctx.theme,
            labels=[str(value) for value in ordered["identity"]],
            right_colours=[ctx.theme.colour("morphology")] * len(ordered),
            left_label="Matched non-rhythmic noise", right_label="Observed cell",
        )
        axes["quantified"].set_xlabel("Returns in repeated sequences (fraction)")
        axes["quantified"].set_ylabel("Cell identity")
        common.semantic_legend(
            axes["quantified"], ctx.theme, location="above", columns=2,
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=(
            f"{len(walls)} cells; each state combines {len(metrics)} "
            f"{str(settings['detrend']).replace('_', ' ')} detrended, "
            f"standardised measurements. Darker map values are more similar; black outlines "
            f"enclose the closest {quantile:.0%}, counted as returns."
        ),
        footnote=(
            "Time separation is elapsed time between two observations of the same cell. "
            "The self-comparison diagonal is not counted. A curve above the grey band returns "
            "more often than matched non-rhythmic noise; a repeated sequence contains at least "
            "two consecutive similar-state pairs."
        ),
        auxiliary={"recurrence_quantification.csv": quantification},
    )


if __name__ == "__main__":
    run_figure("recurrence-wall")
