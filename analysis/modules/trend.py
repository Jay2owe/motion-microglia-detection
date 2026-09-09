"""How far each cell drifted over the whole recording, per measurement.

The windowed tables answer a different question. They cut the recording in two
and compare the halves, which is the right shape when something happened in the
middle - a drug went in, the temperature changed - because the cut is where the
event was. When nothing happened, the cut is arbitrary, and an arbitrary cut
throws away every frame's position in favour of which side of the line it fell.

This module fits a line instead. One slope per cell per measurement, using
every frame the cell was present for. The everyday version is the difference
between weighing yourself in January and again in July, and weighing yourself
every week and drawing the line: the two agree about the direction, but only
the line tells you whether it was steady, and only the line uses the twenty-four
weeks in between.

**Two slopes, not one, and they are not interchangeable.**

``slope_per_hour``
    Theil-Sen: the median of the slopes of every pair of points. One wild
    frame moves it by almost nothing, and it comes with an interval of its own
    rather than borrowing one from an assumption about the noise.
``slope_least_squares_per_hour``
    Ordinary least squares, the familiar line of best fit, reported beside the
    robust one so a disagreement between them is visible rather than hidden by
    whichever this module happened to prefer. A large gap means outliers are
    driving the answer.
A slope says a measurement moved and how fast. It does not say why. Biology,
photobleaching and focus drift all produce a falling line, and nothing in this
package tells them apart. A rhythm caught at an unlucky phase can also resemble
a slope. Rhythm removal belongs in the configurable Circadian Workbench analysis,
where the period is estimated rather than assumed to be 24 hours.

**One row per cell per measurement.** ``metric`` is a column, so adding a
measurement to the list adds rows rather than columns, and every figure that
names a column keeps working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.registry import Column, MeasurementContext, Output, register_derived

DEFAULTS = {
    # The same signals ``rhythms`` fits by default, and for the same reason:
    # these are the measurements a cell can be said to have a level of. A
    # configuration that changes the rhythm list almost always wants to change
    # this one too, and is expected to say so rather than have this module
    # reach into another module's settings.
    "metrics": [
        "area_px",
        "corrected_mean",
        "solidity",
        "ramification_index",
        "turnover_index",
        "step_px",
    ],
    # Require enough points that isolated observations cannot dominate a slope.
    "min_observations": 24,
}

PRODUCES = (
    # Three columns ``rhythms`` also writes, declared identically on purpose:
    # the registry refuses two modules that describe one column differently,
    # and these mean exactly what they mean there.
    Column("metric", "Which measurement this row is about", "column name", "reference"),
    Column("observations", "Frames the fit used", "frames", "reference"),
    Column("span_hours", "Time from first to last frame used", "h", "reference"),

    Column("level_median", "Middle value of the trace being fitted", "", "reference"),
    Column("slope_per_hour", "Change per hour, robust to outliers", "per h", "variable"),
    Column("slope_low_per_hour", "Low end of the robust slope interval", "per h", "reference"),
    Column("slope_high_per_hour", "High end of the robust slope interval", "per h", "reference"),
    Column("slope_least_squares_per_hour", "Change per hour, line of best fit", "per h", "reference"),
    Column("change_over_recording", "Slope times the span, in the metric's own unit", "", "variable"),
    Column("monotone_rho", "Spearman correlation of the trace against time", "correlation", "fit"),
    Column("monotone_p_value", "Chance of that correlation from noise", "p", "significant"),
)

WRITES = (
    Output("trend", grain=("identity", "metric")),
)


@register_derived(
    name="trend",
    description="Robust and least-squares drift per cell per measurement over the whole recording",
    needs_columns=("identity", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("trend")}
    metrics = [metric for metric in params["metrics"] if metric in cell_frame.columns]
    minimum = int(params["min_observations"])
    rows: list[dict] = []
    for metric in metrics:
        for identity, group in cell_frame.groupby("identity", sort=True):
            usable = group[["hours", metric]].dropna()
            if len(usable) < minimum:
                continue
            hours = usable["hours"].to_numpy(float)
            values = usable[metric].to_numpy(float)
            span = float(hours.max() - hours.min())
            # A flat trace has a slope of zero and no correlation at all. Both
            # are true, and ``theilslopes`` says so, but ``spearmanr`` divides
            # by a zero spread and warns while doing it.
            flat = bool(np.allclose(values, values[0]))

            slope, _, low, high = stats.theilslopes(values, hours, alpha=0.95)
            least_squares = stats.linregress(hours, values)
            rows.append(
                {
                    "identity": int(identity),
                    "metric": metric,
                    "observations": int(len(usable)),
                    "span_hours": span,
                    "level_median": float(np.median(values)),
                    "slope_per_hour": float(slope),
                    "slope_low_per_hour": float(low),
                    "slope_high_per_hour": float(high),
                    "slope_least_squares_per_hour": float(least_squares.slope),
                    "change_over_recording": float(slope) * span,
                    "monotone_rho": float("nan") if flat else float(
                        stats.spearmanr(hours, values).statistic),
                    "monotone_p_value": float("nan") if flat else float(
                        stats.spearmanr(hours, values).pvalue),
                }
            )

    table = pd.DataFrame(rows, columns=[
        "identity", "metric", "observations", "span_hours", "level_median",
        "slope_per_hour", "slope_low_per_hour", "slope_high_per_hour",
        "slope_least_squares_per_hour", "change_over_recording",
        "monotone_rho", "monotone_p_value",
    ])
    return {"trend": table.sort_values(["metric", "identity"]).reset_index(drop=True)}
