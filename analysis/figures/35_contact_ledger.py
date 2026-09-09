"""Figure 35: whether contacting cells share rhythms, size, or behaviour.

Every requested metric can come from ``cell_summary.csv`` or the selected row
of ``rhythms.csv``. A base metric such as ``area_px`` resolves to its per-cell
median when the summary table holds ``area_px_median``.

    python analysis/figures/35_contact_ledger.py <run>
    ... --metrics best_period_hours,area_px,mean_speed
    ... --rhythm-metric corrected_mean
    ... --dilation 1 --min-hours 1 --max-pairs 30
    ... --panels pairs
"""
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from analysis.contrasts import _benjamini_hochberg
from analysis.modules.coupling import contact_metric_permutation_test
from _metrics import describe
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import common
from panels import coupling as coupling_panels


DEFAULT_METRICS = [
    "rhythmic",
    "best_period_hours",
    "best_phase_fraction",
    "area_px",
    "corrected_mean",
    "mean_speed",
    "turnover_index",
    "ramification_index",
]

LABELS = {
    "rhythmic": "Rhythm status",
    "best_period_hours": "Detected period",
    "best_phase_fraction": "Peak position",
    "area_px": "Cell area",
    "corrected_mean": "Reporter intensity",
    "mean_speed": "Movement speed",
    "turnover_index": "Footprint turnover",
    "ramification_index": "Ramification",
}

MATRIX_LABELS = {
    "rhythmic": "Rhythm\nstatus",
    "best_period_hours": "Detected\nperiod",
    "best_phase_fraction": "Peak\nposition",
    "area_px": "Cell\narea",
    "corrected_mean": "Reporter\nintensity",
    "mean_speed": "Movement\nspeed",
    "turnover_index": "Footprint\nturnover",
    "ramification_index": "Ramification",
}

RHYTHM_COLUMNS = {"rhythmic", "best_period_hours", "best_phase_fraction"}
RHYTHM_METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}

RANDOM_SEED = 24_051_986
ALPHA = 0.05


def _compatible_rhythms(rhythms: pd.DataFrame) -> pd.DataFrame:
    """Give older rhythm tables the primary-test columns used by current runs."""
    fits = rhythms.copy()
    fallbacks = {
        "rhythmic": "rhythmic_lombscargle",
        "best_period_hours": "lombscargle_period_hours",
        "best_phase_hours": "free_cosinor_peak_hour",
        "best_p_value": "lombscargle_false_alarm",
    }
    for current, legacy in fallbacks.items():
        if current not in fits and legacy in fits:
            fits[current] = fits[legacy]
    if "best_phase_fraction" not in fits:
        period = pd.to_numeric(fits.get("best_period_hours"), errors="coerce")
        phase = pd.to_numeric(fits.get("best_phase_hours"), errors="coerce")
        fits["best_phase_fraction"] = np.mod(phase, period) / period
    if "primary_rhythm_test" not in fits:
        fits["primary_rhythm_test"] = "lomb"
    if "period_estimation_method" not in fits:
        fits["period_estimation_method"] = "lomb"
    return fits


def _rhythm_label(fits: pd.DataFrame) -> str:
    method = (
        str(fits["primary_rhythm_test"].dropna().mode().iloc[0])
        if "primary_rhythm_test" in fits and fits["primary_rhythm_test"].notna().any()
        else "lomb"
    )
    return RHYTHM_METHOD_LABELS.get(method, method)


def _estimator_label(fits: pd.DataFrame) -> str:
    if "best_method_label" in fits and fits["best_method_label"].notna().any():
        return str(fits["best_method_label"].dropna().mode().iloc[0])
    method = (
        str(fits["period_estimation_method"].dropna().mode().iloc[0])
        if "period_estimation_method" in fits
        and fits["period_estimation_method"].notna().any()
        else "lomb"
    )
    return RHYTHM_METHOD_LABELS.get(method, method)


def _metric_values(
    requested: str,
    summary: pd.DataFrame,
    fits: pd.DataFrame | None,
    interval_minutes: float,
) -> tuple[pd.Series, dict[str, str | float | None]]:
    """Resolve one user metric to per-cell values and explicit comparison rules."""
    if requested in RHYTHM_COLUMNS:
        if fits is None or requested not in fits:
            raise SystemExit(
                f"--metrics {requested} needs rhythms.csv with the current or legacy primary-test columns"
            )
        values = fits.set_index("identity")[requested]
        if requested in {"best_period_hours", "best_phase_fraction"}:
            rhythmic = fits.set_index("identity")["rhythmic"].fillna(False).astype(bool)
            values = values.where(rhythmic)
        circular_period = 1.0 if requested == "best_phase_fraction" else None
        units = {
            "rhythmic": "mismatch: 0 same, 1 different",
            "best_period_hours": "h",
            "best_phase_fraction": "fraction of own cycle",
        }[requested]
        return pd.to_numeric(values, errors="coerce"), {
            "metric": requested,
            "source_column": requested,
            "label": LABELS[requested],
            "unit": units,
            "circular_period": circular_period,
            "rhythmic_pairs_only": requested in {"best_period_hours", "best_phase_fraction"},
        }

    candidates = [requested, f"{requested}_median", f"{requested}_mean"]
    resolved = next((column for column in candidates if column in summary), None)
    source = summary
    if resolved is None and fits is not None and requested in fits:
        resolved = requested
        source = fits
    if resolved is None:
        available = sorted(set(summary.columns) | (set() if fits is None else set(fits.columns)))
        preview = ", ".join(available[:24])
        raise SystemExit(
            f"--metrics {requested} is unavailable; the first available columns are: {preview}"
        )
    metric = describe(resolved)
    label = LABELS.get(requested, metric.label)
    unit = metric.unit_text(interval_minutes)
    values = source.set_index("identity")[resolved]
    return pd.to_numeric(values, errors="coerce"), {
        "metric": requested,
        "source_column": resolved,
        "label": label,
        "unit": unit,
        "circular_period": 1.0 if requested.endswith("phase_fraction") else None,
        "rhythmic_pairs_only": False,
    }


def _q_label(value: float) -> str:
    if not np.isfinite(value):
        return "q unavailable"
    if value < 0.001:
        return "q < 0.001"
    return f"q = {value:.3g}"


def _forest_labels(summary: pd.DataFrame, q_column: str) -> list[str]:
    return [
        f"{row.metric_label}\n{int(row.pairs)} pairs; {_q_label(float(getattr(row, q_column)))}"
        for row in summary.itertuples()
    ]


def _statistics_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for item in summary.to_dict("records"):
        shared = {
            "metric": item["metric"],
            "metric_label": item["metric_label"],
            "source_column": item["source_column"],
            "unit": item["unit"],
            "cells": item["cells_available"],
            "pairs": item["pairs"],
            "shuffles": item["shuffles"],
            "random_state": item["random_state"],
            "alternative": "two-sided",
            "correction": "Benjamini-Hochberg false-discovery rate within analysis family",
            "alpha": ALPHA,
            "status": item["status"],
        }
        rows.append({
            **shared,
            "analysis_family": "overall contact-pair similarity",
            "test": "cell-label permutation test of mean absolute pair difference",
            "estimate_name": "observed mean difference / shuffled mean difference",
            "estimate": item.get("difference_ratio", np.nan),
            "observed_raw": item.get("observed_mean_difference", np.nan),
            "null_mean": item.get("random_mean_difference", np.nan),
            "null_interval_low": item.get("similarity_null_lo", np.nan),
            "null_interval_high": item.get("similarity_null_hi", np.nan),
            "p_value": item.get("similarity_p_value", np.nan),
            "q_value": item.get("similarity_q_value", np.nan),
        })
        rows.append({
            **shared,
            "analysis_family": "contact duration and pair similarity",
            "test": "cell-label permutation test of Spearman rank correlation",
            "estimate_name": "Spearman rho: contact hours versus pair difference",
            "estimate": item.get("duration_spearman_rho", np.nan),
            "observed_raw": item.get("duration_spearman_rho", np.nan),
            "null_mean": 0.0,
            "null_interval_low": item.get("duration_null_lo", np.nan),
            "null_interval_high": item.get("duration_null_hi", np.nan),
            "p_value": item.get("duration_p_value", np.nan),
            "q_value": item.get("duration_q_value", np.nan),
        })
    return pd.DataFrame(rows)


@figure(
    number=35,
    slug="contact-ledger",
    summary="whether contacting cells share rhythms, size, reporter level, or movement",
    title="Which cell traits are shared across contacts?",
    reads=(Table("contacts.csv", module="contacts"),
           Table("cell_summary.csv", module="summary"),
           Table("rhythms.csv", module="rhythms", optional=True)),
    panels=(
        Panel("pairs", coupling_panels.pair_metric_ledger,
              min_width_inches=14.0, min_height_inches=9.0,
              title="Longest contacts: how different is each pair?"),
        Panel("overall", common.forest,
              min_width_inches=7.2, min_height_inches=6.4,
              title="Are contacting pairs more alike\nthan shuffled cells?"),
        Panel("duration", common.forest,
              min_width_inches=7.2, min_height_inches=6.4,
              title="Do longer contacts link\nmore alike cells?"),
    ),
    options=(
        Option("metrics", default=DEFAULT_METRICS),
        Option("rhythm_metric", default="corrected_mean"),
        Option("dilation", default=None, metavar="PX",
               help="one measured contact radius; unset takes the strictest radius"),
        Option("min_hours", default=1.0),
        Option("max_pairs", default=30),
        Option("shuffles", default=10_000),
    ),
    grammar="contact-pair metric-difference matrix with permutation-effect forest",
)
def build(ctx: FigureContext) -> FigureResult:
    requested_metrics = list(ctx.option("metrics"))
    if not requested_metrics:
        raise SystemExit("--metrics needs at least one cell-level measurement")
    contacts_all = ctx.table("contacts.csv")
    summary = ctx.table("cell_summary.csv")
    rhythms = ctx.optional_table("rhythms.csv")
    rhythm_metric = str(ctx.option("rhythm_metric"))
    fits = None
    if rhythms is not None:
        selected = rhythms[rhythms["metric"] == rhythm_metric].copy()
        if not selected.empty:
            fits = _compatible_rhythms(selected)

    available_radii = sorted(int(value) for value in contacts_all["dilation_px"].unique())
    requested_radius = ctx.option_or("dilation", [float(available_radii[0])])
    if len(requested_radius) != 1:
        raise SystemExit("--dilation accepts one measured radius on the contact-similarity ledger")
    radius = int(requested_radius[0])
    if radius not in available_radii:
        raise SystemExit(
            f"--dilation {radius} was not measured; available radii: {available_radii}"
        )
    minimum_hours = float(ctx.option("min_hours"))
    if minimum_hours < 0:
        raise SystemExit("--min-hours cannot be negative")
    contacts = contacts_all[
        contacts_all["dilation_px"].eq(radius)
        & contacts_all["hours_in_contact"].ge(minimum_hours)
    ].copy()
    handoffs = contacts.get("handoff_coincident", pd.Series(False, index=contacts.index))
    excluded_handoffs = int(handoffs.fillna(False).astype(bool).sum())
    contacts = contacts[~handoffs.fillna(False).astype(bool)].copy()
    contacts["identity_a"] = contacts["identity_a"].astype(int)
    contacts["identity_b"] = contacts["identity_b"].astype(int)
    contacts["pair"] = (
        contacts["identity_a"].astype(str) + "-" + contacts["identity_b"].astype(str)
    )
    if contacts.empty:
        raise SystemExit(
            f"no non-handoff pairs contacted for at least {minimum_hours:g} h at radius {radius} px"
        )

    shuffles = int(ctx.option("shuffles"))
    pair_tables: list[pd.DataFrame] = []
    result_rows: list[dict] = []
    metric_labels: dict[str, str] = {}
    matrix_labels: dict[str, str] = {}
    for index, requested in enumerate(requested_metrics):
        values, info = _metric_values(
            requested, summary, fits, float(ctx.interval),
        )
        pair_data, result = contact_metric_permutation_test(
            contacts, values,
            circular_period=info["circular_period"],
            shuffles=shuffles,
            random_state=RANDOM_SEED + index,
        )
        label = str(info["label"])
        metric_labels[requested] = label
        matrix_labels[requested] = MATRIX_LABELS.get(
            requested, textwrap.fill(label, width=13)
        )
        if not pair_data.empty:
            pair_data.insert(0, "metric", requested)
            pair_data.insert(1, "metric_label", label)
            pair_data.insert(2, "source_column", str(info["source_column"]))
            pair_data.insert(3, "unit", str(info["unit"]))
            pair_data["pair"] = (
                pair_data["identity_a"].astype(int).astype(str)
                + "-" + pair_data["identity_b"].astype(int).astype(str)
            )
            pair_tables.append(pair_data)
        result_rows.append({
            "metric": requested,
            "metric_label": label,
            "source_column": str(info["source_column"]),
            "unit": str(info["unit"]),
            "rhythmic_pairs_only": bool(info["rhythmic_pairs_only"]),
            **result,
        })

    all_pair_metrics = (
        pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    )
    tests = pd.DataFrame(result_rows)
    missing_values = pd.Series(np.nan, index=tests.index, dtype=float)
    tests["similarity_q_value"] = _benjamini_hochberg(
        pd.to_numeric(
            tests.get("similarity_p_value", missing_values), errors="coerce",
        ).to_numpy()
    )
    tests["duration_q_value"] = _benjamini_hochberg(
        pd.to_numeric(
            tests.get("duration_p_value", missing_values), errors="coerce",
        ).to_numpy()
    )
    tests["similarity_significant"] = tests["similarity_q_value"] <= ALPHA
    tests["duration_significant"] = tests["duration_q_value"] <= ALPHA
    statistics = _statistics_rows(tests)

    pair_order = contacts.sort_values(
        ["hours_in_contact", "identity_a", "identity_b"],
        ascending=[False, True, True], kind="mergesort",
    )["pair"].tolist()
    maximum_pairs = int(ctx.option("max_pairs"))
    if maximum_pairs < 0:
        raise SystemExit("--max-pairs cannot be negative")
    shown_pairs = pair_order if maximum_pairs == 0 else pair_order[:maximum_pairs]
    figure_data = all_pair_metrics[all_pair_metrics["pair"].isin(shown_pairs)].copy()

    panels = ctx.panels()
    ledger_height = max(9.0, 0.28 * len(shown_pairs))
    if panels.keys == ["pairs", "overall", "duration"]:
        fig, axes = ctx.grid_layout(
            panels,
            {
                "pairs": (0, 0, 1, 2),
                "overall": (1, 0, 1, 1),
                "duration": (1, 1, 1, 1),
            },
            minimum_sizes={"pairs": (14.0, ledger_height)},
            top_inches=2.5,
            bottom_inches=2.8,
            left_inches=3.35,
            right_inches=1.55,
            horizontal_gap_inches=3.0,
            vertical_gap_inches=2.1,
        )
    else:
        minimum_sizes = ({"pairs": (14.0, ledger_height)}
                         if "pairs" in panels else None)
        fig, axes = ctx.layout(
            panels, minimum_sizes=minimum_sizes, gap_inches=1.4,
            left_inches=3.35, right_inches=1.55,
        )

    if "pairs" in axes:
        ledger = coupling_panels.pair_metric_ledger(
            axes["pairs"], figure_data, ctx.theme,
            pair_order=shown_pairs,
            metric_order=requested_metrics,
            metric_labels=matrix_labels,
        )
        common.inset_colour_bar(
            axes["pairs"], ledger.extra["handle"], ctx.theme,
            label="Pair difference / shuffled mean",
            ticks=[0, 1, 2],
            ticklabels=["0\nsame", "1\nshuffled mean", "2+\nmore different"],
        )
        axes["pairs"]._semantic_legend_handled = True

    similarity_labels = _forest_labels(tests, "similarity_q_value")
    refusals = [
        "" if status == "ok" else "Too few eligible contact pairs for this metric"
        for status in tests["status"]
    ]
    if "overall" in axes:
        common.forest(
            axes["overall"], tests["difference_ratio"], ctx.theme,
            labels=similarity_labels,
            lo=tests["similarity_null_lo"], hi=tests["similarity_null_hi"],
            significant=tests["similarity_significant"], refusals=refusals,
            no_effect=1.0,
            x_label="Mean pair difference / shuffled mean\n<1 more alike | >1 more different",
        )
        axes["overall"].set_xlim(left=0)
        axes["overall"]._semantic_legend_handled = True

    if "duration" in axes:
        common.forest(
            axes["duration"], tests["duration_spearman_rho"], ctx.theme,
            labels=_forest_labels(tests, "duration_q_value"),
            lo=tests["duration_null_lo"], hi=tests["duration_null_hi"],
            significant=tests["duration_significant"], refusals=refusals,
            no_effect=0.0,
            x_label="Spearman correlation: contact hours vs pair difference\n<0 longer contacts are more alike",
        )
        axes["duration"].set_xlim(-1, 1)
        axes["duration"]._semantic_legend_handled = True

    radius_text = f"{ctx.scale.length(radius):g} {ctx.length_label}"
    subtitle = (
        f"{len(contacts)} pairs within {radius_text} for at least {minimum_hours:g} h; "
        f"the {len(shown_pairs)} longest are shown above and all pairs are tested below."
    )
    rhythm_label = _rhythm_label(fits) if fits is not None else "the configured primary rhythm test"
    estimator_label = _estimator_label(fits) if fits is not None else "the configured period estimator"
    span = float(ctx.summary.get("hours_covered", np.nan))
    span_note = (
        f" The {span:g} h recording makes long-period estimates exploratory."
        if np.isfinite(span) else ""
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        auxiliary={
            "all_pair_metrics.csv": all_pair_metrics,
            "metric_tests.csv": tests,
            "statistics.csv": statistics,
            "contact_pairs.csv": contacts,
        },
        subtitle=subtitle,
        footnote=(
            f"Cell measurements were shuffled {shuffles:,} times across the unchanged contact network; "
            "whiskers are central 95% shuffle intervals, not confidence intervals. "
            "Two-sided p-values were corrected by the Benjamini-Hochberg false-discovery rate within each lower panel; purple rows have q <= 0.05. "
            f"Period and peak position come from {estimator_label} and use only pairs where both cells were rhythmic by {rhythm_label}; grey ledger cells lack an eligible rhythmic pair."
            f"{span_note} Handoff-coincident pairs excluded: {excluded_handoffs}."
        ),
    )


if __name__ == "__main__":
    run_figure("contact-ledger")
