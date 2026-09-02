"""Does this group differ from that one, and by how much.

Every p-value the rest of this package produces asks whether *one cell* has a
rhythm. None of them asks whether treated differs from control. This is the pass
that does, and it is the last one: it reads the pooled tables and computes
nothing about a cell that was not already measured.

Three rules shape all of it.

**Nothing is tested that was not named.** There is no "test every metric" mode.
The package declares 449 columns; a handful of contrasts across all of them is
thousands of tests, and a correction applied across a thousand tests nobody
meant to run destroys the power to find the twenty that were the point.

**The unit of replication has no default.** Eighty-three cells from one movie
are not eighty-three independent samples. With ``unit: "cell"`` almost
everything comes out significant, and that is the single most likely way for
this package to produce a confidently wrong result. It is a required setting,
and where it is coarser than a cell the statistic that reduces cells to a unit
is required too, because that choice changes the answer as well.

**A test that cannot be run is a row, not a silence.** Too few units, a group
that is not there, a column one movie never measured - each is written with the
reason in ``note`` and everything else blank.

Where this borrows
------------------
The lab's own ``CircadianWorkbench`` v0.5.0 (MIT) carries a group-statistics
layer, and its ``_group_comparison`` compares *profiles on a shared bin grid* -
every recording contributing a length-k vector over identical bins. These
contrasts are *scalar metrics per unit*, so the calling shape does not
transfer and calling it would make every contrast a one-bin ANOVA with its
sphericity machinery idling.

What does transfer is its guards, and they are copied with the source line
cited: ``_hedges_g`` and the scale-relative variance noise floor it rests on
(``analysis.py`` lines 3577 and 3528), the refusal when a group has fewer than
three units (line 3704), the Bonferroni and Sidak corrections (line 3511), and
the aggregation of everything sharing a subject id to one value before testing,
which its own docstring cites as the Lazic 2010 / Hughes 2017 unit-of-analysis
guard. ``analysis/test_contrasts.py`` runs Hedges' g and both corrections
through this module and through the workbench and requires the same answer -
skipped rather than failed when the workbench is absent, so it is a check and
not a dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.config import AnalysisConfig, ContrastConfig
from analysis.pool import MEASURED_FOLDER, POOLED_FOLDER

#: Fewer units than this in any group and no p-value is written, only the
#: reason. Copied from ``circadian_workbench.analysis._small_n_result``
#: (analysis.py:3704), which refuses at the same count for the same reason: with
#: two units a difference is a pair of numbers, not evidence.
MIN_UNITS = 3

#: The written file. One row per result, never one column per contrast, and
#: shaped so it is `plot-that`'s ``data/der/statistics.csv`` without renaming.
STATISTICS_FILE = "statistics.csv"

COLUMNS = [
    "contrast", "family", "description",
    "table", "window", "metric",
    "group_by", "groups", "group_a", "group_b",
    "unit", "aggregate", "n_groups", "n_units", "n_per_group",
    "n_a", "n_b", "cells_a", "cells_b",
    "test", "statistic", "p_value",
    "effect", "effect_kind", "effect_lo", "effect_hi",
    "correction", "p_corrected", "alpha", "significant",
    "note", "methods",
]

#: How several rows become one value for a unit. Both are offered and neither is
#: a default, on the same reasoning ``summarise._STATISTICS`` writes four rather
#: than one: across the metrics this package summarises, mean and median rank
#: the cells differently in 48 of 50.
AGGREGATES = {
    "median": lambda values: float(np.nanmedian(values)),
    "mean": lambda values: float(np.nanmean(values)),
}

#: What identifies one independent unit. Identity numbers are per movie - cell
#: 12 of one movie is not cell 12 of another - so the cell key is
#: ``(stem, identity)`` and never ``identity`` alone.
UNIT_KEYS = {
    "cell": ("stem", "identity"),
    "movie": ("stem",),
    "subject": ("subject",),
}


# --------------------------------------------------------------- effect sizes


def _variance_noise_floor(*samples) -> float:
    """Variance below which a within-group spread is floating-point dust.

    Copied from ``circadian_workbench.analysis._variance_noise_floor``
    (analysis.py:3528). Scale-relative rather than an absolute ``== 0`` test,
    because catastrophic cancellation defeats the absolute one: on a cohort of
    identical values the exact zeros cancel to about 1e-29 rather than to zero,
    and an effect size of -1.3e14 reached a figure legend before this existed.
    """
    parts = [np.asarray(s, dtype=float).ravel() for s in samples]
    pooled = np.concatenate(parts) if parts else np.empty(0, dtype=float)
    pooled = pooled[np.isfinite(pooled)]
    scale = float(np.max(np.abs(pooled))) if pooled.size else 0.0
    sd_tol = max(scale, 1.0) * 1e-12
    return sd_tol * sd_tol


def hedges_g(x, y) -> float:
    """Two-sample Hedges' g with the small-sample bias correction J.

    Copied from ``circadian_workbench.analysis._hedges_g`` (analysis.py:3577).
    NaN when either sample has fewer than two finite observations or when the
    pooled variance is at or below the numerical noise floor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = int(x.size), int(y.size)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = float(np.var(x, ddof=1))
    s2 = float(np.var(y, ddof=1))
    sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if not np.isfinite(sp2) or sp2 <= _variance_noise_floor(x, y):
        return float("nan")
    d = (float(np.mean(x)) - float(np.mean(y))) / math.sqrt(sp2)
    dof = n1 + n2 - 2
    correction = 1.0 - 3.0 / (4.0 * dof - 1.0)
    return float(correction * d)


def paired_hedges_g(differences) -> float:
    """The same correction applied to a within-unit difference.

    A paired effect is standardised by the spread of the *differences*, not by
    the spread of either group: the pairing is what removed the between-unit
    variation, and dividing by it again would undo the design.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    spread = float(np.var(values, ddof=1))
    if not np.isfinite(spread) or spread <= _variance_noise_floor(values):
        return float("nan")
    dof = values.size - 1
    correction = 1.0 - 3.0 / (4.0 * dof - 1.0)
    return float(correction * float(np.mean(values)) / math.sqrt(spread))


def _median_difference(x, y) -> float:
    return float(np.nanmedian(x) - np.nanmedian(y))


def _epsilon_squared(statistic: float, n_units: int, n_groups: int) -> float:
    """Effect size for a Kruskal-Wallis H, on the same 0-1 scale as eta-squared.

    ``(H - k + 1) / (n - k)``: the share of the rank variation the grouping
    accounts for, corrected so that an H at its chance expectation gives zero
    rather than something small and positive.
    """
    denominator = n_units - n_groups
    if denominator <= 0 or not np.isfinite(statistic):
        return float("nan")
    return float((statistic - n_groups + 1) / denominator)


def _eta_squared(statistic: float, n_units: int, n_groups: int) -> float:
    """Effect size for a one-way F, as the share of variance between groups."""
    df_between, df_within = n_groups - 1, n_units - n_groups
    if df_within <= 0 or not np.isfinite(statistic):
        return float("nan")
    return float((statistic * df_between) / (statistic * df_between + df_within))


def _bootstrap_interval(values: list[np.ndarray], effect, resamples: int,
                        generator: np.random.Generator) -> tuple[float, float]:
    """A percentile interval around any effect size, by resampling units.

    One method for every test rather than an analytic formula per test. The
    formulas differ by test and each is a chance to be subtly wrong; resampling
    the units is the same operation whatever was computed from them, and the
    units are what the interval is about.

    Every group is resampled independently, which is what makes this the
    interval for a between-group effect. A paired effect passes one array of
    differences and gets its pairs resampled together, which is the same
    statement one level down.
    """
    if resamples <= 0:
        return float("nan"), float("nan")
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        drawn = [group[generator.integers(0, len(group), len(group))]
                 for group in values]
        try:
            draws[index] = effect(*drawn)
        except Exception:
            draws[index] = np.nan
    usable = draws[np.isfinite(draws)]
    if usable.size < resamples // 2:
        # Half the resamples failing means the effect is not estimable on this
        # data rather than merely uncertain, and a percentile of what survived
        # would be an interval around the subset that happened to work.
        return float("nan"), float("nan")
    return float(np.percentile(usable, 2.5)), float(np.percentile(usable, 97.5))


# --------------------------------------------------------------------- tests


def _mannwhitney(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _kruskal(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.kruskal(*groups)
    return float(result.statistic), float(result.pvalue)


def _welch_t(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    return float(result.statistic), float(result.pvalue)


def _anova(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.f_oneway(*groups)
    return float(result.statistic), float(result.pvalue)


def _wilcoxon(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.wilcoxon(groups[0], groups[1])
    return float(result.statistic), float(result.pvalue)


def _paired_t(groups: list[np.ndarray]) -> tuple[float, float]:
    result = stats.ttest_rel(groups[0], groups[1])
    return float(result.statistic), float(result.pvalue)


#: The small honest set. Which test is run is a declared setting and never a
#: property of the data: running a normality test and choosing accordingly is a
#: garden of forking paths with a friendly face.
TESTS = {
    "mannwhitney": {
        "run": _mannwhitney, "groups": 2, "paired": False,
        "effect": _median_difference, "effect_kind": "median_difference",
        "label": "Mann-Whitney U",
    },
    "kruskal": {
        "run": _kruskal, "groups": None, "paired": False,
        "effect": None, "effect_kind": "epsilon_squared",
        "label": "Kruskal-Wallis H",
    },
    "welch_t": {
        "run": _welch_t, "groups": 2, "paired": False,
        "effect": hedges_g, "effect_kind": "hedges_g",
        "label": "Welch's t",
    },
    "anova": {
        "run": _anova, "groups": None, "paired": False,
        "effect": None, "effect_kind": "eta_squared",
        "label": "one-way ANOVA",
    },
    "wilcoxon": {
        "run": _wilcoxon, "groups": 2, "paired": True,
        "effect": None, "effect_kind": "median_paired_difference",
        "label": "Wilcoxon signed-rank",
    },
    "paired_t": {
        "run": _paired_t, "groups": 2, "paired": True,
        "effect": None, "effect_kind": "hedges_g_paired",
        "label": "paired t",
    },
}

PAIRED_TESTS = tuple(name for name, spec in TESTS.items() if spec["paired"])


# ---------------------------------------------------------------- corrections


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """The step-up false-discovery-rate procedure, as adjusted p-values.

    Adjusted rather than a threshold, so a row carries a number a reader can
    compare against any alpha rather than a verdict against the one that
    happened to be set. Enforced monotone on the way back down, which is what
    makes the adjusted value interpretable at all.
    """
    p = np.asarray(p_values, dtype=float)
    out = p.copy()
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m <= 0:
        return out
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    out[finite] = restored
    return out


def _bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Copied from ``circadian_workbench.analysis._apply_pointwise_correction``
    (analysis.py:3511)."""
    p = np.asarray(p_values, dtype=float)
    out = p.copy()
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m <= 0:
        return out
    out[finite] = np.minimum(p[finite] * m, 1.0)
    return out


def _sidak(p_values: np.ndarray) -> np.ndarray:
    """Copied from the same function as :func:`_bonferroni`."""
    p = np.asarray(p_values, dtype=float)
    out = p.copy()
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m <= 0:
        return out
    clipped = np.clip(p[finite], 0.0, 1.0)
    out[finite] = 1.0 - np.power(1.0 - clipped, m)
    return out


def _uncorrected(p_values: np.ndarray) -> np.ndarray:
    return np.asarray(p_values, dtype=float).copy()


CORRECTIONS = {
    "benjamini_hochberg": _benjamini_hochberg,
    "bonferroni": _bonferroni,
    "sidak": _sidak,
    "none": _uncorrected,
}


# ------------------------------------------------------- reducing to units


def to_units(rows: pd.DataFrame, metric: str, unit: str,
             aggregate: str | None) -> pd.DataFrame:
    """One value per independent unit, and the count of rows behind each.

    ``cell``    each row is a unit. Only defensible when cells come from many
                movies and the movie effect is what is being tested.
    ``movie``   each movie contributes one value and n is the number of movies.
    ``subject`` the same, one level up, because two movies from one animal are
                not two samples. This is the aggregation
                ``circadian_workbench`` cites as the Lazic 2010 / Hughes 2017
                unit-of-analysis guard, and it is the reason ``subject`` is
                stamped onto every row this package writes.

    Returns a table with the unit key, ``value`` and ``cells``. A unit whose
    rows are all blank is dropped rather than carried as NaN: it contributes no
    observation, and a NaN in a group would silently shrink n without saying so.
    """
    keys = list(UNIT_KEYS[unit])
    missing = [key for key in keys if key not in rows.columns]
    if missing:
        raise KeyError(f"unit={unit!r} needs {missing}, which the table does not carry")
    if unit == "cell":
        out = rows[keys].copy()
        out["value"] = pd.to_numeric(rows[metric], errors="coerce").to_numpy(float)
        out["cells"] = 1
        return out[np.isfinite(out["value"])].reset_index(drop=True)

    reduce = AGGREGATES[aggregate]
    collected = []
    for key, block in rows.groupby(keys, sort=True, dropna=False):
        values = pd.to_numeric(block[metric], errors="coerce").to_numpy(dtype=float)
        usable = values[np.isfinite(values)]
        if not usable.size:
            continue
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record["value"] = reduce(usable)
        record["cells"] = int(usable.size)
        collected.append(record)
    return pd.DataFrame(collected, columns=keys + ["value", "cells"])


# ----------------------------------------------------------------- one result


def _blank_row(contrast: ContrastConfig, metric: str, note: str) -> dict:
    """A test that could not run, written as a result.

    Everything but the reason is blank, so nothing on this row can be mistaken
    for a measurement and nothing downstream can average it in.
    """
    row = {column: "" for column in COLUMNS}
    row.update({
        "contrast": contrast.name,
        "family": contrast.family,
        "description": contrast.description,
        "table": contrast.table,
        "window": contrast.window or "",
        "metric": metric,
        "group_by": contrast.group_by,
        "groups": "+".join(contrast.groups),
        "unit": contrast.unit,
        "aggregate": contrast.aggregate or "",
        "correction": contrast.correction,
        "alpha": contrast.alpha,
        "test": contrast.test,
        "significant": False,
        "note": note,
    })
    return row


def _methods(contrast: ContrastConfig, metric: str, spec: dict,
             counts: dict[str, int]) -> str:
    """The sentence a methods section needs, in the row that made the claim.

    In the row rather than in a docstring, because the group size, the unit of
    replication and the correction are what decide whether a p-value means
    anything, and a reader who has only the CSV must not have to come back here
    to find them.
    """
    sizes = ", ".join(f"{group} n={count}" for group, count in counts.items())
    parts = [
        f"{spec['label']} on {metric}",
        f"unit of replication = {contrast.unit}",
        sizes,
    ]
    if contrast.unit == "subject":
        parts.append("recordings sharing a subject averaged to one value before "
                     "testing (Hughes et al. 2017; Lazic 2010)")
    elif contrast.unit == "cell":
        parts.append("unit of replication = cell: cells within one movie are not "
                     "independent samples and the pseudoreplication risk should "
                     "be reported explicitly")
    if contrast.aggregate:
        parts.append(f"cells reduced to a unit by their {contrast.aggregate}")
    parts.append(f"{contrast.correction} correction within family "
                 f"{contrast.family!r}, alpha = {contrast.alpha:g}")
    return "; ".join(parts) + "."


def _one_result(contrast: ContrastConfig, metric: str, rows: pd.DataFrame,
                resamples: int, generator: np.random.Generator) -> dict:
    """One metric of one contrast, tested or refused."""
    spec = TESTS[contrast.test]

    present = [group for group in contrast.groups
               if (rows[contrast.group_by].astype(str) == group).any()]
    absent = [group for group in contrast.groups if group not in present]
    if absent:
        available = sorted(rows[contrast.group_by].astype(str).dropna().unique())
        return _blank_row(contrast, metric,
                          f"group(s) {absent} are not in {contrast.group_by}; "
                          f"the table holds {available}")

    by_group = {}
    for group in contrast.groups:
        block = rows[rows[contrast.group_by].astype(str) == group]
        by_group[group] = to_units(block, metric, contrast.unit, contrast.aggregate)

    counts = {group: int(len(table)) for group, table in by_group.items()}
    thin = {group: count for group, count in counts.items() if count < MIN_UNITS}
    if thin:
        return _blank_row(
            contrast, metric,
            f"only {', '.join(f'{n} {contrast.unit}(s) in {g!r}' for g, n in thin.items())}; "
            f"{MIN_UNITS} is the fewest this package will test - with two, a "
            "difference is a pair of numbers rather than evidence")

    if spec["paired"]:
        keys = list(UNIT_KEYS[contrast.unit])
        left, right = (by_group[group] for group in contrast.groups)
        matched = left.merge(right, on=keys, suffixes=("_a", "_b"))
        if len(matched) < MIN_UNITS:
            return _blank_row(
                contrast, metric,
                f"only {len(matched)} {contrast.unit}(s) appear in both "
                f"{contrast.groups[0]!r} and {contrast.groups[1]!r}; a paired "
                "test matches each unit with itself, so an unmatched unit has "
                "nothing to be compared against")
        values = [matched["value_a"].to_numpy(float), matched["value_b"].to_numpy(float)]
        cells = [int(matched["cells_a"].sum()), int(matched["cells_b"].sum())]
        counts = {contrast.groups[0]: len(matched), contrast.groups[1]: len(matched)}
    else:
        values = [by_group[group]["value"].to_numpy(float) for group in contrast.groups]
        cells = [int(by_group[group]["cells"].sum()) for group in contrast.groups]

    if all(np.allclose(block, block[0]) for block in values):
        # Copied from ``circadian_workbench.analysis._degenerate_bin``: only
        # *every* group being flat is a refusal, because one flat group against
        # one with real spread still gives a well-defined pooled variance. With
        # none of them having any, a t or an F is a 0/0 artefact that evaluates
        # to infinity and reports p = 0.0 exactly - for a difference of 1e-15.
        return _blank_row(
            contrast, metric,
            "no group has any spread: every unit within each group carries the "
            "same value; zero within-group variance is not infinite evidence, "
            "it is no evidence")

    try:
        statistic, p_value = spec["run"](values)
    except Exception as error:
        return _blank_row(contrast, metric, f"{contrast.test} could not run: {error}")

    n_units = int(sum(len(block) for block in values))
    if spec["paired"]:
        differences = values[0] - values[1]
        if spec["effect_kind"] == "hedges_g_paired":
            def effect(block):
                return paired_hedges_g(block)
        else:
            def effect(block):
                return float(np.nanmedian(block))
        value = effect(differences)
        low, high = _bootstrap_interval([differences], effect, resamples, generator)
    elif spec["effect"] is not None:
        effect = spec["effect"]
        value = effect(*values)
        low, high = _bootstrap_interval(values, effect, resamples, generator)
    else:
        # An omnibus effect is a function of the statistic and the counts rather
        # than of the samples, so resampling it would describe the resampling
        # and not the data. Written without an interval, and the blank says so.
        if spec["effect_kind"] == "epsilon_squared":
            value = _epsilon_squared(statistic, n_units, len(values))
        else:
            value = _eta_squared(statistic, n_units, len(values))
        low, high = float("nan"), float("nan")

    # `n_a` and `n_b` name a pair, so on an omnibus test over three or more
    # groups they are left blank rather than reporting the first two as though
    # they were the comparison. `n_per_group` is always written, so the group
    # sizes - which are what decide whether the p-value means anything - are on
    # every row whatever the shape of the test.
    pair = len(values) == 2
    row = _blank_row(contrast, metric, "")
    row.update({
        "group_a": contrast.groups[0] if pair else "",
        "group_b": contrast.groups[1] if pair else "",
        "n_groups": len(values),
        "n_units": n_units,
        "n_per_group": "+".join(f"{group}={count}" for group, count in counts.items()),
        "n_a": len(values[0]) if pair else "",
        "n_b": len(values[1]) if pair else "",
        "cells_a": cells[0] if pair else "",
        "cells_b": cells[1] if pair else "",
        "statistic": statistic,
        "p_value": p_value,
        "effect": value,
        "effect_kind": spec["effect_kind"],
        "effect_lo": low,
        "effect_hi": high,
        "methods": _methods(contrast, metric, spec, counts),
    })
    return row


# --------------------------------------------------------------- the whole run


def _pooled_table(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / POOLED_FOLDER / MEASURED_FOLDER / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _columns_missing(run_dir: Path, table: str) -> dict[str, list[str]]:
    """Which movie lacked which column, straight from the pooling manifest.

    A metric blank in one movie because that movie declared no object set is not
    missing data - it is a movie the question does not apply to. Dropping those
    rows silently would compare the movies that have the column against each
    other and call it a group result.
    """
    path = run_dir / POOLED_FOLDER / "manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest.get("tables", {}).get(table, {})
    return record.get("columns_missing", {})


def run_contrasts(run_dir: str | Path, config: AnalysisConfig,
                  resamples: int = 2000, random_state: int = 20260902) -> pd.DataFrame:
    """Every declared contrast, tested or refused, one row per result.

    The correction is applied **within** each declared family and never across
    the run: a family is the set of results the reader meant to look at
    together, and correcting across families they did not would make the
    twenty results anyone cared about vanish among the ones nobody asked for.
    """
    run_dir = Path(run_dir)
    generator = np.random.default_rng(random_state)
    rows: list[dict] = []

    for contrast in config.contrasts:
        table = _pooled_table(run_dir, contrast.table)
        if table is None:
            for metric in contrast.metrics:
                rows.append(_blank_row(
                    contrast, metric,
                    f"{contrast.table} is not in {POOLED_FOLDER}/{MEASURED_FOLDER}; "
                    "pool the run first, or name a table a module writes"))
            continue

        if contrast.window is not None:
            if "window" not in table.columns:
                for metric in contrast.metrics:
                    rows.append(_blank_row(
                        contrast, metric,
                        f"{contrast.table} has no `window` column, so it cannot "
                        f"be filtered to {contrast.window!r}; name a windowed "
                        "table or drop the window"))
                continue
            table = table[table["window"].astype(str) == contrast.window]
            if table.empty:
                for metric in contrast.metrics:
                    rows.append(_blank_row(
                        contrast, metric,
                        f"no rows carry window={contrast.window!r}"))
                continue

        if contrast.group_by not in table.columns:
            for metric in contrast.metrics:
                rows.append(_blank_row(
                    contrast, metric,
                    f"{contrast.table} has no column {contrast.group_by!r} to "
                    "group by"))
            continue

        missing_in = _columns_missing(run_dir, contrast.table)
        for metric in contrast.metrics:
            if metric not in table.columns:
                rows.append(_blank_row(
                    contrast, metric,
                    f"{contrast.table} has no column {metric!r}"))
                continue
            lacking = sorted(stem for stem, columns in missing_in.items()
                             if metric in columns)
            if lacking:
                rows.append(_blank_row(
                    contrast, metric,
                    f"{metric!r} was not measured in {lacking} - the pooled "
                    "column is blank there because the movie never produced it, "
                    "not because the answer was zero"))
                continue
            rows.append(_one_result(contrast, metric, table, resamples, generator))

    statistics = pd.DataFrame(rows, columns=COLUMNS)
    if statistics.empty:
        return statistics

    # Corrected within the family and never across it.
    statistics["p_corrected"] = np.nan
    for family, block in statistics.groupby("family", sort=False):
        p_values = pd.to_numeric(block["p_value"], errors="coerce").to_numpy(float)
        method = block["correction"].iloc[0]
        statistics.loc[block.index, "p_corrected"] = CORRECTIONS[method](p_values)
    alpha = pd.to_numeric(statistics["alpha"], errors="coerce")
    statistics["significant"] = (
        pd.to_numeric(statistics["p_corrected"], errors="coerce") < alpha).fillna(False)
    return statistics
