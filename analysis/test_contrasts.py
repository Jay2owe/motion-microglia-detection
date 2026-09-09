"""Every offered test against a known answer, and every refusal.

Two things are being checked here and they are different. The first is that the
arithmetic is right: each test and each correction is run against a worked
example whose answer can be written down without running this code. The second,
and the one that matters more, is that the package refuses rather than guesses -
the unit of replication decides the answer, and a test that quietly ran at the
wrong unit would look exactly like one that ran at the right one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.config import AnalysisConfig, ContrastConfig, _contrasts
from analysis.contrasts import (CORRECTIONS, MIN_UNITS, PAIRED_TESTS,
                                SINGLE_GROUP_TESTS, TESTS, _benjamini_hochberg,
                                _bonferroni, _sidak, hedges_g, paired_hedges_g,
                                run_contrasts, to_units)
from analysis.pool import MEASURED_FOLDER, POOLED_FOLDER

TEST_NAMES = tuple(TESTS)
CORRECTION_NAMES = tuple(CORRECTIONS)


def _contrast(**kwargs) -> ContrastConfig:
    groups = kwargs.pop("metric_groups", {})
    block = {
        "name": "c", "table": "cell_summary", "metrics": ["area_px_median"],
        "group_by": "condition", "groups": ["control", "treated"],
        "unit": "cell", "test": "mannwhitney",
    }
    block.update(kwargs)
    return ContrastConfig.from_dict(block, TEST_NAMES, PAIRED_TESTS,
                                    CORRECTION_NAMES, SINGLE_GROUP_TESTS, groups)


def _pooled(tmp_path, table: pd.DataFrame, name: str = "cell_summary",
            columns_missing: dict | None = None):
    """A pooled run folder holding one table, without measuring anything."""
    folder = tmp_path / POOLED_FOLDER / MEASURED_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f"{name}.csv", index=False)
    manifest = {"movies": sorted(table["stem"].unique()), "tables": {
        name: {"columns_missing": columns_missing or {}}}}
    (tmp_path / POOLED_FOLDER / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _rows(groups: dict[str, list[float]], metric: str = "area_px_median",
          subjects: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """One row per cell, stamped the way a pooled table is."""
    records = []
    for condition, values in groups.items():
        for index, value in enumerate(values):
            stem = (subjects[condition][index] if subjects
                    else f"{condition}_{index}")
            records.append({"stem": stem, "condition": condition,
                            "subject": stem, "identity": index + 1,
                            metric: value})
    return pd.DataFrame(records)


def _paired_rows(before: list[float], after: list[float],
                 metric: str = "area_px_median") -> pd.DataFrame:
    """The same cells in two windows, which is what a paired test needs.

    A paired test matches each unit with itself in the other group, so the unit
    key has to appear on both sides. Two groups of six *different* cells is an
    unpaired design however it is labelled.
    """
    records = []
    for window, values in (("baseline", before), ("treatment", after)):
        for index, value in enumerate(values):
            records.append({"stem": "m", "condition": "control", "subject": "s",
                            "identity": index + 1, "window": window, metric: value})
    return pd.DataFrame(records)


def _config(*contrasts: ContrastConfig) -> AnalysisConfig:
    return AnalysisConfig(dataset="t", frame_interval_min=30.0, movies=[],
                          output_root=".", contrasts=list(contrasts))


# ------------------------------------------------ each test against scipy itself


@pytest.mark.parametrize("test, runner", [
    ("mannwhitney", lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided")),
    ("welch_t", lambda a, b: stats.ttest_ind(a, b, equal_var=False)),
])
def test_a_two_group_test_reproduces_the_reference_statistic(tmp_path, test, runner):
    """The answer is scipy's, and this proves nothing was reshaped on the way.

    Written against the library rather than against a hard-coded number so that
    the check is of *this package's* plumbing - which rows reached the test, in
    what order - rather than of scipy's arithmetic, which is not ours to test.
    """
    control = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0]
    treated = [15.0, 17.0, 16.0, 18.0, 14.5, 19.0]
    table = _rows({"control": control, "treated": treated})
    run = _pooled(tmp_path, table)

    statistics = run_contrasts(run, _config(_contrast(test=test)), resamples=50)
    row = statistics.iloc[0]

    expected = runner(np.array(control), np.array(treated))
    assert row["note"] == ""
    assert row["statistic"] == pytest.approx(float(expected.statistic))
    assert row["p_value"] == pytest.approx(float(expected.pvalue))
    assert row["n_a"] == 6 and row["n_b"] == 6
    assert row["unit"] == "cell"


@pytest.mark.parametrize("test, runner", [
    ("wilcoxon", lambda a, b: stats.wilcoxon(a, b)),
    ("paired_t", lambda a, b: stats.ttest_rel(a, b)),
])
def test_a_paired_test_reproduces_the_reference_statistic(tmp_path, test, runner):
    """The same cells before and after, matched on (stem, identity)."""
    before = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0]
    after = [15.0, 17.0, 16.0, 11.0, 14.5, 19.0]
    run = _pooled(tmp_path, _paired_rows(before, after), name="cell_summary_windowed")

    contrast = _contrast(table="cell_summary_windowed", group_by="window",
                         groups=["baseline", "treatment"], test=test)
    row = run_contrasts(run, _config(contrast), resamples=50).iloc[0]

    expected = runner(np.array(before), np.array(after))
    assert row["note"] == ""
    assert row["statistic"] == pytest.approx(float(expected.statistic))
    assert row["p_value"] == pytest.approx(float(expected.pvalue))
    assert row["n_a"] == 6 and row["n_b"] == 6


@pytest.mark.parametrize("test, runner", [
    ("kruskal", stats.kruskal),
    ("anova", stats.f_oneway),
])
def test_a_three_group_test_reproduces_the_reference_statistic(tmp_path, test, runner):
    groups = {"a": [1.0, 2.0, 3.0, 2.5], "b": [4.0, 5.0, 6.0, 5.5],
              "c": [7.0, 8.0, 9.0, 8.5]}
    run = _pooled(tmp_path, _rows(groups))

    statistics = run_contrasts(
        run, _config(_contrast(test=test, groups=["a", "b", "c"])), resamples=0)
    row = statistics.iloc[0]

    expected = runner(*(np.array(v) for v in groups.values()))
    assert row["statistic"] == pytest.approx(float(expected.statistic))
    assert row["p_value"] == pytest.approx(float(expected.pvalue))
    assert row["n_groups"] == 3
    assert row["n_units"] == 12
    # An omnibus test names no pair, and the pair columns stay blank rather than
    # naming the first two groups as though they were the comparison. The group
    # sizes are still on the row, because they are what decides whether the
    # p-value means anything.
    assert row["group_a"] == "" and row["group_b"] == ""
    assert row["n_a"] == "" and row["n_b"] == ""
    assert row["n_per_group"] == "a=4+b=4+c=4"


def test_every_result_carries_an_effect_size(tmp_path):
    """A p-value alone is not a result, for every test this package offers."""
    unpaired = _pooled(tmp_path / "u", _rows({"control": [1.0, 2.0, 3.0, 4.0],
                                              "treated": [5.0, 6.5, 7.0, 8.0]}))
    paired = _pooled(tmp_path / "p",
                     _paired_rows([1.0, 2.0, 3.0, 4.0, 5.0],
                                  [3.0, 5.0, 4.0, 8.0, 6.0]),
                     name="cell_summary_windowed")

    for test, spec in TESTS.items():
        if spec["paired"]:
            contrast = _contrast(test=test, table="cell_summary_windowed",
                                 group_by="window", groups=["baseline", "treatment"])
            run = paired
        elif spec["groups"] == 1:
            # One group against zero: the values must sit away from zero for
            # there to be an effect at all, so the treated block is the one to
            # hand it.
            contrast = _contrast(test=test, groups=["treated"])
            run = unpaired
        else:
            contrast = _contrast(test=test, groups=["control", "treated"])
            run = unpaired
        row = run_contrasts(run, _config(contrast), resamples=40).iloc[0]
        assert row["note"] == "", test
        assert np.isfinite(float(row["effect"])), test
        assert row["effect_kind"], test


def test_a_two_group_effect_carries_an_interval_and_an_omnibus_one_does_not(tmp_path):
    """An omnibus effect is a function of the statistic, not of the samples.

    Resampling it would describe the resampling rather than the data, so the
    interval is left blank and the blank is the honest answer.
    """
    run = _pooled(tmp_path, _rows({"a": [1.0, 2.0, 3.0, 2.5],
                                   "b": [4.0, 5.0, 6.0, 5.5],
                                   "c": [7.0, 8.0, 9.0, 8.5]}))
    pair = run_contrasts(run, _config(_contrast(groups=["a", "b"])), resamples=200)
    omnibus = run_contrasts(
        run, _config(_contrast(test="kruskal", groups=["a", "b", "c"])), resamples=200)

    assert np.isfinite(float(pair.iloc[0]["effect_lo"]))
    assert np.isfinite(float(pair.iloc[0]["effect_hi"]))
    assert float(pair.iloc[0]["effect_lo"]) <= float(pair.iloc[0]["effect"])
    assert not np.isfinite(float(omnibus.iloc[0]["effect_lo"]))


# ---------------------------------------------------------- the unit of replication


def test_the_unit_reduction_uses_the_stem_and_identity_together():
    """Cell 12 of one movie is not cell 12 of another.

    Grouping by ``identity`` alone would silently average two unrelated cells
    into one, and the two movies would then look more alike than they are.
    """
    rows = pd.DataFrame({
        "stem": ["a", "a", "b", "b"], "subject": ["s1", "s1", "s2", "s2"],
        "identity": [1, 2, 1, 2], "m": [10.0, 20.0, 30.0, 40.0]})

    cells = to_units(rows, "m", "cell", None)
    assert len(cells) == 4
    assert sorted(cells["value"]) == [10.0, 20.0, 30.0, 40.0]

    movies = to_units(rows, "m", "movie", "median")
    assert len(movies) == 2
    assert sorted(movies["value"]) == [15.0, 35.0]
    assert sorted(movies["cells"]) == [2, 2]


def test_two_movies_from_one_animal_are_one_subject():
    """The guard ``circadian_workbench`` cites as Lazic 2010 / Hughes 2017.

    Without it, running the same animal twice doubles n and halves the p-value
    for nothing.
    """
    rows = pd.DataFrame({
        "stem": ["a", "b", "c"], "subject": ["s1", "s1", "s2"],
        "identity": [1, 1, 1], "m": [10.0, 20.0, 60.0]})

    movies = to_units(rows, "m", "movie", "mean")
    subjects = to_units(rows, "m", "subject", "mean")

    assert len(movies) == 3
    assert len(subjects) == 2
    assert sorted(subjects["value"]) == [15.0, 60.0]
    assert sorted(subjects["cells"]) == [1, 2]


def test_the_aggregate_changes_the_answer_which_is_why_it_is_required():
    rows = pd.DataFrame({
        "stem": ["a"] * 4, "subject": ["s"] * 4, "identity": [1, 2, 3, 4],
        "m": [1.0, 1.0, 1.0, 97.0]})

    assert to_units(rows, "m", "movie", "median")["value"].iloc[0] == 1.0
    assert to_units(rows, "m", "movie", "mean")["value"].iloc[0] == 25.0


def test_a_unit_with_nothing_but_blanks_is_dropped_rather_than_carried():
    """A NaN in a group would shrink n without saying so."""
    rows = pd.DataFrame({
        "stem": ["a", "a", "b", "b"], "subject": ["s", "s", "t", "t"],
        "identity": [1, 2, 1, 2], "m": [np.nan, np.nan, 3.0, 5.0]})

    movies = to_units(rows, "m", "movie", "median")
    assert list(movies["stem"]) == ["b"]


def test_the_unit_reaches_the_written_row(tmp_path):
    """`unit: cell` and `unit: movie` on one dataset must not give one answer."""
    subjects = {"control": ["m1", "m1", "m1", "m2", "m2", "m2"],
                "treated": ["m3", "m3", "m3", "m4", "m4", "m4"]}
    table = _rows({"control": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                   "treated": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}, subjects=subjects)
    table["identity"] = list(range(1, 7)) * 2
    run = _pooled(tmp_path, table)

    by_cell = run_contrasts(run, _config(_contrast(unit="cell")), resamples=0).iloc[0]
    by_movie = run_contrasts(
        run, _config(_contrast(unit="movie", aggregate="median")), resamples=0).iloc[0]

    assert by_cell["n_a"] == 6 and by_cell["cells_a"] == 6
    assert by_cell["note"] == ""
    # The same data at the honest unit has two movies a side, which is below the
    # floor - so it refuses rather than reporting the twelve-cell answer again
    # under a different name. This is the whole point of the setting.
    assert by_movie["note"] != ""
    assert "2 movie(s)" in by_movie["note"]
    assert by_movie["p_value"] == ""


# ------------------------------------------------------------------ refusals


def test_a_group_with_too_few_units_is_a_row_and_not_a_crash(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0, 2.0, 3.0, 4.0],
                                   "treated": [5.0, 6.0]}))

    row = run_contrasts(run, _config(_contrast()), resamples=0).iloc[0]

    assert row["note"] != ""
    assert "treated" in row["note"]
    assert str(MIN_UNITS) in row["note"]
    assert row["p_value"] == "" and row["effect"] == ""
    assert row["significant"] is False or row["significant"] == False  # noqa: E712


def test_a_group_that_is_not_there_names_what_is(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0, 2.0, 3.0, 4.0],
                                   "wildtype": [5.0, 6.0, 7.0, 8.0]}))

    row = run_contrasts(run, _config(_contrast()), resamples=0).iloc[0]

    assert "'treated'" in row["note"]
    assert "wildtype" in row["note"]


def test_a_constant_column_is_refused_rather_than_reported_as_certain(tmp_path):
    """Zero within-group variance is not infinite evidence.

    Left to scipy, a t-test on two flat groups returns an infinite statistic and
    p = 0.0 *exactly*, for a difference that may be one part in 1e15.
    """
    run = _pooled(tmp_path, _rows({"control": [5.0] * 4, "treated": [7.0] * 4}))

    row = run_contrasts(run, _config(_contrast(test="welch_t")), resamples=0).iloc[0]

    assert "no group has any spread" in row["note"]
    assert row["p_value"] == ""


def test_one_flat_group_against_one_with_spread_still_tests(tmp_path):
    """Only *every* group being flat is a refusal, which is the workbench's rule."""
    run = _pooled(tmp_path, _rows({"control": [5.0] * 4,
                                   "treated": [6.0, 7.0, 8.0, 9.0]}))

    row = run_contrasts(run, _config(_contrast(test="welch_t")), resamples=0).iloc[0]

    assert row["note"] == ""
    assert np.isfinite(float(row["p_value"]))


def test_a_metric_absent_from_one_movie_is_named_rather_than_dropped(tmp_path):
    """Not missing data - a movie the question does not apply to.

    Dropping those rows silently would compare the movies that have the column
    against each other and call it a group result.
    """
    table = _rows({"control": [1.0, 2.0, 3.0, 4.0], "treated": [5.0, 6.0, 7.0, 8.0]})
    run = _pooled(tmp_path, table,
                  columns_missing={"treated_0": ["area_px_median"]})

    row = run_contrasts(run, _config(_contrast()), resamples=0).iloc[0]

    assert "was not measured in" in row["note"]
    assert "treated_0" in row["note"]
    assert "not because the answer was zero" in row["note"]


def test_a_metric_the_table_does_not_have_is_named(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0] * 4, "treated": [2.0] * 4}))

    row = run_contrasts(run, _config(_contrast(metrics=["nonesuch"])),
                        resamples=0).iloc[0]

    assert "no column 'nonesuch'" in row["note"]


def test_a_table_that_was_never_pooled_is_named(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0] * 4, "treated": [2.0] * 4}))

    row = run_contrasts(run, _config(_contrast(table="nonesuch")),
                        resamples=0).iloc[0]

    assert "pool the run first" in row["note"]


def test_every_row_carries_either_a_result_or_a_reason(tmp_path):
    """No row is blank on both counts, which is the whole contract of the file."""
    table = _rows({"control": [1.0, 2.0, 3.0, 4.0], "treated": [5.0, 6.0]})
    run = _pooled(tmp_path, table)

    statistics = run_contrasts(
        run, _config(_contrast(metrics=["area_px_median", "nonesuch"])), resamples=0)

    assert len(statistics) == 2
    for _, row in statistics.iterrows():
        has_result = row["p_value"] != "" and np.isfinite(float(row["p_value"] or "nan"))
        has_reason = str(row["note"]) != ""
        assert has_result or has_reason
        assert not (has_result and has_reason)


# ------------------------------------------------------------ windows and pairs


def test_a_window_contrast_pairs_the_same_cells_across_two_windows(tmp_path):
    """The paired shape: the grouping column is the window, and a cell is its own control."""
    records = []
    for window, offset in (("baseline", 0.0), ("treatment", 3.0)):
        for index in range(6):
            records.append({"stem": "m", "condition": "control", "subject": "s",
                            "identity": index + 1, "window": window,
                            "area_px_median": 10.0 + index + offset})
    run = _pooled(tmp_path, pd.DataFrame(records), name="cell_summary_windowed")

    contrast = _contrast(table="cell_summary_windowed", group_by="window",
                         groups=["treatment", "baseline"], test="paired_t")
    row = run_contrasts(run, _config(contrast), resamples=100).iloc[0]

    assert row["note"] == ""
    assert row["n_a"] == 6 and row["n_b"] == 6
    # Every cell rose by exactly three, so the paired difference has no spread
    # and the standardised effect is not finite - which is the honest answer,
    # and the p-value is the one scipy gives for a constant difference.
    assert row["test"] == "paired_t"


def test_a_paired_test_refuses_when_the_units_do_not_match_up(tmp_path):
    # Both groups clear the three-unit floor; what they do not do is overlap, so
    # the refusal has to come from the pairing rather than from the count.
    records = []
    for index in range(6):
        records.append({"stem": "m", "condition": "c", "subject": "s",
                        "identity": index + 1, "window": "baseline",
                        "area_px_median": 10.0 + index})
    for index in range(4):
        records.append({"stem": "m", "condition": "c", "subject": "s",
                        "identity": index + 101, "window": "treatment",
                        "area_px_median": 20.0 + index})
    run = _pooled(tmp_path, pd.DataFrame(records), name="cell_summary_windowed")

    contrast = _contrast(table="cell_summary_windowed", group_by="window",
                         groups=["baseline", "treatment"], test="wilcoxon")
    row = run_contrasts(run, _config(contrast), resamples=0).iloc[0]

    assert "appear in both" in row["note"]


def test_naming_a_window_on_a_table_without_one_is_refused(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0] * 4, "treated": [2.0] * 4}))

    row = run_contrasts(run, _config(_contrast(window="baseline")),
                        resamples=0).iloc[0]

    assert "has no `window` column" in row["note"]


def test_a_window_is_filtered_to_before_testing(tmp_path):
    records = []
    for window, values in (("baseline", [1.0, 2.0, 3.0, 4.0]),
                           ("treatment", [90.0, 91.0, 92.0, 93.0])):
        for condition in ("control", "treated"):
            for index, value in enumerate(values):
                records.append({
                    "stem": f"{condition}_{index}", "condition": condition,
                    "subject": f"{condition}_{index}", "identity": 1,
                    "window": window,
                    "area_px_median": value + (5.0 if condition == "treated" else 0.0)})
    run = _pooled(tmp_path, pd.DataFrame(records), name="cell_summary_windowed")

    contrast = _contrast(table="cell_summary_windowed", window="treatment")
    row = run_contrasts(run, _config(contrast), resamples=0).iloc[0]

    assert row["window"] == "treatment"
    assert row["n_a"] == 4
    # The baseline rows were left out entirely rather than averaged in.
    assert row["cells_a"] == 4


# --------------------------------------------------------------- corrections


def test_benjamini_hochberg_against_a_hand_worked_set():
    """Four p-values, the step-up procedure done on paper.

    p = .01, .02, .03, .04 at m = 4 gives raw*m/rank = .04, .04, .04, .04, and
    the monotone pass down leaves all four at .04.
    """
    assert _benjamini_hochberg(np.array([0.01, 0.02, 0.03, 0.04])) == \
        pytest.approx([0.04, 0.04, 0.04, 0.04])
    # A clear winner among noise keeps its rank and stays below the rest. Here
    # raw*m/rank is .004, 1.0, .8, .7, and the monotone pass *down* from the
    # largest pulls the middle two back to .7 - the step that makes an adjusted
    # p-value interpretable at all, and the one easiest to leave out.
    adjusted = _benjamini_hochberg(np.array([0.001, 0.5, 0.6, 0.7]))
    assert adjusted == pytest.approx([0.004, 0.7, 0.7, 0.7])


def test_a_correction_never_returns_a_probability_above_one():
    for name, correct in CORRECTIONS.items():
        out = correct(np.array([0.4, 0.5, 0.9]))
        assert (out <= 1.0).all(), name


def test_a_correction_leaves_a_blank_blank():
    """A refusal row has no p-value, and it must not be counted in m either.

    Counting it would divide the evidence by the number of tests that could not
    be run, which is the wrong direction of conservative.
    """
    values = np.array([0.01, np.nan, 0.02])
    for name, correct in CORRECTIONS.items():
        out = correct(values)
        assert np.isnan(out[1]), name
    assert _bonferroni(values)[0] == pytest.approx(0.02)     # m = 2, not 3


def test_the_correction_is_within_a_family_and_never_across_one(tmp_path):
    """Proved against what a single pooled correction would have given.

    Two families of two tests each: corrected within, every p is multiplied by
    two under Bonferroni. Corrected across, every p would be multiplied by four.
    """
    table = _rows({"control": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                   "treated": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]})
    table["other_median"] = table["area_px_median"] * 2.0
    run = _pooled(tmp_path, table)

    config = _config(
        _contrast(name="primary", family="primary", correction="bonferroni",
                  metrics=["area_px_median", "other_median"]),
        _contrast(name="secondary", family="secondary", correction="bonferroni",
                  metrics=["area_px_median", "other_median"]),
    )
    statistics = run_contrasts(run, config, resamples=0)

    assert len(statistics) == 4
    for _, row in statistics.iterrows():
        raw, corrected = float(row["p_value"]), float(row["p_corrected"])
        assert corrected == pytest.approx(min(raw * 2, 1.0)), row["contrast"]
        assert corrected != pytest.approx(min(raw * 4, 1.0)) or raw * 4 >= 1.0


def test_significance_is_read_off_the_corrected_value(tmp_path):
    run = _pooled(tmp_path, _rows({"control": [1.0, 2.0, 3.0, 4.0, 5.0],
                                   "treated": [6.0, 7.0, 8.0, 9.0, 10.0]}))

    statistics = run_contrasts(run, _config(_contrast(alpha=0.05)), resamples=0)
    row = statistics.iloc[0]

    assert bool(row["significant"]) == (float(row["p_corrected"]) < 0.05)


# ------------------------------------------------------------- the declarations


@pytest.mark.parametrize("block, message", [
    ({"name": "Primary"}, "lower-case identifier"),
    ({"metrics": []}, "non-empty list"),
    ({"metrics": "area_px_median"}, "non-empty list"),
    ({"groups": ["only_one"]}, "at least two"),
    ({"groups": ["a", "a"]}, "named twice"),
    ({"unit": "dish"}, "not a unit of replication"),
    ({"test": "chi2"}, "not one this package offers"),
    ({"correction": "holm"}, "not one this package offers"),
    ({"alpha": 1.5}, "not between 0 and 1"),
    ({"unit": "cell", "aggregate": "mean"}, "nothing to aggregate"),
    ({"unit": "movie"}, "does not choose the statistic"),
    ({"unit": "movie", "aggregate": "mode"}, "not one this package knows"),
    ({"test": "wilcoxon", "groups": ["a", "b", "c"], "unit": "cell"},
     "paired test and compares two groups"),
])
def test_a_contrast_that_cannot_be_read_is_refused(block, message):
    base = {"name": "c", "table": "cell_summary", "metrics": ["m"],
            "group_by": "condition", "groups": ["a", "b"], "unit": "cell",
            "test": "mannwhitney"}
    base.update(block)
    with pytest.raises(ValueError, match=message):
        ContrastConfig.from_dict(base, TEST_NAMES, PAIRED_TESTS,
                                     CORRECTION_NAMES, SINGLE_GROUP_TESTS, {})


@pytest.mark.parametrize("key", ["table", "metrics", "group_by", "groups", "unit", "test"])
def test_every_required_setting_is_required(key):
    base = {"name": "c", "table": "cell_summary", "metrics": ["m"],
            "group_by": "condition", "groups": ["a", "b"], "unit": "cell",
            "test": "mannwhitney"}
    base.pop(key)
    with pytest.raises(ValueError, match="does not say"):
        ContrastConfig.from_dict(base, TEST_NAMES, PAIRED_TESTS,
                                     CORRECTION_NAMES, SINGLE_GROUP_TESTS, {})


def test_two_contrasts_in_one_family_may_not_disagree_about_the_correction():
    """The corrected p-value is the number anyone quotes.

    Whichever contrast was read last would otherwise silently decide for both.
    """
    blocks = [
        {"name": "a", "table": "t", "metrics": ["m"], "group_by": "condition",
         "groups": ["x", "y"], "unit": "cell", "test": "mannwhitney",
         "family": "primary", "correction": "bonferroni"},
        {"name": "b", "table": "t", "metrics": ["m"], "group_by": "condition",
         "groups": ["x", "y"], "unit": "cell", "test": "mannwhitney",
         "family": "primary", "correction": "benjamini_hochberg"},
    ]
    with pytest.raises(ValueError, match="one correction and one alpha"):
        _contrasts(blocks, {})


def test_two_contrasts_with_one_name_are_refused():
    block = {"name": "a", "table": "t", "metrics": ["m"], "group_by": "condition",
             "groups": ["x", "y"], "unit": "cell", "test": "mannwhitney"}
    with pytest.raises(ValueError, match="declared twice"):
        _contrasts([block, dict(block)], {})


def test_no_contrasts_parses_to_nothing_and_writes_nothing():
    assert _contrasts(None, {}) == []
    assert _contrasts([], {}) == []


def test_the_written_columns_are_what_plot_that_asks_for(tmp_path):
    """`statistics.csv` has to be `plot-that`'s `data/der/statistics.csv` as written.

    The skill requires the group n values, the test name, the effect size, the
    exact p and q, the correction method and alpha in the same row as the
    result. Renaming columns on the way into a figure bundle is where a
    correction method silently becomes the wrong one.
    """
    run = _pooled(tmp_path, _rows({"control": [1.0, 2.0, 3.0, 4.0],
                                   "treated": [5.0, 6.0, 7.0, 8.0]}))
    statistics = run_contrasts(run, _config(_contrast()), resamples=0)

    for column in ("test", "n_a", "n_b", "p_value", "p_corrected", "correction",
                   "alpha", "effect", "effect_kind", "significant", "methods"):
        assert column in statistics.columns, column
    assert statistics.iloc[0]["methods"].endswith(".")
    assert "unit of replication" in statistics.iloc[0]["methods"]


# ------------------------------- the same answer as the lab's own implementation


def test_hedges_g_agrees_with_circadian_workbench():
    """Copied, and checked against the original rather than trusted.

    Skipped rather than failed when the workbench is absent, so this is a check
    and not a dependency.
    """
    workbench = pytest.importorskip("circadian_workbench").statistics

    for x, y in (([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 9.0]),
                 ([10.0, 10.5, 11.0], [10.2, 10.4, 10.6]),
                 ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])):
        ours = hedges_g(x, y)
        theirs = workbench.hedges_g(x, y)
        if np.isnan(theirs):
            assert np.isnan(ours)
        else:
            assert ours == pytest.approx(theirs, rel=0, abs=0)


def test_bonferroni_and_sidak_agree_with_circadian_workbench():
    workbench = pytest.importorskip("circadian_workbench").statistics

    values = np.array([0.001, 0.02, np.nan, 0.4, 0.9])
    assert np.allclose(_bonferroni(values),
                       workbench.adjust_pvalues(values, "bonferroni"),
                       rtol=0, atol=0, equal_nan=True)
    assert np.allclose(_sidak(values),
                       workbench.adjust_pvalues(values, "sidak"),
                       rtol=0, atol=0, equal_nan=True)


def test_the_small_n_floor_is_the_workbenchs():
    """Three units, the same count its ``_small_n_result`` refuses below."""
    assert MIN_UNITS == 3


def test_a_paired_effect_is_standardised_by_the_spread_of_the_differences():
    """Not by either group's spread - the pairing is what removed that.

    Dividing by the between-unit variation again would undo the design and
    report a smaller effect than the experiment actually has.
    """
    differences = np.array([2.0, 3.0, 2.5, 3.5, 2.2])
    expected_d = float(np.mean(differences)) / float(np.std(differences, ddof=1))
    correction = 1.0 - 3.0 / (4.0 * (len(differences) - 1) - 1.0)

    assert paired_hedges_g(differences) == pytest.approx(correction * expected_d)
    assert np.isnan(paired_hedges_g([1.0]))
    assert np.isnan(paired_hedges_g([2.0, 2.0, 2.0]))


# ------------------------------------------- one group, against zero not a group


def test_a_one_group_test_compares_against_zero(tmp_path):
    """The shape of question a single-group table supports, and the only one.

    A slope, a change and a difference all have a meaningful zero. Asking
    whether a set of them sits away from it is a real comparison; asking a
    two-group test the same question means inventing a second group.
    """
    run = _pooled(tmp_path, _rows({"treated": [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]}))
    row = run_contrasts(run, _config(_contrast(test="signed_rank",
                                               groups=["treated"])),
                        resamples=200).iloc[0]

    assert row["note"] == ""
    assert float(row["p_value"]) < 0.05
    assert row["effect_kind"] == "median_against_zero"
    assert float(row["effect"]) == pytest.approx(0.65)
    assert float(row["statistic"]) == pytest.approx(
        stats.wilcoxon([0.4, 0.5, 0.6, 0.7, 0.9, 1.1]).statistic)


def test_a_one_group_result_names_one_group_and_leaves_the_other_blank(tmp_path):
    """There is no second group, and a blank says so where a name would lie."""
    run = _pooled(tmp_path, _rows({"treated": [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]}))
    row = run_contrasts(run, _config(_contrast(test="signed_rank",
                                               groups=["treated"])),
                        resamples=40).iloc[0]

    assert row["group_a"] == "treated"
    assert row["group_b"] == ""
    assert int(row["n_a"]) == 6
    assert row["n_b"] == ""
    assert int(row["n_groups"]) == 1


def test_a_one_group_effect_carries_a_resampled_interval(tmp_path):
    run = _pooled(tmp_path, _rows({"treated": [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]}))
    row = run_contrasts(run, _config(_contrast(test="signed_rank",
                                               groups=["treated"])),
                        resamples=400).iloc[0]
    assert float(row["effect_lo"]) <= float(row["effect"]) <= float(row["effect_hi"])


def test_values_sitting_on_zero_are_not_called_a_change(tmp_path):
    """The null this test is against, drawn from the same machinery."""
    run = _pooled(tmp_path, _rows({"treated": [-0.6, 0.5, -0.4, 0.45, -0.5, 0.55]}))
    row = run_contrasts(run, _config(_contrast(test="signed_rank",
                                               groups=["treated"])),
                        resamples=200).iloc[0]
    assert float(row["p_value"]) > 0.05
    assert not bool(row["significant"])


def test_a_one_group_test_named_with_two_groups_is_refused():
    """Refused where it is written rather than ignored where it is run.

    A second group parses perfectly well and would then be dropped in silence,
    leaving a row that says it compared two things and did not.
    """
    with pytest.raises(ValueError, match="one group against zero"):
        _contrast(test="signed_rank", groups=["control", "treated"])


def test_a_two_group_test_still_needs_two_groups():
    with pytest.raises(ValueError, match="at least two"):
        _contrast(test="mannwhitney", groups=["treated"])


def test_a_one_group_test_with_too_few_cells_is_refused_like_any_other(tmp_path):
    run = _pooled(tmp_path, _rows({"treated": [0.4, 0.5]}))
    row = run_contrasts(run, _config(_contrast(test="signed_rank",
                                               groups=["treated"])),
                        resamples=40).iloc[0]
    assert row["p_value"] == ""
    assert str(MIN_UNITS) in row["note"]


# ------------------------------- the table has to exist before anything is run


def _declared(table: str, enabled: list[str] | None = None):
    from analysis.cli import contrast_table_state, table_writers

    config = AnalysisConfig(dataset="t", frame_interval_min=30.0, movies=[],
                            output_root=".", enabled_modules=enabled or [])
    return contrast_table_state(config, _contrast(table=table), table_writers())


def test_a_contrast_against_a_table_nothing_writes_is_caught_before_measuring():
    """The failure it replaces costs an hour and points at the wrong step.

    A contrast naming a table that will not exist runs perfectly, writes a
    refusal per metric, and gives "not in pooled/tables" as the reason - which
    reads as a pooling problem rather than as the misspelling it is.
    """
    state, writer = _declared("trned")
    assert state == "NO SUCH TABLE"
    assert writer is None


def test_a_contrast_against_a_switched_off_module_is_caught_too():
    """The one that actually happened: the module exists and is not enabled."""
    state, writer = _declared("trend", enabled=["morphology", "rhythms"])
    assert state == "MODULE IS OFF"
    assert writer == "trend"


def test_a_contrast_against_an_enabled_module_passes():
    assert _declared("trend", enabled=["morphology", "trend"]) == ("", "trend")


def test_an_empty_enabled_list_means_every_module_rather_than_none():
    """The same reading `run` takes; the two must not disagree."""
    assert _declared("trend", enabled=[])[0] == ""


def test_the_windowed_tables_belong_to_a_step_that_cannot_be_switched_off():
    """They are not written by a module, so looking one up must not fail."""
    state, writer = _declared("cell_summary_windowed", enabled=["morphology"])
    assert state == ""
    assert writer == "window"


# ------------------------------------------------------------- metric groups


def test_a_contrast_may_name_a_metric_group_instead_of_listing_columns():
    """One set of columns, written once, tested and drawn from the same place."""
    from analysis.metric_groups import build

    groups = build({"circadian": ["cosinor_amplitude", "m10", "l5"]})
    contrast = _contrast(metrics=["@circadian"], metric_groups=groups)
    assert contrast.metrics == ("cosinor_amplitude", "m10", "l5")


def test_a_group_reference_mixes_with_plain_column_names():
    from analysis.metric_groups import build

    groups = build({"circadian": ["m10", "l5"]})
    contrast = _contrast(metrics=["area_px", "@circadian"], metric_groups=groups)
    assert contrast.metrics == ("area_px", "m10", "l5")


def test_a_column_in_two_groups_is_not_tested_twice():
    """Otherwise the correction is applied across the same column twice."""
    from analysis.metric_groups import build

    groups = build({"a": ["m10", "l5"], "b": ["l5", "area_px"]})
    contrast = _contrast(metrics=["@a", "@b"], metric_groups=groups)
    assert contrast.metrics == ("m10", "l5", "area_px")


def test_a_contrast_naming_a_group_nobody_declared_says_which_contrast():
    with pytest.raises(ValueError, match="contrast 'c' metrics"):
        _contrast(metrics=["@nope"])
