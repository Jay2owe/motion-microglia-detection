"""Named sets of measured columns: what resolves, and what is refused.

The point of a group is that a set of columns is written once and checked once.
Every test here is about the checking half: a group that quietly resolved to
nothing, or to a column no module writes, would draw a blank figure and say
nothing about why - which is the failure this package exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.config import AnalysisConfig
from analysis.metric_groups import (COLUMN_SET_SETTINGS, PREFIX, SELECTORS,
                                    build, group_name, is_reference,
                                    resolve_metrics, resolve_one, resolve_setting)

#: Six real columns, all written by the rhythms module and all role `rhythmic`.
CIRCADIAN = ["cosinor_amplitude", "cosinor_relative_amplitude",
             "cosinor_peak_hour", "relative_amplitude", "m10", "l5"]


def _config_file(tmp_path: Path, **blocks) -> Path:
    """The smallest configuration that loads, plus whatever is being tested."""
    body = {
        "dataset": "test",
        "frame_interval_min": 30,
        "movies": [{"stem": "m1", "labels": "labels.tif", "raw": "raw.tif"}],
    }
    body.update(blocks)
    path = tmp_path / "analysis_config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ------------------------------------------------------------ the two forms


def test_a_list_group_is_taken_literally_and_in_the_order_written():
    groups = build({"circadian": CIRCADIAN})
    assert list(groups["circadian"].columns) == CIRCADIAN
    assert groups["circadian"].declared == CIRCADIAN


def test_a_selector_group_reads_what_the_modules_declare():
    groups = build({"drift": {"module": "trend"}})
    assert "slope_per_hour" in groups["drift"].columns
    assert "cosinor_amplitude" not in groups["drift"].columns


def test_two_selector_keys_are_anded_not_ored():
    both = build({"g": {"module": "rhythms", "role": "rhythmic"}})["g"].columns
    by_module = build({"g": {"module": "rhythms"}})["g"].columns
    assert set(both) < set(by_module)
    assert set(both) == {c for c in by_module if c in set(both)}


def test_a_selector_group_comes_out_in_a_stable_alphabetical_order():
    """The same configuration has to draw the same figures in the same order.

    A group's order is the order its figures appear in on disk and in the plan
    manifest, so an order that depended on which module registered first would
    shuffle a plan every time a module was added.
    """
    columns = build({"drift": {"module": "trend"}})["drift"].columns
    assert list(columns) == sorted(columns)
    assert build({"drift": {"module": "trend"}})["drift"].columns == columns


def test_the_declaration_is_kept_beside_what_it_resolved_to():
    group = build({"drift": {"module": "trend"}})["drift"]
    assert group.declared == {"module": "trend"}
    assert len(group.columns) > 1


# ------------------------------------------------------------- what is refused


def test_a_list_naming_a_column_no_module_writes_is_refused_by_name():
    with pytest.raises(ValueError, match="area_pxx"):
        build({"shape": ["area_px", "area_pxx"]})


def test_a_near_miss_is_offered_the_column_it_probably_meant():
    with pytest.raises(ValueError, match="did you mean"):
        build({"shape": ["area_pix"]})


def test_a_selector_key_this_package_does_not_offer_is_refused():
    with pytest.raises(ValueError, match=f"selects by {' or '.join(SELECTORS)}"):
        build({"shape": {"grammar": "histogram"}})


def test_a_group_by_table_is_refused_with_the_reason_it_cannot_work():
    """`table` is the selector everybody reaches for first, and it cannot work.

    A module may write several tables and a column declaration does not say
    which one it lands in, so honouring it would mean guessing.
    """
    with pytest.raises(ValueError, match="may write several tables"):
        build({"shape": {"table": "cell_frame"}})


def test_a_selector_matching_nothing_is_refused_rather_than_returned_empty():
    with pytest.raises(ValueError, match="matches no declared column"):
        build({"shape": {"module": "nonesuch"}})


def test_a_selector_that_selects_nothing_at_all_is_refused():
    with pytest.raises(ValueError, match="every column ever measured"):
        build({"shape": {}})


def test_a_group_may_not_be_built_out_of_another_group():
    with pytest.raises(ValueError, match="One level only"):
        build({"a": ["area_px"], "b": [f"{PREFIX}a"]})


def test_an_empty_group_is_refused():
    with pytest.raises(ValueError, match="is empty"):
        build({"shape": []})


def test_a_group_name_has_to_be_a_plain_lower_case_identifier():
    with pytest.raises(ValueError, match="plain lower-case identifier"):
        build({"Circadian": ["area_px"]})


def test_a_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(TypeError, match="block of name"):
        build(["circadian"])


def test_a_member_that_is_not_a_column_name_is_refused():
    with pytest.raises(ValueError, match="expected column names"):
        build({"shape": [7]})


# ------------------------------------------------------------- the references


def test_a_reference_is_a_string_that_starts_with_the_prefix():
    assert is_reference("@circadian")
    assert not is_reference("circadian")
    assert not is_reference("@")          # a prefix and nothing after it
    assert not is_reference(7)
    assert group_name("@circadian") == "circadian"


def test_a_plain_column_name_resolves_to_itself_in_a_list_of_one():
    assert resolve_one("area_px", {}) == ["area_px"]


def test_a_reference_to_a_group_nobody_declared_lists_the_ones_that_were():
    groups = build({"circadian": CIRCADIAN})
    with pytest.raises(ValueError, match="declared groups: circadian"):
        resolve_one("@cicadian", groups)


def test_a_group_is_expanded_in_place_keeping_the_order_written():
    groups = build({"circadian": CIRCADIAN})
    resolved = resolve_metrics(["area_px", "@circadian", "perimeter_px"],
                               groups, where="test")
    assert list(resolved) == ["area_px", *CIRCADIAN, "perimeter_px"]


def test_a_column_in_two_groups_keeps_its_first_position_and_appears_once():
    """A figure drawing one column twice is never what was meant.

    On a contrast it is worse than untidy: the column would be tested twice and
    the correction applied across it twice.
    """
    groups = build({"a": ["area_px", "m10"], "b": ["m10", "l5"]})
    assert list(resolve_metrics(["@a", "@b"], groups, where="test")) == [
        "area_px", "m10", "l5"]


def test_the_setting_a_bad_reference_was_written_in_is_named():
    with pytest.raises(ValueError, match="figures.rhythm-strength.options.metrics"):
        resolve_metrics(["@nope"], {},
                        where="figures.rhythm-strength.options.metrics")


def test_a_metrics_list_that_is_not_a_list_is_refused():
    with pytest.raises(ValueError, match="expected a list of column names"):
        resolve_metrics("a,b", {}, where="test")


def test_a_value_holding_no_reference_comes_back_exactly_as_written():
    """`"a,b"` is for the option's own cast to split, not for this to interpret."""
    groups = build({"circadian": CIRCADIAN})
    assert resolve_setting("a,b", groups, where="test") == "a,b"
    assert resolve_setting(["a", "b"], groups, where="test") == ["a", "b"]
    assert resolve_setting(30, groups, where="test") == 30


def test_a_bare_reference_becomes_the_whole_group():
    groups = build({"circadian": CIRCADIAN})
    assert resolve_setting("@circadian", groups, where="test") == CIRCADIAN


# ----------------------------------------------------------- the configuration


def test_a_configuration_with_no_metric_groups_block_declares_none(tmp_path):
    config = AnalysisConfig.load(_config_file(tmp_path))
    assert config.metric_groups == {}
    assert config.metric_group_problems() == []


def test_the_figures_block_may_name_a_group(tmp_path):
    config = AnalysisConfig.load(_config_file(
        tmp_path,
        metric_groups={"circadian": CIRCADIAN},
        figures={"rhythm-strength": {"options": {"metrics": "@circadian"}}},
    ))
    assert config.figures["rhythm-strength"]["options"]["metrics"] == CIRCADIAN


def test_the_figures_block_is_left_alone_when_no_group_is_named(tmp_path):
    """A configuration that declares no groups must come out byte-identical."""
    block = {"rhythm-strength": {"title": "t", "options": {"metrics": "a,b",
                                                          "bins": 30}}}
    config = AnalysisConfig.load(_config_file(tmp_path, figures=block))
    assert config.figures == block


def test_groups_are_read_before_the_blocks_that_reference_them(tmp_path):
    """JSON key order must not decide whether a reference resolves.

    Parsing `contrasts` before `metric_groups` would fail with a message about
    a group nobody had misspelled, which is the kind of error that costs an
    afternoon.
    """
    written = {
        "metric_groups": {"circadian": CIRCADIAN},
        "figures": {"rhythm-strength": {"options": {"metrics": "@circadian"}}},
    }
    for order in (("metric_groups", "figures"), ("figures", "metric_groups")):
        body = {
            "dataset": "test", "frame_interval_min": 30,
            "movies": [{"stem": "m1", "labels": "l.tif", "raw": "r.tif"}],
        }
        for key in order:
            body[key] = written[key]
        path = tmp_path / f"{'_'.join(order)}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        config = AnalysisConfig.load(path)
        assert config.figures["rhythm-strength"]["options"]["metrics"] == CIRCADIAN


def test_a_group_naming_an_undeclared_column_stops_the_load(tmp_path):
    with pytest.raises(ValueError, match="no module says it writes"):
        AnalysisConfig.load(_config_file(
            tmp_path, metric_groups={"shape": ["area_pxx"]}))


def test_the_doctor_gets_lines_rather_than_a_traceback(tmp_path):
    """`metric_group_problems` reports; it must never raise.

    The doctor prints every fault it can find in one pass, so the first one
    must not hide the rest.
    """
    path = _config_file(tmp_path, metric_groups={"circadian": CIRCADIAN})
    config = AnalysisConfig.load(path)
    assert config.metric_group_problems() == []
    # Break the file under it, the way an edit between two doctor runs would.
    body = json.loads(path.read_text(encoding="utf-8"))
    body["metric_groups"] = {"shape": {"table": "cell_frame"}}
    path.write_text(json.dumps(body), encoding="utf-8")
    problems = config.metric_group_problems()
    assert problems and all(p.startswith("metric_groups: ") for p in problems)


# ----------------------------------------------------------------- the layering


def test_this_module_never_imports_the_drawing_half():
    """A run that only measures must stay possible without matplotlib."""
    source = (Path(__file__).resolve().parent / "metric_groups.py").read_text(
        encoding="utf-8")
    lines = [line for line in source.splitlines()
             if line.startswith(("import ", "from "))]
    assert not [line for line in lines if "figures" in line]
    assert not [line for line in lines if "matplotlib" in line]


def test_a_column_holding_one_name_is_not_a_place_to_drop_a_whole_group():
    """`size` sets point size from one column; a set of columns is not one."""
    assert "size" not in COLUMN_SET_SETTINGS
