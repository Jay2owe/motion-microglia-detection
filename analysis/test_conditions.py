"""Tests for the condition system.

Run with ``python -m pytest analysis/test_conditions.py``.

Most of these pin a *refusal*. Getting a condition wrong is the one mistake in
this package that produces a complete, plausible, entirely incorrect result, so
the interesting behaviour is not what it assigns but what it declines to.
"""

from __future__ import annotations

import json

import pytest

from analysis.conditions import ConditionSet, resolve_colour
from analysis.theme import BASE_COLOURS, load_theme


def _treatment() -> ConditionSet:
    return ConditionSet.from_config({"HCQ": r"_A\d", "vehicle": r"_B\d"})


def _crossed() -> ConditionSet:
    return ConditionSet.from_config([
        {"name": "HCQ", "label": "+HCQ", "factor": "treatment", "match": r"_A\d"},
        {"name": "vehicle", "label": "Vehicle", "factor": "treatment",
         "match": r"_B\d", "control": True},
        {"name": "young", "factor": "age", "match": r"^3m"},
        {"name": "old", "factor": "age", "match": r"^18m"},
    ])


# ------------------------------------------------------------------ deriving

def test_the_shorthand_form_is_a_name_and_a_regex():
    """The whole point: two lines of configuration and the names do the rest."""
    conditions = _treatment()
    assert conditions.assign("VID95_A3").condition == "HCQ"
    assert conditions.assign("VID95_B1").condition == "vehicle"


def test_a_declared_condition_beats_a_derived_one():
    """One awkward file is fixed by naming it, not by contorting the pattern."""
    assignment = _treatment().assign("VID95_A3", declared="vehicle")
    assert assignment.condition == "vehicle"
    assert assignment.source == "declared"


def test_a_declared_condition_must_be_one_that_exists():
    assignment = _treatment().assign("VID95_A3", declared="saline")
    assert not assignment.ok
    assert "not declared" in assignment.problem


def test_matching_is_case_insensitive():
    assert _treatment().assign("vid95_a3").condition == "HCQ"


def test_the_assignment_records_which_pattern_matched():
    """A table saying only 'HCQ' cannot be checked; one saying why can."""
    assignment = _treatment().assign("VID95_A3")
    assert assignment.evidence == {"condition": r"_A\d"}


# ------------------------------------------------------------------ refusing

def test_two_patterns_matching_one_name_is_an_error_not_a_race():
    """No first-hit-wins ordering to remember, and no silent wrong answer."""
    assignment = _treatment().assign("VID95_A3_B1")
    assert not assignment.ok
    assert assignment.source == "ambiguous"
    assert "must not overlap" in assignment.problem


def test_a_name_no_pattern_matches_is_unassigned_and_says_so():
    assignment = _treatment().assign("VID95_C2")
    assert not assignment.ok
    assert assignment.condition == "unassigned"
    assert assignment.source == "unassigned"
    assert r"HCQ=/_A\d/" in assignment.problem, "the message must show what was tried"


def test_no_conditions_declared_is_not_an_error():
    """An experiment with no groups is a legitimate experiment."""
    assignment = ConditionSet().assign("VID95_A3")
    assert assignment.ok
    assert assignment.condition == "unassigned"


def test_a_bare_condition_string_survives_with_no_conditions_block():
    """Backwards compatible: a movie that names its own group keeps it."""
    assignment = ConditionSet().assign("VID95_A3", declared="control")
    assert assignment.condition == "control"
    assert assignment.source == "declared"


def test_an_invalid_regex_is_a_configuration_error():
    with pytest.raises(ValueError, match="invalid regular expression"):
        ConditionSet.from_config({"broken": "_A(\\d"})


def test_a_name_cannot_be_declared_twice():
    with pytest.raises(ValueError, match="declared twice"):
        ConditionSet.from_config([
            {"name": "HCQ", "match": r"_A\d"},
            {"name": "HCQ", "match": r"_C\d"},
        ])


def test_a_factor_has_at_most_one_control():
    with pytest.raises(ValueError, match="one reference group"):
        ConditionSet.from_config([
            {"name": "a", "match": "a", "control": True},
            {"name": "b", "match": "b", "control": True},
        ])


# ------------------------------------------------------------------- crossed

def test_two_factors_are_resolved_independently_and_combined():
    conditions = _crossed()
    assert conditions.factors() == ("treatment", "age")
    assignment = conditions.assign("3m_VID95_A3")
    assert assignment.condition == "HCQ_young"
    assert assignment.label == "+HCQ / young"
    assert assignment.factors == {"treatment": "HCQ", "age": "young"}


def test_one_unresolved_factor_leaves_the_whole_movie_unassigned():
    """Half a condition is not a condition; it must not be pooled with anything."""
    assignment = _crossed().assign("VID95_A3")
    assert not assignment.ok
    assert assignment.condition == "unassigned"
    assert "age" in assignment.problem


def test_the_control_is_found_within_its_own_factor():
    assert _crossed().control("treatment").name == "vehicle"


# ------------------------------------------------------------------- colours

def test_an_unnamed_condition_gets_a_colourblind_safe_colour():
    colours = _treatment().colours()
    assert set(colours) == {"HCQ", "vehicle"}
    assert len(set(colours.values())) == 2
    for value in colours.values():
        assert value in BASE_COLOURS.values()


def test_auto_colours_follow_declaration_order_not_alphabet():
    """Adding a group at the end must not recolour the groups before it."""
    first = ConditionSet.from_config({"HCQ": "a", "vehicle": "b"}).colours()
    later = ConditionSet.from_config({"HCQ": "a", "vehicle": "b", "wash": "c"}).colours()
    assert later["HCQ"] == first["HCQ"] and later["vehicle"] == first["vehicle"]


def test_a_named_colour_is_resolved_when_the_configuration_is_read():
    conditions = ConditionSet.from_config({"HCQ": {"match": "a", "colour": "red"}})
    assert conditions.colours()["HCQ"] == BASE_COLOURS["red"]


def test_a_nonsense_colour_is_rejected_at_configuration_time():
    with pytest.raises(KeyError, match="neither a"):
        resolve_colour("nrable", "HCQ")


def test_more_conditions_than_safe_colours_refuses_rather_than_repeats():
    many = {f"c{i}": f"_{i}_" for i in range(9)}
    with pytest.raises(ValueError, match="would look"):
        ConditionSet.from_config(many).colours()


def test_a_condition_colour_never_reaches_a_role():
    """analysis_kit's rule: declaring a condition called 'ink' recolours that
    condition, and not the axes of every figure."""
    theme = load_theme("house").with_conditions({"ink": "#ff00ff"})
    assert theme.condition_colour("ink") == "#ff00ff"
    assert theme.colour("ink") == BASE_COLOURS["black"]


def test_a_role_colour_never_reaches_a_condition_by_accident():
    """A group nobody declared still draws, in a safe slot chosen by index."""
    theme = load_theme("house")
    first = theme.condition_colour("mystery", 0)
    second = theme.condition_colour("other", 1)
    assert first.startswith("#") and first != second


def test_a_theme_loaded_from_a_configuration_carries_its_condition_colours(tmp_path):
    """A figure is handed one object, not a theme plus a colour table."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "dataset": "x", "frame_interval_min": 30, "movies": [],
        "conditions": {"HCQ": {"match": "a", "colour": "red"}},
    }), encoding="utf-8")
    theme = load_theme(path)
    assert theme.condition_colour("HCQ") == BASE_COLOURS["red"]
    assert theme.preset == "house", "a conditions block must not disturb the look"


# --------------------------------------------------------------------- audit

def test_the_audit_warns_about_a_design_nothing_can_be_tested_on():
    conditions = _treatment()
    rows = [conditions.assign("VID95_A3")]
    notes = " ".join(conditions.audit(rows))
    assert "one movie" in notes
    assert "at least two groups" in notes


def test_the_audit_warns_when_no_control_is_named():
    conditions = _treatment()
    notes = " ".join(conditions.audit([conditions.assign("VID95_A3")]))
    assert "no reference group" in notes
    assert not any("reference group" in n for n in _crossed().audit([]))


def test_the_design_is_recorded_in_full():
    """What ends up in the manifest must be enough to re-derive the assignment."""
    recorded = _crossed().as_dict()
    assert recorded["factors"] == ["treatment", "age"]
    entry = next(c for c in recorded["conditions"] if c["name"] == "HCQ")
    assert entry["match"] == [r"_A\d"] and entry["colour"].startswith("#")
    assert json.dumps(recorded), "the design must be JSON-serialisable for the manifest"


def test_the_recorded_design_round_trips():
    """A run folder's conditions.json must rebuild the same set months later."""
    original = _crossed()
    rebuilt = ConditionSet.from_config(original.as_dict()["conditions"])
    assert rebuilt.colours() == original.colours()
    assert rebuilt.assign("18m_VID95_B1").condition == "vehicle_old"


# ----------------------------------------------------------------- synthetic

def _synthetic() -> ConditionSet:
    return ConditionSet.from_config({
        "synthetic": True,
        "conditions": {"treated": r"_A\d", "control": r"_B\d"},
    })


def test_a_synthetic_design_says_so_in_every_place_it_is_recorded():
    """Invented groups reading as real ones is the failure this module exists
    to stop, so the flag travels with the design rather than with the folder."""
    conditions = _synthetic()
    assert conditions.synthetic
    assert conditions.as_dict()["synthetic"] is True
    assert any("SYNTHETIC" in note for note in conditions.audit([]))


def test_a_real_design_is_never_marked_synthetic_by_accident():
    assert _treatment().as_dict()["synthetic"] is False
    assert not any("SYNTHETIC" in note for note in _treatment().audit([]))


def test_the_synthetic_flag_survives_the_round_trip():
    """A run folder that loses the flag becomes a folder of fabricated results
    indistinguishable from measured ones."""
    rebuilt = ConditionSet.from_config(_synthetic().as_dict())
    assert rebuilt.synthetic is True
    assert rebuilt.assign("demo_A1").condition == "treated"
