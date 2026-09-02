"""Tests for figure wording.

Run with ``python -m pytest analysis/test_figure_text.py``.

The property under test is a negative one: nothing in the package decides what
a figure means. A builder supplies a description of its axes, and every word is
replaceable by the user.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from _text import SLOTS, figure_text  # noqa: E402


def _run(tmp_path: Path, block: dict | None = None) -> Path:
    if block is not None:
        (tmp_path / "figures.json").write_text(
            json.dumps({"figures": block}), encoding="utf-8")
    return tmp_path


def test_the_builder_default_is_used_when_nobody_says_otherwise(tmp_path):
    text = figure_text(_run(tmp_path), "demo", argv=[], title="Area against time")
    assert text.title == "Area against time"
    assert text.source["title"] == "default"


def test_a_configuration_entry_replaces_the_default(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "Cells get bigger"}})
    text = figure_text(run, "demo", argv=[], title="Area against time")
    assert text.title == "Cells get bigger"
    assert text.source["title"] == "config"


def test_a_flag_replaces_the_configuration(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "From config"}})
    text = figure_text(run, "demo", argv=["--title", "From flag"], title="Default")
    assert text.title == "From flag"
    assert text.source["title"] == "flag"


def test_replacing_one_slot_leaves_the_others_alone(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "Mine"}})
    text = figure_text(run, "demo", argv=[], title="Theirs", subtitle="83 cells")
    assert text.title == "Mine" and text.subtitle == "83 cells"


def test_an_empty_entry_takes_the_text_off_the_figure(tmp_path):
    """Replacing a default is not enough; a user must be able to delete one."""
    run = _run(tmp_path, {"demo": {"footnote": ""}})
    text = figure_text(run, "demo", argv=[], footnote="Both spreads are IQRs.")
    assert text.footnote == "" and text.source["footnote"] == "config"


def test_an_absent_entry_is_not_an_empty_one(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "mine"}})
    text = figure_text(run, "demo", argv=[], footnote="kept")
    assert text.footnote == "kept" and text.source["footnote"] == "default"


def test_a_flag_can_blank_a_slot_too(tmp_path):
    text = figure_text(_run(tmp_path), "demo", argv=["--subtitle="], subtitle="83 cells")
    assert text.subtitle == ""


def test_an_equals_form_works_too(tmp_path):
    text = figure_text(_run(tmp_path), "demo", argv=["--title=Inline"], title="Default")
    assert text.title == "Inline"


def test_a_figures_block_for_another_figure_is_ignored(tmp_path):
    run = _run(tmp_path, {"other-figure": {"title": "Not mine"}})
    text = figure_text(run, "demo", argv=[], title="Mine")
    assert text.title == "Mine"


# ------------------------------------------------------------------ refusing

def test_a_typo_in_a_flag_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown option"):
        figure_text(_run(tmp_path), "demo", argv=["--titel", "oops"])


def test_a_typo_in_the_configuration_is_rejected(tmp_path):
    run = _run(tmp_path, {"demo": {"titel": "oops"}})
    with pytest.raises(SystemExit, match="unknown key"):
        figure_text(run, "demo", argv=[])


def test_a_builder_cannot_invent_a_slot(tmp_path):
    with pytest.raises(KeyError, match="unknown text slot"):
        figure_text(_run(tmp_path), "demo", argv=[], conclusion="cells move")


def test_a_flag_without_a_value_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="needs a value"):
        figure_text(_run(tmp_path), "demo", argv=["--title"])


# ------------------------------------------------------------------- claims

def test_a_claim_has_no_default(tmp_path):
    """The package does not interpret figures, so it has nothing to say here."""
    text = figure_text(_run(tmp_path), "demo", argv=[], title="Area against time")
    assert text.claim == ""


def test_a_claim_is_reproduced_exactly_as_written(tmp_path):
    written = "Cells in HCQ are 30% smaller (p = 0.02, n = 6)."
    run = _run(tmp_path, {"demo": {"claim": written}})
    assert figure_text(run, "demo", argv=[]).claim == written


def test_every_slot_is_recorded_with_where_it_came_from(tmp_path):
    run = _run(tmp_path, {"demo": {"subtitle": "s"}})
    text = figure_text(run, "demo", argv=["--title", "t"], footnote="f")
    assert set(text.source) == set(SLOTS)
    assert text.source["title"] == "flag"
    assert text.source["subtitle"] == "config"
    assert text.source["footnote"] == "default"
    assert text.source["claim"] == "default"
    assert json.dumps(text.as_dict()), "the wording goes into the bundle"


# ------------------------------------------------- nothing infers, anywhere

def test_no_builder_reads_a_removed_verdict_module():
    """The rules that chose a conclusion are gone; nothing may quietly return."""
    figures = Path(__file__).resolve().parent / "figures"
    for builder in sorted(figures.glob("[0-9][0-9]_*.py")):
        source = builder.read_text(encoding="utf-8")
        for banned in ("_verdicts", "analysis.verdict", "VERDICT", "decide("):
            assert banned not in source, f"{builder.name} still references {banned}"


def test_the_package_ships_no_conclusion_rules():
    package = Path(__file__).resolve().parent
    assert not (package / "verdict.py").exists()
    assert not (package / "figures" / "_verdicts.py").exists()


# ---------------------------------------------- the options half of the block

# Options resolve the same three ways as the wording and share the same
# `figures.<slug>` entry, under an `options` key. They resolve against a
# different vocabulary - whatever that figure declared, rather than the five
# slots this module owns - so both halves refuse a typo by name.

import pathlib  # noqa: E402
import _schema  # noqa: E402
from _schema import FigureContext, FigureSpec, Option, Panel, Table  # noqa: E402
from _text import OPTIONS_KEY, _from_run  # noqa: E402


def _panel(ax, values, theme):
    return None


def _demo_spec(**overrides) -> FigureSpec:
    defaults = dict(
        slug="demo", number=99, title="A demo", grammar="histogram",
        build=lambda ctx: None,
        panels=(Panel("only", _panel),),
        reads=(Table("cell_frame.csv", module="motility"),),
        options=(Option("bins", default=45),
                 Option("metrics", default=["area_px"])),
        source=Path(__file__),
    )
    defaults.update(overrides)
    return FigureSpec(**defaults)


def _demo_context(run: Path, argv: list[str], spec: FigureSpec | None = None
                  ) -> FigureContext:
    return FigureContext(
        spec=spec or _demo_spec(), run=run, tables=run, theme=None, bundle=run,
        summary={"minutes_per_frame": 30.0, "stem": "demo"}, field={}, stem=None,
        argv=list(argv))


def test_a_flag_beats_a_configured_option(tmp_path):
    run = _run(tmp_path, {"demo": {OPTIONS_KEY: {"bins": 60}}})
    ctx = _demo_context(run, ["--bins", "12"])
    assert ctx.option("bins") == 12
    assert ctx.option_source["bins"] == "flag"


def test_a_configured_option_beats_the_spec_default(tmp_path):
    run = _run(tmp_path, {"demo": {OPTIONS_KEY: {"bins": 60}}})
    ctx = _demo_context(run, [])
    assert ctx.option("bins") == 60
    assert ctx.option_source["bins"] == "config"


def test_an_absent_option_gives_the_spec_default(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "t"}})
    ctx = _demo_context(run, [])
    assert ctx.option("bins") == 45
    assert ctx.option_source["bins"] == "default"


def _side(tmp_path: pathlib.Path, name: str, block: dict) -> pathlib.Path:
    folder = tmp_path / name
    folder.mkdir()
    return _run(folder, block)


def test_a_number_and_the_same_number_as_text_arrive_the_same(tmp_path):
    typed = _demo_context(_side(tmp_path, "a", {"demo": {OPTIONS_KEY: {"bins": 60}}}), [])
    quoted = _demo_context(_side(tmp_path, "b", {"demo": {OPTIONS_KEY: {"bins": "60"}}}), [])
    assert typed.option("bins") == quoted.option("bins") == 60


def test_a_list_and_a_comma_separated_string_arrive_the_same(tmp_path):
    listed = _demo_context(
        _side(tmp_path, "a", {"demo": {OPTIONS_KEY: {"metrics": ["a", "b"]}}}), [])
    joined = _demo_context(
        _side(tmp_path, "b", {"demo": {OPTIONS_KEY: {"metrics": "a,b"}}}), [])
    assert listed.option("metrics") == joined.option("metrics") == ["a", "b"]


def test_a_null_option_means_use_the_default(tmp_path):
    """Unlike a text slot, where null is the empty string and takes a line off."""
    run = _run(tmp_path, {"demo": {OPTIONS_KEY: {"bins": None}}})
    ctx = _demo_context(run, [])
    assert ctx.option("bins") == 45
    assert ctx.option_source["bins"] == "default"


def test_a_configured_option_the_figure_does_not_declare_is_refused(tmp_path):
    run = _run(tmp_path, {"demo": {OPTIONS_KEY: {"bin": 60}}})
    ctx = _demo_context(run, [])
    with pytest.raises(SystemExit) as raised:
        ctx.option("bins")
    assert "'bin'" in str(raised.value)
    assert "bins" in str(raised.value) and "metrics" in str(raised.value)


def test_a_text_slot_typo_is_still_refused_by_name(tmp_path):
    run = _run(tmp_path, {"demo": {"titel": "oops"}})
    with pytest.raises(SystemExit, match="titel"):
        figure_text(run, "demo", argv=[], title="Area against time")


def test_the_options_block_is_not_mistaken_for_a_text_slot(tmp_path):
    run = _run(tmp_path, {"demo": {"title": "t", OPTIONS_KEY: {"bins": 60}}})
    text = figure_text(run, "demo", argv=[])
    assert text.title == "T"
    wording, options = _from_run(run, "demo")
    assert wording == {"title": "t"} and options == {"bins": 60}


def test_a_run_that_recorded_nothing_leaves_every_option_at_its_default(tmp_path):
    ctx = _demo_context(tmp_path, [])
    assert ctx.option("bins") == 45


# ------------------------------------------- a panel's own default for an option


def test_a_panel_keeps_its_own_default_while_nobody_has_chosen(tmp_path):
    """Figure 1's phase histogram is 24 h wide and steps 6 h, not 24.

    `option_or` is how a panel says "mine, unless somebody asked": the flag
    still drives everything, and only the untouched case differs.
    """
    spec = _demo_spec(options=(Option("hour_ticks", default=None),))
    run = _run(tmp_path, {"demo": {"title": "t"}})
    ctx = _demo_context(run, [], spec)
    assert ctx.option("hour_ticks") is None
    assert ctx.option_or("hour_ticks", 6.0) == 6.0


def test_a_configured_step_beats_the_panels_own_default(tmp_path):
    """The old test was "was the flag given", which missed the config block."""
    spec = _demo_spec(options=(Option("hour_ticks", default=None),))
    run = _run(tmp_path, {"demo": {OPTIONS_KEY: {"hour_ticks": 12}}})
    ctx = _demo_context(run, [], spec)
    assert ctx.option_or("hour_ticks", 6.0) == 12


def test_a_flag_beats_the_panels_own_default(tmp_path):
    spec = _demo_spec(options=(Option("hour_ticks", default=None),))
    run = _run(tmp_path, {"demo": {"title": "t"}})
    ctx = _demo_context(run, ["--hour-ticks", "8"], spec)
    assert ctx.option_or("hour_ticks", 6.0) == 8.0
