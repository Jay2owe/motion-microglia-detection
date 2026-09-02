"""Tests for the figure schema.

Run with ``python -m pytest analysis/test_figure_schema.py``.

The property under test: a figure's declaration and its behaviour cannot drift
apart. Every option a builder honours is one it declared, every option it
declared is in the shared vocabulary, and ``--help`` lists exactly those. The
failure these prevent is silent - a flag accepted and ignored draws the wrong
figure and says nothing - so each one is written as the sentence it is checking.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

FIGURES = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(FIGURES))

from _options import OPTIONS, SWITCHES, exactly  # noqa: E402
from _text import SLOTS  # noqa: E402
import _schema  # noqa: E402
from _schema import (FigureContext, FigureResult, FigureSpec, Input, Option,  # noqa: E402
                     Panel, Stack, Table, catalogue, figure, load_all)


#: Panels the build function still draws itself, rather than through a function
#: in `panels/`. Every one is a panel no other page can compose with. The list
#: may shrink; a new entry has to be added deliberately, which is the point.
INLINE_PANELS: tuple[str, ...] = (
    "breath-trace.small_multiples",
    "breakout-triggered-average.alignment",
    "independence-map.measures",
    "negative-space.stages",
    "tissue-tectonics.contested",
    "tissue-tectonics.handoffs",
)


# ------------------------------------------------------------------ a fixture


def _panel(ax, values, theme):
    """A panel-shaped callable, for specs that never draw anything."""
    return None


def _block_panel(figure, rect, values, theme):
    return None


def _spec(**overrides) -> FigureSpec:
    """A declaration that is not registered, so a test cannot disturb the set."""
    defaults = dict(
        slug="demo", number=99, title="A demo", grammar="scatter",
        build=lambda ctx: None,
        panels=(Panel("left", _panel), Panel("right", _panel)),
        reads=(Table("cell_frame.csv", module="motility"),),
        options=(Option("bins", default=45), Option("metrics", default="area_px")),
        summary="a demo",
        source=Path(__file__),
    )
    defaults.update(overrides)
    return FigureSpec(**defaults)


def _context(spec: FigureSpec, argv: list[str], tables: Path | None = None
             ) -> FigureContext:
    return FigureContext(
        spec=spec, run=Path("."), tables=tables or Path("."), theme=None,
        bundle=Path("."), summary={"minutes_per_frame": 30.0, "stem": "demo"},
        field={}, stem=None, argv=list(argv),
    )


# ------------------------------------------------------------- the declaration


def test_every_numbered_builder_registers_exactly_one_figure():
    """A file in the numbered set is a figure, or it is on its way to being one."""
    rows = catalogue()
    assert len(rows) == 36, "the numbered set changed size; update this number"
    for number, path, spec in rows:
        if spec is None:
            continue
        assert spec.number == number, (
            f"{path.name} declares number {spec.number} but is filed under {number}")
        assert spec.source == path.resolve()


def test_no_two_figures_share_a_slug_or_a_number():
    specs = list(load_all().values())
    slugs = [spec.slug for spec in specs]
    numbers = [spec.number for spec in specs]
    assert len(set(slugs)) == len(slugs), "two figures claim one slug"
    assert len(set(numbers)) == len(numbers), "two figures claim one number"


def test_every_declared_option_is_in_the_shared_vocabulary():
    for spec in load_all().values():
        for declared in spec.options:
            assert declared.name in OPTIONS, (
                f"{spec.slug} declares --{declared.name}, which no other figure "
                f"can mean anything by")


def test_an_option_outside_the_vocabulary_is_refused_at_declaration_time():
    with pytest.raises(KeyError, match="shared vocabulary"):
        @figure(slug="not-registered", number=98, title="x", grammar="scatter",
                options=(Option("binz", default=1),))
        def build(ctx):
            return None


def test_every_spec_declares_at_least_one_source():
    """A figure that reads nothing drew nothing, or read it behind the bundle's back.

    A source is a table, an image stack a module wrote, or one of the movie's
    own stacks. A page of pixels is as much in need of a recorded source as a
    page of numbers - more, because a picture cannot be re-derived from itself.
    """
    for spec in load_all().values():
        assert spec.reads, (
            f"{spec.slug} declares no source, so its bundle cannot say where "
            f"the figure came from")


def test_every_panel_names_a_callable_that_takes_an_axes_or_a_figure():
    for spec in load_all().values():
        for panel in spec.panels:
            if panel.draw is None:
                continue        # drawn inline; counted by the test below
            first = list(inspect.signature(panel.draw).parameters)[0]
            expected = "figure" if panel.block else "ax"
            assert first == expected, (
                f"{spec.slug}.{panel.key} draws with {panel.draw.__name__}, whose "
                f"first parameter is {first!r}; a "
                f"{'block' if panel.block else 'plain'} panel takes {expected!r}")


def test_build_all_passes_through_exactly_what_every_figure_takes():
    """`build_all` writes the shared set out rather than importing it.

    It launches thirty-six subprocesses and has no other reason to load
    matplotlib, so the list is duplicated on purpose. This is what keeps the
    duplicate honest.
    """
    import build_all
    assert build_all.SHARED == {"stem", *_schema.UNIVERSAL_SWITCHES}


def test_the_panels_still_drawn_inline_are_the_ones_on_the_list():
    """A panel with no function in `panels/` is a panel no other page can use.

    The house rule is that a builder draws nothing itself. These are the pages
    that still do, listed rather than hidden, so the number can only go down.
    """
    inline = sorted(f"{spec.slug}.{panel.key}"
                    for spec in load_all().values()
                    for panel in spec.panels if panel.draw is None)
    assert len(inline) <= len(INLINE_PANELS), (
        "a new panel is drawn inline: " + ", ".join(set(inline) - set(INLINE_PANELS)))
    assert not set(inline) - set(INLINE_PANELS), (
        "unexpected inline panel(s): " + ", ".join(sorted(set(inline) - set(INLINE_PANELS))))


def test_every_panel_a_figure_declares_needs_something_it_reads():
    for spec in load_all().values():
        declared = {source.name for source in spec.reads}
        for panel in spec.panels:
            # A need may narrow to one column of a table it reads.
            missing = [need for need in panel.needs
                       if need.partition(":")[0] not in declared]
            assert not missing, (
                f"{spec.slug}.{panel.key} needs {', '.join(missing)}, which the "
                f"figure does not declare in reads=")


# ---------------------------------------------------------------- the refusals


def test_reading_an_undeclared_option_is_refused_not_defaulted():
    ctx = _context(_spec(), [])
    with pytest.raises(KeyError, match="does not declare"):
        ctx.option("quantile")


def test_an_unaccepted_flag_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit, match="not an option of demo"):
        _schema._check_argv(_spec(), ["run", "--quantile", "0.1"])


def test_the_refusal_names_the_options_this_figure_does_accept():
    with pytest.raises(SystemExit) as raised:
        _schema._check_argv(_spec(), ["run", "--quantile", "0.1"])
    assert "--bins" in str(raised.value) and "--metrics" in str(raised.value)


def test_text_slots_and_switches_are_accepted_by_every_figure():
    spec = _spec(options=())
    _schema._check_argv(spec, ["run", "--title", "x", "--draft", "--stem", "a"])


def test_a_switch_value_is_not_mistaken_for_a_flag():
    """``--draft <run>`` must not make the run folder look like an option."""
    _schema._check_argv(_spec(), ["--draft", "outputs/a03", "--bins", "12"])


def test_a_panel_this_figure_does_not_have_names_the_ones_it_does():
    ctx = _context(_spec(), ["run", "--panels", "middle"])
    with pytest.raises(SystemExit) as raised:
        ctx.panels()
    assert "left" in str(raised.value) and "right" in str(raised.value)


def test_panels_defaults_to_every_panel_in_declaration_order():
    ctx = _context(_spec(), ["run"])
    assert [panel.key for panel in ctx.panels()] == ["left", "right"]


def test_one_named_panel_can_be_the_whole_figure():
    ctx = _context(_spec(), ["run", "--panels", "right"])
    assert [panel.key for panel in ctx.panels()] == ["right"]


def test_a_panel_whose_table_is_missing_is_skipped_and_said_out_loud(tmp_path, capsys):
    spec = _spec(
        reads=(Table("cell_frame.csv", module="motility"),
               Table("presence.csv", module="presence", optional=True)),
        panels=(Panel("left", _panel),
                Panel("right", _panel, needs=("presence.csv",))),
    )
    (tmp_path / "cell_frame.csv").write_text("identity\n1\n", encoding="utf-8")
    ctx = _context(spec, ["run"], tables=tmp_path)
    assert [panel.key for panel in ctx.panels()] == ["left"]
    assert ctx.skipped_panels == [("right", ["presence.csv"])]
    assert "presence.csv" in capsys.readouterr().out


def test_asking_for_a_panel_the_run_cannot_draw_stops_rather_than_drawing_nothing(tmp_path):
    spec = _spec(
        reads=(Table("presence.csv", module="presence", optional=True),),
        panels=(Panel("right", _panel, needs=("presence.csv",)),),
    )
    ctx = _context(spec, ["run", "--panels", "right"], tables=tmp_path)
    with pytest.raises(SystemExit, match="presence.csv"):
        ctx.panels()


def test_reading_a_table_the_figure_did_not_declare_is_refused():
    ctx = _context(_spec(), [])
    with pytest.raises(KeyError, match="does not declare"):
        ctx.table("presence.csv")


# ------------------------------------------------------------- the resolution


def test_a_flag_beats_the_declared_default():
    ctx = _context(_spec(), ["run", "--bins", "12"])
    assert ctx.option("bins") == 12
    assert ctx.option_source["bins"] == "flag"


def test_an_absent_flag_gives_the_declared_default():
    ctx = _context(_spec(), ["run"])
    assert ctx.option("bins") == 45
    assert ctx.option_source["bins"] == "default"


def test_a_figure_may_narrow_the_shared_cast():
    """``--metrics`` is a list everywhere and a single column on a one-metric page."""
    spec = _spec(options=(Option("metrics", default="area_px", cast=str),))
    ctx = _context(spec, ["run", "--metrics", "perimeter_px"])
    assert ctx.option("metrics") == "perimeter_px"


def test_a_value_the_cast_refuses_stops_the_build_naming_the_flag():
    ctx = _context(_spec(), ["run", "--bins", "many"])
    with pytest.raises(SystemExit, match="--bins"):
        ctx.option("bins")


# ------------------------------------------------------------------- the help


def test_help_lists_this_figures_options_and_no_others():
    text = _spec().usage()
    assert "--bins" in text and "--metrics" in text
    assert "--quantile" not in text
    assert "--clusters" not in text


def test_help_names_the_panels_the_tables_and_the_defaults():
    text = _spec().usage()
    assert "left" in text and "right" in text
    assert "cell_frame.csv" in text and "motility" in text
    assert "45" in text


def test_help_says_which_options_every_figure_takes():
    text = _spec().usage()
    for slot in SLOTS:
        assert f"--{slot}" in text
    assert "--stem" in text
    assert "--draft" in text


def test_a_switch_this_figure_does_not_honour_is_not_offered_or_accepted():
    """``--overlay`` on a page with nothing to overlay is the same silence as
    ``--bins`` on a page with no histogram."""
    assert "--overlay" not in _spec().usage()
    with pytest.raises(SystemExit, match="not an option of demo"):
        _schema._check_argv(_spec(), ["run", "--overlay"])


def test_a_declared_switch_is_offered_and_accepted():
    spec = _spec(switches=("overlay",))
    assert "--overlay" in spec.usage()
    _schema._check_argv(spec, ["run", "--overlay"])


def test_a_figure_with_no_declared_options_still_prints_a_help_block():
    text = _spec(options=()).usage()
    assert "this figure declares none" in text


# ------------------------------------------------------------------ the title


def test_the_declared_title_is_filled_from_the_builds_numbers():
    spec = _spec(title="{metric} over {interval}")
    result = FigureResult(figure=None, axes=[], figure_data=pd.DataFrame(),
                          title_fields={"metric": "Area", "interval": "30 min"})
    assert _schema._default_title(spec, result) == "Area over 30 min"


def test_a_title_placeholder_the_build_did_not_supply_is_named():
    spec = _spec(title="{metric} over {interval}")
    result = FigureResult(figure=None, axes=[], figure_data=pd.DataFrame(),
                          title_fields={"metric": "Area"})
    with pytest.raises(SystemExit, match="interval"):
        _schema._default_title(spec, result)


def test_a_title_with_no_placeholders_is_left_exactly_as_written():
    spec = _spec(title="Half of all steps are under {0.31} px")
    result = FigureResult(figure=None, axes=[], figure_data=pd.DataFrame())
    assert _schema._default_title(spec, result) == "Half of all steps are under {0.31} px"


# ------------------------------------------------------------------- the note


def test_a_note_is_drawn_only_where_the_figure_declared_room_for_one():
    """`--note` on a page with nowhere to put it draws nothing, quietly.

    The alternative is worse: a note placed at a default position on a page
    laid out without one lands across the plot, and the user has to work out
    that their own flag did it.
    """
    from analysis.theme import load_theme
    import matplotlib.pyplot as plt

    theme = load_theme(None)
    figure_ = plt.figure(figsize=(4, 3))
    assert _schema._place_note(figure_, theme, "words", None, 4, 3) is None
    assert _schema._place_note(figure_, theme, "", {"x": 0.8, "y": 0.6}, 4, 3) is None
    artist = _schema._place_note(figure_, theme, "words", {"x": 0.8, "y": 0.6}, 4, 3)
    assert artist is not None and artist.get_position() == (0.8, 0.6)
    plt.close(figure_)


def test_a_declared_note_position_may_override_the_house_look():
    from analysis.theme import load_theme
    import matplotlib.pyplot as plt

    theme = load_theme(None)
    figure_ = plt.figure(figsize=(4, 3))
    artist = _schema._place_note(
        figure_, theme, "words",
        {"x": 0.5, "y": 0.5, "fontsize": 7.5, "linespacing": 1.4}, 4, 3)
    assert artist.get_fontsize() == 7.5
    plt.close(figure_)


# --------------------------------------------------------- a list-valued cast


def test_a_list_option_of_the_wrong_length_is_refused_by_its_own_cast():
    """Figure 3's `--metrics` is the x column then the y column, not a list."""
    spec = _spec(options=(Option("metrics", default=("a", "b"),
                                 cast=exactly(2)),))
    ctx = _context(spec, ["run", "--metrics", "only_one"])
    with pytest.raises(SystemExit, match="exactly 2"):
        ctx.option("metrics")


def test_the_refusal_of_a_wrong_length_list_names_the_flag():
    spec = _spec(options=(Option("metrics", default=("a", "b"), cast=exactly(2)),))
    ctx = _context(spec, ["run", "--metrics", "a,b,c"])
    with pytest.raises(SystemExit, match="--metrics"):
        ctx.option("metrics")


def test_a_list_of_the_declared_length_passes_through_unchanged():
    spec = _spec(options=(Option("metrics", default=("a", "b"), cast=exactly(2)),))
    ctx = _context(spec, ["run", "--metrics", "x,y"])
    assert ctx.option("metrics") == ["x", "y"]


# ------------------------------------------------------------ the panel contract


#: Functions in `panels/` that take an axes but are not panels. A key explains a
#: mapping something else already recorded - a colour bar points at the raster
#: that holds the numbers - so it gives back the matplotlib object rather than a
#: table of its own.
PANEL_KEYS: tuple[str, ...] = ("colour_bar", "inset_colour_bar", "semantic_legend")


def _panel_functions():
    """Every public function in `panels/` that takes an axes or a figure."""
    import importlib
    import panels

    for module_name in ("common", "coupling", "intensity", "morphology", "motility",
                        "presence", "rhythms", "surveillance", "territory"):
        module = importlib.import_module(f"panels.{module_name}")
        for name in getattr(module, "__all__", []):
            function = getattr(module, name, None)
            if not callable(function) or not inspect.isfunction(function):
                continue
            if function.__module__ != module.__name__:
                continue        # re-exported from common; tested where it lives
            first = list(inspect.signature(function).parameters)[:1]
            if first and first[0] in ("ax", "figure"):
                yield module_name, name, function


def test_every_panel_returns_the_table_it_drew():
    """One return shape, so a builder never rebuilds the table it just plotted.

    Read off the annotation rather than by calling: a panel needs an axes, a
    theme and real numbers, and a test that supplied all three for sixty panels
    would be testing its own fixtures.
    """
    from panels import PanelResult

    wrong = []
    for module_name, name, function in _panel_functions():
        if name in PANEL_KEYS:
            continue
        annotation = inspect.signature(function).return_annotation
        if annotation not in (PanelResult, "PanelResult"):
            wrong.append(f"panels.{module_name}.{name} -> {annotation}")
    assert not wrong, "not returning PanelResult: " + ", ".join(wrong)


def test_a_key_is_not_a_panel_and_says_so():
    """The four that take an axes and are not panels are the four on the list."""
    from panels import PanelResult

    keys = [f"{module}.{name}" for module, name, function in _panel_functions()
            if inspect.signature(function).return_annotation
            not in (PanelResult, "PanelResult")]
    assert sorted(name.split(".")[-1] for name in keys) == sorted(PANEL_KEYS), (
        "a panel stopped returning PanelResult, or a key was added without a "
        f"line on PANEL_KEYS saying why: {keys}")


def test_a_figure_that_names_no_table_gets_its_leading_panels():
    """`ctx.drew` is what makes `figure_data` optional."""
    from panels import PanelResult

    ctx = _context(_spec(), ["run"])
    ctx.panels()
    ctx.drew("left", PanelResult(data=pd.DataFrame({"x": [1, 2]})))
    ctx.drew("right", PanelResult(data=pd.DataFrame({"y": [3]})))
    assert ctx.leading_table()["x"].tolist() == [1, 2]


def test_a_build_that_records_nothing_and_names_nothing_has_no_table():
    ctx = _context(_spec(), ["run"])
    assert ctx.leading_table() is None


# ------------------------------------------------------- placement, both units


def test_a_placement_needs_its_unit():
    """0.2 of a page and 0.2 inches are both plausible; guessing moves the words.

    The defaults are not all in one unit - the title is inches, the footnote a
    share - so there is no sensible unit to assume, and assuming the wrong one
    is a figure whose words moved by an inch without anybody asking.
    """
    from _options import length

    with pytest.raises(ValueError, match="needs a unit"):
        length("0.2")
    with pytest.raises(ValueError, match="not a number"):
        length("wideish in")


def test_a_placement_reads_both_units():
    from _options import Length, length

    assert length("2%") == Length(0.02, "page")
    assert length("0.25in") == Length(0.25, "in")
    assert length(' 0.25 IN ') == Length(0.25, "in")


def test_the_two_units_meet_on_a_page_of_a_known_size():
    """An inch of a ten-inch page is a tenth of it, and the reverse."""
    from _options import Length

    assert Length(1.0, "in").fraction(10.0) == pytest.approx(0.1)
    assert Length(0.1, "page").fraction(10.0) == pytest.approx(0.1)
    assert Length(0.1, "page").fraction(4.0) == pytest.approx(0.1), (
        "a share does not know how big the page is; that is the point of it")


def test_a_bare_number_on_a_result_still_means_a_share_of_the_page():
    """The builders wrote `header_x=0.085` before there was a second unit."""
    from _options import Length, as_length

    assert as_length(0.085) == Length(0.085, "page")
    assert as_length(Length(0.25, "in")) == Length(0.25, "in")


def test_every_figure_accepts_the_placement_flags_without_declaring_them():
    """They are about the sheet, not about what the figure draws."""
    from _options import PLACEMENTS

    for slug in ("identity-trajectories", "step-size-distribution"):
        spec = _schema.get_figure(slug)
        assert not {o.name for o in spec.options} & set(PLACEMENTS)
        for name in PLACEMENTS:
            _schema._check_argv(spec, ["run", f"--{name.replace('_', '-')}", "1%"])


def test_a_placement_flag_beats_the_builders_own_value():
    """And is recorded, so a bundle whose title moved says who moved it."""
    from _options import Length

    ctx = _context(_spec(), ["run", "--header-x", "0.5in"])
    assert _schema._placement(ctx, "header_x", 0.02) == Length(0.5, "in")
    assert ctx.option_source["header_x"] == "flag"

    ctx = _context(_spec(), ["run"])
    assert _schema._placement(ctx, "header_x", 0.085) == Length(0.085, "page")
    assert ctx.option_source["header_x"] == "default"


def test_a_placement_flag_that_makes_no_sense_stops_the_build():
    ctx = _context(_spec(), ["run", "--title-y", "quite far down"])
    with pytest.raises(SystemExit, match="--title-y"):
        _schema._placement(ctx, "title_y", 0.2)


def test_the_house_placements_are_the_ones_that_were_hard_coded():
    """The defaults have to reproduce the render they replaced, exactly."""
    from _options import Length

    result = FigureResult(figure=None, axes=[])
    assert result.title_y == Length(0.20, "in")
    assert result.subtitle_y == Length(0.70, "in")
    # An eight-inch sheet is what most of the set is; the numbers below are the
    # fractions `_finish` computed before the placements were configurable.
    assert 1.0 - result.title_y.fraction(8.0) == pytest.approx(1.0 - 0.20 / 8.0)
    assert 1.0 - result.subtitle_y.fraction(8.0) == pytest.approx(1.0 - 0.70 / 8.0)
    from _options import as_length
    assert as_length(result.header_x).fraction(12.0) == pytest.approx(0.02)
    assert as_length(result.footnote_y).fraction(8.0) == pytest.approx(0.008)


def test_a_note_may_be_placed_in_inches_too():
    """Half an inch up a three-inch sheet is a sixth of the way up it."""
    from analysis.theme import load_theme
    from _options import Length
    import matplotlib.pyplot as plt

    figure_ = plt.figure(figsize=(4, 3))
    artist = _schema._place_note(
        figure_, load_theme(None), "words",
        {"x": Length(1.0, "in"), "y": Length(0.5, "in")}, 4, 3)
    assert artist.get_position() == pytest.approx((0.25, 1 / 6))
    plt.close(figure_)


def test_an_inch_placed_footnote_keeps_its_inches_when_the_sheet_grows():
    """A share-placed one does not, and both stay clear of the drawing."""
    from _options import Length
    from _schema import _clear_footnote
    import matplotlib.pyplot as plt

    for unit, held in (("in", True), ("page", False)):
        figure = plt.figure(figsize=(8.0, 6.0))
        ax = figure.add_axes([0.1, 0.16, 0.8, 0.74])
        ax.set_xlabel("a label with descenders: pqgy")
        at = Length(0.048, unit) if unit == "in" else Length(0.008, unit)
        footnote = figure.text(0.02, at.fraction(6.0),
                               "\n".join(f"line {i}" for i in range(9)),
                               fontsize=10, va="bottom")
        _clear_footnote(figure, footnote, at, [])
        taller = figure.get_figheight()
        assert taller > 6.0, "the footnote should have forced the page taller"
        inches_up = footnote.get_position()[1] * taller
        if held:
            assert inches_up == pytest.approx(0.048, abs=1e-6)
        else:
            assert inches_up > 0.008 * 6.0
        plt.close(figure)


def test_the_words_go_where_the_two_units_say_they_do():
    """On figure 8's sheet: 14.6in wide, 12.8in tall.

    Both directions and both units in one assertion, because the mistakes here
    are silent. The title and subtitle count down from the top edge; the
    footnote counts up from the bottom; and 0.6in on this sheet is not 0.6 of
    it. The house numbers on the left of each pair are the ones `_finish` had
    hard-coded before any of this was configurable.
    """
    from _options import Length
    from _schema import _header_positions

    house = _header_positions(14.6, 12.8, Length(0.02), Length(0.20, "in"),
                              Length(0.70, "in"), Length(0.008))
    assert house == pytest.approx((0.02, 1 - 0.20 / 12.8, 1 - 0.70 / 12.8, 0.008))

    moved = _header_positions(14.6, 12.8, Length(0.9, "in"), Length(0.60, "in"),
                              Length(0.03), Length(0.35, "in"))
    left, title, subtitle, footnote = moved
    assert left == pytest.approx(0.9 / 14.6), "inches in from the left edge"
    assert title == pytest.approx(1 - 0.60 / 12.8), "inches down from the top"
    assert subtitle == pytest.approx(0.97), "a share, measured down from the top"
    assert footnote == pytest.approx(0.35 / 12.8), "inches up from the bottom"
    assert title < house[1], "a larger title_y moves the title down the page"


def test_a_run_can_set_a_placement_for_every_build_of_a_figure(tmp_path):
    """The three-level resolution every other option gets: flag, run, builder.

    A project that wants its titles lower should say so once in the run rather
    than on every command line, and the flag has to still win over it - that is
    what makes a one-off adjustment possible without editing the run.
    """
    import json
    from _options import Length

    spec = _spec()
    (tmp_path / "figures.json").write_text(json.dumps(
        {"figures": {spec.slug: {"options": {"title_y": "0.55in"}}}}), encoding="utf-8")

    ctx = FigureContext(
        spec=spec, run=tmp_path, tables=tmp_path, theme=None, bundle=tmp_path,
        summary={"minutes_per_frame": 30.0, "stem": "demo"}, field={},
        stem=None, argv=["run"],
    )
    assert _schema._placement(ctx, "title_y", Length(0.2, "in")) == Length(0.55, "in")
    assert ctx.option_source["title_y"] == "config"

    ctx.argv = ["run", "--title-y", "0.60in"]
    assert _schema._placement(ctx, "title_y", Length(0.2, "in")) == Length(0.60, "in")
    assert ctx.option_source["title_y"] == "flag"


def test_a_placement_in_a_run_is_not_mistaken_for_a_figures_own_option(tmp_path):
    """It belongs to every figure, so no figure may refuse it as undeclared."""
    import json

    spec = _spec()
    (tmp_path / "figures.json").write_text(json.dumps(
        {"figures": {spec.slug: {"options": {"header_x": "3%"}}}}), encoding="utf-8")
    ctx = FigureContext(
        spec=spec, run=tmp_path, tables=tmp_path, theme=None, bundle=tmp_path,
        summary={"minutes_per_frame": 30.0, "stem": "demo"}, field={},
        stem=None, argv=["run"],
    )
    assert ctx._from_run_options() == {}, "a placement is not this figure's option"
