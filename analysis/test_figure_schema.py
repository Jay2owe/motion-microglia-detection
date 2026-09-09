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
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from analysis import circadian as workbench

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
    "independence-map.measures",
)


AMBIGUOUS_DISPLAY_TERMS: tuple[str, ...] = (
    "breakout", "excursion", "upheaval", "programme", "report card",
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
    assert len(rows) == 45, "the numbered set changed size; update this number"
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


def test_user_facing_figure_names_avoid_undefined_metaphors():
    """Stable machine slugs may stay; text printed on figures must name measurements."""
    for spec in load_all().values():
        displayed = [spec.title, spec.summary, *(panel.heading() for panel in spec.panels)]
        for text in displayed:
            lowered = text.lower()
            assert not any(term in lowered for term in AMBIGUOUS_DISPLAY_TERMS), (
                f"{spec.slug} displays an undefined metaphor in {text!r}"
            )


def test_review_figures_live_in_the_review_module():
    """A quality-control plot cannot silently return to the result set."""
    review = FIGURES / "review"
    for spec in load_all().values():
        expected = review if spec.purpose == "review" else FIGURES
        assert spec.source.parent == expected, (
            f"{spec.slug} declares purpose={spec.purpose!r} but lives in "
            f"{spec.source.parent}")


def test_only_result_or_review_is_a_valid_figure_purpose():
    with pytest.raises(ValueError, match="purpose must be"):
        _spec(purpose="decoration")


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

    It launches one subprocess per builder and has no other reason to load
    matplotlib, so the list is duplicated on purpose. This is what keeps the
    duplicate honest.
    """
    import build_all
    assert build_all.SHARED == {"stem", *_schema.UNIVERSAL_SWITCHES}


def test_build_all_can_select_result_or_review_builders():
    import build_all
    from _builders import is_review_builder

    review = [path for path in build_all.BUILDERS if is_review_builder(path)]
    results = [path for path in build_all.BUILDERS if not is_review_builder(path)]
    assert len(review) == 6
    assert any(path.name == "03_surveillance_not_translocation.py" for path in review)
    assert any(path.name == "16_territory_anchoring.py" for path in review)
    assert len(results) == 39


@pytest.mark.parametrize(
    "builder",
    ["08_identity_trajectories.py", "22_independence_map.py"],
)
def test_circadian_builders_launch_directly_from_the_project_root(builder):
    """Builders must find the analysis package before `_schema` is imported.

    `build_all.py` launches each builder as a script.  These two pages import
    the Circadian Workbench adapter before `_schema` can put the project root
    on `sys.path`, so their own launch preamble must do it first.
    """
    done = subprocess.run(
        [sys.executable, str(FIGURES / builder), "--help"],
        cwd=FIGURES.parents[1],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_cells_on_screen_owns_the_lifespan_views_and_the_old_page_is_gone():
    specs = load_all()
    assert "lifespan-gantt" not in specs
    panels = {panel.key: panel.draw for panel in specs["cells-on-screen"].panels}
    assert {"lifespan", "arrivals", "coverage"} <= set(panels)
    from panels import common, presence
    assert panels["lifespan"] is presence.lifespan_bars
    assert panels["arrivals"] is common.histogram
    assert panels["coverage"] is common.histogram


def test_repeated_motion_evidence_budget_is_not_a_figure():
    assert "motion-evidence-budget" not in load_all()


def test_negative_space_has_one_configurable_coverage_map():
    spec = load_all()["negative-space"]
    assert [panel.key for panel in spec.panels] == ["coverage", "composition"]
    from panels import territory
    assert spec.panel("coverage").draw is territory.coverage_history
    assert spec.panel("composition").draw is territory.coverage_composition
    assert spec.option("coverage_view").default == "stages"


def test_patch_ledger_declares_reusable_maps_and_a_configurable_permutation_test():
    spec = load_all()["patch-ledger"]
    from panels import territory
    assert spec.panel("revisit").draw is territory.revisit_grid
    assert spec.panel("revisit").block
    assert spec.panel("coverage").draw is territory.coverage_curve
    assert spec.option("shuffles").default == 1000


def test_pixel_fate_flow_accepts_named_event_times_through_shared_options():
    spec = load_all()["pixel-fate-flow"]
    from panels import territory
    assert spec.panel("flow").draw is territory.fate_flow
    assert spec.option("events").default == []
    assert spec.option("event_times").default == []


def test_tissue_tectonics_has_the_six_canonical_maps_in_order():
    spec = load_all()["tissue-tectonics"]
    from analysis.circadian import CIRCADIAN_ANALYSIS_OPTIONS
    from panels import territory

    assert [panel.key for panel in spec.panels] == [
        "first_coverage", "cumulative_occupancy", "unique_cells", "speed",
        "significant_period", "splitting_events",
    ]
    assert spec.panel("first_coverage").draw is territory.first_coverage_time_map
    assert spec.panel("cumulative_occupancy").draw is territory.cumulative_occupancy_map
    assert spec.panel("unique_cells").draw is territory.owner_count_map
    assert spec.panel("speed").draw is territory.cell_metric_map
    assert spec.panel("significant_period").draw is territory.cell_metric_map
    assert spec.panel("significant_period").title == (
        "All significant intensity periods\nBlack boundary: supported subset"
    )
    assert spec.panel("splitting_events").draw is territory.split_event_map
    assert {option.name for option in spec.options} >= {
        "map_summary", "map_assignment", "map_luts", "map_range",
        *CIRCADIAN_ANALYSIS_OPTIONS,
    }
    assert spec.option("fit_method").default == "lomb"
    assert spec.option("significance_method").default == "lomb"
    assert spec.option("multiple_testing").default == "none"


def test_tissue_tectonics_uses_uniform_map_text_and_outer_coordinate_ticks():
    import matplotlib.pyplot as plt
    from analysis.theme import load_theme

    spec = load_all()["tissue-tectonics"]
    style_map_grid = spec.build.__globals__["_style_map_grid"]
    theme = load_theme()
    figure, raw_axes = plt.subplots(2, 3)
    keys = [
        "first_coverage", "cumulative_occupancy", "unique_cells",
        "speed", "significant_period", "splitting_events",
    ]
    axes = dict(zip(keys, raw_axes.ravel()))
    positions = {key: divmod(index, 3) for index, key in enumerate(keys)}
    for ax in axes.values():
        ax.set_title("Panel")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    style_map_grid(axes, positions, theme)

    assert {ax.title.get_fontsize() for ax in axes.values()} == {theme.size("panel")}
    assert {ax.xaxis.label.get_fontsize() for ax in axes.values()} == {
        theme.size("annotation")
    }
    assert {label.get_fontsize() for ax in axes.values()
            for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]} == {
        theme.size("caption")
    }
    assert not any(label.get_visible() for ax in raw_axes[0]
                   for label in ax.get_xticklabels())
    assert all(label.get_visible() for ax in raw_axes[1]
               for label in ax.get_xticklabels())
    assert all(label.get_visible() for ax in raw_axes[:, 0]
               for label in ax.get_yticklabels())
    assert not any(label.get_visible() for ax in raw_axes[:, 1:].ravel()
                   for label in ax.get_yticklabels())
    plt.close(figure)


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


def test_all_detrending_figures_expose_and_inherit_the_complete_control_set(monkeypatch):
    from analysis.circadian import DETREND_DEFAULTS
    load_all()
    spec = _schema.get_figure("metric-rhythm-matrix")
    assert DETREND_DEFAULTS.keys() <= {option.name for option in spec.options}
    monkeypatch.setattr(_schema, "module_params", lambda *args: {
        "detrend_polynomial_degree": 3, "detrend_filter_order": 4})
    ctx = _context(spec, ["--detrend-polynomial-degree", "6"])
    effective = ctx.module_params("rhythms")
    assert effective["detrend_polynomial_degree"] == 6
    assert effective["detrend_filter_order"] == 4


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


# --------------------------------------------- tables that belong to the run

from _bundle import (POOLED_FOLDER as BUNDLE_POOLED, any_movie,  # noqa: E402
                     require_run_table, require_table, tables_for)

def _run_with(tmp_path, movie_tables=(), run_files=(), pooled_tables=()):
    """A run folder shaped the way ``analysis.run`` writes one."""
    for name in movie_tables:
        path = tmp_path / "m_a" / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a\n1\n", encoding="utf-8")
    for name in run_files:
        (tmp_path / name).write_text("a\n1\n", encoding="utf-8")
    for name in pooled_tables:
        path = tmp_path / "pooled" / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a\n1\n", encoding="utf-8")
    return tmp_path / "m_a" / "tables"


def test_a_run_scoped_table_resolves_from_the_run_root(tmp_path):
    tables = _run_with(tmp_path, movie_tables=["cell_frame.csv"],
                       run_files=["statistics.csv"])
    found = require_run_table(tables, "statistics.csv", "contrasts")
    assert found == tmp_path / "statistics.csv"


def test_a_movie_scoped_table_is_not_found_at_the_run_root(tmp_path):
    """The reason the two searches are separate functions.

    One search that tried both would make every movie-level table findable at
    the run root, and a page claiming one movie would draw the pooled stack of
    all of them.
    """
    tables = _run_with(tmp_path, movie_tables=["cell_frame.csv"],
                       run_files=["statistics.csv"])
    with pytest.raises(SystemExit, match="rhythms"):
        require_table(tables, "rhythms.csv", "rhythms")


def test_a_pooled_table_resolves_at_run_scope(tmp_path):
    tables = _run_with(tmp_path, movie_tables=["cell_frame.csv"],
                       pooled_tables=["cell_summary.csv"])
    found = require_run_table(tables, "cell_summary.csv", "presence")
    assert found == tmp_path / "pooled" / "tables" / "cell_summary.csv"


def test_a_missing_run_table_says_a_contrasts_block_is_what_writes_one(tmp_path):
    tables = _run_with(tmp_path, movie_tables=["cell_frame.csv"])
    with pytest.raises(SystemExit, match="contrasts"):
        require_run_table(tables, "statistics.csv", "contrasts")


def test_a_table_scope_outside_the_two_is_refused_at_declaration_time():
    with pytest.raises(ValueError, match="movie.*run"):
        Table("x.csv", module="m", scope="everywhere")


def test_the_default_scope_is_the_movie_so_nothing_already_written_moved():
    assert Table("cell_frame.csv", module="motility").scope == "movie"


def test_a_page_is_run_level_only_when_every_source_is():
    """A stem it does not need is a stem it must not ask for."""
    run_only = _spec(reads=(Table("statistics.csv", module="c", scope="run"),))
    mixed = _spec(reads=(Table("statistics.csv", module="c", scope="run"),
                         Table("cell_frame.csv", module="m")))
    assert run_only.run_level
    assert not mixed.run_level
    assert not _spec(reads=()).run_level


def test_the_pooled_folder_is_never_mistaken_for_a_movie(tmp_path):
    """It holds a tables folder too, and would otherwise demand a --stem.

    On a pooled run of one movie every figure would refuse to draw until given
    a stem it should not need; on a run of several, ``--stem pooled`` would
    draw every movie at once on a page whose words say one.
    """
    _run_with(tmp_path, movie_tables=["cell_frame.csv"],
              pooled_tables=["cell_frame.csv"])
    assert tables_for(tmp_path) == tmp_path / "m_a" / "tables"
    assert any_movie(tmp_path) == "m_a"


def test_the_pooled_folder_has_one_name_across_the_package():
    """Restated in _bundle so drawing does not import the measurement package."""
    from analysis.pool import POOLED_FOLDER as measured

    assert BUNDLE_POOLED == measured


def _forest_page():
    import importlib.util

    path = FIGURES / "review" / "39_contrast_forest.py"
    spec = importlib.util.spec_from_file_location("_forest_page", path)
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)
    return page


def test_the_forest_page_agrees_with_the_floor_the_contrasts_step_applies():
    """The figure restates it rather than importing it, so this is the check."""
    from analysis.contrasts import MIN_UNITS as applied

    assert _forest_page().MIN_UNITS == applied


def test_every_effect_a_test_can_report_has_words_on_the_forest():
    """A new test brings a new effect kind, and the axis is where it surfaces.

    Without this the page falls back to the column name with its underscores
    turned into spaces, which reads as an axis label right up until somebody
    tries to say what it means.
    """
    from analysis.contrasts import TESTS

    labels = _forest_page().EFFECT_LABELS
    undescribed = sorted({spec["effect_kind"] for spec in TESTS.values()}
                         - set(labels))
    assert not undescribed, (
        f"no words for {undescribed}; add an entry to EFFECT_LABELS in "
        f"39_contrast_forest.py")


def test_a_bundle_that_read_nothing_still_writes_a_usable_source_index(tmp_path):
    """A page can legitimately read no table, and its bundle must still be valid.

    pandas gives a frame built from no rows no columns either, so the index came
    out headerless and the ReproFig writer refused it for missing every required
    field. The page that hits this is the contrast forest on a run whose
    configuration declared no comparisons.
    """
    from _bundle import make_bundle

    table = make_bundle(tmp_path, {})
    written = pd.read_csv(tmp_path / "data" / "sources.csv")
    for column in ("short_name", "copied_path", "file_name",
                   "modification_time", "byte_size", "sha256"):
        assert column in table.columns
        assert column in written.columns
    assert "original_path" not in table.columns
    assert "original_path" not in written.columns
    assert written.empty


def test_a_bundle_source_index_does_not_disclose_its_original_path(tmp_path):
    from _bundle import make_bundle

    source = tmp_path / "private" / "measurements.csv"
    source.parent.mkdir()
    source.write_text("value\n1\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    make_bundle(bundle, {"measurements.csv": source})

    assert str(source) not in (bundle / "data" / "sources.csv").read_text(encoding="utf-8")
    assert str(source) not in (bundle / "data" / "sources.md").read_text(encoding="utf-8")


def test_a_source_can_be_recorded_without_being_a_declared_table(tmp_path):
    """The escape hatch for a page whose claim is that a file is absent."""
    evidence = tmp_path / "manifest.json"
    evidence.write_text("{}", encoding="utf-8")
    ctx = _context(_spec(), [])

    ctx.record_source("manifest.json", evidence)
    assert ctx.sources["manifest.json"] == evidence
    with pytest.raises(SystemExit, match="is not there"):
        ctx.record_source("gone.json", tmp_path / "gone.json")


def test_a_long_table_contrast_is_blocked_by_the_thing_it_measured():
    """Two slopes both reading "per h" are not two points on one ruler.

    ``trend`` and ``window_change`` put the measurement in a column, so a
    contrast on one of them tests a single column across groups that are
    themselves measurements. Without this, a slope of an area and a slope of a
    brightness five orders of magnitude away share an axis, and the smaller of
    them is drawn on the no-effect line - which reads as no effect, the
    opposite of what it says.
    """
    scale = _forest_page()._scale
    area = scale("median_against_zero", "slope_per_hour", 30.0,
                 group_by="metric", group="area_px")
    reporter = scale("median_against_zero", "slope_per_hour", 30.0,
                     group_by="metric", group="corrected_mean")

    assert area != reporter
    assert "px per h" in area
    assert "camera units per h" in reporter


def test_grouping_by_something_that_is_not_a_measurement_changes_nothing():
    """The ordinary case: a condition is a group, not a unit."""
    scale = _forest_page()._scale
    assert scale("median_difference", "area_px_median", 30.0,
                 group_by="condition", group="treated") == scale(
                     "median_difference", "area_px_median", 30.0)


def test_a_metric_group_nothing_has_words_for_is_left_alone():
    """An undocumented group name would otherwise invent a unit for itself."""
    scale = _forest_page()._scale
    assert scale("median_against_zero", "slope_per_hour", 30.0,
                 group_by="metric", group="not_a_column") == scale(
                     "median_against_zero", "slope_per_hour", 30.0)


# ------------------------------------------------ physical panel dimensions

def test_breath_trace_declares_tall_individual_panels():
    spec = load_all()["breath-trace"]
    assert {panel.key: panel.min_height_inches for panel in spec.panels} == {
        "breath": 7.0,
        "small_multiples": 7.0,
    }


def _layout_context(*panels):
    """A drawing context using the unscaled house theme."""
    from analysis.theme import load_theme

    ctx = _context(_spec(panels=tuple(panels)), [])
    ctx.theme = load_theme("house")
    return ctx


def _axes_inches(figure_, axis):
    """The physical drawing rectangle, excluding the page margins."""
    box = axis.get_position()
    return box.width * figure_.get_figwidth(), box.height * figure_.get_figheight()


def test_adding_a_grid_column_grows_the_page_without_shrinking_a_plot():
    """A composite is a larger sheet, not a thumbnail maker."""
    left = Panel("left", _panel, min_width_inches=6.0, min_height_inches=4.0)
    right = Panel("right", _panel, min_width_inches=6.0, min_height_inches=4.0)
    wide = Panel("wide", _panel, min_width_inches=13.0, min_height_inches=3.0)
    ctx = _layout_context(left, right, wide)

    single, single_axes = ctx.grid_layout([left], {"left": (0, 0, 1, 1)})
    composite, composite_axes = ctx.grid_layout(
        [left, right, wide],
        {"left": (0, 0, 1, 1), "right": (0, 1, 1, 1),
         "wide": (1, 0, 1, 2)},
    )

    single_width, single_height = _axes_inches(single, single_axes["left"])
    composite_width, composite_height = _axes_inches(
        composite, composite_axes["left"])
    assert single.get_figwidth() == pytest.approx(13.8)
    assert composite.get_figwidth() > single.get_figwidth()
    assert composite_width >= single_width
    assert composite_height >= single_height
    for key, minimum in {"left": (6.0, 4.0), "right": (6.0, 4.0),
                         "wide": (13.0, 3.0)}.items():
        width, height = _axes_inches(composite, composite_axes[key])
        assert width >= minimum[0] - 1e-9
        assert height >= minimum[1] - 1e-9

    import matplotlib.pyplot as plt
    plt.close(single)
    plt.close(composite)


def test_stacked_panels_keep_their_declared_height_when_the_page_grows():
    first = Panel("first", _panel, min_width_inches=8.0, min_height_inches=4.5)
    second = Panel("second", _panel, min_width_inches=11.0, min_height_inches=6.0)
    ctx = _layout_context(first, second)

    single, single_axes = ctx.layout([first])
    stacked, stacked_axes = ctx.layout([first, second])

    assert _axes_inches(single, single_axes["first"])[1] == pytest.approx(4.5)
    assert _axes_inches(stacked, stacked_axes["first"])[1] == pytest.approx(4.5)
    assert _axes_inches(stacked, stacked_axes["second"])[1] == pytest.approx(6.0)
    assert stacked.get_figheight() > single.get_figheight()

    import matplotlib.pyplot as plt
    plt.close(single)
    plt.close(stacked)


def test_a_small_multiple_grid_adds_rows_and_columns_at_the_item_size():
    panel = Panel(
        "maps", _panel, item_width_inches=3.0, item_height_inches=2.5,
        item_gap_inches=0.25)

    assert panel.grid_minimum(1, 4) == pytest.approx((11.0, 5.0))
    assert panel.grid_minimum(8, 4) == pytest.approx((12.75, 5.25))
    assert panel.grid_minimum(12, 4) == pytest.approx((12.75, 8.0))


@pytest.mark.parametrize("width,height", [(0, 5), (5, 0), (-1, 5),
                                           (float("nan"), 5)])
def test_a_panel_refuses_a_non_physical_minimum_size(width, height):
    with pytest.raises(ValueError, match="finite positive inches"):
        Panel("bad", _panel, min_width_inches=width, min_height_inches=height)


def test_predictability_clock_names_its_descriptive_rule_and_configured_cycle():
    from panels import common

    spec = load_all()["predictability-clock"]
    assert spec.panel("dial").draw is common.rose
    assert spec.option("period_hours").default == 24.0
    assert spec.option("hour_ticks").default == 6.0
    assert "metrics" not in {option.name for option in spec.options}
    assert "prediction" not in spec.title.lower()


def test_the_spatial_phase_map_uses_the_generic_chart_panel():
    from panels import common

    spec = load_all()["independence-map"]
    assert spec.panel("field").draw is common.phase_map


def test_recurrence_wall_uses_plain_time_wording_and_a_noise_comparison():
    from panels import common

    spec = load_all()["recurrence-wall"]
    assert spec.panel("quantified").draw is common.dumbbell
    assert spec.option("cells").default == "9"
    assert spec.option("hour_ticks").default == 3.0
    assert "lag" not in " ".join(panel.heading().lower() for panel in spec.panels)


def test_radial_occupancy_rhythms_is_a_configurable_generic_trace_matrix():
    from panels import common

    spec = load_all()["radial-occupancy-rhythms"]
    assert spec.panel("matrix").draw is common.raster
    assert spec.option("scaling").default == "cell"
    assert spec.option("display").default == "raw"
    assert spec.option("period_min_hours").default == 2.0
    assert spec.option("period_max_hours").default == 48.0
    assert spec.option("fit_method").default is None
    assert spec.option("multiple_testing").default == "bh"


def test_metric_rhythm_matrix_accepts_metrics_and_the_full_period_test_controls():
    from panels import rhythms

    spec = load_all()["metric-rhythm-matrix"]
    assert spec.panel("matrix").draw is rhythms.period_status_matrix
    assert spec.option("metrics").default == [
        "corrected_mean", "area_px", "speed", "reach_p95"
    ]
    assert spec.option("fit_method").default is None  # inherit the main analysis estimator
    assert spec.option("detrend").default is None
    assert spec.option("detrend_window_hours").default is None
    assert spec.option("period_min_hours").default == 2.0
    assert spec.option("period_max_hours").default == 48.0
    assert spec.option("multiple_testing").default == "bh"
    assert spec.option("correction_scope").default == "matrix"
    assert spec.option("column_label_rotation").default == 0.0
    assert spec.option("column_label_wrap").default == 18


def test_cd68_rhythm_card_refits_broadly_and_keeps_matrix_plus_histogram():
    from panels import rhythms

    spec = load_all()["cd68-reporter-rhythm"]
    assert spec.panel("period_histogram").draw is rhythms.significant_period_histogram
    assert spec.panel("period_peak_matrix").draw is rhythms.timing_by_period
    assert [panel.key for panel in spec.panels] == [
        "raster", "period_peak_matrix", "period_histogram",
    ]
    assert spec.option("fit_method").default is None
    assert spec.option("significance_method").default is None
    assert spec.option("secondary_significance_method").default is None
    assert spec.option("period_min_hours").default == 2.0
    assert spec.option("period_max_hours").default == 48.0
    assert spec.option("multiple_testing").default == "none"
    assert spec.option("order").default == ["principal_component"]
    builder = sys.modules["01_cd68_reporter_rhythm"]
    assert builder._resolved_order(["principal_component"]) == [
        "pattern_rank", "period_hours", "identity",
    ]
    assert builder._resolved_order(["spectral"]) == [
        "spectral_rank", "period_hours", "identity",
    ]
    assert builder._resolved_order(["onset", "period"]) == [
        "displayed_onset_hours", "period_hours", "identity",
    ]
    assert builder._resolved_order(["period"]) == [
        "period_hours", "identity",
    ]


def test_cd68_dual_test_statistics_keep_both_tests_and_the_final_verdict():
    load_all()
    builder = sys.modules["01_cd68_reporter_rhythm"]
    fits = pd.DataFrame({
        "identity": [1, 2],
        "metric": ["corrected_mean", "corrected_mean"],
        "period_estimation_method": ["mesa", "mesa"],
        "period_hours": [10.0, 18.0],
        "primary_significance_method": ["lomb", "lomb"],
        "primary_test_status": ["ok", "ok"],
        "primary_p_value": [0.001, 0.2],
        "primary_q_value": [0.002, 0.2],
        "primary_significant": [True, False],
        "significance_period_hours": [10.1, 18.2],
        "secondary_significance_method": ["f", "f"],
        "secondary_test_status": ["ok", "ok"],
        "secondary_p_value": [0.003, 0.04],
        "secondary_q_value": [0.006, 0.08],
        "secondary_significant": [True, False],
        "secondary_significance_period_hours": [10.0, 9.0],
        "significant": [True, False],
        "rhythm_status": ["rhythmic", "not rhythmic"],
    })

    statistics = builder._statistics_table(fits)

    assert len(statistics) == 4
    assert set(statistics["test_role"]) == {"primary", "secondary"}
    assert set(statistics["significance_method"]) == {"lomb", "f"}
    assert statistics.groupby("identity")["final_cell_significant"].nunique().eq(1).all()


def test_every_figure_that_recalculates_a_rhythm_inherits_both_detrend_controls():
    specs = load_all()
    recalculating = (
        "cd68-reporter-rhythm",
        "radial-occupancy-rhythms",
        "metric-rhythm-matrix",
        "own-clock-composite",
        "null-channel-phase-test",
        "cell-report-card",
    )
    for slug in recalculating:
        spec = specs[slug]
        assert spec.option("detrend").default is None
        assert spec.option("detrend_window_hours").default is None


def test_every_fresh_circadian_figure_has_the_same_analysis_controls():
    specs = load_all()
    shared = set(workbench.CIRCADIAN_ANALYSIS_OPTIONS)
    recalculating = [
        spec for spec in specs.values()
        if any(option.name in {"fit_method", "significance_method"}
               for option in spec.options)
    ]
    assert recalculating
    for spec in recalculating:
        declared = {option.name for option in spec.options}
        assert shared <= declared, spec.slug
        assert spec.option("period_min_hours").default == 2.0, spec.slug
        assert spec.option("period_max_hours").default == 48.0, spec.slug
        assert spec.option("rhythmic_alpha").default == 0.05, spec.slug
