"""Tests for the aesthetic engine.

Run with ``python -m pytest analysis/test_theme.py``.

The point of most of these is not that the numbers are right - it is that they
cannot change by accident. A theme is the one part of an analysis package where
a silent change produces work that looks fine and is not comparable with what
came before it.
"""

from __future__ import annotations

import json

import pytest

from analysis.theme import (
    BASE_COLOURS,
    PRESETS,
    ROLES,
    Theme,
    conformance_report,
    load_theme,
)


# ------------------------------------------------------------------- contract

def test_the_contract_is_not_offered_as_a_preset():
    """It is a conformance reference, not a look. Offering it invites a choice
    between two things a user cannot tell apart from the name."""
    assert "pyflash" not in PRESETS
    with pytest.raises(KeyError, match="unknown theme preset"):
        load_theme("pyflash")


def test_the_contract_matches_analysis_kit_when_it_is_installed():
    """The vendored contract numbers must not drift from the shared contract.

    Skipped rather than failed where analysis_kit is absent, because an external
    user is expected not to have it - that is why the numbers are vendored.
    """
    report = conformance_report()
    if not report["available"]:
        pytest.skip(report["reason"])
    assert report["conformant"], report["differences"]


def test_every_preset_resolves_every_role():
    for name in PRESETS:
        theme = load_theme(name)
        for role in ROLES:
            value = theme.colour(role)
            assert isinstance(value, str) and value.startswith("#"), (name, role, value)


#: The five measurement families. These are the package's nouns and each must
#: keep a colour of its own, or a reader who learned "teal is turnover" in one
#: figure misreads the next one.
FAMILIES = ("reporter", "morphology", "motility", "surveillance", "evidence")

#: Roles that describe a test outcome. They must not borrow a family colour, for
#: the same reason.
STATES = ("rhythmic", "significant")


def test_the_measurement_families_never_share_a_colour():
    for name in PRESETS:
        if name == "mono":
            continue        # see test_mono_is_documented_as_lossy
        theme = load_theme(name)
        used = [theme.colour(family) for family in FAMILIES]
        assert len(set(used)) == len(FAMILIES), (name, dict(zip(FAMILIES, used)))


def test_a_test_outcome_never_wears_a_family_colour():
    for name in PRESETS:
        if name == "mono":
            continue
        theme = load_theme(name)
        families = {theme.colour(family) for family in FAMILIES}
        for state in STATES:
            assert theme.colour(state) not in families, (name, state)


def test_mono_is_documented_as_lossy():
    """Greyscale cannot carry five families plus states; say so rather than pretend.

    This test exists to stop someone "fixing" mono by inventing greys that are
    not actually distinguishable in print. The fix is a linestyle or a hatch in
    the builder, not another shade.
    """
    theme = load_theme("mono")
    used = [theme.colour(family) for family in FAMILIES]
    assert len(set(used)) < len(FAMILIES), "if mono ever separates all five, update the docs"


def test_every_preset_separates_the_four_presence_states():
    """Four states is where a colour-only encoding starts to fail. Check it.

    A name on screen, a name the tracker reconstructed, a gap and time outside a
    cell's life are drawn side by side on one raster. If two of them resolve to
    the same value in any preset the figure silently merges two different
    statements, so this is checked in every preset rather than only the default.
    """
    states = ("named", "inferred", "unclaimed", "missing")
    for name in PRESETS:
        theme = load_theme(name)
        used = [theme.colour(state) for state in states]
        assert len(set(used)) == len(states), f"{name} merges two presence states"


def test_the_reconstructed_state_is_a_qualified_name_not_a_new_family():
    """``inferred`` must not wear a measurement family's colour.

    It qualifies ``named``; borrowing the morphology or reporter colour would
    make a provenance statement look like a measurement. Mono is excluded for
    the reason the preset is already documented as lossy: greyscale has too few
    distinguishable values to keep every family apart, let alone a state as
    well.
    """
    for name in PRESETS:
        if name == "mono":
            continue
        theme = load_theme(name)
        assert theme.colour("inferred") not in {theme.colour(family) for family in FAMILIES}


def test_every_role_has_a_purpose_and_a_default():
    from analysis.theme import _ROLE_BASE

    assert set(ROLES) == set(_ROLE_BASE), "a role must have both a description and a default"
    for role, base in _ROLE_BASE.items():
        assert base in BASE_COLOURS, f"{role} defaults to {base}, which is not a base colour"


# --------------------------------------------------------------------- tuning

def test_a_role_override_reaches_the_figure():
    theme = load_theme({"theme": {"roles": {"reporter": "#123456"}}})
    assert theme.colour("reporter") == "#123456"
    assert theme.colour("morphology") == BASE_COLOURS["blue"], "one override must not move the rest"


def test_a_named_palette_colour_can_be_used_for_a_role():
    theme = load_theme({"theme": {"palette": {"house_purple": "#7a1fa2"},
                                  "roles": {"reporter": "house_purple"}}})
    assert theme.colour("reporter") == "#7a1fa2"


def test_a_typo_in_a_role_is_an_error_not_a_no_op():
    with pytest.raises(KeyError, match="unknown colour role"):
        load_theme({"theme": {"roles": {"reportr": "#123456"}}})


def test_a_typo_in_a_setting_is_an_error_not_a_no_op():
    with pytest.raises(KeyError, match="unknown theme setting"):
        load_theme({"theme": {"lien_width": 3.0}})


def test_an_analysis_config_without_a_theme_block_is_the_house_default():
    config = {"dataset": "x", "frame_interval_min": 30, "movies": [], "line_width": 99}
    theme = load_theme(config)
    assert theme.preset == "house"
    assert theme["line_width"] == load_theme("house")["line_width"], (
        "config keys must not be read as theme settings")


def test_scales_move_type_and_strokes_independently():
    base = load_theme("house")
    theme = load_theme({"theme": {"type_scale": 2.0}})
    assert theme["tick_size"] == pytest.approx(base["tick_size"] * 2)
    assert theme["line_width"] == pytest.approx(base["line_width"])


# ------------------------------------------------------------------ derived

def test_the_type_ladder_scales_with_the_preset():
    house, talk = load_theme("house"), load_theme("talk")
    for step in ("note", "subtitle", "title"):
        assert talk.size(step) > house.size(step), step
    # every step keeps its order whatever the preset
    for theme in (house, talk, load_theme("print")):
        steps = [theme.size(s) for s in ("note", "subtitle", "annotation", "panel", "title")]
        assert steps == sorted(steps)


def test_the_canvas_scales_with_the_figure_size():
    assert load_theme("house").canvas(14.0, 10.0) == (14.0, 10.0)
    narrow = load_theme("print")
    width, height = narrow.canvas(14.0, 10.0)
    assert width < 14.0 and height / width == pytest.approx(10.0 / 14.0), "aspect must be preserved"


def test_unknown_steps_and_strokes_are_rejected():
    theme = load_theme()
    with pytest.raises(KeyError):
        theme.size("enormous")
    with pytest.raises(KeyError):
        theme.stroke("wiggly")


def test_the_role_cycle_refuses_to_repeat_itself():
    theme = load_theme()
    assert len(theme.cycle()) == len(set(theme.cycle())), "a repeat would draw two series alike"
    with pytest.raises(ValueError, match="one colour"):
        theme.cycle(99)


# ---------------------------------------------------------------------- ticks

def test_value_ticks_follow_the_pyflash_five_tick_rule():
    """The maximum rounds up on the same significant-5 grid as PyFLASH."""
    assert load_theme().value_ticks(2.0, 113.0) == [
        0.0, 37.5, 75.0, 112.5, 150.0,
    ]


def test_signed_value_ticks_keep_zero_and_symmetric_limits():
    assert load_theme().value_ticks(-168.0, 121.0) == [
        -200.0, -100.0, 0.0, 100.0, 200.0,
    ]


def test_value_tick_structure_can_be_set_in_the_theme():
    theme = load_theme({"theme": {
        "value_tick_count": 3,
        "value_tick_round_to": 2,
        "value_tick_start": 10,
    }})
    assert theme.value_ticks(12.0, 38.0) == [10.0, 24.0, 38.0]
    assert theme.stamp()["value_ticks"] == {
        "count": 3, "round_to": 2.0, "start": 10.0,
    }


def test_colour_map_keys_have_one_tunable_physical_shape():
    theme = load_theme({"theme": {
        "colour_bar_width_inches": 0.2,
        "colour_bar_height_inches": 1.8,
        "colour_bar_gap_inches": 0.12,
    }})
    assert theme.stamp()["colour_bar"] == {
        "width_inches": 0.2, "height_inches": 1.8, "gap_inches": 0.12,
    }


@pytest.mark.parametrize("setting,value", [
    ("value_tick_count", 1),
    ("value_tick_count", 4.5),
    ("value_tick_round_to", 0),
    ("value_tick_start", float("nan")),
])
def test_invalid_value_tick_settings_stop_at_theme_loading(setting, value):
    with pytest.raises(ValueError, match="theme tick settings"):
        load_theme({"theme": {setting: value}})


# -------------------------------------------------------------------- legend

def test_above_the_axes_is_the_package_default():
    """Every preset puts the legend above the axes.

    ``best`` picks the emptiest corner of the data, so it moves when the data
    moves and two runs of the same figure can disagree about where it is.
    """
    for name in PRESETS:
        assert load_theme(name)["legend_location"] == "above", name


def _legend_top(theme, **kwargs) -> tuple[float, float]:
    """Draw a one-series panel and return (legend bottom, axes top) in pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    legend = theme.legend(ax, **kwargs)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bottom = legend.get_window_extent(renderer).y0
    top = ax.get_window_extent().y1
    plt.close(figure)
    return bottom, top


def test_the_default_puts_the_legend_clear_of_the_axes():
    bottom, top = _legend_top(load_theme("house"))
    assert bottom >= top, "an 'above' legend must sit entirely outside the data area"


def test_an_explicit_loc_wins_over_the_theme_anchor():
    """A builder that names a corner has opted out; it must not get both."""
    bottom, top = _legend_top(load_theme("house"), loc="lower right")
    assert bottom < top, "an explicit loc must place the legend inside the axes"


def test_an_unknown_legend_location_is_rejected():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = load_theme()
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    with pytest.raises(KeyError, match="unknown legend location"):
        theme.legend(ax, location="somewhere nice")
    plt.close(figure)


# ------------------------------------------------------------------ recording

def test_the_stamp_records_every_resolved_role():
    stamp = load_theme("colourblind").stamp()
    assert stamp["preset"] == "colourblind"
    assert set(stamp["resolved_roles"]) == set(ROLES)
    assert json.dumps(stamp), "a stamp must be JSON-serialisable for the manifest"


def test_rcparams_never_leave_a_colour_unresolved():
    for name in PRESETS:
        params = load_theme(name).rcparams()
        for key in ("axes.edgecolor", "xtick.color", "ytick.color"):
            assert str(params[key]).startswith("#") or params[key] in ("black", "none")


def test_a_theme_is_immutable():
    theme = load_theme()
    with pytest.raises(Exception):
        theme.preset = "talk"          # type: ignore[misc]
    assert isinstance(theme, Theme)
