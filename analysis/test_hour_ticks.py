"""Tests for ticks on a time axis.

Run with ``python -m pytest analysis/test_hour_ticks.py``.

The property under test: every tick on an axis measured in hours falls at the
same time of day. Matplotlib's default locator counts in tens, which puts the
third tick of a 50 h recording at 06:00 and the reader has no way to see it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.theme import load_theme  # noqa: E402
from _options import OPTIONS, SWITCHES, option, switch  # noqa: E402
from _text import figure_text  # noqa: E402

THEME = load_theme()


# ------------------------------------------------------------------ defaults

def test_a_day_is_the_default_step():
    assert THEME["hours_per_tick"] == 24.0


def test_a_two_day_recording_gets_a_tick_per_day():
    assert THEME.hour_ticks(0.5, 49.0) == [24.0, 48.0]


def test_ticks_start_at_zero_when_the_axis_does():
    assert THEME.hour_ticks(0.0, 72.0) == [0.0, 24.0, 48.0, 72.0]


def test_a_tick_exactly_on_the_limit_is_kept():
    """A 48 h recording must show its last day boundary, not stop at 24."""
    assert THEME.hour_ticks(0.0, 48.0)[-1] == 48.0


def test_hours_that_are_not_quite_round_still_land_on_the_day():
    """Frame intervals give hours like 24.000000001; the tick is still 24."""
    ticks = THEME.hour_ticks(0.4999999, 48.0000001)
    assert ticks == [24.0, 48.0]


# ------------------------------------------------------------- other steps

@pytest.mark.parametrize("step", [1, 2, 3, 4, 6, 8, 12, 24, 48, 72])
def test_every_factor_and_multiple_of_a_day_is_allowed(step):
    ticks = THEME.hour_ticks(0, 72, step)
    assert ticks and all(abs((t / step) - round(t / step)) < 1e-9 for t in ticks)


def test_a_fractional_step_that_divides_the_day_is_allowed():
    assert THEME.hour_ticks(0, 6, 1.5) == [0.0, 1.5, 3.0, 4.5, 6.0]


@pytest.mark.parametrize("step", [5, 7, 10, 9.5, 36, 0, -24])
def test_a_step_that_is_neither_a_factor_nor_a_multiple_is_refused(step):
    with pytest.raises(ValueError, match="factor or a multiple of 24"):
        THEME.hour_ticks(0, 72, step)


@pytest.mark.parametrize("step", [1, 1.5, 2, 3, 4, 6, 8, 12])
def test_a_sub_day_step_marks_every_day_boundary(step):
    """The whole point. A 10 h step never puts a tick on 24, 48 or 72."""
    assert {24.0, 48.0, 72.0, 96.0} <= set(THEME.hour_ticks(0, 96, step))


@pytest.mark.parametrize("step", [24, 48, 96])
def test_a_multi_day_step_puts_every_tick_on_a_day_boundary(step):
    """48 h marks every second midnight, which is the point of asking for 48."""
    assert all(tick % 24 == 0 for tick in THEME.hour_ticks(0, 96, step))


@pytest.mark.parametrize("step", [1, 1.5, 2, 3, 4, 6, 8, 12])
def test_a_sub_day_step_divides_the_day_into_whole_intervals(step):
    """Ticks within a day are at the same times on every day, not drifting."""
    ticks = THEME.hour_ticks(0, 48, step)
    first_day = [t for t in ticks if t < 24]
    second_day = [round(t - 24, 6) for t in ticks if 24 <= t < 48]
    assert first_day == second_day and len(first_day) == round(24 / step)


# ----------------------------------------------------------- configuration

def test_the_step_can_be_set_in_the_theme_block():
    theme = load_theme({"hours_per_tick": 12})
    assert theme.hour_ticks(0, 48) == [0.0, 12.0, 24.0, 36.0, 48.0]


def test_a_bad_step_stops_the_configuration_rather_than_the_figure():
    with pytest.raises(ValueError, match="factor or a multiple of 24"):
        load_theme({"hours_per_tick": 10})


def test_the_step_is_recorded_with_the_run():
    assert load_theme({"hours_per_tick": 6}).stamp()["hours_per_tick"] == 6.0


# --------------------------------------------------------- the command line

def test_the_flag_is_read_with_either_spelling():
    assert option("hour_ticks", cast=float, argv=["run", "--hour-ticks", "12"]) == 12.0
    assert option("hour_ticks", cast=float, argv=["run", "--hour_ticks=8"]) == 8.0


def test_no_flag_means_no_opinion():
    assert option("hour_ticks", cast=float, argv=["run"]) is None


def test_a_value_that_is_not_a_number_is_refused_readably():
    with pytest.raises(SystemExit, match="hour-ticks"):
        option("hour_ticks", cast=float, argv=["--hour-ticks", "daily"])


def test_a_builder_option_is_not_mistaken_for_wording(tmp_path):
    """`--hour-ticks 12` must not read as an unknown text slot, nor eat a title."""
    text = figure_text(
        tmp_path, "demo",
        argv=["run", "--hour-ticks", "12", "--title", "Kept"],
        title="Default",
    )
    assert text.title == "Kept"


def test_wording_is_not_mistaken_for_a_builder_option():
    assert option("identity", default=44, cast=int,
                  argv=["run", "--title", "--identity is not here"]) == 44


def test_a_typo_is_still_refused_now_that_some_options_are_skipped():
    with pytest.raises(SystemExit, match="unknown option"):
        figure_text(Path("."), "demo", argv=["--hour-tick", "12"])


def test_every_declared_option_has_a_description():
    assert all(entry.help for entry in OPTIONS.values())
    assert all(description for description in SWITCHES.values())


def test_every_declared_option_says_how_its_text_becomes_a_value():
    """A cast and a metavar, so a figure's ``--help`` can be true without them."""
    assert all(callable(entry.cast) for entry in OPTIONS.values())
    assert all(entry.metavar for entry in OPTIONS.values())


def test_every_declared_option_is_in_the_readme():
    """A flag nobody can find is a flag that does not exist."""
    readme = (Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    missing = [
        name for name in {**OPTIONS, **SWITCHES}
        if f"--{name.replace('_', '-')}" not in readme
    ]
    assert not missing, "declared but undocumented: " + ", ".join(sorted(missing))


def test_a_switch_does_not_swallow_the_next_token():
    """``--draft <run>`` must not read the run folder as the switch's value."""
    assert option("identity", default=44, cast=int,
                  argv=["--draft", "outputs/a03", "--identity", "7"]) == 7
    assert switch("draft", argv=["--draft", "outputs/a03"]) is True
    assert switch("draft", argv=["outputs/a03"]) is False
