"""Tests for the capital at the start of a title or an axis label.

Run with ``python -m pytest analysis/test_capitalisation.py``.

The property under test: a figure never opens a title or an axis label on a
lower-case letter, and never corrupts a name that carries its own capitals to
get there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.theme import load_theme, sentence_case  # noqa: E402
from _text import figure_text  # noqa: E402


# ------------------------------------------------------------------- the rule

@pytest.mark.parametrize(
    "written, shown",
    [
        ("hours from start of recording", "Hours from start of recording"),
        ("cell", "Cell"),
        ("area (px)", "Area (px)"),
        ("one cell's own spread", "One cell's own spread"),
    ],
)
def test_a_lower_case_label_is_raised(written, shown):
    assert sentence_case(written) == shown


@pytest.mark.parametrize(
    "written",
    [
        "pH of the medium",
        "mCherry intensity",
        "iNOS-positive cells",
        "CD68 reporter intensity",
        "Hours from start of recording",
    ],
)
def test_a_name_that_carries_its_own_capitals_is_left_alone(written):
    """Forcing the first letter of pH or mCherry would be worse than the gap."""
    assert sentence_case(written) == written


@pytest.mark.parametrize("written", ["24 h blocks", "µm per frame", "(px) per cell"])
def test_a_label_that_does_not_open_on_a_letter_is_left_alone(written):
    """These are already correct; the first letter inside them is not the start."""
    assert sentence_case(written) == written


def test_only_the_first_character_changes():
    """The second line of a two-line label is a continuation, not a new label."""
    assert sentence_case("fraction replaced\nper 30 min") == "Fraction replaced\nper 30 min"


def test_nothing_is_invented_for_empty_text():
    assert sentence_case("") == ""
    assert sentence_case("   ") == "   "


def test_leading_space_is_kept():
    assert sentence_case("  hours") == "  Hours"


# ------------------------------------------------------------------ the axes

def _axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    return figure, ax


def test_finish_raises_an_axis_label_a_builder_wrote_in_lower_case():
    theme = load_theme()
    figure, ax = _axes()
    try:
        ax.set_xlabel("hours from start of recording")
        ax.set_ylabel("cells")
        theme.finish(ax)
        assert ax.get_xlabel() == "Hours from start of recording"
        assert ax.get_ylabel() == "Cells"
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_finish_raises_an_axes_title_without_resizing_it():
    """``set_title`` would take the font back to matplotlib's default."""
    theme = load_theme()
    figure, ax = _axes()
    try:
        ax.set_title("cells called rhythmic", fontsize=31)
        theme.finish(ax)
        assert ax.get_title() == "Cells called rhythmic"
        assert ax.title.get_fontsize() == 31
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_finish_leaves_a_label_with_its_own_capitals_alone():
    theme = load_theme()
    figure, ax = _axes()
    try:
        ax.set_xlabel("pH of the medium")
        theme.finish(ax)
        assert ax.get_xlabel() == "pH of the medium"
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


# ----------------------------------------------------------------- the title

def test_a_builder_default_title_gets_its_capital(tmp_path):
    text = figure_text(tmp_path, "demo", argv=[], title="soma movement against turnover")
    assert text.title == "Soma movement against turnover"


def test_a_title_from_the_command_line_gets_its_capital(tmp_path):
    text = figure_text(tmp_path, "demo", argv=["--title", "microglia stay put"], title="Area")
    assert text.title == "Microglia stay put"
    assert text.source["title"] == "flag"


def test_a_title_from_the_configuration_gets_its_capital(tmp_path):
    import json

    (tmp_path / "figures.json").write_text(
        json.dumps({"figures": {"demo": {"title": "cells get bigger"}}}), encoding="utf-8")
    text = figure_text(tmp_path, "demo", argv=[], title="Area against time")
    assert text.title == "Cells get bigger"
    assert text.source["title"] == "config"


def test_a_deleted_title_stays_deleted(tmp_path):
    """Capitalisation must not resurrect text the user took off the figure."""
    text = figure_text(tmp_path, "demo", argv=["--title="], title="Area against time")
    assert text.title == ""


def test_the_other_slots_are_left_as_written(tmp_path):
    """A footnote is body text; only the title is a title."""
    text = figure_text(tmp_path, "demo", argv=[], footnote="both spreads are IQRs.")
    assert text.footnote == "both spreads are IQRs."


# --------------------------------------------------------------- the builders

def test_no_builder_writes_a_lower_case_axis_label():
    """The theme is the backstop; the source should already read correctly."""
    import re

    offenders = []
    pattern = re.compile(r"""set_(?:x|y)label\(f?["']([a-z])""")
    for builder in sorted((Path(__file__).resolve().parent / "figures").glob("[0-9]*.py")):
        for number, line in enumerate(builder.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{builder.name}:{number}: {line.strip()}")
    assert not offenders, "axis labels written in lower case:\n" + "\n".join(offenders)


def test_no_builder_writes_a_lower_case_default_title():
    import re

    offenders = []
    pattern = re.compile(r"""^\s*title=f?["']([a-z])""")
    for builder in sorted((Path(__file__).resolve().parent / "figures").glob("[0-9]*.py")):
        for number, line in enumerate(builder.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{builder.name}:{number}: {line.strip()}")
    assert not offenders, "default titles written in lower case:\n" + "\n".join(offenders)
