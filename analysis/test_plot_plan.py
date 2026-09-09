"""The plot plan: what a request expands into, and what is refused.

Not to be confused with ``test_figure_plan.py``, which is about the primitives
a panel is drawn with. This is about the ``plots`` block: a list of requests,
each naming a figure and optionally a setting to vary, expanded into one item
per figure that will actually be drawn.

The difference between ``for_each`` and ``options`` is the whole design and has
a test of its own. Everything else here is a refusal, because the failure this
feature has to avoid is the quiet one: a plan that draws forty pages of the
wrong thing looks exactly like a plan that worked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from analysis.config import AnalysisConfig
from analysis.metric_groups import build
from analysis.plots import (REQUEST_KEYS, TEXT_SLOTS, PlotItem, expand, parse,
                            problems, slugify)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

CIRCADIAN = ["cosinor_amplitude", "m10", "l5"]
GROUPS = build({"circadian": CIRCADIAN})


def _specs() -> dict:
    from _schema import catalogue

    return {spec.slug: spec for _, _, spec in catalogue() if spec is not None}


SPECS = _specs()


def _items(*blocks, groups=GROUPS, specs=None, unconverted=()):
    return expand(parse(list(blocks), groups), SPECS if specs is None else specs,
                  unconverted=unconverted)


def _config_file(tmp_path: Path, **blocks) -> Path:
    body = {
        "dataset": "test",
        "frame_interval_min": 30,
        "movies": [{"stem": "m1", "labels": "labels.tif", "raw": "raw.tif"}],
    }
    body.update(blocks)
    path = tmp_path / "analysis_config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# --------------------------------------------------- the one structural idea


def test_for_each_draws_a_page_per_value_and_options_draws_one_page():
    """The whole design in one test.

    `for_each` is a page per value; `options` is one page with all of them on
    it. Encoding that difference as a key of its own, rather than as how deeply
    one key's value is nested, is what makes a plan readable at a glance.
    """
    fanned = _items({"figure": "rhythm-strength", "for_each": {"metrics": "@circadian"}})
    pooled = _items({"figure": "rhythm-strength", "options": {"metrics": "@circadian"}})

    assert len(fanned) == len(CIRCADIAN)
    assert [item.options["metrics"] for item in fanned] == [[c] for c in CIRCADIAN]
    assert len(pooled) == 1
    assert pooled[0].options["metrics"] == CIRCADIAN


def test_several_for_each_keys_take_the_cross_product_last_key_fastest():
    """Predictable order: it is the order the figures land on disk in."""
    items = _items({"figure": "change-ledger",
                    "for_each": {"stem": ["a", "b"], "metrics": ["m10", "l5"]}})
    assert [item.name for item in items] == [
        "change-ledger/stem=a,metrics=m10",
        "change-ledger/stem=a,metrics=l5",
        "change-ledger/stem=b,metrics=m10",
        "change-ledger/stem=b,metrics=l5",
    ]


def test_a_list_valued_setting_is_wrapped_and_a_scalar_one_is_not():
    """A bare string given to a list-valued option is read one character at a time."""
    fanned = _items({"figure": "cell-report-card", "for_each": {"identity": [7, 12]}})
    assert [item.options["identity"] for item in fanned] == [7, 12]

    metrics = _items({"figure": "rhythm-strength", "for_each": {"metrics": ["m10"]}})
    assert metrics[0].options["metrics"] == ["m10"]


def test_a_for_each_value_that_is_already_a_list_is_not_wrapped_again():
    """Two metrics on each of two pages, rather than two pages of one list."""
    items = _items({"figure": "rhythm-strength",
                    "for_each": {"metrics": [["m10", "l5"], ["cosinor_amplitude"]]}})
    assert [item.options["metrics"] for item in items] == [
        ["m10", "l5"], ["cosinor_amplitude"]]
    assert items[0].name == "rhythm-strength/metrics=m10+l5"


def test_the_options_of_a_request_are_on_every_item_it_produces():
    items = _items({"figure": "rhythm-strength",
                    "for_each": {"metrics": "@circadian"},
                    "options": {"bins": 30}})
    assert all(item.options["bins"] == 30 for item in items)


def test_the_text_of_a_request_is_on_every_item_it_produces():
    items = _items({"figure": "rhythm-strength", "footnote": "one cell each",
                    "for_each": {"metrics": ["m10", "l5"]}})
    assert all(item.text["footnote"] == "one cell each" for item in items)


# ------------------------------------------------------------------- naming


def test_a_request_that_produces_one_item_is_called_after_its_figure():
    assert _items({"figure": "rhythm-strength"})[0].name == "rhythm-strength"


def test_an_as_template_substitutes_every_for_each_key():
    items = _items({"figure": "cell-report-card",
                    "for_each": {"identity": [7, 12, 40]},
                    "as": "report-card/cell-{identity}"})
    assert [item.name for item in items] == [
        "report-card/cell-7", "report-card/cell-12", "report-card/cell-40"]


def test_an_as_template_naming_a_key_the_request_does_not_vary_is_refused():
    with pytest.raises(ValueError, match="which this request does not vary"):
        _items({"figure": "cell-report-card", "for_each": {"identity": [7]},
                "as": "cell-{stem}"})


def test_a_value_becomes_a_folder_name_that_survives_a_filesystem():
    assert slugify("cosinor_amplitude") == "cosinor-amplitude"
    assert slugify("95_A3") == "95-a3"
    assert slugify(7) == "7"
    assert slugify(["m10", "l5"]) == "m10+l5"


def test_two_items_with_one_name_are_refused_naming_both_requests():
    """Otherwise the second silently overwrites the first item's bundle."""
    with pytest.raises(ValueError, match=r"plots\[0\] and plots\[1\]"):
        _items({"figure": "rhythm-strength"}, {"figure": "rhythm-strength"})


def test_an_as_template_with_no_placeholder_on_a_fanned_request_is_refused():
    """It would name every item the same thing, which is the same failure."""
    with pytest.raises(ValueError, match="both draw an item called"):
        _items({"figure": "rhythm-strength", "for_each": {"metrics": ["m10", "l5"]},
                "as": "one-name"})


# ---------------------------------------------------------------- refusals


def test_a_request_naming_no_figure_is_refused():
    with pytest.raises(ValueError, match="does not say which figure"):
        parse([{"for_each": {"metrics": ["m10"]}}], GROUPS)


def test_an_unknown_figure_is_refused_with_the_known_slugs_listed():
    with pytest.raises(ValueError, match="no figure declares this slug"):
        _items({"figure": "rhythm-strenth"})


def test_a_builder_not_on_the_schema_is_refused_with_its_own_message():
    """It declares no options, so nothing about the request can be checked."""
    with pytest.raises(ValueError, match="not on the figure schema"):
        _items({"figure": "41_new_idea"}, unconverted={"41_new_idea.py"})


def test_an_unknown_setting_is_refused_with_what_the_figure_does_accept():
    with pytest.raises(ValueError, match="not a setting of this figure"):
        _items({"figure": "rhythm-strength", "options": {"nonesuch": 3}})


def test_a_switch_may_not_be_fanned_over():
    with pytest.raises(ValueError, match="is a switch"):
        _items({"figure": "rhythm-strength", "for_each": {"draft": [True, False]}})


def test_a_stray_key_on_a_request_is_refused_against_the_ones_it_takes():
    with pytest.raises(ValueError, match="is not part of a request"):
        parse([{"figure": "rhythm-strength", "foreach": {"metrics": ["m10"]}}],
              GROUPS)


def test_an_empty_for_each_list_is_refused():
    with pytest.raises(ValueError, match="draws nothing"):
        parse([{"figure": "rhythm-strength", "for_each": {"metrics": []}}], GROUPS)


def test_a_for_each_given_a_single_value_says_what_to_do_instead():
    with pytest.raises(ValueError, match="move it to options"):
        parse([{"figure": "cell-report-card", "for_each": {"identity": 7}}], GROUPS)


def test_one_setting_may_not_be_in_both_for_each_and_options():
    with pytest.raises(ValueError, match="in both for_each and options"):
        parse([{"figure": "rhythm-strength", "for_each": {"metrics": ["m10"]},
                "options": {"metrics": ["l5"]}}], GROUPS)


def test_a_group_may_not_be_named_for_a_setting_that_holds_no_columns():
    with pytest.raises(ValueError, match="does not take one"):
        parse([{"figure": "cell-report-card", "for_each": {"identity": "@circadian"}}],
              GROUPS)


def test_a_whole_group_may_not_be_given_to_a_setting_that_holds_one_column():
    with pytest.raises(ValueError, match="put size in for_each instead"):
        parse([{"figure": "stable-traits-versus-states",
                "options": {"size": "@circadian"}}], GROUPS)


def test_a_run_level_figure_may_not_be_fanned_over_stem():
    """It reads nothing one movie wrote, so a page per movie is one page twice."""
    with pytest.raises(ValueError, match="belongs to one movie"):
        _items({"figure": "contrast-forest", "for_each": {"stem": ["a", "b"]}})


def test_a_cast_this_module_does_not_recognise_is_refused_not_guessed():
    """Guessing scalar turns a fanned list setting into a bare string."""
    from analysis.plots import _is_list_valued

    class _Spec:
        slug = "made-up"
        options = ()

    class _Vocabulary:
        cast = staticmethod(lambda text: text)

    with pytest.raises(ValueError, match="does not recognise"):
        _is_list_valued(_Spec(), "metrics", {"metrics": _Vocabulary()})


# ------------------------------------------------------------ the untested one


def test_fanning_over_panels_draws_one_page_per_panel():
    """Universal and list-valued, so it works; decided here rather than found."""
    items = _items({"figure": "cell-report-card",
                    "for_each": {"panels": ["tiles", "traces"]}})
    assert [item.options["panels"] for item in items] == [["tiles"], ["traces"]]


# ----------------------------------------------------------- the configuration


def test_the_text_slots_still_agree_with_the_ones_the_figures_own():
    """The mirror in `plots.py` must not go stale."""
    from _text import SLOTS

    assert set(TEXT_SLOTS) == set(SLOTS)


def test_a_configuration_with_no_plots_block_expands_to_nothing(tmp_path):
    config = AnalysisConfig.load(_config_file(tmp_path))
    assert config.plots == []
    assert config.plot_items() == []
    assert config.plot_problems() == []


def test_a_plan_in_a_configuration_expands(tmp_path):
    config = AnalysisConfig.load(_config_file(
        tmp_path,
        metric_groups={"circadian": CIRCADIAN},
        plots=[{"figure": "rhythm-strength", "for_each": {"metrics": "@circadian"}}],
    ))
    assert config.plot_problems() == []
    assert [item.name for item in config.plot_items()] == [
        f"rhythm-strength/metrics={slugify(c)}" for c in CIRCADIAN]


def test_the_doctor_gets_one_line_per_faulty_request_not_just_the_first(tmp_path):
    config = AnalysisConfig.load(_config_file(tmp_path, plots=[
        {"figure": "nonesuch"},
        {"figure": "rhythm-strength", "options": {"nonesuch": 1}},
    ]))
    found = config.plot_problems()
    assert len(found) == 2
    assert "no figure declares this slug" in found[0]
    assert "not a setting of this figure" in found[1]


def test_a_plan_that_cannot_be_expanded_reports_rather_than_raises(tmp_path):
    config = AnalysisConfig.load(_config_file(
        tmp_path, plots=[{"figure": "rhythm-strength"},
                         {"figure": "rhythm-strength"}]))
    assert any("both draw an item called" in line
               for line in config.plot_problems())


def test_the_request_keys_are_the_ones_the_message_offers():
    assert set(REQUEST_KEYS) == {"figure", "for_each", "options", "as"}


# -------------------------------------------------------------- the item shape


def test_an_item_survives_a_round_trip_through_json():
    """It is written into the run folder and read back by a builder."""
    item = _items({"figure": "rhythm-strength",
                   "for_each": {"metrics": ["m10"]},
                   "options": {"bins": 30}, "title": "t"})[0]
    again = PlotItem.from_dict(json.loads(json.dumps(item.as_dict())))
    assert again.name == item.name
    assert again.options == item.options
    assert again.text == item.text
    assert again.varied == item.varied


def test_the_varied_settings_render_for_a_manifest_column():
    item = _items({"figure": "change-ledger",
                   "for_each": {"stem": ["95_A3"], "metrics": ["m10"]}})[0]
    assert item.described() == "stem=95-a3;metrics=m10"


def test_problems_reports_nothing_for_a_plan_that_is_fine():
    assert problems(parse([{"figure": "rhythm-strength"}], GROUPS), SPECS) == []


# ------------------------------------------------------- carrying an item


def _run_folder(tmp_path: Path, figures: dict | None = None,
                items: list | None = None) -> Path:
    """A run folder holding only a `figures.json`, which is all transport needs."""
    body = {"figures": figures or {}}
    if items is not None:
        body["plots"] = items
    (tmp_path / "figures.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_no_item_reads_exactly_what_it_read_before(tmp_path):
    from _text import _from_run

    run = _run_folder(tmp_path, {"demo": {"title": "t", "options": {"bins": 30}}})
    assert _from_run(run, "demo") == ({"title": "t"}, {"bins": 30})


def test_an_item_lays_its_settings_over_the_per_slug_block(tmp_path):
    """The per-slug block is what every drawing shares; the item is the difference."""
    from _text import _from_run

    run = _run_folder(
        tmp_path, {"demo": {"options": {"bins": 30, "metrics": ["area_px"]}}},
        [{"name": "demo/metrics=m10", "figure": "demo",
          "options": {"metrics": ["m10"]}}])
    text, options = _from_run(run, "demo", "demo/metrics=m10")
    assert options == {"bins": 30, "metrics": ["m10"]}
    assert text == {}


def test_an_item_may_carry_its_own_wording(tmp_path):
    from _text import figure_text

    run = _run_folder(tmp_path, {"demo": {"footnote": "shared"}},
                      [{"name": "one", "figure": "demo",
                        "text": {"footnote": "this one only"}}])
    assert figure_text(run, "demo", argv=[], item="one").footnote == "this one only"


def test_a_flag_still_beats_both(tmp_path):
    from _text import figure_text

    run = _run_folder(tmp_path, {"demo": {"title": "from config"}},
                      [{"name": "one", "figure": "demo",
                        "text": {"title": "from the item"}}])
    text = figure_text(run, "demo", argv=["--title", "from the flag"], item="one")
    assert text.title == "From the flag"


def test_an_item_naming_a_figure_other_than_this_one_is_refused(tmp_path):
    from _text import item_settings

    run = _run_folder(tmp_path, {}, [{"name": "one", "figure": "other"}])
    with pytest.raises(SystemExit, match="is an item of other"):
        item_settings(run, "demo", "one")


def test_an_unknown_item_lists_the_items_the_run_does_have(tmp_path):
    from _text import item_settings

    run = _run_folder(tmp_path, {}, [{"name": "demo/metrics=m10", "figure": "demo"},
                                     {"name": "demo/metrics=l5", "figure": "demo"}])
    with pytest.raises(SystemExit, match="demo/metrics=l5"):
        item_settings(run, "demo", "demo/metrics=m11")


def test_a_run_with_no_plan_says_so_rather_than_listing_nothing(tmp_path):
    from _text import item_settings

    with pytest.raises(SystemExit, match="has no plot plan"):
        item_settings(_run_folder(tmp_path), "demo", "anything")


def test_an_item_may_set_stem_and_panels_without_the_figure_declaring_them(tmp_path):
    """They are read before a context exists, so they must not reach the
    options block that refuses an option the figure does not declare."""
    from _text import _from_run, item_settings

    run = _run_folder(tmp_path, {}, [{"name": "one", "figure": "demo",
                                      "options": {"stem": "95_A3",
                                                  "panels": ["tiles"],
                                                  "bins": 30}}])
    assert _from_run(run, "demo", "one")[1] == {"bins": 30}
    assert item_settings(run, "demo", "one")["stem"] == "95_A3"


def test_a_plan_written_beside_the_bundles_is_read_too(tmp_path):
    """`plots --plan <file>` records what it drew where the bundles are."""
    from _text import plan_items

    run = _run_folder(tmp_path, {})
    (run / "figures").mkdir()
    (run / "figures" / "plan.json").write_text(
        json.dumps({"plots": [{"name": "one", "figure": "demo"}]}), encoding="utf-8")
    assert [entry["name"] for entry in plan_items(run)] == ["one"]


def test_a_nested_item_name_is_a_folder_and_a_flat_filename():
    from _bundle import bundle_for, flat

    assert flat("rhythm-strength/metrics=m10") == "rhythm-strength_metrics=m10"
    bundle = bundle_for(Path("outputs") / "g02", "rhythm-strength/metrics=m10")
    assert bundle == Path("outputs") / "g02" / "figures" / "rhythm-strength" / "metrics=m10"


def test_the_context_is_called_after_its_item_and_falls_back_to_the_slug():
    from _schema import FigureContext

    spec = SPECS["rhythm-strength"]
    plain = FigureContext(spec=spec, run=Path("."), tables=Path("."), theme=None,
                          bundle=Path("."), summary={}, field={}, stem=None)
    assert plain.name == "rhythm-strength"
    fanned = FigureContext(spec=spec, run=Path("."), tables=Path("."), theme=None,
                           bundle=Path("."), summary={}, field={}, stem=None,
                           item="rhythm-strength/metrics=m10")
    assert fanned.name == "rhythm-strength/metrics=m10"


def test_every_figure_accepts_item_without_declaring_it():
    from _schema import UNIVERSAL, _check_argv

    assert "item" in UNIVERSAL
    _check_argv(SPECS["rhythm-strength"], ["--item", "one"])


def test_none_of_the_numbered_builders_needed_an_edit():
    """The seam is `_from_run`. If a builder needs changing, it is in the wrong
    place - and editing every builder is not a refactor, it is a rewrite."""
    builders = sorted(FIGURES_DIR.glob("[0-9][0-9]_*.py"))
    assert len(builders) == 47
    assert not [p.name for p in builders
                if "item" in p.read_text(encoding="utf-8")
                and "--item" in p.read_text(encoding="utf-8")]


def test_a_configuration_with_no_plan_never_reaches_matplotlib(tmp_path):
    """The measuring half of the package has never imported the plotting half.

    A run that only measures must stay possible where matplotlib is not
    installed, so a plotless configuration must not touch the figure catalogue
    on the way past.
    """
    import subprocess
    import sys as _sys

    path = _config_file(tmp_path)
    script = (
        "import sys\n"
        "class Blocked:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] == 'matplotlib':\n"
        "            raise ImportError('matplotlib is not installed here')\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        return self.find_module(name, path)\n"
        "sys.meta_path.insert(0, Blocked())\n"
        "from analysis.config import AnalysisConfig\n"
        f"config = AnalysisConfig.load(r'{path}')\n"
        "assert config.plot_items() == []\n"
        "assert config.plot_problems() == []\n"
        "assert 'matplotlib' not in sys.modules\n"
        "print('ok')\n"
    )
    done = subprocess.run([_sys.executable, "-c", script],
                          cwd=str(Path(__file__).resolve().parents[1]),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


# ------------------------------------------------- the guard and the command


def _bare_config(**kwargs) -> AnalysisConfig:
    return AnalysisConfig(dataset="t", frame_interval_min=30.0, movies=[],
                          output_root=".", **kwargs)


def test_a_queued_figure_whose_module_is_off_is_named_before_anything_is_measured():
    """The forty minutes run `g01` lost, caught by `doctor` instead."""
    from analysis.cli import figure_table_state, table_writers

    config = _bare_config(enabled_modules=["morphology"])
    faults = figure_table_state(config, SPECS["rhythm-strength"], table_writers())
    assert faults == [("rhythms.csv", "MODULE IS OFF", "rhythms")]


def test_with_every_module_on_no_figure_reads_a_table_that_will_not_exist():
    """Also the check that a `Stack` or an `Input` is not looked up as a table.

    Their `module` is prose - "the movie" - so treating one as a table would
    report every image-drawing page as broken by a module nobody switched off.
    """
    from analysis.cli import figure_table_state, table_writers

    config, writers = _bare_config(), table_writers()
    broken = {slug: figure_table_state(config, spec, writers)
              for slug, spec in SPECS.items()}
    assert not {slug: faults for slug, faults in broken.items() if faults}


def test_an_optional_table_whose_module_is_off_is_not_a_fault():
    """A missing optional table is a panel left out with a note, by design."""
    from analysis.cli import figure_table_state, table_writers

    config = _bare_config(enabled_modules=["motility", "presence"])
    faults = figure_table_state(config, SPECS["cells-on-screen"], table_writers())
    assert "history_lifespans.csv" not in [name for name, _, _ in faults]


def test_the_csv_suffix_is_stripped_before_the_writer_is_looked_up():
    """`Table.name` carries it; the writer index does not."""
    from analysis.cli import figure_table_state, table_writers

    faults = figure_table_state(_bare_config(), SPECS["rhythm-strength"],
                                table_writers())
    assert faults == []          # would be NO SUCH TABLE if the suffix stayed on


def _planned_run(tmp_path: Path, items: list) -> Path:
    (tmp_path / "figures.json").write_text(
        json.dumps({"figures": {}, "plots": items}), encoding="utf-8")
    return tmp_path


def _plots(*argv) -> tuple[int, str]:
    import contextlib
    import io

    from analysis.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            code = main(["plots", *argv])
        except SystemExit as stop:
            code = 1
            out.write(str(stop.code))
    return code, out.getvalue()


ITEMS = [
    {"name": "rhythm-strength/metrics=m10", "figure": "rhythm-strength",
     "varied": {"metrics": ["m10"]}, "options": {"metrics": ["m10"]}},
    {"name": "report-card/cell-7", "figure": "cell-report-card",
     "varied": {"identity": 7}, "options": {"identity": 7}},
]


def test_a_dry_run_prints_every_item_and_draws_nothing(tmp_path):
    run = _planned_run(tmp_path, ITEMS)
    code, output = _plots(str(run), "--dry-run")
    assert code == 0
    assert "rhythm-strength/metrics=m10" in output
    assert "report-card/cell-7" in output
    assert "2 item(s) would be drawn" in output
    assert not (run / "figures" / "plan.csv").exists()


def test_only_selects_by_item_name(tmp_path):
    run = _planned_run(tmp_path, ITEMS)
    _, output = _plots(str(run), "--dry-run", "--only", "report-card/cell-7")
    assert "1 item(s)" in output
    assert "rhythm-strength/metrics=m10" not in output


def test_only_selects_every_item_of_a_figure(tmp_path):
    run = _planned_run(tmp_path, [*ITEMS, {"name": "report-card/cell-9",
                                           "figure": "cell-report-card"}])
    _, output = _plots(str(run), "--dry-run", "--only", "cell-report-card")
    assert "2 item(s)" in output


def test_an_unknown_only_value_lists_the_names_that_do_exist(tmp_path):
    run = _planned_run(tmp_path, ITEMS)
    code, output = _plots(str(run), "--dry-run", "--only", "report-card/cell-8")
    assert code == 1
    assert "report-card/cell-7" in output


def test_a_run_with_no_plan_says_what_to_do_about_it(tmp_path):
    (tmp_path / "figures.json").write_text(json.dumps({"figures": {}}),
                                           encoding="utf-8")
    code, output = _plots(str(tmp_path), "--dry-run")
    assert code == 1
    assert "has no plot plan" in output


def test_a_failing_item_does_not_stop_the_rest_and_the_manifest_says_why(tmp_path):
    """Run against a folder with no tables at all, so every item fails fast."""
    from analysis.cli import PLAN_COLUMNS

    run = _planned_run(tmp_path, ITEMS)
    code, output = _plots(str(run))
    assert code == 1
    assert "--- rhythm-strength/metrics=m10" in output
    assert "--- report-card/cell-7" in output

    import csv

    with (run / "figures" / "plan.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == [item["name"] for item in ITEMS]
    assert {row["status"] for row in rows} == {"failed"}
    assert all(row["error"] for row in rows)
    assert list(rows[0]) == list(PLAN_COLUMNS)


def test_a_later_only_pass_keeps_the_rows_it_is_not_about(tmp_path):
    """Re-drawing one item after a file lock must not erase the other records."""
    import csv

    run = _planned_run(tmp_path, ITEMS)
    _plots(str(run))
    _plots(str(run), "--only", "report-card/cell-7")
    with (run / "figures" / "plan.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["name"] for row in rows} == {item["name"] for item in ITEMS}


def test_a_plan_file_is_expanded_and_recorded_beside_the_bundles(tmp_path):
    run = _planned_run(tmp_path, [])
    plan = tmp_path / "plan-of-my-own.json"
    plan.write_text(json.dumps({
        "metric_groups": {"circadian": CIRCADIAN},
        "plots": [{"figure": "rhythm-strength", "for_each": {"metrics": "@circadian"}}],
    }), encoding="utf-8")
    code, output = _plots(str(run), "--plan", str(plan), "--dry-run")
    assert code == 0
    assert f"{len(CIRCADIAN)} item(s)" in output
    # A dry run writes nothing into the run folder, the record included.
    assert not (run / "figures" / "plan.json").exists()

    _plots(str(run), "--plan", str(plan), "--only", "rhythm-strength")
    recorded = json.loads((run / "figures" / "plan.json").read_text(encoding="utf-8"))
    assert recorded["source"] == str(plan.resolve())
    assert len(recorded["plots"]) == len(CIRCADIAN)


def test_the_three_tables_the_run_writes_itself_are_in_the_writer_index():
    """`cell_frame` is the most-read table in the package and no module writes it.

    It is the join of every module's per-cell-frame output, built by the run.
    Without it in the index, every figure that reads it reports as broken.
    """
    from analysis.cli import JOIN_STEP, JOINED, table_writers

    writers = table_writers()
    assert all(writers.get(name) == JOIN_STEP for name in JOINED)


def test_a_figure_drawing_statistics_needs_contrasts_declared():
    """The contrasts step is switched on by declaring contrasts, not a module."""
    from analysis.cli import table_state, table_writers

    writers = table_writers()
    assert table_state(_bare_config(), "statistics", writers)[0] == "MODULE IS OFF"


def test_the_doctor_names_the_table_and_the_module_a_queued_figure_is_missing(tmp_path):
    """One line a user can act on, before a single pixel is drawn."""
    import contextlib
    import io

    from analysis.cli import main

    path = _config_file(
        tmp_path,
        enabled_modules=["morphology"],
        plots=[{"figure": "rhythm-strength"}],
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["doctor", "--config", str(path)])
    output = out.getvalue()
    assert code == 1
    assert "plots      1 request(s) -> 1 item(s)" in output
    assert "MODULE IS OFF" in output
    assert "rhythms.csv is written by 'rhythms'" in output


def test_the_doctor_prints_a_metric_group_resolved(tmp_path):
    import contextlib
    import io

    from analysis.cli import main

    path = _config_file(tmp_path, metric_groups={"circadian": CIRCADIAN})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(["doctor", "--config", str(path)])
    assert "group      circadian" in out.getvalue()
    assert "3 column(s)" in out.getvalue()
