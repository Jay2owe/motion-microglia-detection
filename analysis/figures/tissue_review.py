"""Registered flat review bundle for the canonical six-panel tissue tectonics figure.

Adapted from the registered tissue-rhythm review producer. The copied producer
rebuilds the actual panels from frozen numeric sources and captured code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tifffile


def finish_display(ctx, result):
    from _schema import _header_positions, _clear_footnote
    from _options import as_length
    from _bundle import wrap_title, wrap_footnote
    from _text import figure_text
    text = figure_text(ctx.run, ctx.spec.slug, argv=ctx.argv, title=ctx.spec.title,
                       subtitle=result.subtitle, footnote=result.footnote, note=result.note)
    canvas_text = text.on_canvas()
    fig = result.figure
    under = as_length(result.footnote_y)
    left, title_y, subtitle_y, footnote_y = _header_positions(
        fig.get_figwidth(), fig.get_figheight(), as_length(result.header_x),
        as_length(result.title_y), as_length(result.subtitle_y), under)
    header = [fig.text(left, title_y, wrap_title(ctx.theme, fig, canvas_text.title),
                       fontsize=ctx.theme.size("title"), fontweight="bold", va="top")]
    if canvas_text.subtitle:
        header.append(fig.text(
            left, subtitle_y,
            wrap_footnote(ctx.theme, fig, canvas_text.subtitle, "subtitle"),
            fontsize=ctx.theme.size("subtitle"), va="top"))
    foot = None
    if canvas_text.footnote:
        foot = fig.text(
            left, footnote_y, wrap_footnote(ctx.theme, fig, canvas_text.footnote),
            fontsize=ctx.theme.size("note"), color=ctx.theme.colour("caption"),
            va="bottom")
    for ax in result.axes:
        ctx.theme.finish(ax, keep_spines=result.keep_spines)
    if foot is not None:
        _clear_footnote(fig, foot, under, header)


def pixel_table(ctx, result):
    del ctx
    return result.figure_data.copy()


def reproduce(bundle, output=None, verify_only=False):
    """Execute only this trusted captured producer, never arbitrary embedded code."""
    import runpy
    import reprofig
    bundle = Path(bundle).resolve()
    settings = json.loads(pd.read_csv(bundle / "der_render_settings.csv").settings_json.iloc[0])
    with tempfile.TemporaryDirectory(prefix="tissue-map-reproduce-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(bundle / "src_code.zip") as archive:
            for info in archive.infolist():
                if not (root / info.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("captured code archive contains an unsafe path")
            archive.extractall(root)
        sys.path[:0] = [str(root), str(root / "analysis/figures")]
        runpy.run_path(str(root / "analysis/figures/28_tissue_tectonics.py"))
        from _schema import FigureContext, get_figure
        from analysis.theme import Theme

        class CapturedContext(FigureContext):
            def _from_run_options(self):
                return settings["options"]

            def module_params(self, module):
                return settings["module_params"].get(module, {})

            def stack(self, name, module=None):
                return tifffile.imread(bundle / ("src_" + name))

            def input_path(self, kind):
                return bundle / ("src_" + kind + ".tif")

            def table(self, name, module=None):
                return pd.read_csv(bundle / ("src_" + name))

            def optional_table(self, name):
                path = bundle / ("src_" + name)
                return pd.read_csv(path) if path.is_file() else None

        theme = Theme(**json.loads((bundle / "src_theme.json").read_text(encoding="utf-8")))
        theme.apply()
        ctx = CapturedContext(spec=get_figure("tissue-tectonics"), run=bundle, tables=bundle,
            bundle=bundle, theme=theme, summary=settings["summary"], field=settings["field"],
            stem=None, argv=settings["argv"])
        result = ctx.spec.build(ctx)
        actual = pixel_table(ctx, result)
        expected = pd.read_csv(bundle / "figure_data.csv", low_memory=False)
        actual_checked = actual.astype(object).where(actual.notna(), "<missing>")
        expected_checked = expected.astype(object).where(expected.notna(), "<missing>")
        pd.testing.assert_frame_equal(
            actual_checked, expected_checked, check_dtype=False, check_exact=False,
            atol=1e-10, rtol=1e-10,
        )
        finish_display(ctx, result)
        result.figure.canvas.draw()
        if not verify_only:
            record = reprofig.extract_record(bundle / "tissue-tectonics.svg")
            reprofig.save_figure(result.figure, output or bundle / "reproduced.svg", record=record,
                                savefig_kwargs={"bbox_inches": "tight", "transparent": True})
        plt.close(result.figure)
        print(f"Verified and rendered {len(actual):,} exact panel rows from the captured sources.")


def export(run, assignment, speed_summary):
    from types import SimpleNamespace
    from dataclasses import asdict
    import circadian_workbench
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "analysis/figures"))
    from _schema import (FigureContext, load_all, tables_for, theme_for, summary_for, field_for)
    from _flat_bundle import finish
    run = Path(run).resolve()
    bundle = run / "figures/tissue-tectonics-current"
    spec = load_all()["tissue-tectonics"]
    argv = ["--map-summary", speed_summary, "--map-assignment", assignment,
            "--title", "Cell behaviour and occupancy across the tissue field"]
    ctx = FigureContext(spec=spec, run=run, tables=tables_for(run, None),
        theme=theme_for(run), summary=summary_for(run, None), field=field_for(run, None),
        bundle=bundle, stem=None, argv=argv)
    ctx.theme.apply()
    result = spec.build(ctx)
    finish_display(ctx, result)
    result.figure_data = pixel_table(ctx, result)
    settings = dict(summary=ctx.summary, field=ctx.field, argv=argv,
                    options={option.name: ctx.option(option.name) for option in spec.options},
                    module_params={"rhythms": ctx.module_params("rhythms")})
    result.auxiliary["render_settings.csv"] = pd.DataFrame([{"settings_json": json.dumps(settings)}])
    bundle.mkdir(parents=True, exist_ok=True)
    result.readme = ("# Tissue tectonics: canonical six-panel review\n\n"
        "The saved review follows the canonical order: first coverage time, cumulative occupied "
        "time, unique cells, speed, significant Lomb–Scargle intensity period, and split-event area.\n\n"
        + result.readme +
        "\n## Reproduction\n\nThe standalone producer reconstructs the actual panels from copied numeric "
        "sources and captured analysis code. It verifies every panel's encoded values before saving. "
        "The older tissue-tectonics bundles remain unchanged for recovery.\n")
    (bundle / "README.md").write_text(result.readme, encoding="utf-8")
    archive_path = bundle / "src_code.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        packages = [(root / "analysis", "analysis"),
                    (Path(circadian_workbench.__file__).parent, "circadian_workbench")]
        for package, prefix in packages:
            for path in package.rglob("*.py"):
                if path.name.startswith("test_") or "__pycache__" in path.parts:
                    continue
                archive.write(path, str(Path(prefix) / path.relative_to(package)))
    theme_path = bundle / "src_theme.json"
    theme_path.write_text(json.dumps(asdict(ctx.theme)), encoding="utf-8")
    result.standalone_producer = Path(__file__).read_text(encoding="utf-8")
    result.producer_sources.update({"code.zip": archive_path, "theme.json": theme_path,
                                    "producer.py": Path(__file__), "manifest.json": run / "manifest.json"})
    result.heading = ("Six canonical tissue maps show first coverage, occupied time, unique cells, "
                      "speed, significant intensity periods and split-event areas.")
    finished = finish(SimpleNamespace(run=run, bundle=bundle, name="tissue-tectonics", spec=spec,
                                      sources=ctx.sources), result)
    print(finished)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--map-assignment", default="occupancy_weighted_mean")
    parser.add_argument("--map-summary", default="median")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.run:
        export(args.run, args.map_assignment, args.map_summary)
    else:
        reproduce(Path(__file__).resolve().parent, args.output, args.verify_only)
