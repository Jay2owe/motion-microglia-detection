"""Command line.

    python -m analysis modules                       what can be measured
    python -m analysis figures                       what can be drawn
    python -m analysis theme      --list             how figures will look
    python -m analysis conditions --config <file>    which movie is in which group
    python -m analysis doctor     --config <file>    are the inputs readable and pinned
    python -m analysis run        --config <file> --out outputs/<run>
    python -m analysis window     <run> --config <file>   before and after, separately
    python -m analysis pool       <run>               every movie's tables in one
    python -m analysis contrasts  <run> --config <file>   which groups differ, and by how much
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

import analysis.modules  # noqa: F401  - registers every module
from analysis.config import AnalysisConfig
from analysis.io import (PROVENANCE_ADDED, PROVENANCE_INFERRED,
                         PROVENANCE_UNRESOLVED, sha256_of)
from analysis.registry import list_derived, list_modules
from analysis.run import run as run_analysis
from analysis.units import resolve_scale


def _tables_of(module) -> None:
    """The tables a module declares, and what one row of each is.

    Beside the columns for the same reason the columns are here: a reader
    asking what a module leaves behind wants the files as well as the headings,
    and a table's grain is the thing they need before they can join it to
    anything. A folded table says so, because it does not arrive as a file.
    """
    for output in module.writes:
        grain = ", ".join(output.grain) or "not stated - a copied table"
        note = " (folded into a roll-up)" if output.fold else ""
        optional = " (not written for every movie)" if output.optional else ""
        print(f"  {'':<18} {output.name + '.csv':<32} one row per {grain}{note}{optional}")


def _columns_of(module) -> None:
    """The columns a module declares, printed under it.

    Worth the lines because this is the only place a reader can see what a
    module actually leaves behind without opening its source, and a column
    missing here is a column that will get an invented label on every figure
    that draws it.
    """
    if module.writes:
        print(f"  {'':<16} writes {len(module.writes)} table(s):")
        _tables_of(module)
    if not module.produces:
        print(f"  {'':<16} writes: nothing declared yet")
        return
    print(f"  {'':<16} writes {len(module.produces)} columns:")
    for column in module.produces:
        unit = f"  ({column.unit})" if column.unit else ""
        print(f"  {'':<18} {column.name:<32} {column.label}{unit}")


def _cmd_modules(_: argparse.Namespace) -> int:
    print("measurement modules (pixels -> tables)")
    for module in list_modules():
        needs = ", ".join(module.requires)
        print(f"  {module.name:<16} {module.description}")
        print(f"  {'':<16} needs: {needs}")
        _columns_of(module)
    print()
    print("derived modules (tables -> tables)")
    for module in list_derived():
        print(f"  {module.name:<16} {module.description}")
        _columns_of(module)
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    """The numbered figure set, and what each one accepts.

    The counterpart of ``modules``. A figure declares its panels, its source
    tables and its options on itself, so this reads the declarations rather
    than a list kept here - which is the only arrangement where the catalogue
    cannot drift from the figures.

    A builder not yet on the schema still gets a row. It is in the folder and a
    user can run it; leaving it out would read as though it had gone.
    """
    figures_dir = Path(__file__).resolve().parent / "figures"
    if str(figures_dir) not in sys.path:
        sys.path.insert(0, str(figures_dir))
    import _schema  # noqa: E402  - needs the path above, and pulls in matplotlib

    rows = _schema.catalogue()
    if args.slug:
        try:
            print(_schema.get_figure(args.slug).usage())
        except KeyError as error:
            print(str(error).strip("'"))
            return 1
        return 0

    print("figures (tables -> one audited bundle each)")
    for number, path, spec in rows:
        if spec is None:
            # The filename, not a slug guessed from it: a builder off the
            # schema has not said what its slug is, and the two are allowed to
            # differ.
            print(f"  {number:>2}  {path.name:<32} not on the schema yet")
            continue
        print(f"  {number:>2}  {spec.slug:<32} {spec.summary}")
        print(f"      {'':<32} panels: "
              f"{', '.join(p.key for p in spec.panels) or 'one'}")
        print(f"      {'':<32} options: "
              f"{' '.join(o.flag for o in spec.options) or 'none'}")
    declared = sum(1 for _, _, spec in rows if spec is not None)
    print()
    print(f"{declared} of {len(rows)} declared; "
          f"python -m analysis figures --slug <slug> for one figure's options")
    return 0


def _cmd_theme(args: argparse.Namespace) -> int:
    from analysis.theme import PRESETS, ROLES, conformance_report, load_theme, write_swatch

    if args.check:
        report = conformance_report()
        if not report["available"]:
            print(f"conformance not checked: {report['reason']}")
            print("the vendored contract numbers are used as-is")
            return 0
        print(
            f"lab contract (analysis_kit 'pyflash'): {report['matches']} of "
            f"{report['keys_compared']} keys match"
        )
        for key, sides in report["differences"].items():
            print(f"  DRIFT {key}: contract {sides['contract']!r}, ours {sides['ours']!r}")
        return 0 if report["conformant"] else 1

    source = args.config or args.preset
    theme = load_theme(source)

    if args.preview:
        path = write_swatch(theme, args.preview)
        print(f"wrote {path}")
        print(f"and   {path.with_suffix('.png')}")
        return 0

    if args.list:
        print("presets")
        for name in sorted(PRESETS):
            other = load_theme(name)
            print(
                f"  {name:<13} type {other['tick_size']:g}/{other['axis_size']:g}/"
                f"{other['title_size']:g} pt   line {other['line_width']:g}   "
                f"{other['figure_size'][0]:g}x{other['figure_size'][1]:g} in"
            )
        print()

    print(f"active theme: {theme.preset}  (from {theme.source})")
    print("roles")
    for role, purpose in ROLES.items():
        print(f"  {role:<16} {theme.colour(role):<9} {purpose}")
    print()
    print("colour maps")
    for key in ("diverging_cmap", "sequential_cmap", "image_cmap"):
        print(f"  {key.replace('_cmap', ''):<16} {theme[key]}")
    return 0


def _cmd_conditions(args: argparse.Namespace) -> int:
    """Show which movie lands in which experimental group, without running anything.

    This is the command to run before `run`, every time a pattern changes. A
    regular expression that matches one stem too many is invisible in every
    figure downstream and obvious here.
    """
    config = AnalysisConfig.load(args.config)
    conditions = config.conditions

    print(f"dataset: {config.dataset}")
    if not conditions:
        print("\nno conditions block; every movie carries whatever it declares")
        for movie in config.movies:
            print(f"  {movie.stem:<24} {movie.condition or '-'}")
        print("\nadd a conditions block to derive them from the file names:")
        print('  "conditions": {"treated": "_A\\\\d", "control": "_B\\\\d"}')
        return 0

    if conditions.synthetic:
        print("\n*** SYNTHETIC DESIGN - invented to exercise the pipeline ***")
    print(f"matching against: the movie {conditions.match_against}")
    colours = conditions.colours()

    # Widths from the data: a long label must not shunt the colour column into
    # the one beside it, which is exactly when a table stops being checkable.
    name_width = max(len("condition"), *(len(c.name) for c in conditions)) + 2
    label_width = max(len("label"), *(len(c.label) for c in conditions)) + 2

    for factor, levels in conditions.by_factor().items():
        print(f"\nfactor: {factor}")
        print(f"  {'condition':<{name_width}}{'label':<{label_width}}"
              f"{'colour':<10}{'ref':<5}patterns")
        for entry in levels:
            patterns = " ".join(f"/{p}/" for p in entry.match) or "(declared only)"
            print(
                f"  {entry.name:<{name_width}}{entry.label:<{label_width}}"
                f"{colours[entry.name]:<10}{'yes' if entry.control else '':<5}{patterns}"
            )

    rows = [config.assignment(movie) for movie in config.movies]
    rows += [conditions.assign(stem) for stem in (args.try_stem or [])]

    print("\nassignment")
    for assignment in rows:
        arrow = assignment.condition if assignment.ok else "NOT RESOLVED"
        how = assignment.source
        if assignment.evidence:
            how += "  " + " ".join(f"/{p}/" for p in assignment.evidence.values())
        print(f"  {assignment.stem:<24} -> {arrow:<20} {how}")
        if assignment.problem:
            print(f"  {'':<24}    {assignment.problem}")

    notes = conditions.audit(rows)
    if notes:
        print()
        for note in notes:
            print(f"  ! {note}")

    problems = sum(1 for a in rows if not a.ok)
    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = AnalysisConfig.load(args.config)
    print(f"dataset: {config.dataset}")
    print(f"frame interval: {config.frame_interval_min} min")

    problems = 0
    for movie in config.movies:
        assignment = config.assignment(movie)
        subject = movie.subject or movie.stem
        print(f"\n{movie.stem}  (condition: {assignment.condition} "
              f"[{assignment.source}], subject: {subject})")
        if assignment.problem:
            print(f"  condition  UNRESOLVED  {assignment.problem}")
            problems += 1
        for key in ("labels", "raw", "unclaimed", "evidence", "provenance",
                    "valid_mask"):
            path = getattr(movie, key)
            if path is None:
                print(f"  {key:<10} not configured")
                continue
            path = Path(path)
            if not path.exists():
                # Said out loud, because a misspelled optional path otherwise
                # just makes a module skip and looks like success.
                print(f"  {key:<10} CONFIGURED BUT NOT FOUND  {path}")
                problems += 1
                continue
            digest = sha256_of(path)
            expected = movie.expected_sha256.get(key)
            state = "ok"
            if expected and digest != expected:
                state = "FINGERPRINT MISMATCH"
                problems += 1
            elif expected:
                state = "ok, pinned"
            else:
                state = "ok, not pinned"
            print(f"  {key:<10} {state}  {digest[:16]}  {path.name}")
            if key == "provenance":
                packed = tifffile.imread(path)
                inferred_px = int(np.count_nonzero(packed & PROVENANCE_INFERRED))
                unresolved_px = int(np.count_nonzero(packed & PROVENANCE_UNRESOLVED))
                added_px = int(np.count_nonzero(packed & PROVENANCE_ADDED))
                print(f"             {packed.shape[0]} frames, "
                      f"{inferred_px} reconstructed px, "
                      f"{unresolved_px} unresolved px, "
                      f"{added_px} px the detection never saw")

        # The declared inputs, which the loop above cannot cover because there
        # is no fixed number of them. Named here rather than left to the run,
        # because a misspelled path in one of these makes a module skip - and a
        # skip reads exactly like a movie that has nothing to measure.
        for label, declared, detail in (
            ("channel", movie.channels,
             lambda spec: f"channel_index={spec.channel_index}"),
            ("object", movie.objects,
             lambda spec: f"static={spec.static}"),
            ("side", movie.side_tables,
             lambda spec: f"keyed on {spec.keyed_on}"),
        ):
            for spec in declared:
                name = f"{label}:{spec.name}"
                if not Path(spec.path).exists():
                    print(f"  {name:<10} CONFIGURED BUT NOT FOUND  {spec.path}")
                    problems += 1
                    continue
                digest = sha256_of(spec.path)
                if spec.expected_sha256 and digest != spec.expected_sha256:
                    state = "FINGERPRINT MISMATCH"
                    problems += 1
                else:
                    state = "ok, pinned" if spec.expected_sha256 else "ok, not pinned"
                print(f"  {name:<10} {state}  {digest[:16]}  "
                      f"{Path(spec.path).name}  {detail(spec)}")

        if movie.history is None:
            print(f"  {'history':<10} not configured")
        elif not Path(movie.history).is_dir():
            print(f"  {'history':<10} CONFIGURED BUT NOT FOUND  {movie.history}")
            problems += 1
        else:
            found = sum(1 for _ in Path(movie.history).rglob("*.csv"))
            print(f"  {'history':<10} ok  {found} decision table(s)  "
                  f"{Path(movie.history).name}")

        scale = resolve_scale(
            config.frame_interval_min,
            config.microns_per_pixel,
            [Path(movie.raw), *config.calibration_tiffs],
        )
        if scale.calibrated:
            print(f"  scale      {scale.microns_per_pixel:.4f} um/px from {scale.source}")
        else:
            print("  scale      uncalibrated - every spatial number is reported in pixels")
            print("             set microns_per_pixel in the configuration to convert them")

        # Declared windows, checked here rather than after a run. A window
        # holding no frames is a mistyped window, and finding that out from an
        # empty summary an hour later is the whole failure mode `doctor` exists
        # to prevent.
        windows = config.windows_for(movie)
        if windows and Path(movie.labels).exists():
            try:
                with tifffile.TiffFile(movie.labels) as handle:
                    n_frames = int(handle.series[0].shape[0])
            except Exception:                       # a broken stack is the run's problem
                n_frames = 0
            hours = np.array([scale.hours(i + movie.source_frame_offset)
                              for i in range(n_frames)])
            for window in windows:
                if window.in_frames:
                    span = f"frames {window.from_frame}-{window.to_frame}"
                    held = int(((np.arange(n_frames) >= window.from_frame)
                                & (np.arange(n_frames) < window.to_frame)).sum())
                else:
                    span = f"{window.from_hours}-{window.to_hours} h"
                    held = int(((hours >= window.from_hours)
                                & (hours < window.to_hours)).sum())
                against = f", against {window.baseline}" if window.baseline else ""
                state = "ok" if held else "CATCHES NO FRAMES"
                if not held:
                    problems += 1
                print(f"  window     {state}  {window.name:<14} {span:<20} "
                      f"{held} of {n_frames} frames{against}")

    # The declared contrasts, checked against the design rather than against a
    # run. How many units each group would have is knowable from the
    # configuration alone when the grouping is `condition` or `subject`, and a
    # contrast that could never reach three units in a group is a contrast that
    # will only ever write refusals - worth knowing before measuring, not after.
    if config.contrasts:
        print()
        assignments = {movie.stem: config.assignment(movie).condition
                       for movie in config.movies}
        subjects = {movie.stem: (movie.subject or movie.stem) for movie in config.movies}
        for contrast in config.contrasts:
            units_per_group: dict[str, int] = {}
            for group in contrast.groups:
                stems = [stem for stem, condition in assignments.items()
                         if str(condition) == group] \
                    if contrast.group_by == "condition" else []
                if contrast.group_by != "condition":
                    units_per_group[group] = -1        # not knowable without a run
                elif contrast.unit == "movie":
                    units_per_group[group] = len(stems)
                elif contrast.unit == "subject":
                    units_per_group[group] = len({subjects[stem] for stem in stems})
                else:
                    units_per_group[group] = -1        # cells need the measuring
            sizes = ", ".join(
                f"{group}={'?' if count < 0 else count}"
                for group, count in units_per_group.items())
            known = [count for count in units_per_group.values() if count >= 0]
            state = "ok"
            if known and min(known) < 3:
                state = "TOO FEW UNITS"
                problems += 1
            elif any(count == 0 for count in units_per_group.values()):
                state = "GROUP IS EMPTY"
                problems += 1
            print(f"  contrast   {state}  {contrast.name:<26} "
                  f"{len(contrast.metrics)} metric(s)  by {contrast.group_by}  "
                  f"[{sizes}] {contrast.unit}(s)  {contrast.test}  "
                  f"family {contrast.family!r}")

    # The figures block, checked before anything is measured rather than by the
    # thirty-sixth build an hour later. A misspelled option is the failure that
    # looks like success: the figure draws, with the defaults.
    figure_notes = config.figure_problems()
    if figure_notes:
        print()
        for note in figure_notes:
            print(f"  figures    {note}")
        problems += len(figure_notes)

    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = AnalysisConfig.load(args.config)
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = config.output_root / output_dir.name if args.out_is_name else output_dir
    manifest = run_analysis(
        config,
        output_dir=output_dir,
        stems=args.stem or None,
        modules=args.module or None,
    )
    for note in manifest["design"].get("warnings", []):
        print(f"  ! {note}")
    for movie in manifest["movies"]:
        summary = movie["summary"]
        print(
            f"{movie['stem']} [{movie['condition']['condition']}]: "
            f"{summary['identities']} identities, "
            f"{summary['cell_frame_rows']} cell-frames, "
            f"{summary['hours_covered']:.1f} h, {movie['seconds']:.1f} s"
        )
        for record in movie["modules"]:
            if record["status"] == "skipped":
                print(f"  skipped {record['module']}: {record['reason']}")
    print(f"\nwritten to {output_dir}")
    if args.print_summary:
        for movie in manifest["movies"]:
            print(json.dumps(movie["summary"], indent=2, default=str))
    return 0


def _cmd_contrasts(args: argparse.Namespace) -> int:
    """Run the declared contrasts against a run that already exists.

    Separate from ``run`` for the same reason ``pool`` and ``window`` are: the
    question you want to ask is usually decided after you have looked at the
    measurements, and re-measuring to ask it would be absurd.
    """
    from analysis.contrasts import STATISTICS_FILE, run_contrasts
    from analysis.run import _write_table

    config = AnalysisConfig.load(args.config)
    if not config.contrasts:
        print("no contrasts declared; add a `contrasts` block to the configuration")
        return 0
    run_dir = Path(args.run)
    path = run_dir / STATISTICS_FILE
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; analysis runs are immutable, delete it "
            "deliberately or test into a new run")

    statistics = run_contrasts(run_dir, config)
    _write_table(statistics, path)

    refused = statistics[statistics["note"].astype(str) != ""]
    tested = statistics[statistics["note"].astype(str) == ""]
    print(f"{len(statistics)} result(s) from {len(config.contrasts)} contrast(s): "
          f"{len(tested)} tested, {len(refused)} refused")
    for _, row in tested.iterrows():
        mark = "*" if row["significant"] else " "
        print(f" {mark} {row['contrast']:<24} {row['metric']:<34} "
              f"{row['test']:<12} p={row['p_value']:.4g} "
              f"q={row['p_corrected']:.4g}  {row['effect_kind']}="
              f"{row['effect']:.4g}  n={row['n_a']}/{row['n_b']} {row['unit']}(s)")
    for _, row in refused.iterrows():
        print(f" ! {row['contrast']:<24} {row['metric']:<34} {row['note']}")
    print(f"\nwritten to {path}")
    return 0


def _cmd_window(args: argparse.Namespace) -> int:
    """Compute windowed summaries for a run that already exists.

    Separate from ``run`` so that windows can be declared after the measuring is
    done, which is the usual order: you find out that something changed at hour
    twenty by looking at the whole-recording result first.

    It re-reads each movie's inputs to rebuild the measurement context - the
    per-frame roll-up counts pixels in the label stack - but re-measures
    nothing. The numbers come from the run's own ``cell_frame.csv``.
    """
    import pandas as pd

    from analysis.io import load_movie
    from analysis.run import MEASURED_FOLDER, _stamp, _write_table
    from analysis.windows import WINDOWED, windowed_summaries

    config = AnalysisConfig.load(args.config)
    run_dir = Path(args.run)
    by_name = {output.name: output for output in WINDOWED}
    wrote = 0
    for movie in config.movies:
        table_dir = run_dir / movie.stem / MEASURED_FOLDER
        if not table_dir.is_dir():
            continue
        windows = config.windows_for(movie)
        if not windows:
            print(f"{movie.stem}: no windows declared")
            continue
        existing = [name for name in by_name if (table_dir / f"{name}.csv").exists()]
        if existing:
            raise FileExistsError(
                f"{table_dir} already holds {existing}; analysis runs are "
                "immutable, delete them deliberately or window a new run"
            )
        context, _ = load_movie(config, movie)
        cell_frame = pd.read_csv(table_dir / "cell_frame.csv")
        assignment = config.assignment(movie)
        produced = windowed_summaries(cell_frame, context, windows)
        for name, table in sorted(produced.items()):
            _write_table(_stamp(table, movie, assignment.condition),
                         table_dir / f"{name}.csv", by_name[name])
            print(f"{movie.stem}: {name}.csv  {len(table)} rows")
            wrote += 1
    if not wrote:
        print("nothing written; declare a `windows` block in the configuration")
    return 0


def _cmd_pool(args: argparse.Namespace) -> int:
    """Concatenate an existing run's per-movie tables into ``pooled/``.

    Separate from ``run`` so that runs measured before pooling existed can be
    pooled without re-measuring, which is the whole reason a concatenation is
    worth its own entry point.
    """
    from analysis.pool import pool_run

    fragment = pool_run(Path(args.run))
    stems = fragment["movies"]
    print(f"pooled {len(fragment['tables'])} table(s) from {len(stems)} movie(s): "
          f"{', '.join(stems)}")
    for name, record in sorted(fragment["tables"].items()):
        counts = ", ".join(f"{stem} {rows}" for stem, rows in record["rows_per_movie"].items())
        line = f"  {record['folder']}/{name + '.csv':<34} {record['rows']:>7} rows  ({counts})"
        print(line)
        # Printed rather than left in the manifest: a column one movie lacks is
        # the difference between "no events" and "not measured", and it is worth
        # seeing at the moment the pooled file is made.
        for stem, columns in record.get("columns_missing", {}).items():
            print(f"      ! {stem} had no {columns}")
        if record.get("movies_absent"):
            print(f"      ! not produced by {', '.join(record['movies_absent'])}")
    print(f"\nwritten to {Path(args.run) / 'pooled'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    modules = sub.add_parser("modules", help="list registered analysis modules")
    modules.set_defaults(handler=_cmd_modules)

    figures = sub.add_parser("figures", help="list the numbered figures and their options")
    figures.add_argument("--slug", help="print one figure's full option list instead")
    figures.set_defaults(handler=_cmd_figures)

    theme = sub.add_parser("theme", help="show, preview or check the figure aesthetic")
    theme.add_argument("--config", help="read the theme block from this analysis configuration")
    theme.add_argument("--preset", help="show a named preset instead")
    theme.add_argument("--list", action="store_true", help="list every preset first")
    theme.add_argument("--preview", metavar="SVG",
                       help="draw a swatch card of the theme to this path")
    theme.add_argument("--check", action="store_true",
                       help="check the vendored pyflash contract still matches analysis_kit")
    theme.set_defaults(handler=_cmd_theme)

    groups = sub.add_parser(
        "conditions", help="show which movie lands in which experimental group")
    groups.add_argument("--config", required=True)
    groups.add_argument("--try", dest="try_stem", action="append", metavar="STEM",
                        help="also test the patterns against this name, repeatable")
    groups.set_defaults(handler=_cmd_conditions)

    doctor = sub.add_parser("doctor", help="check inputs, fingerprints and spatial scale")
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(handler=_cmd_doctor)

    runner = sub.add_parser("run", help="measure the configured movies into a new run folder")
    runner.add_argument("--config", required=True)
    runner.add_argument("--out", required=True, help="output run folder, must not already exist")
    runner.add_argument("--stem", action="append", help="limit to this stem, repeatable")
    runner.add_argument("--module", action="append", help="limit to this module, repeatable")
    runner.add_argument("--out-is-name", action="store_true",
                        help="treat --out as a run name under the configured output_root")
    runner.add_argument("--print-summary", action="store_true",
                        help="print the movie summary JSON when the run finishes")
    runner.set_defaults(handler=_cmd_run)

    windower = sub.add_parser(
        "window", help="compute windowed summaries for a run that already exists")
    windower.add_argument("run", help="the run folder to add windowed summaries to")
    windower.add_argument("--config", required=True,
                          help="the configuration holding the `windows` block")
    windower.set_defaults(handler=_cmd_window)

    pooler = sub.add_parser(
        "pool", help="concatenate an existing run's per-movie tables into pooled/")
    pooler.add_argument("run", help="the run folder to pool, which must not already hold pooled/")
    pooler.set_defaults(handler=_cmd_pool)

    tester = sub.add_parser(
        "contrasts", help="run the declared group comparisons against an existing run")
    tester.add_argument("run", help="the run folder to test, which must hold pooled/")
    tester.add_argument("--config", required=True,
                        help="the configuration holding the `contrasts` block")
    tester.set_defaults(handler=_cmd_contrasts)
    return parser


def main(argv: list[str] | None = None) -> int:
    # A Windows console still defaults to cp1252, which cannot encode the α in
    # "Motion scaling exponent, α" - so listing the modules would end in a
    # traceback on the one platform this is most often run on. The labels
    # themselves stay as written, because they are what goes on the figure.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
