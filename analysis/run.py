"""Run the analysis for one or more movies into an immutable output folder.

The output folder mirrors the convention the tracking pipeline already uses: a
run is written once, never edited, and carries a manifest that fingerprints
both what went in and what came out. Re-running with the same inputs and the
same configuration must produce the same numbers, and the manifest is how that
is checked rather than assumed.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import analysis.modules  # noqa: F401  - importing registers every module
from analysis import __version__
from analysis.config import AnalysisConfig, MovieConfig
from analysis.io import load_movie, sha256_of
from analysis.registry import (MeasurementContext, Output, declared_tables,
                               get_derived, get_module, list_derived, list_modules)
from analysis.summarise import (ROLLUPS, build_cell_summary, build_frame_summary,
                                build_movie_summary, join_side, side_column_names)
from analysis.theme import load_theme
from analysis.windows import WINDOWED, window_extent, windowed_summaries

#: Which roll-up a folded table's columns belong in, looked up by grain. The
#: roll-ups declare their own grain in ``summarise.ROLLUPS``; a module declares
#: its table's grain beside the code that fills it. Nothing here names a module.
_ROLLUP_BY_GRAIN = {output.grain: output.name for output in ROLLUPS}
_ROLLUP_BY_NAME = {output.name: output for output in ROLLUPS}
_WINDOWED_BY_NAME = {output.name: output for output in WINDOWED}


def _fold_target(name: str, declared: dict[str, Output]) -> str | None:
    """The roll-up this table's columns belong in, or ``None`` if it is a file.

    Both the join that does the folding and the writer that skips what was
    folded ask this one question, so they cannot disagree. They used to read one
    hard-coded set of six table names, which meant the answer to "will my module
    get its own CSV?" was in ``run.py`` rather than in the module.

    ``fold`` is checked as well as the grain, and deliberately: ``presence``
    shares ``cell_frame``'s grain and must stay a file of its own, because it
    carries a row for every frame a cell was *absent* as well.
    """
    output = declared.get(name)
    if output is None or not output.fold:
        return None
    return _ROLLUP_BY_GRAIN.get(output.grain)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(table: pd.DataFrame, movie: MovieConfig, condition: str) -> pd.DataFrame:
    """Put who and what at the front of every table.

    Pooling ten movies then becomes a concatenation rather than a join against
    the configuration, and no downstream script can lose track of which group a
    row belongs to. ``subject`` defaults to the stem: two movies are separate
    subjects unless the configuration says they are the same animal, because
    the opposite default silently turns pseudo-replication into replication.
    """
    stamped = table.copy()
    for column, value in (
        ("subject", movie.subject or movie.stem),
        ("condition", condition),
        ("stem", movie.stem),
    ):
        if column in stamped.columns:
            stamped = stamped.drop(columns=[column])
        stamped.insert(0, column, value)
    return stamped


def _write_table(table: pd.DataFrame, path: Path, output: Output | None = None) -> dict:
    """Write one table and describe what was written.

    The manifest carries the grain and the origin as well as the fingerprint, so
    a run folder says what one row of each of its tables is without anyone
    having to open the package that made it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Nine significant figures. A mean of 16-bit pixel values printed to
    # seventeen digits records the arithmetic rather than the measurement, and
    # the extra digits were a quarter of a run's size. Nine rather than six
    # because a figure that re-derives something from the file - a z-score, a
    # circular difference, a surrogate band - subtracts nearly-equal numbers,
    # and six left those visibly rounded; nine holds every value in the pinned
    # movie to 5e-9 of itself, which no such derivation can amplify into
    # anything drawable. %g keeps the exponent, so a p-value of
    # 1.45517642e-15 survives intact. The arithmetic itself is unchanged -
    # this is the precision of the file, not of the calculation.
    table.to_csv(path, index=False, float_format="%.9g")
    record = {
        "path": path.name,
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "sha256": sha256_of(path),
    }
    if output is not None:
        record["grain"] = list(output.grain)
        record["origin"] = output.origin
        record["folder"] = path.parent.name
    return record


#: A run's copied tables live beside the measured ones rather than among them.
#: The only thing telling the two apart used to be a ``history_`` prefix applied
#: by convention; now the folder says it, and ``Output.origin`` chooses.
TRACKER_FOLDER = "tracker"
MEASURED_FOLDER = "tables"


def _folder_for(output: Output | None) -> str:
    return TRACKER_FOLDER if output is not None and output.origin == "tracker" else MEASURED_FOLDER


def _join_cell_frame(tables: dict[str, pd.DataFrame], context: MeasurementContext) -> pd.DataFrame:
    declared = declared_tables()
    keys = ["identity", "frame_index"]
    joined: pd.DataFrame | None = None
    for name in sorted(tables):
        if _fold_target(name, declared) != "cell_frame":
            continue
        table = tables[name]
        if table.empty:
            continue
        if joined is None:
            joined = table.copy()
            continue
        overlap = [c for c in table.columns if c in joined.columns and c not in keys]
        joined = joined.merge(table.drop(columns=overlap), on=keys, how="outer")

    if joined is None:
        joined = pd.DataFrame(columns=keys)

    frames = context.frame_table()
    # The time axis belongs to the context, not to any one module. A module
    # that carried its own copy of `hours` along gets it dropped here rather
    # than merged, or the two would collide and every column downstream would
    # be reading `hours_x`.
    shared = [c for c in (*frames.columns, "stem") if c != "frame_index" and c in joined.columns]
    joined = joined.drop(columns=shared)
    joined = joined.merge(frames, on="frame_index", how="left")
    joined.insert(0, "stem", context.stem)
    ordered = ["stem", "identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours"]
    rest = [c for c in joined.columns if c not in ordered]
    return joined[ordered + rest].sort_values(["identity", "frame_index"]).reset_index(drop=True)


def _fold_derived(cell_frame: pd.DataFrame, tables: dict[str, pd.DataFrame],
                  names: set[str]) -> pd.DataFrame:
    """Merge a derived module's cell-frame-grain columns into the joined table.

    A derived module reads ``cell_frame`` rather than the pixels, so by the time
    it produces anything the join is already finished and it cannot take part in
    it. Its columns still belong beside the ones they were computed from -
    ``regimes`` is four columns describing each cell-frame, not a table in its
    own right - so they are merged in here instead.

    The merge is a left join on a grain the declaration promises is unique, so
    no row is added and none is lost: a derived table covers the rows it could
    compute, and a row it could not stays exactly as it was, with a blank.
    """
    declared = declared_tables()
    keys = ["identity", "frame_index"]
    for name in sorted(names):
        if _fold_target(name, declared) != "cell_frame":
            continue
        table = tables.get(name)
        if table is None or table.empty or any(key not in table.columns for key in keys):
            continue
        extra = [c for c in table.columns if c not in cell_frame.columns]
        if not extra:
            continue
        cell_frame = cell_frame.merge(table[keys + extra], on=keys, how="left")
    return cell_frame


def _granted_side_columns(name: str, context: MeasurementContext,
                          supplied: set[str]) -> set[str]:
    """The side columns one derived module is allowed to read.

    Named per module in the configuration, under that module's own settings::

        "modules": {"coupling": {"side_columns": ["schedule_dose"]}}

    A column that no side table supplied stops the run rather than being
    ignored. Ignoring it would produce exactly the result the reader was trying
    to get - a run that finishes, with the computation they asked for quietly
    missing - and a typo in a column name is the most likely way to get here.
    """
    asked = context.module_params(name).get("side_columns")
    if not asked:
        return set()
    if not isinstance(asked, (list, tuple)):
        raise TypeError(
            f"{name}: side_columns must be a list of column names, not "
            f"{type(asked).__name__}"
        )
    granted = {str(column) for column in asked}
    unknown = sorted(granted - supplied)
    if unknown:
        offered = sorted(supplied)
        tables = sorted(context.side)
        raise ValueError(
            f"{name} is granted the side column(s) {unknown}, which no side "
            f"table supplied. This movie declared the side table(s) {tables}, "
            f"between them providing {offered}. A side column is written under "
            "its table's name, so a `dose` column in a table called `schedule` "
            "is `schedule_dose`."
        )
    return granted


def _side_columns_used(granted: set[str], produced: dict[str, pd.DataFrame]) -> set[str]:
    """Which of the granted side columns actually reached the module's output.

    A column can arrive two ways: as a column of a produced table, or as a
    *value* in one - ``coupling`` names the two series of a lag profile in
    ``metric_a`` and ``metric_b``, so the column name is data there. Both are
    checked, because the point of the record is to expose a grant nobody used,
    and half a check would report a false one every time.
    """
    if not granted:
        return set()
    used: set[str] = set()
    for table in produced.values():
        if table is None or table.empty:
            continue
        used |= granted & set(table.columns)
        for column in table.columns:
            if table[column].dtype == object:
                used |= granted & set(table[column].dropna().astype(str).unique())
    return used


def analyse_movie(
    config: AnalysisConfig,
    movie: MovieConfig,
    output_dir: Path,
    modules: list[str] | None = None,
) -> dict:
    started = time.perf_counter()
    context, provenance = load_movie(config, movie)

    requested = modules or config.enabled_modules or [m.name for m in list_modules()]
    measurement_names = [m.name for m in list_modules()]
    derived_names = [m.name for m in list_derived()]

    tables: dict[str, pd.DataFrame] = {}
    stacks: dict[str, np.ndarray] = {}
    module_records: list[dict] = []

    for name in requested:
        if name not in measurement_names:
            continue
        module = get_module(name)
        available, reason = module.available(context)
        if not available:
            module_records.append({"module": name, "status": "skipped", "reason": reason})
            continue
        module_started = time.perf_counter()
        produced = module.measure(context)
        table_outputs = {
            key: value for key, value in produced.items() if isinstance(value, pd.DataFrame)
        }
        stack_outputs = {
            key: np.asarray(value) for key, value in produced.items()
            if isinstance(value, np.ndarray)
        }
        unknown = set(produced) - set(table_outputs) - set(stack_outputs)
        if unknown:
            raise TypeError(
                f"{name} returned unsupported output(s) {sorted(unknown)}; "
                "modules may return pandas tables or numpy image stacks"
            )
        tables.update(table_outputs)
        stacks.update(stack_outputs)
        module_records.append(
            {
                "module": name,
                "status": "done",
                "seconds": round(time.perf_counter() - module_started, 3),
                "tables": {key: int(len(value)) for key, value in table_outputs.items()},
                "stacks": {key: list(value.shape) for key, value in stack_outputs.items()},
                "parameters": {**module.defaults, **context.module_params(name)},
            }
        )

    cell_frame = _join_cell_frame(tables, context)

    # Side tables before the derived modules, so a user's own series can be
    # something the package computes with rather than only something it shows.
    #
    # The order used to be the other way round, and for a good reason: a derived
    # module that scanned cell_frame for columns would pick up a user column
    # whose meaning is not ours and treat it as one of our own. That protection
    # is now explicit rather than positional - a derived module is handed a
    # table with every side column removed *except* the ones the configuration
    # named for it under `side_columns`, and the manifest records which were
    # offered and which were used. Scanning is what was unsafe, not proximity.
    #
    # Both keys reach cell_frame. Per-frame data belongs on every row plainly
    # enough; per-cell data repeats one fact down a cell's whole track, which
    # is convenient and slightly dishonest about being one fact. To stop that,
    # drop the "identity" line below - cell_summary still carries it.
    cell_frame = join_side(cell_frame, context, "frame_index")
    cell_frame = join_side(cell_frame, context, "identity")
    supplied = side_column_names(context) & set(cell_frame.columns)

    side_use: dict[str, dict] = {}
    derived_produced: set[str] = set()
    for name in requested:
        if name not in derived_names:
            continue
        module = get_derived(name)
        # Fold what the derived modules before this one wrote, so a derived
        # module can read a column an earlier one added: `sequence_distance`
        # compares the state numbers `regimes` writes. Folding only once after
        # the loop put those columns in the written cell_frame.csv and never in
        # the table the modules were handed, so the dependency was expressible
        # in `needs_columns` and impossible to satisfy.
        #
        # It makes the order of `enabled_modules` matter, which is why a module
        # whose column has not arrived yet is skipped with the column named
        # rather than failing: the reader is told to move it down the list.
        cell_frame = _fold_derived(cell_frame, tables, derived_produced)

        granted = _granted_side_columns(name, context, supplied)
        # The enforcement, and it is a filter rather than a convention: a module
        # cannot read what it was not handed. Nothing is copied when no side
        # column is hidden, which is every run that declares no side tables, so
        # this costs such a run neither time nor a moved number.
        hidden = sorted(supplied - granted)
        visible = cell_frame.drop(columns=hidden) if hidden else cell_frame

        # Checked against what the module will actually be given rather than
        # against the whole table, so the two can never disagree about whether a
        # column is there.
        missing = [c for c in module.needs_columns if c not in visible.columns]
        if missing:
            module_records.append(
                {"module": name, "status": "skipped", "reason": f"cell_frame is missing {missing}"}
            )
            continue

        module_started = time.perf_counter()
        produced = module.derive(visible, context)
        tables.update(produced)
        derived_produced.update(produced)
        side_use[name] = {
            "offered": sorted(granted),
            "used": sorted(_side_columns_used(granted, produced)),
        }
        module_records.append(
            {
                "module": name,
                "status": "done",
                "seconds": round(time.perf_counter() - module_started, 3),
                "tables": {key: int(len(value)) for key, value in produced.items()},
                "parameters": {**module.defaults, **context.module_params(name)},
            }
        )

    cell_frame = _fold_derived(cell_frame, tables, derived_produced)

    # The user's own columns keep their place at the end of the table. They are
    # joined early now so a module can compute with them, and a reader's eye
    # still finds the package's columns first and the spreadsheet's after -
    # moving *when* the join happens must not move where the columns land.
    if supplied:
        cell_frame = cell_frame[[c for c in cell_frame.columns if c not in supplied]
                                + [c for c in cell_frame.columns if c in supplied]]

    cell_summary = build_cell_summary(cell_frame, tables, context)
    frame_summary = build_frame_summary(cell_frame, context, tables)
    movie_summary = build_movie_summary(cell_frame, cell_summary, frame_summary, tables, context)

    assignment = config.assignment(movie)
    movie_summary["condition"] = assignment.condition
    movie_summary["condition_label"] = assignment.label
    movie_summary["subject"] = movie.subject or movie.stem

    def stamp(table: pd.DataFrame) -> pd.DataFrame:
        return _stamp(table, movie, assignment.condition)

    movie_dir = output_dir / movie.stem
    table_dir = movie_dir / MEASURED_FOLDER
    declared = declared_tables()
    written = {
        name: _write_table(stamp(table), table_dir / f"{name}.csv", _ROLLUP_BY_NAME[name])
        for name, table in (("cell_frame", cell_frame),
                            ("cell_summary", cell_summary),
                            ("frame_summary", frame_summary))
    }

    # The same roll-ups again, over a filter. Declaring no windows costs
    # nothing and writes nothing, so a run made without them is the run this
    # package always made.
    windows = config.windows_for(movie)
    for name, table in sorted(windowed_summaries(cell_frame, context, windows).items()):
        written[name] = _write_table(
            stamp(table), table_dir / f"{name}.csv", _WINDOWED_BY_NAME[name])
    for name, table in sorted(tables.items()):
        # A folded table is already inside a roll-up; writing it again would be
        # two files holding one set of numbers, which is two places for them to
        # disagree.
        if _fold_target(name, declared) is not None or table is None or table.empty:
            continue
        output = declared.get(name)
        written[name] = _write_table(
            stamp(table), movie_dir / _folder_for(output) / f"{name}.csv", output)

    written_stacks: dict[str, dict] = {}
    stack_dir = movie_dir / "stacks"
    for name, stack in sorted(stacks.items()):
        path = stack_dir / f"{name}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, stack, compression="zlib")
        written_stacks[name] = {
            "path": str(path.relative_to(movie_dir)).replace("\\", "/"),
            "shape": list(stack.shape),
            "dtype": str(stack.dtype),
            "sha256": sha256_of(path),
        }

    (movie_dir / "movie_summary.json").write_text(
        json.dumps(movie_summary, indent=2, default=str), encoding="utf-8"
    )

    return {
        "stem": movie.stem,
        "condition": assignment.as_dict(),
        "subject": movie.subject or movie.stem,
        "provenance": provenance,
        "modules": module_records,
        # Which of the user's own columns each derived module was offered and
        # which it actually reached for. A column offered and unused is a
        # misconfiguration - the computation the reader wanted did not happen -
        # and this is where it is visible rather than hidden.
        "side_column_use": side_use,
        # What each declared window actually caught, beside the tables it
        # produced. A window that asked for twenty hours and got four frames is
        # a mistyped window, and this is where that is visible without opening
        # a CSV.
        "windows": [
            {"name": w.name, "baseline": w.baseline, "description": w.description,
             "frames": window_extent(w, context)[0],
             "hours": window_extent(w, context)[1]}
            for w in windows
        ],
        "tables": written,
        "stacks": written_stacks,
        "summary": movie_summary,
        "seconds": round(time.perf_counter() - started, 3),
    }


def run(
    config: AnalysisConfig,
    output_dir: Path,
    stems: list[str] | None = None,
    modules: list[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    # Empty directory shells survive a delete on a Dropbox-backed folder for a
    # while, so the guard asks whether any *file* is present rather than
    # whether the folder exists.
    if output_dir.exists() and any(p.is_file() for p in output_dir.rglob("*")):
        raise FileExistsError(
            f"{output_dir} already holds results; analysis runs are immutable, "
            "choose a new run name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [m for m in config.movies if not stems or m.stem in stems]
    if not selected:
        raise ValueError(f"no configured movies match {stems}")

    # A movie in the wrong group is the one error no figure will reveal, so it
    # stops the run rather than being measured and sorted out later. `python -m
    # analysis conditions --config <file>` shows the same table without running.
    assignments = [config.assignment(movie) for movie in selected]
    broken = [f"{a.stem}: {a.problem}" for a in assignments if a.problem is not None]
    if broken:
        raise ValueError(
            "the experimental design cannot be resolved:\n  "
            + "\n  ".join(broken)
            + "\nrun `python -m analysis conditions --config <file>` to see the whole table"
        )

    started = _now()
    clock = time.perf_counter()
    results = [analyse_movie(config, movie, output_dir, modules) for movie in selected]

    # Imported here rather than at the top because `pool` reads this module's
    # layout - which folder a table lands in, and how one is written - so the
    # two would import each other. The call has to live in `run` and the
    # constants have to live beside the writer that uses them, and this is the
    # smaller of the two compromises.
    from analysis.pool import pool_run

    # Pooling reads the per-movie folders back off disk, so it runs after every
    # movie is written. Its failure is recorded rather than raised, on the same
    # reasoning as the theme block below: a run is twenty-five minutes of
    # measurement and a concatenation that went wrong must not take it with it.
    # It is recorded rather than swallowed, because a missing `pooled/` folder
    # with no stated reason is the silence this package does not permit.
    try:
        pooled = pool_run(output_dir)
    except Exception as error:
        pooled = {"error": str(error)}

    # The contrasts read the pooled tables, so they run last. Declaring none
    # writes no `statistics.csv` and leaves the run exactly as it was.
    tested: dict = {"contrasts": 0, "rows": 0, "refused": 0}
    if config.contrasts:
        from analysis.contrasts import STATISTICS_FILE, run_contrasts

        statistics = run_contrasts(output_dir, config)
        _write_table(statistics, output_dir / STATISTICS_FILE)
        tested = {
            "contrasts": len(config.contrasts),
            "rows": int(len(statistics)),
            "refused": int((statistics["note"].astype(str) != "").sum()),
            "path": STATISTICS_FILE,
        }

    # The look is recorded beside the numbers, so a figure rebuilt from this run
    # a year later is the same figure and not merely the same data.
    try:
        theme_stamp = load_theme(config.theme).with_conditions(
            config.conditions.colours() if config.conditions else {}
        ).stamp()
        # The block itself, beside the tables, so `figures/build_all.py <run>`
        # draws in the same theme months later without the configuration. The
        # condition colours ride along: a group's colour is part of the look.
        (output_dir / "theme.json").write_text(
            json.dumps({"theme": config.theme,
                        "conditions": config.conditions.as_dict()["conditions"]},
                       indent=2),
            encoding="utf-8",
        )
    except Exception as error:                       # a broken theme must not lose a run
        theme_stamp = {"error": str(error)}

    # The design, beside the numbers. Which movie was which group, and how that
    # was decided, is as much a part of the result as the measurements are.
    design = {
        **config.conditions.as_dict(),
        "assignments": [a.as_dict() for a in assignments],
        "subjects": {m.stem: (m.subject or m.stem) for m in selected},
        "warnings": config.conditions.audit(assignments),
    }
    (output_dir / "conditions.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )

    # Wording and options per figure, beside the tables for the same reason the
    # theme is: rebuilding a run's figures later must reproduce what was written
    # and what was drawn at the time rather than whatever the configuration says
    # now. The package supplies only descriptive defaults; anything interpretive
    # is the user's, and lives here.
    #
    # Copied verbatim, options sub-block included. Nothing here reads it - a
    # figure resolves its own settings against its own declaration, and a block
    # this run cannot check is still a block the next reader can see.
    (output_dir / "figures.json").write_text(
        json.dumps({"figures": config.figures}, indent=2), encoding="utf-8"
    )

    manifest = {
        "analysis_package_version": __version__,
        "generated_at_utc": started,
        "theme": theme_stamp,
        "design": design,
        "finished_at_utc": _now(),
        "elapsed_seconds": round(time.perf_counter() - clock, 3),
        "dataset": config.dataset,
        "config_path": str(config.source_path),
        "config_sha256": sha256_of(config.source_path) if config.source_path else None,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "registered_modules": [
            {"name": m.name, "kind": "measurement", "description": m.description,
             "requires": list(m.requires)}
            for m in list_modules()
        ]
        + [
            {"name": m.name, "kind": "derived", "description": m.description}
            for m in list_derived()
        ],
        "movies": results,
        "pooled": pooled,
        "statistics": tested,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest
