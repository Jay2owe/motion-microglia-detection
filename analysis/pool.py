"""Every movie's tables, concatenated once, so a comparison is a filter.

A run measures each movie on its own and writes each one its own folder. That is
right for measuring - a movie is the unit that gets loaded, checked and
fingerprinted - and wrong for reading, because the first thing anyone wants is
two movies side by side.

Nothing here computes. Every row is already stamped with the movie it came from,
the condition it belongs to and the subject it was taken from, so pooling is
stacking. What this module adds is the record of what was stacked: which movies
contributed to each table, how many rows each gave, which columns one movie had
that another did not, and which tables a movie never produced at all.

That record is the load-bearing part. A movie that declared no object set has no
object columns, and a pooled table that quietly fills them with NaN looks
exactly like a movie whose cells touched no objects. The manifest is what
separates the two, and anything that tests a column must read it first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.registry import declared_tables
from analysis.run import MEASURED_FOLDER, TRACKER_FOLDER, _write_table

#: Where a pooled run is written, beside the per-movie folders rather than
#: inside one of them. A movie folder is identified by holding a ``tables``
#: folder, so this name must never be one.
POOLED_FOLDER = "pooled"

#: The two folders a movie's tables are split across, and the split is kept
#: inside ``pooled/`` rather than flattened. ``tracker`` holds tables this
#: package copied from the tracking pass rather than measured, and the
#: ``origin`` contract that distinguishes them would be quietly broken if
#: pooling merged them into one heap at the last step.
SOURCE_FOLDERS = (MEASURED_FOLDER, TRACKER_FOLDER)

#: What ``run._stamp`` puts on the front of every written table. Pooling is only
#: a concatenation because these are already there; a table without them cannot
#: be pooled into anything a later stage can group by, so their absence stops
#: the pooling rather than producing a table nobody can use.
STAMP_COLUMNS = ("stem", "condition", "subject")


def movie_folders(run_dir: Path) -> list[Path]:
    """The per-movie folders in a run, in a fixed order.

    A movie folder is one holding a ``tables`` folder, which is what separates
    it from ``figures/`` without this module having to carry a list of names it
    must skip.

    ``pooled/`` is the exception and has to be named, because once it exists it
    holds a ``tables`` folder too and would otherwise be pooled into itself -
    every row twice, and the second pooling would look exactly like a run with
    one extra movie called `pooled`. The refusal in :func:`pool_run` catches the
    ordinary case; this catches the one where someone deleted the manifest and
    left the tables.
    """
    return [p for p in sorted(Path(run_dir).iterdir())
            if p.is_dir() and p.name != POOLED_FOLDER and (p / MEASURED_FOLDER).is_dir()]


def _ordered_union(columns_per_movie: list[list[str]]) -> list[str]:
    """The union of several column lists, in first-seen order.

    Order matters because a pooled file is read by people as well as by code,
    and a column order that depended on dictionary iteration would change
    between runs of the same data. Movies are visited in sorted order, so this
    is deterministic: the first movie's columns in its own order, then whatever
    later movies added, in theirs.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for columns in columns_per_movie:
        for column in columns:
            if column not in seen:
                seen.add(column)
                ordered.append(column)
    return ordered


def _read(path: Path) -> pd.DataFrame:
    """One movie's table, read back exactly as it was written.

    Nothing here converts dtypes. A column that was integral in one movie and
    absent in another becomes floating point when the two are stacked, because
    that is what a missing integer is.
    """
    return pd.read_csv(path)


def pool_run(run_dir: str | Path) -> dict:
    """Write ``<run>/pooled/`` from the per-movie folders already in ``run_dir``.

    Idempotent in the sense that matters: it reads only per-movie folders and
    writes only into ``pooled/``, so re-running it after adding a movie is
    correct. It still refuses to overwrite, in line with immutable runs - delete
    ``pooled/`` deliberately or pool into a new run.

    Returns the manifest fragment, which the caller also writes into the run
    manifest; the same fragment is written to ``pooled/manifest.json`` so a
    pooled folder can be read on its own.
    """
    run_dir = Path(run_dir)
    pooled_dir = run_dir / POOLED_FOLDER
    if pooled_dir.exists() and any(p.is_file() for p in pooled_dir.rglob("*")):
        raise FileExistsError(
            f"{pooled_dir} already holds pooled tables; analysis runs are "
            "immutable, delete it deliberately or pool into a new run"
        )

    movies = movie_folders(run_dir)
    if not movies:
        raise ValueError(
            f"{run_dir} holds no movie folders; a movie folder is one containing "
            f"a '{MEASURED_FOLDER}' folder"
        )
    stems = [movie.name for movie in movies]

    # Collected first, written second, so a table that turns out to be
    # unpoolable stops the whole thing before half a pooled folder exists.
    collected: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for movie in movies:
        for folder in SOURCE_FOLDERS:
            source = movie / folder
            if not source.is_dir():
                continue
            for path in sorted(source.glob("*.csv")):
                table = _read(path)
                missing = [c for c in STAMP_COLUMNS if c not in table.columns]
                if missing:
                    raise ValueError(
                        f"{path} is missing {missing}, so its rows cannot be told "
                        "apart from another movie's once pooled; every table a run "
                        "writes is stamped by run._stamp, so this file was not "
                        "written by this package"
                    )
                collected.setdefault((folder, path.stem), {})[movie.name] = table

    declared = declared_tables()
    record: dict[str, dict] = {}
    for (folder, name), by_movie in sorted(collected.items()):
        contributing = [stem for stem in stems if stem in by_movie]
        union = _ordered_union([list(by_movie[stem].columns) for stem in contributing])
        # Reindexing rather than letting concat align: the column order of a
        # pooled file is then stated here rather than being a property of how
        # pandas happened to merge them, and every movie's block has the same
        # shape, so a column one movie lacked is a blank and never a silently
        # dropped one.
        frames = [by_movie[stem].reindex(columns=union) for stem in contributing]
        pooled = pd.concat(frames, ignore_index=True, sort=False)

        written = _write_table(pooled, pooled_dir / folder / f"{name}.csv", declared.get(name))
        written["folder"] = folder
        written["movies"] = contributing
        written["rows_per_movie"] = {stem: int(len(by_movie[stem])) for stem in contributing}
        columns_missing = {
            stem: [c for c in union if c not in by_movie[stem].columns]
            for stem in contributing
        }
        written["columns_missing"] = {k: v for k, v in columns_missing.items() if v}
        # A table one movie never wrote at all is the same failure mode one step
        # up: pooling three movies' contacts when only two produced any looks
        # like a study in which the third movie's cells never touched.
        written["movies_absent"] = [stem for stem in stems if stem not in by_movie]
        record[name] = written

    fragment = {
        "movies": stems,
        "folders": list(SOURCE_FOLDERS),
        "tables": record,
    }
    pooled_dir.mkdir(parents=True, exist_ok=True)
    (pooled_dir / "manifest.json").write_text(
        json.dumps(fragment, indent=2), encoding="utf-8"
    )
    return fragment
