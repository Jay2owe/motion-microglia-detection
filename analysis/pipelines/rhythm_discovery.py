"""Free measurement choices and the saved-input contract for rhythm discovery.

Configuration parsing is structural and does not import the scientific engine.
``resolve_request`` checks actual table grain, numeric columns and input identity,
then resolves all scientific options through ``analysis.circadian``. Neither
operation fits a trace, summarises a comparison measurement or expands plots.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from .contracts import (CellKey, CellMeasurementKey, InputIdentity, Measurement,
                        MeasurementPair, PipelineRecipe, Record, SampleAssignment,
                        Settings, StepSpec, cell_number, text_key)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class MeasurementChoice(Record):
    column: str
    table: str | None = None
    summary: str | None = None


def _object(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    return value


def _known_keys(block: dict, keys: set[str], where: str) -> None:
    unknown = set(block) - keys
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {', '.join(sorted(unknown))}")


def _choices(values: object, groups: dict, where: str) -> tuple[MeasurementChoice, ...]:
    from analysis.metric_groups import resolve_metrics

    if not isinstance(values, list):
        raise ValueError(f"{where}: expected a list of measurements")
    found: dict[str, MeasurementChoice] = {}
    for entry in values:
        if isinstance(entry, str):
            text_key(entry, where)
            expanded = [MeasurementChoice(column) for column in
                        resolve_metrics([entry], groups, where=where)]
        elif isinstance(entry, dict):
            _known_keys(entry, {"column", "table", "summary"}, where)
            column = text_key(entry.get("column"), f"{where}.column")
            if column.startswith("@"):
                raise ValueError(f"{where}: put a group directly in the list; "
                                 "a table-qualified choice names one column")
            table = entry.get("table")
            summary = entry.get("summary")
            if table is not None:
                text_key(table, f"{where}.table")
            if summary is not None:
                text_key(summary, f"{where}.summary")
            expanded = [MeasurementChoice(column, table, summary)]
        else:
            raise ValueError(f"{where}: expected a column name, group or choice object")
        for choice in expanded:
            previous = found.get(choice.column)
            if previous is not None and previous != choice:
                raise ValueError(f"{where}: {choice.column!r} has conflicting table/summary choices")
            found.setdefault(choice.column, choice)
    return tuple(found.values())


def _pairs(block: object, measurements: tuple[str, ...]) -> tuple[MeasurementPair, ...]:
    if block is None:
        block = {"mode": "all"}
    block = _object(block, "pairs")
    _known_keys(block, {"mode", "pairs", "reference"}, "pairs")
    mode = block.get("mode", "all")
    if mode == "all":
        if set(block) - {"mode"}:
            raise ValueError("pairs: all mode accepts only mode")
        chosen = list(combinations(measurements, 2))
    elif mode == "reference":
        if "pairs" in block:
            raise ValueError("pairs: reference mode does not take explicit pairs")
        reference = block.get("reference")
        if reference not in measurements:
            raise ValueError("pairs.reference must name a test measurement")
        chosen = [(reference, name) for name in measurements if name != reference]
    elif mode == "explicit":
        if "reference" in block:
            raise ValueError("pairs: explicit mode takes direction from each pair")
        chosen = block.get("pairs")
        if not isinstance(chosen, list):
            raise ValueError("pairs.pairs must be a list of [reference, target] pairs")
    else:
        raise ValueError("pairs.mode must be all, explicit or reference")
    resolved: dict[tuple[str, str], MeasurementPair] = {}
    for pair in chosen:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each pair must be [reference, target]")
        if any(not isinstance(name, str) or name not in measurements for name in pair):
            raise ValueError("each pair must name two selected test measurements")
        reference, target = pair
        if reference == target:
            raise ValueError("self-pairs are not allowed")
        result = MeasurementPair(reference, target)
        # Reversed duplicates describe one detection pair. First direction wins
        # and is retained explicitly for later timing calculations.
        resolved.setdefault(result.detection_key, result)
    return tuple(resolved.values())


@dataclass(frozen=True)
class RhythmDiscoveryRequest(Record):
    name: str
    declaration: Settings
    test_measurements: tuple[MeasurementChoice, ...]
    comparison_measurements: tuple[MeasurementChoice, ...]
    pairs: tuple[MeasurementPair, ...]
    analysis_options: Settings
    biological_samples: Settings
    cells: tuple[tuple[str, int], ...] | None = None
    correction_scope: str = "all"
    pipeline: str = "rhythm-discovery"

    @classmethod
    def from_dict(cls, block: dict, groups: dict, *, where: str = "pipeline"):
        _known_keys(block, {"pipeline", "name", "test_measurements",
                           "comparison_measurements", "pairs", "analysis_options",
                           "biological_samples", "cells", "correction_scope"}, where)
        if block.get("pipeline") != "rhythm-discovery":
            raise ValueError(f"{where}: expected pipeline 'rhythm-discovery'")
        name = text_key(block.get("name", "rhythm-discovery"), f"{where}.name")
        if re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
            raise ValueError(f"{where}.name must be a lower-case identifier")
        tested = _choices(block.get("test_measurements"), groups,
                          f"{where}.test_measurements")
        if not tested:
            raise ValueError(f"{where}: test_measurements must not be empty")
        if any(choice.summary is not None for choice in tested):
            raise ValueError(f"{where}: test_measurements need time series, not summaries")
        compared = _choices(block.get("comparison_measurements", []), groups,
                            f"{where}.comparison_measurements")
        options = _object(block.get("analysis_options", {}), f"{where}.analysis_options")
        samples = _object(block.get("biological_samples", {}), f"{where}.biological_samples")
        for movie, sample in samples.items():
            text_key(movie, "biological_samples movie")
            text_key(sample, f"biological_samples.{movie}")
        cells = block.get("cells")
        selected = None
        if cells is not None:
            if not isinstance(cells, list):
                raise ValueError(f"{where}.cells must be a list of movie/identity objects")
            selected = []
            for entry in cells:
                entry = _object(entry, f"{where}.cells")
                _known_keys(entry, {"movie", "identity"}, f"{where}.cells")
                key = (text_key(entry.get("movie"), "cells.movie"),
                       cell_number(entry.get("identity")))
                if key not in selected:
                    selected.append(key)
        scope = block.get("correction_scope", "all")
        if scope not in ("all", "measurement", "movie_measurement"):
            raise ValueError("correction_scope must be all, measurement or movie_measurement")
        return cls(name, Settings(block), tested, compared,
                   _pairs(block.get("pairs"), tuple(c.column for c in tested)),
                   Settings(options), Settings(samples),
                   None if selected is None else tuple(selected), scope)


@dataclass(frozen=True)
class ResolvedRhythmRequest(Record):
    request: RhythmDiscoveryRequest
    inputs: InputIdentity
    test_measurements: tuple[Measurement, ...]
    comparison_measurements: tuple[Measurement, ...]
    analysis_options: Settings
    rhythm_params: Settings
    workbench_version: str
    period_methods: Settings

    @property
    def expected_pairs(self) -> tuple[CellMeasurementKey, ...]:
        """Complete screening population, including cells without usable traces."""
        return tuple(CellMeasurementKey(cell, metric.column)
                     for cell in self.inputs.cells for metric in self.test_measurements)


# These are declarations, not implementations of the producer or runner.
# Subsequent stages bind the producer and register additional dependent steps.
RECIPE = PipelineRecipe("rhythm-discovery", 1, (
    StepSpec("rhythm-screen", "rhythm-screen", inputs=("measured-tables",)),
))


def _catalogue(table_grains: Mapping | None):
    import analysis.modules  # noqa: F401 - existing declarations register here.
    from analysis.registry import declared_columns, declared_tables
    from analysis.summarise import ROLLUPS

    grains = {name: output.grain for name, output in declared_tables().items()}
    grains.update({output.name: output.grain for output in ROLLUPS})
    for name, grain in (table_grains or {}).items():
        if not isinstance(grain, (tuple, list)) or not grain:
            raise ValueError(f"table_grains.{name} must name a non-empty row key")
        if any(not isinstance(key, str) or not key for key in grain):
            raise ValueError(f"table_grains.{name} must contain column names")
        if name in grains and set(grains[name]) != set(grain):
            raise ValueError(f"table_grains.{name} conflicts with its existing declaration")
        grains[name] = tuple(grain)
    return declared_columns(), grains


def _shape_problem(frame, grain: tuple[str, ...] | None, *, summary: str | None,
                   testing: bool) -> str | None:
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

    if not isinstance(frame, pd.DataFrame):
        return "input is not a measured table"
    if not frame.columns.is_unique:
        return "duplicate column names"
    if grain is None:
        return "table grain is undeclared; provide its row-key metadata"
    cell_grain = set(grain) - {"stem"}
    trace = cell_grain in ({"identity", "frame_index"}, {"identity", "hours"})
    scalar = cell_grain == {"identity"}
    if testing and not trace:
        return "wrong grain: a test needs one row per cell and observation"
    if not testing and not (scalar or trace):
        return "wrong grain: a comparison needs a per-cell value or declared trace summary"
    if not testing and trace and summary is None:
        return "time-varying comparison requires an explicit summary operation"
    if scalar and summary is not None:
        return "per-cell values are already summaries; remove the summary operation"
    required = set(grain) | {"stem", "identity"} | ({"hours"} if trace else set())
    missing = required - set(frame.columns)
    if missing:
        return "missing identity/time columns: " + ", ".join(sorted(missing))
    keys = list(dict.fromkeys(("stem", *grain)))
    if frame[keys].isna().any().any():
        return "missing values in the declared row key"
    if frame.duplicated(keys).any():
        return "duplicate rows at the declared grain; select the intended table/slice"
    if trace and not frame.empty and (not is_numeric_dtype(frame["hours"].dtype)
                                     or is_bool_dtype(frame["hours"].dtype)
                                     or is_complex_dtype(frame["hours"].dtype)):
        return "observation hours must be real numeric coordinates"
    return None


def _resolve_measurement(choice, tables, declarations, grains, *, testing):
    from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

    if choice.table is not None and choice.table not in tables:
        raise ValueError(f"{choice.column!r}: table {choice.table!r} is unavailable")
    names = [choice.table] if choice.table is not None else list(tables)
    present = [name for name in names if choice.column in tables[name].columns]
    if not present:
        detail = "declared but unavailable in these inputs" if choice.column in declarations \
            else "unavailable in the actual input tables"
        raise ValueError(f"measurement {choice.column!r} is {detail}")
    eligible = []
    problems = []
    for name in present:
        frame = tables[name]
        problem = _shape_problem(frame, grains.get(name), summary=choice.summary, testing=testing)
        if problem is None:
            values = frame[choice.column]
            dtype = values.dtype
            # A header-only CSV has object columns. With no observed values,
            # retain its expected cells for the producer's untestable records.
            missing_only = values.isna().all()
            if ((not is_numeric_dtype(dtype) and not missing_only)
                    or is_bool_dtype(dtype) or is_complex_dtype(dtype)):
                problem = "measurement is not a real numeric column"
        if problem is None and choice.column in set(grains[name]) | {"stem", "hours"}:
            problem = "identity/time coordinates are not measured outcomes"
        if problem:
            problems.append(f"{name}: {problem}")
        else:
            eligible.append(name)
    if not eligible:
        raise ValueError(f"measurement {choice.column!r} is unsuitable: " + "; ".join(problems))
    if len(eligible) > 1:
        raise ValueError(f"measurement {choice.column!r} is ambiguous across "
                         f"{', '.join(eligible)}; select its table explicitly")
    table = eligible[0]
    column = declarations.get(choice.column)
    return Measurement(choice.column, table, tuple(grains[table]),
                       column.label if column else choice.column,
                       column.unit if column else "", column is not None, choice.summary)


def _population(source_run, tables, names, request):
    cells: dict[CellKey, None] = {}
    subjects: dict[str, set[str]] = {}
    for name in names:
        frame = tables[name]
        for movie, identity in frame[["stem", "identity"]].drop_duplicates().itertuples(index=False, name=None):
            cells.setdefault(CellKey(source_run, movie, identity), None)
        if "subject" in frame:
            for movie, subject in frame[["stem", "subject"]].dropna().itertuples(index=False, name=None):
                subjects.setdefault(movie, set()).add(str(subject))
    if request.cells is not None:
        selected = tuple(CellKey(source_run, movie, identity) for movie, identity in request.cells)
        missing = [cell for cell in selected if cell not in cells]
        if missing:
            raise ValueError(f"requested cell is unavailable: {missing[0].movie}/{missing[0].identity}")
    else:
        selected = tuple(sorted(cells))
    movies = sorted({cell.movie for cell in selected})
    known_movies = {cell.movie for cell in cells}
    unknown = set(request.biological_samples) - known_movies
    if unknown:
        raise ValueError("biological_samples names unavailable movies: " + ", ".join(sorted(unknown)))
    samples = []
    for movie in movies:
        observed = subjects.get(movie, set())
        if len(observed) > 1:
            raise ValueError(f"movie {movie!r} has conflicting subject labels in its input tables")
        sample = request.biological_samples.get(movie)
        samples.append(SampleAssignment(movie, sample, sample is not None,
                                        next(iter(observed), None)))
    return selected, tuple(samples)


def resolve_request(request: RhythmDiscoveryRequest, *, source_run: str,
                    tables: Mapping[str, "pd.DataFrame"], input_hashes: Mapping[str, str],
                    rhythm_params: Mapping | None = None,
                    table_grains: Mapping | None = None) -> ResolvedRhythmRequest:
    """Validate a request against measured tables without changing or fitting them.

    ``input_hashes`` contains the verified file/content SHA-256 for each used
    table, obtained by the caller loading the run. Pooled tables retain ``stem``.
    Use ``table_grains`` for additional tables with explicit row-key metadata;
    existing declared grains cannot be overridden. A summary names an operation
    to be implemented/validated by the comparison producer, never an implicit
    reduction performed here.
    """
    import pandas as pd
    from analysis import circadian
    from analysis.registry import get_derived

    text_key(source_run, "source_run")
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame)
           for name, frame in tables.items()):
        raise ValueError("tables must map table names to measured DataFrames")
    declarations, grains = _catalogue(table_grains)
    tests = tuple(_resolve_measurement(c, tables, declarations, grains, testing=True)
                  for c in request.test_measurements)
    comparisons = tuple(_resolve_measurement(c, tables, declarations, grains, testing=False)
                        for c in request.comparison_measurements)
    population_tables = sorted({m.table for m in tests})
    if "cell_summary" in tables:
        problem = _shape_problem(tables["cell_summary"], grains["cell_summary"],
                                 summary=None, testing=False)
        if problem:
            raise ValueError(f"cell_summary population: {problem}")
        population_tables = sorted(set(population_tables) | {"cell_summary"})
    cells, samples = _population(source_run, tables, population_tables, request)
    used = sorted(set(population_tables) | {m.table for m in comparisons})
    fingerprints = {}
    for name in used:
        fingerprint = input_hashes.get(name)
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
            raise ValueError(f"input_hashes.{name}: verified SHA-256 is required for result identity")
        fingerprints[name] = fingerprint.lower()

    options = request.analysis_options.as_dict()
    _known_keys(options, set(circadian.CIRCADIAN_ANALYSIS_OPTIONS), "analysis_options")
    inherited = {**get_derived("rhythms").defaults, **dict(rhythm_params or {})}
    # Resolve an unset option from the run, including an explicitly chosen test
    # even when the estimator itself happens to offer significance.
    resolved = circadian.resolve_analysis_options(
        inherited, lambda name: options.get(name, {} if name == "period_config" else None))
    applied = {name: resolved.get(name) for name in circadian.CIRCADIAN_ANALYSIS_OPTIONS}
    applied.update(fit_method=resolved["method"],
                   significance_method=resolved["significance_method"],
                   period_config=resolved["params"]["workbench_config"])
    return ResolvedRhythmRequest(
        request, InputIdentity(source_run, Settings(fingerprints), cells, samples),
        tests, comparisons, Settings(applied), Settings(resolved["params"]),
        circadian.WORKBENCH_VERSION, Settings(circadian.PERIOD_METHODS))
