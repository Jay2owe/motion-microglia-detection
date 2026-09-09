"""Immutable records shared by scientific producers and dependent renderers.

No tables are fitted or files written here. A scientific result refers to its
complete input population and settings; a display selection refers back to that
result rather than redefining its statistical family.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral, Real
from typing import Any, Literal


def plain(value: Any) -> Any:
    """A detached JSON value, including nested contract records."""
    if isinstance(value, Settings):
        return value.as_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("record object keys must be strings")
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise ValueError(f"record values must be JSON values, not {type(value).__name__}")


def _json(value: Any) -> str:
    try:
        return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"record must contain finite JSON values: {error}") from error


def content_id(value: Any) -> str:
    """An identity for the complete record, independent of mapping key order."""
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class Settings(Mapping):
    """A detached immutable JSON object; reading a nested value returns a copy."""

    _json: str

    def __init__(self, values: Mapping | None = None):
        if values is not None and not isinstance(values, Mapping):
            raise ValueError("settings must be an object")
        object.__setattr__(self, "_json", _json({} if values is None else values))

    def as_dict(self) -> dict:
        return json.loads(self._json)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


class Record:
    def as_dict(self) -> dict:
        return plain(self)

    @property
    def record_id(self) -> str:
        return content_id(self)


def text_key(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def cell_number(value: Any) -> int:
    """Accept integer-valued CSV numbers, preserving cell identity exactly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"cell identity must be an integer, got {value!r}")
    if not isinstance(value, Integral):
        import math

        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"cell identity must be an integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, order=True)
class CellKey(Record):
    source_run: str
    movie: str
    identity: int

    def __post_init__(self):
        text_key(self.source_run, "source_run")
        text_key(self.movie, "movie")
        object.__setattr__(self, "identity", cell_number(self.identity))


@dataclass(frozen=True)
class CellMeasurementKey(Record):
    cell: CellKey
    measurement: str


@dataclass(frozen=True)
class SampleAssignment(Record):
    movie: str
    sample: str | None = None
    confirmed: bool = False
    observed_subject: str | None = None

    def __post_init__(self):
        text_key(self.movie, "movie")
        if self.confirmed:
            text_key(self.sample, "confirmed biological sample")


@dataclass(frozen=True)
class InputIdentity(Record):
    source_run: str
    table_hashes: Settings
    cells: tuple[CellKey, ...]
    samples: tuple[SampleAssignment, ...]


@dataclass(frozen=True)
class Measurement(Record):
    column: str
    table: str
    grain: tuple[str, ...]
    label: str
    unit: str = ""
    declared: bool = False
    summary: str | None = None


@dataclass(frozen=True)
class MeasurementPair(Record):
    """Detection is unordered; timing keeps the declared reference direction."""

    reference: str
    target: str

    @property
    def detection_key(self) -> tuple[str, str]:
        return tuple(sorted((self.reference, self.target)))


@dataclass(frozen=True)
class ArtifactRef(Record):
    """An input/output file and the scientific result it belongs to."""

    name: str
    path: str
    sha256: str
    scientific_id: str
    grain: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionRecord(Record):
    """Typed cell/measurement keys or explicit keys for another pipeline."""

    name: str
    scientific_id: str
    rule: Settings
    members: tuple[CellMeasurementKey | CellKey | MeasurementPair | Settings, ...] = ()


@dataclass(frozen=True)
class StepSpec(Record):
    """A finite recipe step; the runner later binds its producer by name."""

    name: str
    producer: str
    prerequisites: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    selection: str | None = None
    requires_selected_rows: bool = False
    kind: Literal["science", "render"] = "science"


@dataclass(frozen=True)
class PipelineRecipe(Record):
    name: str
    version: int
    steps: tuple[StepSpec, ...]


@dataclass(frozen=True)
class StepResult(Record):
    step: str
    scientific_id: str
    status: Literal["completed", "reused", "skipped-empty", "unavailable", "failed"]
    reason: str
    artifacts: tuple[ArtifactRef, ...] = ()
    selections: tuple[SelectionRecord, ...] = ()
    provenance: Settings = field(default_factory=Settings)

    def __post_init__(self):
        if self.status not in {"completed", "reused", "skipped-empty", "unavailable", "failed"}:
            raise ValueError(f"unknown pipeline step status {self.status!r}")
        text_key(self.reason, "step outcome reason")
