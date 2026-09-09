"""Reusable row ordering for matrices and cell-by-time rasters.

Ordering is a display operation: these functions never shift a trace, refit a
model, change a significance verdict, or replace the values drawn in a matrix.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TRACE_ORDER_ALIASES = {
    "principal_component": "principal_component_gradient",
    "pattern": "principal_component_gradient",
    "pca": "principal_component_gradient",
    "spectral": "spectral_continuum",
}

TRACE_ORDER_LABELS = {
    "principal_component": "principal-component gradient",
    "pattern": "principal-component gradient",
    "pca": "principal-component gradient",
    "spectral": "spectral continuum",
    "onset": "displayed onset",
    "period": "estimated period",
}


def resolve_order(
    tokens: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    append: Sequence[str] = (),
    identity_column: str | None = "identity",
) -> list[str]:
    """Resolve named sort keys and add stable fallback columns."""
    aliases = dict(aliases or {})
    resolved = []
    for token in tokens:
        descending = str(token).startswith("-")
        name = str(token)[1:] if descending else str(token)
        column = aliases.get(name, name)
        resolved.append(("-" if descending else "") + column)
    present = {value.removeprefix("-") for value in resolved}
    for column in append:
        if column not in present:
            resolved.append(column)
            present.add(column)
    if identity_column is not None and identity_column not in present:
        resolved.append(identity_column)
    return resolved


def order_label(
    tokens: Sequence[str], labels: Mapping[str, str] | None = None,
) -> str:
    """Readable description of a requested sequence of sort keys."""
    labels = dict(labels or {})
    parts = []
    for token in tokens:
        descending = str(token).startswith("-")
        name = str(token)[1:] if descending else str(token)
        label = labels.get(name, name.replace("_", " "))
        parts.append(f"{label} descending" if descending else label)
    return ", then ".join(parts)


def ordered_identities(
    summary: pd.DataFrame,
    order_by: Sequence[str],
    *,
    identity_column: str = "identity",
) -> list[Any]:
    """Return identity values after a stable multi-column row sort."""
    if identity_column not in summary.columns:
        raise KeyError(f"ordering table has no {identity_column!r} column")
    keys = list(order_by) or [identity_column]
    columns: list[str] = []
    ascending: list[bool] = []
    for key in keys:
        text = str(key).strip()
        descending = text.startswith("-")
        column = text[1:] if descending else text
        if not column:
            raise ValueError("an ordering column cannot be empty")
        if column not in summary.columns:
            available = ", ".join(map(str, summary.columns))
            raise KeyError(
                f"ordering column {column!r} is absent; available columns: {available}"
            )
        columns.append(column)
        ascending.append(not descending)
    ordered = summary.sort_values(
        columns, ascending=ascending, na_position="last", kind="mergesort",
    )
    return ordered[identity_column].tolist()


def smoothed_ordering_copy(
    matrix: pd.DataFrame, smooth_points: int = 5,
) -> pd.DataFrame:
    """Centred smoothing used only to derive a row order."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix ordering needs a pandas DataFrame")
    smooth_points = max(1, int(smooth_points))
    interpolated = matrix.interpolate(axis=1, limit_area="inside")
    smoothed = interpolated.T.rolling(
        smooth_points, center=True, min_periods=1,
    ).mean().T
    values = matrix.to_numpy(float)
    within_span = np.zeros(values.shape, dtype=bool)
    for row, source in enumerate(values):
        finite = np.flatnonzero(np.isfinite(source))
        if len(finite):
            within_span[row, finite[0]:finite[-1] + 1] = True
    return smoothed.where(within_span)


def trace_comparable(matrix: pd.DataFrame, minimum_points: int = 3) -> pd.Series:
    """Whether each matrix row has enough displayed values to compare."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix ordering needs a pandas DataFrame")
    usable = np.isfinite(matrix.to_numpy(float)).sum(axis=1) >= int(minimum_points)
    return pd.Series(usable, index=matrix.index, name="trace_comparable")


def _filled_ordering_values(matrix: pd.DataFrame) -> np.ndarray:
    interpolated = matrix.interpolate(axis=1, limit_area="inside")
    return interpolated.fillna(0.0).to_numpy(float)


def _correlation_distances(values: np.ndarray) -> np.ndarray:
    rows = len(values)
    distances = np.zeros((rows, rows), dtype=float)
    for left in range(rows):
        for right in range(left + 1, rows):
            overlap = np.isfinite(values[left]) & np.isfinite(values[right])
            if overlap.sum() < 3:
                distance = 1.0
            else:
                x = values[left, overlap]
                y = values[right, overlap]
                x = x - x.mean()
                y = y - y.mean()
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                if denominator <= np.finfo(float).eps:
                    distance = 0.0 if np.allclose(x, y) else 1.0
                else:
                    correlation = float(np.dot(x, y) / denominator)
                    distance = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
            distances[left, right] = distances[right, left] = distance
    return distances


def _principal_component_order(matrix: pd.DataFrame) -> list[Any]:
    comparable = trace_comparable(matrix)
    usable = matrix.loc[comparable]
    unavailable = matrix.index[~comparable].tolist()
    if len(usable) < 2:
        return [*usable.index.tolist(), *unavailable]

    filled = _filled_ordering_values(usable)
    centred = filled - filled.mean(axis=0, keepdims=True)
    left_vectors, singular_values, loadings = np.linalg.svd(
        centred, full_matrices=False,
    )
    coordinate = left_vectors[:, 0] * singular_values[0]
    time_direction = np.linspace(-1.0, 1.0, loadings.shape[1])
    if float(np.dot(loadings[0], time_direction)) < 0:
        coordinate = -coordinate
    positions = np.argsort(coordinate, kind="mergesort")
    return [*usable.index[positions].tolist(), *unavailable]


def _positive_centres(matrix: pd.DataFrame) -> np.ndarray:
    positions = matrix.columns.to_numpy(float)
    centres = []
    for values in matrix.to_numpy(float):
        weights = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
        centres.append(
            float(np.dot(weights, positions) / weights.sum())
            if weights.sum() > np.finfo(float).eps else np.nan
        )
    return np.asarray(centres, dtype=float)


def _spectral_continuum_order(matrix: pd.DataFrame) -> list[Any]:
    comparable = trace_comparable(matrix)
    usable = matrix.loc[comparable]
    unavailable = matrix.index[~comparable].tolist()
    if len(usable) < 3:
        return [*usable.index.tolist(), *unavailable]

    distances = _correlation_distances(usable.to_numpy(float))
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-np.square(distances) / (2.0 * scale * scale))
    np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(axis=1)
    inverse = np.diag(1.0 / np.sqrt(np.maximum(degree, np.finfo(float).eps)))
    laplacian = np.eye(len(usable)) - inverse @ similarity @ inverse
    _, vectors = np.linalg.eigh(laplacian)
    coordinate = vectors[:, 1]

    positive_centres = _positive_centres(usable)
    finite = np.isfinite(positive_centres)
    if finite.sum() >= 2:
        correlation = np.corrcoef(
            coordinate[finite], positive_centres[finite],
        )[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            coordinate = -coordinate
    positions = np.argsort(coordinate, kind="mergesort")
    return [*usable.index[positions].tolist(), *unavailable]


def displayed_onsets(
    matrix: pd.DataFrame,
    *,
    smooth_points: int = 5,
    sustain_points: int = 3,
) -> pd.Series:
    """First sustained negative-to-positive transition on the displayed time axis."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("trace onset ordering needs a pandas DataFrame")
    sustain_points = max(1, int(sustain_points))
    smoothed = smoothed_ordering_copy(matrix, smooth_points)
    positions = smoothed.columns.to_numpy(float)
    onsets = []
    for values in smoothed.to_numpy(float):
        finite = np.flatnonzero(np.isfinite(values))
        onset = np.nan
        if len(finite):
            first = int(finite[0])
            initial = values[first:first + sustain_points]
            if (
                len(initial) == sustain_points
                and np.all(np.isfinite(initial))
                and np.all(initial > 0)
            ):
                onset = float(positions[first])
            else:
                for position in finite[1:]:
                    previous = position - 1
                    run = values[position:position + sustain_points]
                    if (
                        np.isfinite(values[previous])
                        and values[previous] <= 0 < values[position]
                        and len(run) == sustain_points
                        and np.all(np.isfinite(run))
                        and np.all(run > 0)
                    ):
                        onset = float(positions[position])
                        break
        onsets.append(onset)
    return pd.Series(onsets, index=matrix.index, name="displayed_onset_hours")


def trace_pattern_order(
    matrix: pd.DataFrame,
    *,
    method: str = "principal_component_gradient",
    smooth_points: int = 5,
) -> list[Any]:
    """Order whole row patterns without shifting the matrix columns."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("trace pattern ordering needs a pandas DataFrame")
    method = TRACE_ORDER_ALIASES.get(str(method), str(method))
    smoothed = smoothed_ordering_copy(matrix, smooth_points)
    if method == "principal_component_gradient":
        return _principal_component_order(smoothed)
    if method == "spectral_continuum":
        return _spectral_continuum_order(smoothed)
    raise ValueError(
        "trace pattern method must be principal_component_gradient or "
        "spectral_continuum"
    )
