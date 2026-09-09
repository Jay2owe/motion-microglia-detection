"""Arithmetic a figure does on its way to drawing.

Not measurements: nothing here reads a pixel, and nothing here is written to a
table by ``python -m analysis run``. These are the steps between a measured
column and a drawable array - aligning cells on their own events, fitting one
cycle, naming a numbered cluster by its strongest trait - that more than one
page needs and that no panel should have to do, because a panel is given the
numbers and draws.

The split against ``panels/`` is the same one the package already makes
elsewhere: a panel takes an axes and makes marks; this takes a table and
returns a table. A function that did both would be a function neither could
reuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from _options import commas  # noqa: E402

__all__ = ["_aligned_curves", "_regime_labels", "_regime_rows",
           "_require_columns", "_scaled_field", "_selected_identities",
           "_transition_curves"]


def _require_columns(frame: pd.DataFrame, columns, table: str, module: str) -> None:
    """Refuse readably when a roll-up is missing a module's folded columns.

    A module that declares ``fold=True`` writes its columns into a roll-up
    rather than into a file of its own, so a figure asks ``cell_frame`` for
    ``reach_p95`` rather than opening ``sholl_reach.csv``. Two things make the
    column absent: the module was switched off for this run, or the run predates
    the fold and still has the separate file. ``require_table`` says the first
    for a missing file; this says it for a missing column, rather than leaving
    pandas to raise a ``KeyError`` that sends the reader looking for a typo.
    """
    missing = [c for c in columns if c not in frame.columns]
    if not missing:
        return
    raise SystemExit(
        f"{table} has no {', '.join(missing)}: the {module!r} module did not run "
        f"for this movie, or this run was written before that module's columns "
        f"were folded into {table} and still has them in a file of their own. "
        f"Add {module!r} to enabled_modules and re-run."
    )


#: The four columns the ``regimes`` module contributes to ``cell_frame``, and
#: which of them are cluster numbers rather than distances. A number naming a
#: state comes back from cell_frame as a float, because the rows without one
#: are blank; it has to go back to being an integer before anything indexes a
#: colour or a label with it.
_REGIME_COLUMNS = (
    "regime", "regime_label", "regime_size_level", "regime_movement_level",
    "regime_size_percentile", "regime_movement_percentile",
    "regime_distance", "regime_second", "regime_margin",
)
_REGIME_INTEGERS = ("regime", "regime_second")


def _regime_rows(cell_frame: pd.DataFrame) -> pd.DataFrame:
    """The cell-frames that were given a regime, and only those.

    ``regimes`` is a description of each cell-frame rather than a table in its
    own right, so its four columns live in ``cell_frame`` beside the
    measurements they were computed from. It does not describe every row: a
    cell seen in too few frames, or missing one of the features the fit uses,
    is left blank rather than guessed at. Those blanks were never in this
    figure - they were simply absent from the old ``regimes.csv`` - so they are
    dropped here, in one place, rather than in each of the four figures that
    read them.
    """
    _require_columns(cell_frame, ["regime"], "cell_frame.csv", "regimes")
    rows = cell_frame.dropna(subset=["regime"]).copy()
    for column in _REGIME_INTEGERS:
        if column in rows.columns:
            rows[column] = rows[column].astype(int)
    keys = [c for c in ("stem", "condition", "subject", "identity", "frame_index", "hours")
            if c in rows.columns]
    return rows[keys + [c for c in _REGIME_COLUMNS if c in rows.columns]].reset_index(drop=True)


_REGIME_TRAITS: dict[str, tuple[str, str]] = {
    "area_px": ("large", "small"),
    "circularity": ("round", "irregular"),
    "solidity": ("solid", "indented"),
    "ramification_index": ("ramified", "unramified"),
    "aspect_ratio": ("elongated", "compact"),
    "skeleton_branches": ("highly branched", "sparsely branched"),
    "turnover_index": ("high replacement", "stable footprint"),
    "step_px_gapless": ("mobile", "stationary"),
    "punctateness": ("punctate", "diffuse"),
}


def _regime_labels(profiles: pd.DataFrame) -> dict[int, str]:
    """Name each numbered subgroup from its declared levels when available."""
    if "regime_label" in profiles:
        return dict(zip(profiles["regime"].astype(int), profiles["regime_label"].astype(str)))
    features = [column for column in _REGIME_TRAITS if column in profiles]
    if not features:
        return {int(value): f"State {int(value)}" for value in profiles["regime"]}
    values = profiles[features].astype(float)
    spread = values.std(axis=0, ddof=0).replace(0, 1)
    standardised = (values - values.mean(axis=0)) / spread
    labels: dict[int, str] = {}
    for row_index, row in standardised.iterrows():
        feature = row.abs().idxmax()
        high, low = _REGIME_TRAITS[feature]
        trait = high if row[feature] >= 0 else low
        regime = int(profiles.loc[row_index, "regime"])
        labels[regime] = f"State {regime}: {trait}"
    return labels


def _aligned_curves(aligned: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    offsets = np.sort(aligned["offset_frames"].unique()) if not aligned.empty else np.array([])
    matrices = {}
    rows = []
    for metric in metrics:
        if aligned.empty:
            matrices[metric] = np.empty((0, 0), dtype=float)
            continue
        # One cell may now contribute several independently ranked events. The
        # event rank is part of the observational unit; indexing only by cell
        # would average those events together before the plotted mean is made.
        units = ["identity"]
        if "event_rank" in aligned:
            units.append("event_rank")
        pivot = aligned.pivot_table(
            index=units, columns="offset_frames", values=metric
        ).reindex(columns=offsets)
        values = pivot.to_numpy(float)
        matrices[metric] = values
        if values.size:
            mean = np.nanmean(values, axis=0)
            lo, hi = np.nanpercentile(values, [25, 75], axis=0)
            rows.extend({"metric": metric, "offset_frames": int(offset), "mean_z": m,
                         "lo": l, "hi": h,
                         "events": int(np.sum(np.isfinite(values[:, index])))}
                        for index, (offset, m, l, h) in enumerate(zip(offsets, mean, lo, hi)))
    return pd.DataFrame(rows), matrices, offsets


def _transition_curves(joined: pd.DataFrame, metric: str, window: int,
                       random: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-window, window + 1)
    curves = []
    generator = np.random.default_rng(seed)
    for identity, group in joined.groupby("identity", sort=True):
        group = group.sort_values("frame_index").reset_index(drop=True)
        changes = np.flatnonzero(group["regime"].to_numpy()[1:] != group["regime"].to_numpy()[:-1]) + 1
        if random and len(changes):
            valid = np.arange(window, max(window, len(group) - window))
            changes = generator.choice(valid, size=len(changes), replace=len(valid) < len(changes)) if len(valid) else []
        for centre in changes:
            if centre - window < 0 or centre + window >= len(group):
                continue
            values = group.loc[centre - window:centre + window, metric].to_numpy(float)
            if len(values) == len(offsets):
                spread = float(np.nanstd(values)) or 1.0
                curves.append((values - np.nanmean(values)) / spread)
    return offsets, np.asarray(curves, dtype=float)


def _selected_identities(frame: pd.DataFrame, requested: str | int) -> list[int]:
    """Resolve the shared --cells convention against observed support.

    ``requested`` is the figure's already-resolved ``cells`` option: a count of
    the longest-observed identities, or an explicit comma-separated list. It is
    passed in rather than read here so that the figure that honours the flag is
    the figure that declares it.
    """
    if "observed_frames" in frame:
        support = frame.groupby("identity")["observed_frames"].max().sort_values(ascending=False)
    else:
        support = frame.groupby("identity").size().sort_values(ascending=False)
    requested = str(requested)
    try:
        return [int(value) for value in support.index[:max(1, int(requested))]]
    except ValueError:
        wanted = [int(piece) for piece in commas(requested)]
        missing = sorted(set(wanted) - set(int(value) for value in support.index))
        if missing:
            raise SystemExit(f"--cells includes identities not in this table: {missing}")
        return wanted


def _scaled_field(ctx) -> dict[str, float]:
    return {"width": ctx.scale.length(float(ctx.field["width"])),
            "height": ctx.scale.length(float(ctx.field["height"]))}


