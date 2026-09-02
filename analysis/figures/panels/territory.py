"""Panels for per-pixel occupancy history in image coordinates."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult


FATE_ROLES: dict[str, str] = {
    "core": "stable",
    "fringe": "motility",
    "transient": "surveillance",
    "vacated": "missing",
    "transferred": "highlight",
}

FATE_LABELS: dict[str, str] = {
    "core": "Core: high occupancy",
    "fringe": "Fringe: intermediate occupancy",
    "transient": "Transient: low occupancy",
    "vacated": "Vacant in this stage",
    "transferred": "Transferred to another cell",
}

__all__ = [
    "FATE_ROLES", "FATE_LABELS", "revisit_map", "coverage_curve", "ownership_map",
    "never_visited", "never_visited_mark", "fate_flow",
]


def _extent(field: dict) -> tuple[float, float, float, float]:
    return (0, float(field["width"]), float(field["height"]), 0)


def revisit_map(
    ax: Any,
    counts: Any,
    theme: Any,
    *,
    field: dict,
    cmap: Any = None,
    contour_at: float | None = None,
    soma: tuple[float, float] | None = None,
    look: common.Look | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> PanelResult:
    """Number of occupied frames per pixel, at fixed field magnification."""
    values = np.asarray(counts, dtype=float)
    chosen = cmap or (look.cmap if look is not None and look.is_map else theme["sequential_cmap"])
    handle = ax.imshow(values, extent=_extent(field), cmap=chosen, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
    if contour_at is not None:
        ax.contour(values, levels=[float(contour_at)], colors=[theme.colour("ink")],
                   linewidths=theme.stroke("guide"), origin="upper",
                   extent=_extent(field))
    if soma is not None:
        ax.scatter([soma[0]], [soma[1]], marker="+", color=theme.colour("highlight"),
                   s=theme.point_area(1.5), linewidth=theme.stroke("emphasis"))
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    rows, columns = np.indices(values.shape)
    return PanelResult(
        data=pd.DataFrame({"row": rows.ravel(), "column": columns.ravel(),
                           "frames": values.ravel()}),
        axes=ax, extra={"handle": handle},
    )


def coverage_curve(
    ax: Any,
    hours: Sequence[float],
    cumulative: Any,
    theme: Any,
    *,
    denominator: Any = None,
    shuffled: Any = None,
    look: common.Look | None = None,
    hour_ticks: float | None = None,
    y_label: str = "Share of final footprint touched",
) -> PanelResult:
    """Cumulative unique pixels divided by each cell's own union footprint."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(cumulative, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(hours) and values.shape[0] == len(hours):
        values = values.T
    if denominator is None:
        denominators = np.nanmax(values, axis=1)
    else:
        denominators = np.broadcast_to(np.asarray(denominator, dtype=float), (values.shape[0],))
    fractions = np.divide(values, denominators[:, None], out=np.full_like(values, np.nan),
                          where=denominators[:, None] > 0)
    colour = (look or common.Look(colour=theme.colour("surveillance"))).colour
    if shuffled is not None:
        null = np.asarray(shuffled, dtype=float)
        if null.ndim == 1:
            null = null[None, :]
        null_mean = np.nanmean(null, axis=0)
        ax.plot(hours, null_mean, color=theme.colour("reference"),
                linewidth=theme.stroke("guide"), label="Frames in random order")
    for curve in fractions:
        ax.plot(hours, curve, color=colour, alpha=0.18, linewidth=theme.stroke("guide"))
    ax.plot(hours, np.nanmedian(fractions, axis=0), color=colour,
            linewidth=theme.stroke("emphasis"), label="Median observed cell")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    return PanelResult(
        data=pd.DataFrame({
            "identity_row": np.repeat(np.arange(values.shape[0]), len(hours)),
            "hours": np.tile(hours, values.shape[0]),
            "cumulative": values.ravel(),
            "coverage_fraction": fractions.ravel(),
        }),
        axes=ax,
    )


def ownership_map(
    ax: Any,
    first_owner: Any,
    owner_count: Any,
    theme: Any,
    *,
    field: dict,
    contested_role: str = "highlight",
    never_role: str = "missing",
) -> PanelResult:
    """First identity to claim each pixel, highlighting later contest."""
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import to_rgba

    owners = np.asarray(first_owner, dtype=int)
    counts = np.asarray(owner_count, dtype=int)
    identities = sorted(int(value) for value in np.unique(owners) if value)
    roles = ("morphology", "reporter", "surveillance", "motility", "evidence")
    palette = [theme.colour(never_role)] + [theme.colour(roles[index % len(roles)])
                                             for index in range(len(identities))]
    code = np.zeros_like(owners, dtype=int)
    for index, identity in enumerate(identities, start=1):
        code[owners == identity] = index
    handle = ax.imshow(code, cmap=ListedColormap(palette), vmin=0, vmax=max(len(palette) - 1, 1),
                       extent=_extent(field), interpolation="nearest")
    overlay = np.zeros((*owners.shape, 4), dtype=float)
    overlay[counts > 1] = to_rgba(theme.colour(contested_role), 0.75)
    ax.imshow(overlay, extent=_extent(field), interpolation="nearest")
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    rows, columns = np.indices(owners.shape)
    return PanelResult(
        data=pd.DataFrame({"row": rows.ravel(), "column": columns.ravel(),
                           "first_owner": owners.ravel(),
                           "owners": counts.ravel()}),
        axes=ax, extra={"handle": handle, "identities": identities},
    )


def never_visited_mark(over_raw: bool) -> tuple[str, str]:
    """The colour role :func:`never_visited` inks, and what that ink means.

    One place, because the panel draws the mark and the figure writes the legend
    for it, and a legend that names the other side of the mask is worse than no
    legend at all.
    """
    if over_raw:
        return "missing", "Never occupied by a named cell or unclaimed foreground"
    return "visited", "Covered at some point by a named cell or unclaimed foreground"


def never_visited(
    ax: Any,
    mask: Any,
    theme: Any,
    *,
    field: dict,
    background: Any = None,
    look: common.Look | None = None,
) -> PanelResult:
    """Complement of every named or explicitly unclaimed footprint.

    Which side carries the ink depends on what is underneath, because the
    never-visited field is most of the frame. On a blank page the covered ground
    is inked and the never-visited ground is left as page, so the footprint the
    cells actually made is the mark rather than a solid panel with a few holes
    in it. Over a raw frame it is the other way round, in the pale blank colour:
    there the mark's job is to mask off the empty ground so the cells it was
    derived from still show through. :func:`never_visited_mark` says which of
    the two happened, so a legend can be written for the right one.
    """
    from matplotlib.colors import ListedColormap

    values = np.asarray(mask, dtype=bool)
    over_raw = background is not None
    if over_raw:
        ax.imshow(background, cmap=theme["image_cmap"], extent=_extent(field), interpolation="nearest")
    role, _ = never_visited_mark(over_raw)
    drawn = values if over_raw else ~values
    colour = (look or common.Look(colour=theme.colour(role))).colour
    ax.imshow(drawn.astype(int), cmap=ListedColormap(["none", colour]), vmin=0, vmax=1,
              extent=_extent(field), interpolation="nearest",
              alpha=0.85 if over_raw else 1.0)
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    return PanelResult(
        data=pd.DataFrame([{"pixels": int(values.size),
                            "never_visited_px": int(values.sum()),
                            "never_visited_share": float(values.mean())}]),
        axes=ax, extra={"share": float(values.mean())},
    )


def fate_flow(
    figure: Any,
    rect: tuple[float, float, float, float],
    stages: Any,
    theme: Any,
    *,
    stage_labels: Sequence[str],
) -> PanelResult:
    """Core, fringe, transient and vacated bands across time bins."""
    labels = list(FATE_ROLES)
    values = np.asarray(stages)
    if values.dtype.kind not in "iuf":
        mapping = {label: index for index, label in enumerate(labels)}
        values = np.vectorize(mapping.get)(values)
    return common.flow(
        figure, rect, values, theme, class_labels=labels,
        class_roles=[FATE_ROLES[label] for label in labels], stage_labels=stage_labels,
    )
