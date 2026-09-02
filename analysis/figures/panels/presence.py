"""Panels for what the ``presence`` module measures: whether a name is on screen.

This is the register being called. Everything else in the package measures the
cells that answered; presence records who was asked, so an empty seat can be
told apart from a person who was never on the list.

That distinction is the only reason these panels are not general charts. All
three are drawn in fixed colours - a name on screen, a cell inside its own
lifespan with no name, and time outside a cell's life at all - and those states
must mean the same thing on every panel, or the reader learns a colour on one
and misreads it on the next.

A fourth state, ``named_inferred``, splits the first one: the name is on screen,
but which cell owns those pixels was decided by the tracker's reconstruction
rather than by seeing that cell there. It is opt-in. A run with no provenance
draws exactly the three-state figure it always drew, and the three original
codes keep the integers they have always had, so no matrix already written to a
bundle changes meaning.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult

__all__ = ["STATE_CODES", "STATE_ROLES", "STATE_LABELS", "BASE_STATES",
           "PROVENANCE_STATES", "mechanism_colours", "on_screen",
           "unclaimed_area", "persistence_raster", "gap_mechanism_strip",
           "lifespan_bars"]

#: The states, in the order the raster's colour map expects them. The first
#: three integers are fixed for good: changing one would silently reinterpret
#: every matrix already written to a bundle.
STATE_CODES: dict[str, int] = {
    "outside_lifespan": 0, "named": 1, "unclaimed": 2, "named_inferred": 3,
}

STATE_ROLES: dict[str, str] = {
    "outside_lifespan": "missing", "named": "named", "unclaimed": "unclaimed",
    "named_inferred": "inferred",
}

STATE_LABELS: dict[str, str] = {
    "named": "name on screen",
    "named_inferred": "name on screen, ownership reconstructed",
    "unclaimed": "gap inside the cell's own lifespan",
    "outside_lifespan": "before first appearance / after last",
}

#: What a panel draws unless it is told otherwise - the three-state figure.
BASE_STATES: tuple[str, ...] = ("outside_lifespan", "named", "unclaimed")

#: The same, split by whether the outline was observed or reconstructed.
PROVENANCE_STATES: tuple[str, ...] = BASE_STATES + ("named_inferred",)


def mechanism_colours(mechanisms: Sequence[Any], theme: Any) -> dict[str, str]:
    """One colour per gap mechanism, decided in one place.

    Two panels draw the tracker's explanations - the strip beside the raster and
    the cuts in the lifespan bars - and a mechanism that is teal on one and
    orange on the other is worse than no colour at all. Sorted by name so the
    same movie gets the same colours on every rebuild.

    Sampled from the theme's sequential map rather than its role cycle, which
    holds five colours and refuses to repeat itself: there are six mechanisms on
    the reference movie and no promise that a future chain has fewer. The ramp
    carries no meaning - mechanisms have no order - it only has to keep them
    apart, which is also why it works in ``mono``. The ends are trimmed so
    nothing lands on the near-black or near-white that the page furniture uses.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    names = sorted({str(m) for m in mechanisms
                    if m is not None and str(m) not in ("", "nan")})
    if not names:
        return {}
    ramp = colormaps[theme["sequential_cmap"]]
    # Stop short of the dark end. These colours are drawn *inside* a bar in the
    # ``named`` colour, which is dark in every preset, and in ``mono`` the top
    # of the ramp is that same near-black.
    positions = ([0.45] if len(names) == 1
                 else np.linspace(0.12, 0.70, len(names)))
    return {name: to_hex(ramp(position)) for name, position in zip(names, positions)}


def on_screen(
    ax: Any,
    hours: Sequence[float],
    named: Sequence[float],
    expected: Sequence[float],
    theme: Any,
    *,
    hour_ticks: float | None = None,
    total: float | None = None,
    x_label: str = "Hours from start of recording",
    y_label: str = "Cell names\non screen",
    named_label: str = "name on screen",
    expected_label: str = "inside their own lifespan",
    gap_label: str = "exists, but not named this frame",
    show_x: bool = True,
) -> PanelResult:
    """How many names carry a mask each frame, against how many cells exist.

    The dashed line counts every cell between its own first and last appearance,
    so it rises as cells arrive and never counts one before it exists. The area
    between the two lines is the whole point of the panel: those are cells the
    movie shows and the tables do not name, and without the reference line they
    would be invisible - a solid line of 60 looks like a healthy recording
    whether 62 cells are present or 80.
    """
    hours = np.asarray(hours, dtype=float)
    named = np.asarray(named, dtype=float)
    expected = np.asarray(expected, dtype=float)

    ax.plot(
        hours, expected, linestyle=(0, (6, 4)), color=theme.colour("reference"),
        linewidth=theme.stroke("line"), label=expected_label,
    )
    ax.plot(
        hours, named, color=theme.colour("named"), linewidth=theme.stroke("emphasis"),
        label=named_label,
    )
    ax.fill_between(
        hours, named, expected, color=theme.colour("unclaimed"), alpha=0.28,
        linewidth=0, label=gap_label,
    )

    ax.set_ylabel(y_label)
    ceiling = max(float(expected.max()), float(total or 0.0))
    ax.set_ylim(0, ceiling * 1.12)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    if show_x:
        ax.set_xlabel(x_label)
    else:
        ax.set_xticklabels([])
    return PanelResult(
        data=pd.DataFrame({"hours": hours, "named": named, "expected": expected,
                           "unnamed": expected - named}),
        axes=ax,
    )


def unclaimed_area(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
    y_label: str = "Unclaimed\nforeground (px)",
    show_x: bool = True,
) -> PanelResult:
    """Foreground the pipeline agrees is cell signal but attributes to no name.

    Countable only because those pixels are written to their own stack instead
    of being pushed onto the nearest identity. A pipeline that assigned them
    would produce this same figure as a flat zero and be wrong.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)

    ax.fill_between(hours, 0, values, color=theme.colour("unclaimed"), alpha=0.35, linewidth=0)
    ax.plot(hours, values, color=theme.colour("unclaimed"), linewidth=theme.stroke("emphasis"))

    ax.set_ylabel(y_label)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_ylim(0, float(np.nanmax(values)) * 1.18 or 1.0)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    if show_x:
        ax.set_xlabel(x_label)
    else:
        ax.set_xticklabels([])
    return PanelResult(
        data=pd.DataFrame({"hours": hours, "unclaimed_px": values}), axes=ax)


def persistence_raster(
    ax: Any,
    matrix: Any,
    hours: Sequence[float],
    theme: Any,
    *,
    hour_ticks: float | None = None,
    legend: bool = True,
    states: Sequence[str] = BASE_STATES,
    x_label: str = "Hours from start of recording",
    y_label: str = "Cell, ordered by first appearance",
) -> PanelResult:
    """One row per cell, one column per frame, coloured by state.

    The matrix holds the codes in :data:`STATE_CODES`. Rows sorted by first
    appearance turn the arrival of cells into a clean diagonal edge, so a pale
    wedge at the bottom left reads as cells that had not arrived rather than as a
    run of failures.

    ``states`` says which states this matrix may contain, and must be the codes
    ``0..len(states) - 1``. Pass :data:`PROVENANCE_STATES` to split a name on
    screen into one attributed by observation and one attributed by the
    tracker's reconstruction; the default keeps the three-colour figure exactly
    as it was.

    What the panel still does not say is *why* a name is absent. A merged object
    carrying one number instead of two, an outline that fell below threshold and
    a cell that genuinely left all draw the same colour, which is what
    :func:`gap_mechanism_strip` is for.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    matrix = np.asarray(matrix, dtype=float)
    hours = np.asarray(hours, dtype=float)
    order = sorted(states, key=STATE_CODES.get)
    if [STATE_CODES[state] for state in order] != list(range(len(order))):
        raise ValueError(
            f"states must be a contiguous run of codes from 0, got {order}")

    drawn = common.raster(
        ax, matrix, theme, hours=hours,
        cmap=ListedColormap([theme.colour(STATE_ROLES[state]) for state in order]),
        vmin=0, vmax=len(order) - 1,
    )
    ax.set_xlabel(x_label)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_ylabel(y_label)
    ax.set_yticks([0, matrix.shape[0] - 1])
    ax.set_yticklabels(["1", str(matrix.shape[0])])

    if legend:
        # Reading order, not code order: the two "on screen" states sit next to
        # each other so the split between them is the first thing seen.
        shown = [state for state in
                 ("named", "named_inferred", "unclaimed", "outside_lifespan")
                 if state in order]
        ax.legend(
            handles=[
                Patch(facecolor=theme.colour(STATE_ROLES[state]), label=STATE_LABELS[state])
                for state in shown
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.115),
            ncol=3 if len(shown) <= 3 else 2,
            frameon=False, fontsize=theme.size("caption"),
            handlelength=1.6, columnspacing=1.6,
        )
    named = {code: state for state, code in STATE_CODES.items()}
    table = drawn.data.rename(columns={"value": "state_code"})
    table["state"] = table["state_code"].map(named)
    return PanelResult(data=table, axes=ax,
                       extra={"handle": drawn.extra["handle"]})


def gap_mechanism_strip(
    ax: Any,
    mechanisms: Sequence[str | None],
    theme: Any,
    *,
    legend: bool = True,
    x_label: str = "Why",
) -> PanelResult:
    """A one-column strip beside a raster: the tracker's reason for each row's gaps.

    One entry per raster row, in the same order, naming the mechanism that
    accounts for most of that cell's gap frames - or ``None`` for a cell with no
    explained gaps, which is left blank. Deliberately a strip rather than more
    colours on the raster: the raster answers "when", this answers "why", and
    merging them would make one colour carry two questions.

    One row per raster row, naming the mechanism it was coloured by; the names
    in the order they were coloured are in ``extra``.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    palette = mechanism_colours(mechanisms, theme)
    rows = pd.DataFrame({"row": range(len(list(mechanisms))),
                         "mechanism": list(mechanisms)})
    if not palette:
        ax.set_axis_off()
        return PanelResult(data=rows, axes=ax, extra={"mechanisms": []})
    present = list(palette)
    colours = [palette[name] for name in present]
    index = {name: position for position, name in enumerate(present)}
    column = np.array([[index[m] if m in index else np.nan] for m in mechanisms],
                      dtype=float)

    ax.imshow(column, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(colours), vmin=0, vmax=max(len(present) - 1, 1),
              extent=(0, 1, len(mechanisms) - 0.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(x_label, fontsize=theme.size("caption"))
    for spine in ax.spines.values():
        spine.set_visible(False)

    if legend:
        ax.legend(
            handles=[Patch(facecolor=colour, label=name.replace("_", " "))
                     for name, colour in zip(present, colours)],
            loc="upper left", bbox_to_anchor=(2.2, 1.0), ncol=1,
            frameon=False, fontsize=theme.size("caption"),
            handlelength=1.6, title="Most of this cell's gaps",
            title_fontsize=theme.size("caption"),
        )
    return PanelResult(data=rows, axes=ax, extra={"mechanisms": present})


def lifespan_bars(
    ax: Any,
    spans: Any,
    theme: Any,
    *,
    gaps: Any = None,
    order: str = "first_appearance",
    hour_ticks: float | None = None,
    y_label: str = "Cell",
    legend: bool = True,
    mark: Any = None,
) -> PanelResult:
    """One bar per identity from first to last appearance, with gaps cut out.

    A ``mechanism`` column on ``gaps`` colours each cut by the tracker's own
    explanation instead of one flat orange, using :func:`mechanism_colours` so
    a mechanism keeps its colour across panels. A gap with no explanation stays
    the plain unclaimed colour rather than being given a colour that implies
    one.

    ``mark`` is the identities whose last frame needs flagging - typically the
    ones that ended silently, away from the edge of the field, which is the
    difference between a cell leaving and a cell the tracker lost. They get a
    cross at the end of the bar; the caller writes the legend entry, because
    only the caller knows what it marked.
    """
    from matplotlib.patches import Patch
    frame = pd.DataFrame(spans).copy()
    required = {"identity", "first_hour", "last_hour"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"lifespan_bars needs {', '.join(sorted(missing))}")
    if order == "first_appearance":
        frame = frame.sort_values(["first_hour", "identity"])
    elif order == "span":
        frame = frame.assign(_span=frame["last_hour"] - frame["first_hour"]).sort_values(
            ["_span", "identity"], ascending=[False, True]
        )
    elif order == "coverage":
        frame = frame.sort_values(["coverage", "identity"], ascending=[False, True])
    else:
        raise ValueError("order must be first_appearance, span, or coverage")

    gap_frame = pd.DataFrame(gaps).copy() if gaps is not None else pd.DataFrame()
    marked = {int(identity) for identity in (mark or [])}
    gap_hours: dict[int, dict[float, Any]] = {}
    palette: dict[str, str] = {}
    if not gap_frame.empty and {"identity", "hours", "named"}.issubset(gap_frame):
        has_mechanism = "mechanism" in gap_frame.columns
        if has_mechanism:
            palette = mechanism_colours(gap_frame["mechanism"], theme)
        for identity, group in gap_frame[~gap_frame["named"].astype(bool)].groupby("identity"):
            gap_hours[int(identity)] = {
                float(row["hours"]): (str(row["mechanism"]) if has_mechanism else None)
                for _, row in group.iterrows()
            }

    step = float(np.nanmedian(np.diff(np.sort(gap_frame["hours"].unique())))) if (
        not gap_frame.empty and "hours" in gap_frame and gap_frame["hours"].nunique() > 1
    ) else 1.0
    rows = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        identity = int(row["identity"])
        first, last = float(row["first_hour"]), float(row["last_hour"])
        ax.barh(row_index, last - first + step, left=first, height=0.68,
                color=theme.colour("named"), edgecolor="none")
        cuts = gap_hours.get(identity, {})
        for hour in sorted(cuts):
            if first <= hour <= last:
                # No edge. A bar here can be two pixels tall with eighty cells
                # on the axes, and any outline eats the bar it is outlining.
                # Contrast with the bar comes from the ramp being trimmed to
                # stay clear of the ``named`` colour instead.
                ax.barh(row_index, step, left=hour, height=0.68,
                        color=palette.get(cuts[hour], theme.colour("unclaimed")),
                        edgecolor="none")
        if identity in marked:
            ax.plot([last + step], [row_index], marker="x", linestyle="none",
                    color=theme.colour("invalid"),
                    markersize=theme.size("caption") * 0.42,
                    markeredgewidth=theme.stroke("line") * 0.7, clip_on=False)
        rows.append({"identity": identity, "row_order": row_index,
                     "first_hour": first, "last_hour": last,
                     "marked": identity in marked})
    if rows:
        ax.set_xlim(min(row["first_hour"] for row in rows), max(row["last_hour"] for row in rows) + step)
        ticks = theme.hour_ticks(ax.get_xlim()[0], ax.get_xlim()[1], hour_ticks)
        ax.set_xticks(ticks)
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    ax.set_yticks([0, max(len(rows) - 1, 0)])
    ax.set_yticklabels(["1", str(len(rows))] if rows else [])
    ax.invert_yaxis()
    if legend:
        ax.legend(
            handles=[
                Patch(facecolor=theme.colour("named"), label="Cell identity present"),
                Patch(facecolor=theme.colour("unclaimed"), label="Unnamed gap within lifespan"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
            frameon=False, fontsize=theme.size("caption"),
        )
    return PanelResult(data=pd.DataFrame(rows), axes=ax,
                       extra={"mechanism_colours": palette})
