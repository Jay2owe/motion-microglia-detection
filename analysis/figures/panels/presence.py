"""Panels for what the ``presence`` module measures: whether a name is on screen.

This is the register being called. Everything else in the package measures the
cells that answered; presence records who was asked, so an empty seat can be
told apart from a person who was never on the list.

That distinction is the only reason these panels are not general charts. All
three are drawn in fixed colours - black for a name on screen, orange for a
temporary absence inside its lifespan, and grey before first appearance or
after last appearance. Tracker explanations are symbols over the orange state,
not extra colours in the state raster.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult

__all__ = ["STATE_CODES", "STATE_ROLES", "STATE_LABELS", "BASE_STATES",
           "mechanism_colours", "mechanism_label", "on_screen",
           "unclaimed_area", "persistence_values", "persistence_events",
           "persistence_raster", "gap_mechanism_strip", "lifespan_bars"]

#: The states, in the order the raster's colour map expects them. The first
#: three integers are fixed for good: changing one would silently reinterpret
#: every matrix already written to a bundle.
STATE_CODES: dict[str, int] = {
    "outside_lifespan": 0, "named": 1, "unclaimed": 2,
}

STATE_ROLES: dict[str, str] = {
    "outside_lifespan": "missing", "named": "ink", "unclaimed": "unclaimed",
}

STATE_LABELS: dict[str, str] = {
    "named": "present on screen",
    "unclaimed": "temporary gap",
    "outside_lifespan": "before first appearance / after last",
}

#: What a panel draws unless it is told otherwise - the three-state figure.
BASE_STATES: tuple[str, ...] = ("outside_lifespan", "named", "unclaimed")


MECHANISM_LABELS: dict[str, str] = {
    "blue_red_motion_link_candidate": "motion-linked identity candidate",
    "distant_alias_risk": "distant identity alias risk",
    "local_raw_supported_dropout": "raw signal supports a missed outline",
    "long_unproven_gap": "long gap without supporting evidence",
    "same_host_merge_hiding": "cell hidden inside the same merged object",
    "unresolved": "tracker could not classify the gap",
}


def mechanism_label(value: Any) -> str:
    """A tracker's compact mechanism code in words suitable for a legend."""
    name = str(value)
    return MECHANISM_LABELS.get(name, name.replace("_", " "))


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
    gap_label: str = "temporarily off screen",
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
    claimed: Sequence[float] | None = None,
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
    y_label: str = "Detected\nforeground (px)",
    claimed_label: str = "foreground claimed by a cell identity",
    unclaimed_label: str = "foreground without a cell identity",
    show_x: bool = True,
) -> PanelResult:
    """Claimed and unclaimed detected foreground on one common pixel scale.

    With ``claimed``, the filled stack shows the whole detected foreground and
    keeps the unclaimed band in its proper visual proportion. Without it, the
    function retains its older single-series form for callers that only have an
    unclaimed stack.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)

    data = {"hours": hours, "unclaimed_px": values}
    if claimed is None:
        ax.fill_between(
            hours, 0, values, color=theme.colour("unclaimed"), alpha=0.35,
            linewidth=0, label=unclaimed_label,
        )
        ax.plot(
            hours, values, color=theme.colour("unclaimed"),
            linewidth=theme.stroke("emphasis"),
        )
        ceiling = float(np.nanmax(values))
    else:
        claimed_values = np.asarray(claimed, dtype=float)
        if claimed_values.shape != values.shape:
            raise ValueError("claimed and unclaimed foreground must have the same length")
        total = claimed_values + values
        ax.fill_between(
            hours, 0, claimed_values, color=theme.colour("named"), alpha=0.18,
            linewidth=0, label=claimed_label,
        )
        ax.fill_between(
            hours, claimed_values, total, color=theme.colour("unclaimed"), alpha=0.72,
            linewidth=0, label=unclaimed_label,
        )
        ax.plot(
            hours, claimed_values, color=theme.colour("named"),
            linewidth=theme.stroke("emphasis"),
        )
        ax.plot(
            hours, total, color=theme.colour("unclaimed"),
            linewidth=theme.stroke("line"),
        )
        data.update({"claimed_px": claimed_values, "foreground_px": total})
        ceiling = float(np.nanmax(total))

    ax.set_ylabel(y_label)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_ylim(0, ceiling * 1.12 or 1.0)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    if show_x:
        ax.set_xlabel(x_label)
    else:
        ax.set_xticklabels([])
    return PanelResult(data=pd.DataFrame(data), axes=ax)


def persistence_values(
    frame: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    """Prepare one three-state persistence raster from long-form presence rows.

    Returns the exact plotted long table, matrix, frame hours and identity order.
    Keeping this preparation beside the renderer makes the standalone raster and
    any composite page use the same row ordering and state codes.
    """
    presence = pd.DataFrame(frame).copy()
    required = {"identity", "frame_index", "hours", "state",
                "first_frame_index", "last_frame_index"}
    missing = required - set(presence)
    if missing:
        raise ValueError(
            "persistence_values needs " + ", ".join(sorted(missing)))
    unknown = set(presence["state"].dropna().astype(str)) - set(STATE_CODES)
    if unknown:
        raise ValueError("unknown persistence states: " + ", ".join(sorted(unknown)))

    order = (
        presence.groupby("identity")[["first_frame_index", "last_frame_index"]]
        .first().sort_values(["first_frame_index", "last_frame_index"]).index.tolist()
    )
    row_of = {identity: row for row, identity in enumerate(order)}
    presence["row_position"] = presence["identity"].map(row_of).astype(int)
    presence["state_code"] = presence["state"].map(STATE_CODES).astype(int)
    presence = presence.sort_values(["row_position", "frame_index"])
    matrix = (
        presence.pivot(index="row_position", columns="frame_index", values="state_code")
        .sort_index().to_numpy(float)
    )
    hours = (
        presence.groupby("frame_index")["hours"].first().sort_index().to_numpy(float)
    )
    return presence, matrix, hours, [int(identity) for identity in order]


def persistence_events(
    presence: Any,
    gap_frames: Any = None,
    lifespans: Any = None,
) -> pd.DataFrame:
    """One marker per temporary-absence interval and silent ending.

    Every orange run gets one marker at its temporal midpoint. Where the
    tracker's Gantt evidence names a mechanism, that exact code is retained in
    ``mechanism`` and translated only for ``event_label``. Silent mid-field
    endings receive the cross used by the lifespan chart.
    """
    frame = pd.DataFrame(presence).copy()
    required = {"identity", "frame_index", "hours", "state", "row_position"}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            "persistence_events needs " + ", ".join(sorted(missing)))

    history = pd.DataFrame(gap_frames).copy() if gap_frames is not None else pd.DataFrame()
    if not history.empty:
        if "still_missing_in_accepted_labels" in history:
            history = history[history["still_missing_in_accepted_labels"].astype(bool)]
        keep = [column for column in ("identity", "frame_index", "mechanism")
                if column in history]
        if {"identity", "frame_index", "mechanism"}.issubset(keep):
            frame = frame.merge(
                history[keep].drop_duplicates(["identity", "frame_index"]),
                on=["identity", "frame_index"], how="left",
            )
    if "mechanism" not in frame:
        frame["mechanism"] = None

    gaps = frame[frame["state"] == "unclaimed"].copy()
    rows: list[dict[str, Any]] = []
    if not gaps.empty:
        gaps = gaps.sort_values(["identity", "frame_index"])
        new_run = (
            gaps.groupby("identity")["frame_index"].diff().ne(1)
            | gaps["identity"].ne(gaps["identity"].shift())
        )
        gaps["event_run"] = new_run.groupby(gaps["identity"]).cumsum().astype(int)
        for (identity, event_run), group in gaps.groupby(["identity", "event_run"]):
            named = group["mechanism"].dropna().astype(str)
            mechanism = named.mode().iat[0] if not named.empty else None
            label = (mechanism_label(mechanism) if mechanism is not None
                     else "temporary absence, no recorded cause")
            rows.append({
                "identity": int(identity),
                "row_position": int(group["row_position"].iat[0]),
                "event_class": "temporary_absence",
                "event_label": label,
                "mechanism": mechanism,
                "start_frame_index": int(group["frame_index"].min()),
                "end_frame_index": int(group["frame_index"].max()),
                "start_hour": float(group["hours"].min()),
                "end_hour": float(group["hours"].max()),
                "hours": float(group["hours"].mean()),
            })

    life = pd.DataFrame(lifespans).copy() if lifespans is not None else pd.DataFrame()
    if (not life.empty and {"identity", "silent_nonborder_ending"}.issubset(life)):
        silent = set(life.loc[life["silent_nonborder_ending"].astype(bool), "identity"])
        endings = (
            frame[(frame["identity"].isin(silent)) & (frame["state"] == "named")]
            .sort_values("frame_index").groupby("identity").tail(1)
        )
        for _, ending in endings.iterrows():
            rows.append({
                "identity": int(ending["identity"]),
                "row_position": int(ending["row_position"]),
                "event_class": "silent_ending",
                "event_label": "silent ending away from the field edge",
                "mechanism": None,
                "start_frame_index": int(ending["frame_index"]),
                "end_frame_index": int(ending["frame_index"]),
                "start_hour": float(ending["hours"]),
                "end_hour": float(ending["hours"]),
                "hours": float(ending["hours"]),
            })
    columns = ["identity", "row_position", "event_class", "event_label",
               "mechanism", "start_frame_index", "end_frame_index",
               "start_hour", "end_hour", "hours"]
    return pd.DataFrame(rows, columns=columns)


def persistence_raster(
    ax: Any,
    matrix: Any,
    hours: Sequence[float],
    theme: Any,
    *,
    hour_ticks: float | None = None,
    legend: bool = True,
    events: Any = None,
    event_legend: bool = True,
    x_label: str = "Hours from start of recording",
    y_label: str = "Cell, ordered by first appearance",
) -> PanelResult:
    """One row per cell and frame in the fixed black-orange-grey state key.

    The matrix holds the codes in :data:`STATE_CODES`. Rows sorted by first
    appearance turn the arrival of cells into a clean diagonal edge, so a pale
    wedge at the bottom left reads as cells that had not arrived rather than as a
    run of failures.

    ``events`` may add one symbol per interval without adding a fourth state
    colour. It is a table with ``hours``, ``row_position`` and ``event_label``;
    temporary absences receive stable category symbols and a
    ``silent_ending`` event receives a cross.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    matrix = np.asarray(matrix, dtype=float)
    hours = np.asarray(hours, dtype=float)
    order = list(BASE_STATES)
    finite = matrix[np.isfinite(matrix)]
    unknown = sorted(set(finite.astype(int)) - set(STATE_CODES.values()))
    if unknown:
        raise ValueError(f"unknown persistence state codes: {unknown}")

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
        shown = ("named", "unclaimed", "outside_lifespan")
        state_legend = ax.legend(
            handles=[
                Patch(facecolor=theme.colour(STATE_ROLES[state]), label=STATE_LABELS[state])
                for state in shown
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.115),
            ncol=3 if len(shown) <= 3 else 2,
            frameon=False, fontsize=theme.size("caption"),
            handlelength=1.6, columnspacing=1.6,
        )
        ax.add_artist(state_legend)

    event_frame = pd.DataFrame(events).copy() if events is not None else pd.DataFrame()
    event_styles: dict[str, str] = {}
    if not event_frame.empty:
        needed = {"hours", "row_position", "event_label"}
        missing = needed - set(event_frame)
        if missing:
            raise ValueError(
                "persistence events need " + ", ".join(sorted(missing)))
        classes = (event_frame["event_class"].astype(str)
                   if "event_class" in event_frame
                   else pd.Series("temporary_absence", index=event_frame.index))
        ordinary = sorted(event_frame.loc[
            classes != "silent_ending", "event_label"
        ].astype(str).unique())
        marker_cycle = ("o", "s", "^", "D", "v", "P", "h", "*")
        event_styles = {
            label: marker_cycle[index % len(marker_cycle)]
            for index, label in enumerate(ordinary)
        }
        if "event_class" in event_frame and (
                event_frame["event_class"] == "silent_ending").any():
            event_styles["silent ending away from the field edge"] = "x"

        handles = []
        for label in ordinary:
            rows = event_frame[event_frame["event_label"].astype(str) == label]
            marker = event_styles[label]
            ax.scatter(
                rows["hours"], rows["row_position"], marker=marker,
                s=theme.size("caption") ** 2 * 0.34,
                facecolors=theme.colour("page"), edgecolors=theme.colour("ink"),
                linewidths=theme.stroke("line") * 0.55, zorder=4,
            )
            handles.append(Line2D(
                [], [], marker=marker, linestyle="none",
                markerfacecolor=theme.colour("page"),
                markeredgecolor=theme.colour("ink"), label=label,
            ))
        if "event_class" in event_frame:
            rows = event_frame[event_frame["event_class"] == "silent_ending"]
            if not rows.empty:
                ax.scatter(
                    rows["hours"], rows["row_position"], marker="x",
                    s=theme.size("caption") ** 2 * 0.42,
                    color=theme.colour("ink"), linewidths=theme.stroke("line") * 0.7,
                    zorder=5, clip_on=False,
                )
                handles.append(Line2D(
                    [], [], marker="x", linestyle="none", color=theme.colour("ink"),
                    label="silent ending away from the field edge",
                ))
        if event_legend and handles:
            ax.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                ncol=1, frameon=False, fontsize=theme.size("caption"),
                title="Tracker events", title_fontsize=theme.size("caption"),
            )
    named = {code: state for state, code in STATE_CODES.items()}
    table = drawn.data.rename(columns={"value": "state_code"})
    table["state"] = table["state_code"].map(named)
    return PanelResult(data=table, axes=ax,
                       extra={"handle": drawn.extra["handle"],
                              "event_styles": event_styles})


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
            handles=[Patch(facecolor=colour, label=mechanism_label(name))
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
        plotted = {key: value for key, value in row.to_dict().items()
                   if not str(key).startswith("_")}
        plotted.update({"identity": identity, "row_order": row_index,
                        "first_hour": first, "last_hour": last,
                        "marked": identity in marked})
        rows.append(plotted)
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
