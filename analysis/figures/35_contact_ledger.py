"""Figure 35: which identities share boundaries and for how long.

Two cells are in contact when their outlines come within the dilation radius.
The boundary panel compares each cell's exchange while in contact against its
exchange while free, which is the comparison that says whether contact changes
what a cell does.

    python analysis/figures/35_contact_ledger.py <run>
    ... --dilation 1,3           the two contact radii
    ... --min-hours 2            hide pairs that touched only briefly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
import numpy as np
import pandas as pd

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import coupling as coupling_panels


@figure(
    number=35,
    slug="contact-ledger",
    summary="which identities share boundaries and for how long",
    title="Which identities share boundaries and for how long",
    reads=(Table("contacts.csv", module="contacts"),
           Table("contacts_frame.csv", module="contacts"),
           Table("cell_frame.csv", module="contacts")),
    panels=(
        Panel("chord", coupling_panels.contact_chord,
              title="Contact network"),
        Panel("over_time", common.raster, title="Contacts through time"),
        Panel("boundaries", common.paired_slopes,
              title="Boundary exchange with and without contact"),
    ),
    options=(
        Option("dilation", default=None, metavar="PX,PX",
               help="one or two contact radii; unset takes what the run measured"),
        Option("min_hours", default=0.0),
        Option("hour_ticks", default=24.0),
    ),
    grammar="contact chords adjacency raster and paired boundary comparison",
)
def build(ctx: FigureContext) -> FigureResult:
    contacts = ctx.table("contacts.csv")
    frame = ctx.table("contacts_frame.csv")
    available = sorted(int(value) for value in contacts["dilation_px"].unique())
    requested = ctx.option_or("dilation", [float(value) for value in available[:2]])
    dilations = [int(value) for value in requested]
    missing = sorted(set(dilations) - set(available))
    if missing:
        raise SystemExit(f"--dilation {missing} was not measured; available radii: {available}")
    if len(dilations) not in {1, 2}:
        raise SystemExit("--dilation accepts one or two measured radii")
    minimum_hours = float(ctx.option("min_hours"))
    contacts = contacts[(contacts["dilation_px"].isin(dilations)) &
                        (contacts["hours_in_contact"] >= minimum_hours)].copy()
    frame = frame[frame["dilation_px"].isin(dilations)].copy()
    if "overlap_px" not in contacts:
        overlap = frame.groupby(["identity_a", "identity_b", "dilation_px"], as_index=False)["overlap_px"].sum()
        contacts = contacts.merge(overlap, on=["identity_a", "identity_b", "dilation_px"], how="left")
    contacts["coincides_with_handoff"] = contacts.get("handoff_coincident", False).astype(bool)
    figure_columns = ["identity_a", "identity_b", "dilation_px", "frames_in_contact",
                      "hours_in_contact", "first_hour", "last_hour", "longest_bout_frames",
                      "mean_shared_boundary_px", "overlap_px", "coincides_with_handoff"]
    figure_data = contacts[figure_columns]
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "chord" in axes:
        rect = tuple(axes["chord"].get_position().bounds); axes["chord"].remove()
        made = []
        gap = 0.035; width = (rect[2] - gap * (len(dilations) - 1)) / len(dilations)
        for position, dilation in enumerate(dilations):
            subset = contacts[contacts["dilation_px"] == dilation]
            identities = sorted(set(subset["identity_a"]) | set(subset["identity_b"]))
            index = {identity: place for place, identity in enumerate(identities)}
            weights = np.zeros((len(identities), len(identities)), dtype=float)
            flagged = np.zeros_like(weights, dtype=bool)
            for _, row in subset.iterrows():
                left, right = index[int(row["identity_a"])], index[int(row["identity_b"])]
                weights[left, right] = weights[right, left] = row["hours_in_contact"]
                flagged[left, right] = flagged[right, left] = bool(row["coincides_with_handoff"])
            shown = min(18, len(identities))
            chosen_indices = np.argsort(weights.sum(axis=1))[-shown:] if shown else np.array([], dtype=int)
            ax = fig.add_axes([rect[0] + position * (width + gap), rect[1], width, rect[3]])
            coupling_panels.contact_chord(
                ax, weights[np.ix_(chosen_indices, chosen_indices)], ctx.theme,
                labels=[str(identities[value]) for value in chosen_indices],
                flagged=flagged[np.ix_(chosen_indices, chosen_indices)],
            )
            ax.set_title(f"Radius {ctx.scale.length(dilation):g} {ctx.length_label}",
                         fontsize=ctx.theme.size("annotation")); made.append(ax)
        axes["chord"] = made[0] if made else fig.add_axes(rect)
        if made:
            common.semantic_legend(
                made[0], ctx.theme,
                handles=[
                    Line2D([], [], color=ctx.theme.colour("surveillance"),
                           linewidth=ctx.theme.stroke("emphasis")),
                    Line2D([], [], color=ctx.theme.colour("highlight"), marker="x", linestyle="none"),
                ],
                labels=["Contact; line width shows hours", "Coincides with an identity handoff"],
                location="below", columns=1,
            )
    if "over_time" in axes:
        max_frame = int(frame["frame_index"].max()) if not frame.empty else 0
        hours = np.arange(max_frame + 1) * ctx.interval / 60
        row_groups = list(frame.groupby(["dilation_px", "identity_a", "identity_b"], sort=True))
        adjacency = np.zeros((len(row_groups), max_frame + 1), dtype=float)
        labels = []
        for row_index, (keys, group) in enumerate(row_groups):
            adjacency[row_index, group["frame_index"].to_numpy(int)] = 1
            labels.append(f"{ctx.scale.length(int(keys[0])):g} {ctx.length_label}: "
                          f"{int(keys[1])}–{int(keys[2])}")
        common.raster(
            axes["over_time"], adjacency, ctx.theme, hours=hours,
            cmap=matplotlib.colors.ListedColormap([
                ctx.theme.colour("page"), ctx.theme.colour("surveillance")
            ]), vmin=0, vmax=1,
        )
        axes["over_time"].set_xticks(ctx.theme.hour_ticks(float(hours.min()), float(hours.max()), ctx.hour_ticks))
        axes["over_time"].set_xlabel("Hours from start of recording"); axes["over_time"].set_ylabel("Contact pair")
        if labels:
            picks = sorted(set([0, len(labels) - 1])); axes["over_time"].set_yticks(picks)
            axes["over_time"].set_yticklabels([labels[index] for index in picks])
        common.semantic_legend(
            axes["over_time"], ctx.theme,
            handles=[Patch(facecolor=ctx.theme.colour("surveillance"))],
            labels=["Pair in contact"], location="inside",
        )
    ledger = ctx.table("cell_frame.csv")
    ledger = ledger.dropna(subset=["gained_px", "lost_px", "perimeter_px"]).copy()
    ledger["boundary_exchange"] = (
        ledger["gained_px"].map(ctx.scale.area) + ledger["lost_px"].map(ctx.scale.area)
    ) / ledger["perimeter_px"].map(ctx.scale.length).replace(0, np.nan)
    base = dilations[0]
    contact_frames: dict[int, set[int]] = {}
    for _, row in frame[frame["dilation_px"] == base].iterrows():
        contact_frames.setdefault(int(row["identity_a"]), set()).add(int(row["frame_index"]))
        contact_frames.setdefault(int(row["identity_b"]), set()).add(int(row["frame_index"]))
    comparison_rows = []
    for identity, group in ledger.groupby("identity"):
        flags = group["frame_index"].isin(contact_frames.get(int(identity), set()))
        contacting = group.loc[flags, "boundary_exchange"].dropna()
        free = group.loc[~flags, "boundary_exchange"].dropna()
        if contacting.empty or free.empty:
            continue
        comparison_rows.append({"identity": int(identity), "free_boundary_exchange": float(free.median()),
                                "contact_boundary_exchange": float(contacting.median()),
                                "contact_frames": int(flags.sum()), "free_frames": int((~flags).sum())})
    comparison = pd.DataFrame(comparison_rows)
    if "boundaries" in axes and not comparison.empty:
        common.paired_slopes(
            axes["boundaries"], comparison["free_boundary_exchange"],
            comparison["contact_boundary_exchange"], ctx.theme,
            left_label="Free frames", right_label="Contact frames", role="surveillance",
            y_label=f"Exchange ({ctx.length_label})",
        )
    radius_text = ", ".join(f"{ctx.scale.length(value):g} {ctx.length_label}" for value in dilations)
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Contact radii {radius_text}; pairs below {minimum_hours:g} h are hidden, and handoffs are flagged.",
        auxiliary={"contacts_frame.csv": frame, "boundary_comparison.csv": comparison},
    )


if __name__ == "__main__":
    run_figure("contact-ledger")
