"""Figure 14: unusually large cell-footprint changes over time.

A detected transition is one where the cell retains an unusually small share
of the pixels in its preceding footprint. ``jaccard`` is that retained share:
shared footprint pixels divided by pixels occupied in either frame.

    python analysis/figures/14_upheaval_events.py <run>
    ... --quantile 0.02          use the lowest 2% of measurable transitions
    ... --cells 4                draw four matched before/after examples
    ... --cells 55,104,58        draw specified cells' strongest events
    ... --crop-padding 0.35      add 35% empty border on every side
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from _metrics import semantic_label
from _options import commas
from _schema import (FigureContext, FigureResult, Input, Option, Panel, Table,
                     figure, run_figure)

from panels import common
from panels import surveillance as surveillance_panels


def _event_definition(metric: str) -> str:
    if metric == "jaccard":
        return "the fraction of the same cell's footprint retained from the previous frame"
    return semantic_label(metric).lower()


def _events_and_rates(
    data: pd.DataFrame, metric: str, quantile: float
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    if metric not in data:
        raise SystemExit(f"--metrics names {metric!r}, which is not in cell_frame.csv")
    if not 0.0 < quantile < 1.0:
        raise SystemExit("--quantile must be greater than 0 and less than 1")

    values = pd.to_numeric(data[metric], errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan).notna()
    if not finite.any():
        raise SystemExit(f"{metric} has no measurable transitions")
    threshold = float(values[finite].quantile(quantile))
    measured = data.loc[finite].copy()
    measured[metric] = values[finite]
    measured["detected_event"] = measured[metric] <= threshold

    rates = measured.groupby("identity", as_index=False).agg(
        measurable_transitions=(metric, "size"),
        detected_transitions=("detected_event", "sum"),
        first_measurable_hour=("hours", "min"),
        last_measurable_hour=("hours", "max"),
    )
    rates["event_rate"] = rates["detected_transitions"] / rates["measurable_transitions"]
    rates = rates.sort_values(
        ["event_rate", "measurable_transitions", "identity"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    rates["event_rate_rank"] = np.arange(1, len(rates) + 1)

    events = measured[measured["detected_event"]].copy()
    events = events.merge(
        rates[["identity", "measurable_transitions", "detected_transitions",
               "event_rate", "event_rate_rank"]],
        on="identity", how="left", validate="many_to_one",
    )
    events["event_threshold"] = threshold
    events["event_threshold_quantile"] = quantile
    events["event_metric"] = metric
    events["event_definition"] = _event_definition(metric)
    return threshold, events, rates


def _select_examples(
    events: pd.DataFrame, data: pd.DataFrame, metric: str, wanted: str
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    area = data[["identity", "frame_index", "area_px", "hours"]].rename(
        columns={"frame_index": "from_frame_index", "area_px": "before_area_px",
                 "hours": "before_hours"}
    )
    candidates = events.merge(
        area, on=["identity", "from_frame_index"], how="left", validate="many_to_one"
    )
    candidates["largest_pair_area_px"] = candidates[
        ["area_px", "before_area_px"]
    ].max(axis=1)
    candidates = candidates.sort_values(
        [metric, "largest_pair_area_px", "identity", "frame_index"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    strongest = candidates.groupby("identity", sort=False, as_index=False).head(1)
    try:
        return strongest.head(max(0, int(wanted))).reset_index(drop=True)
    except ValueError:
        identities = [int(value) for value in commas(wanted)]
        chosen = strongest[strongest["identity"].astype(int).isin(identities)].copy()
        missing = sorted(set(identities) - set(chosen["identity"].astype(int)))
        if missing:
            raise SystemExit(
                "--cells includes identities with no detected transition: "
                + ", ".join(map(str, missing))
            )
        order = {identity: index for index, identity in enumerate(identities)}
        chosen["requested_order"] = chosen["identity"].astype(int).map(order)
        return chosen.sort_values("requested_order").reset_index(drop=True)


def _crop_box(
    centre: tuple[float, float], side: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    side = min(int(side), int(shape[0]), int(shape[1]))
    top = int(round(float(centre[0]) - side / 2))
    left = int(round(float(centre[1]) - side / 2))
    top = min(max(top, 0), shape[0] - side)
    left = min(max(left, 0), shape[1] - side)
    return top, top + side, left, left + side


def _matched_crops(
    raw: np.ndarray,
    labels: np.ndarray,
    picked: pd.DataFrame,
    *,
    metric: str,
    padding_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray],
           list[tuple[str, str]], pd.DataFrame]:
    if padding_fraction < 0:
        raise SystemExit("--crop-padding cannot be negative")
    prepared: list[dict] = []
    largest_span = 1
    for _, event in picked.iterrows():
        identity = int(event["identity"])
        before_frame = int(event["from_frame_index"])
        after_frame = int(event["frame_index"])
        before_mask = labels[before_frame] == identity
        after_mask = labels[after_frame] == identity
        occupied = np.argwhere(before_mask | after_mask)
        if not occupied.size:
            continue
        low = occupied.min(axis=0)
        high = occupied.max(axis=0)
        largest_span = max(largest_span, int(np.max(high - low + 1)))
        prepared.append({
            "event": event, "identity": identity,
            "before_frame": before_frame, "after_frame": after_frame,
            "before_mask": before_mask, "after_mask": after_mask,
            "centre": tuple((low + high) / 2.0),
        })
    crop_side = max(3, int(np.ceil(largest_span * (1.0 + 2.0 * padding_fraction))))

    before_images: list[np.ndarray] = []
    after_images: list[np.ndarray] = []
    before_masks: list[np.ndarray] = []
    after_masks: list[np.ndarray] = []
    titles: list[tuple[str, str]] = []
    records = []
    for pair, item in enumerate(prepared):
        top, bottom, left, right = _crop_box(
            item["centre"], crop_side, labels.shape[1:]
        )
        before_frame = item["before_frame"]
        after_frame = item["after_frame"]
        event = item["event"]
        before_images.append(raw[before_frame, top:bottom, left:right])
        after_images.append(raw[after_frame, top:bottom, left:right])
        before_masks.append(item["before_mask"][top:bottom, left:right])
        after_masks.append(item["after_mask"][top:bottom, left:right])
        retained = 100.0 * float(event[metric])
        before_hour = float(event["before_hours"])
        titles.append((
            f"Cell {item['identity']} · before ({before_hour:g} h)",
            f"After ({float(event['hours']):g} h) · {retained:.0f}% retained",
        ))
        records.append({
            "pair": pair,
            "identity": item["identity"],
            "before_frame_index": before_frame,
            "after_frame_index": after_frame,
            "before_hours": before_hour,
            "after_hours": float(event["hours"]),
            metric: float(event[metric]),
            "crop_top": top,
            "crop_bottom": bottom,
            "crop_left": left,
            "crop_right": right,
            "crop_size_px": bottom - top,
        })
    return (before_images, after_images, before_masks, after_masks, titles,
            pd.DataFrame(records))


@figure(
    number=14,
    slug="upheaval-events",
    summary="when cells retain unusually little of their preceding footprint",
    title="Large cell-footprint changes through time",
    reads=(Table("cell_frame.csv", module="surveillance"),
           Input("labels"), Input("raw")),
    panels=(
        Panel("raster", surveillance_panels.low_overlap_event_raster,
              min_height_inches=6.0,
              title="Detected changes through time — grey lines show measurable spans"),
        Panel("rates", common.ranked_rate,
              title="Share of measurable transitions detected for each cell"),
        Panel("tiles", common.paired_image_strip, block=True,
              item_width_inches=6.25, item_height_inches=3.2,
              item_gap_inches=0.75,
              title="Strongest matched changes — orange before, teal after"),
    ),
    options=(
        Option("metrics", default="jaccard", cast=str, metavar="COL",
               help="the per-transition column whose lowest values define an event"),
        Option("quantile", default=0.05),
        Option("cells", default="3", cast=str),
        Option("crop_padding", default=0.25),
        Option("hour_ticks", default=None),
    ),
    grammar="event raster with ranked per-cell rate and paired image crops",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    metric = str(ctx.option("metrics"))
    quantile = float(ctx.option("quantile"))
    threshold, events, rates = _events_and_rates(data, metric, quantile)
    panels = ctx.panels()
    requested_cells = str(ctx.option("cells"))
    picked = _select_examples(events, data, metric, requested_cells)
    try:
        requested_count = max(0, int(requested_cells))
    except ValueError:
        requested_count = len(commas(requested_cells))

    raw_path = ctx.input_path("raw") if "tiles" in panels else None
    label_path = ctx.input_path("labels") if "tiles" in panels else None
    tile_panel = ctx.spec.panel("tiles")
    minimum_sizes = ({"tiles": tile_panel.grid_minimum(
        requested_count, max(1, requested_count))} if "tiles" in panels else None)
    fig, axes = ctx.layout(panels, minimum_sizes=minimum_sizes, gap_inches=1.5)
    figure_axes: list = []

    order = rates.sort_values("event_rate_rank")["identity"].tolist()
    if "raster" in axes:
        raster = ctx.drew("raster", surveillance_panels.low_overlap_event_raster(
            axes["raster"], data, ctx.theme,
            jaccard_column=metric, threshold=threshold, order=order,
            hour_ticks=ctx.option("hour_ticks"),
            y_label="Cell rank (highest event rate at top)",
        ))
        events = raster.data.merge(
            rates[["identity", "measurable_transitions", "detected_transitions",
                   "event_rate", "event_rate_rank"]],
            on="identity", how="left", validate="many_to_one",
        )
        events["event_threshold_quantile"] = quantile
        events["event_metric"] = metric
        events["event_definition"] = _event_definition(metric)
        figure_axes.append(axes["raster"])

    population_rate = len(events) / int(rates["measurable_transitions"].sum())
    if "rates" in axes:
        rate_panel = ctx.drew("rates", common.ranked_rate(
            axes["rates"], rates, ctx.theme,
            numerator_column="detected_transitions",
            denominator_column="measurable_transitions",
            unit_column="identity", reference=population_rate,
            reference_label="All measurable transitions",
            x_label="Cells, ranked by detected-transition rate",
            y_label="Detected transitions (%)",
        ))
        figure_axes.append(axes["rates"])
    else:
        rate_panel = common.PanelResult(data=rates)

    tile_table = pd.DataFrame()
    crop_table = pd.DataFrame()
    if "tiles" in axes:
        rect = tuple(axes["tiles"].get_position().bounds)
        axes["tiles"].remove()
        if raw_path and label_path and not picked.empty:
            labels = tifffile.imread(label_path)
            raw_full = tifffile.imread(raw_path)
            offset_values = (
                data["source_imagej_frame"] - data["imagej_frame"]
            ).dropna().astype(int).unique()
            if len(offset_values) != 1:
                raise SystemExit("raw-to-label frame offset is not constant")
            offset = int(offset_values[0])
            raw = raw_full[offset:offset + labels.shape[0]]
            if raw.shape[0] != labels.shape[0]:
                raise SystemExit("raw and label stacks do not cover the same aligned frames")
            (before_images, after_images, before_masks, after_masks, titles,
             crop_table) = _matched_crops(
                raw, labels, picked, metric=metric,
                padding_fraction=float(ctx.option("crop_padding")),
            )
            drawn_tiles = common.paired_image_strip(
                fig, rect, before_images, after_images, ctx.theme,
                before_masks=before_masks, after_masks=after_masks, titles=titles,
                within_pair_gap=(0.08 * ctx.theme.canvas_scale / fig.get_figwidth()),
                between_pair_gap=(0.75 * ctx.theme.canvas_scale / fig.get_figwidth()),
            )
            tile_table = drawn_tiles.data.merge(
                crop_table, on="pair", how="left", validate="many_to_one"
            )
            ctx.drew("tiles", common.PanelResult(
                data=tile_table, axes=drawn_tiles.axes, extra=drawn_tiles.extra
            ))
            figure_axes.extend(drawn_tiles.axes)
            fig.text(
                rect[0], rect[1] + rect[3] + 0.012,
                tile_panel.heading(), fontsize=ctx.theme.size("panel"),
                fontweight="bold", color=ctx.theme.colour("ink"), va="bottom",
            )
        else:
            ax = fig.add_axes(rect)
            ax.text(0.5, 0.5, "No detected transitions to show", ha="center", va="center")
            ax.set_axis_off()
            tile_table = pd.DataFrame(columns=["pair", "state"])
            ctx.drew("tiles", common.PanelResult(data=tile_table, axes=ax))
            figure_axes.append(ax)

    kept = [column for column in (
        "identity", "frame_index", "from_frame_index", "hours", metric,
        "area_px", "event_threshold", "event_threshold_quantile", "event_metric",
        "event_definition", "measurable_transitions", "detected_transitions",
        "event_rate", "event_rate_rank", "row_order",
    ) if column in events]
    definition = _event_definition(metric)
    return FigureResult(
        figure=fig,
        axes=figure_axes,
        figure_data=events[kept],
        auxiliary={
            "event_rates_by_cell.csv": rate_panel.data,
            **({"matched_crop_metadata.csv": crop_table,
                "matched_tile_display.csv": tile_table} if "tiles" in panels else {}),
        },
        subtitle=(
            f"A transition is detected when {definition} is ≤ {threshold:.3g} "
            f"(the lowest {quantile:.0%} of measurable transitions)."
        ),
        footnote=ctx.provenance_footnote(
            "The ranked graph divides detected transitions by all measurable consecutive "
            "transitions for each cell; each dot is one cell.",
            "Matched crops use one pixel scale and one intensity range. On after images, "
            "orange repeats the before outline and teal marks the after outline.",
        ),
        readme=f"""## What is detected

A detected large footprint change is a transition where {definition} is at or
below {threshold:.6g}. This is the recording's {quantile:.0%} quantile, not a
biological cut-off. For Jaccard overlap, retained footprint is shared pixels
divided by pixels occupied in either frame.

## Panels

- The event raster marks when changes were detected; grey spans show when each
  cell had measurable consecutive transitions.
- The ranked rate is detected transitions divided by measurable transitions for
  each cell. It replaces raw counts, which were confounded by observation time.
- The matched crops show the strongest detected transition from distinct cells.
  Orange is the before outline; teal is the after outline.

## Data

`data/der/figure_data.csv` contains every detected transition.
`data/der/event_rates_by_cell.csv` contains the denominator and rate for every
measurable cell. Crop and display metadata are stored beside them; source tables
and image stacks are copied under `data/src/` and fingerprinted in
`data/sources.csv`.
""",
        console=(
            f"{len(events):,} detected of {int(rates['measurable_transitions'].sum()):,} "
            f"measurable transitions across {len(rates):,} cells"
        ),
    )


if __name__ == "__main__":
    run_figure("upheaval-events")
