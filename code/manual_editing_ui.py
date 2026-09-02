from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from common import OUTLINE_COLOURS, label_edges, outline_overlay
from manual_editing import EditingHistoryController, load_edit_batch
from manual_editing_filters import (FILTERABLE_METRICS,
                                    build_track_filter_metrics,
                                    evaluate_track_filters,
                                    validate_filter_set)
from manual_editing_theme import DARK, apply_dark_theme


ZOOM_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
PERCENT_FILTER_METRICS = frozenset({
    "area_cv", "intensity_cv", "coverage_fraction",
    "merge_contact_fraction", "border_touch_fraction",
    "fragmented_mask_fraction", "manual_pixel_fraction",
    "centroid_assisted_pixel_fraction",
})
FILTER_METRIC_DISPLAY_NAMES = {
    "net_displacement": "start-to-end distance",
    "furthest_point_distance": "furthest-point distance",
}
FILTER_METRIC_LABELS = {
    metric: (
        f"{FILTER_METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' '))} (%)"
        if metric in PERCENT_FILTER_METRICS
        else FILTER_METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ")))
    for metric in FILTERABLE_METRICS}
FILTER_METRIC_KEYS = {
    label: metric for metric, label in FILTER_METRIC_LABELS.items()}
FILTER_OPERATOR_KEYS = {
    "<": "lt", "≤": "lte", ">": "gt", "≥": "gte", "=": "eq",
    "between": "between", "outside": "outside",
}


def build_selection_filter_spec(
        rows: list[tuple[str, str, str]], combine: str = "all") -> dict:
    """Convert compact editor rows into a validated Motion filter set."""
    conditions = []
    for index, (metric, operator, raw_value) in enumerate(rows, start=1):
        raw = raw_value.strip()
        if operator in {"between", "outside"}:
            value: float | list[float] = [
                float(part.strip()) for part in raw.split(",")]
        else:
            value = float(raw)
        # REGRESSION GUARD: Percentage-like metrics are stored as ratios.
        # Inline values use percentages so entering 10 consistently means 10%.
        if metric in PERCENT_FILTER_METRICS:
            value = ([item / 100.0 for item in value]
                     if isinstance(value, list) else value / 100.0)
        conditions.append({
            "name": f"filter_{index}",
            "metric": metric,
            "operator": operator,
            "value": value,
        })
    return validate_filter_set({
        "schema": "motion.manual-editing-filter-set",
        "schema_version": 1,
        "combine": combine,
        "filters": conditions,
    })


def create_history_icon(action: str, size: int = 18) -> Image.Image:
    """Draw a compact dark-theme undo, redo, or save toolbar icon."""
    if action not in {"undo", "redo", "save"}:
        raise ValueError(f"unknown history icon: {action}")
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    colour = DARK["foreground"]
    if action == "save":
        draw.rounded_rectangle(
            (3, 2, size - 3, size - 2), radius=1,
            outline=colour, width=2)
        draw.rectangle((6, 2, size - 6, 7), outline=colour, width=1)
        draw.rectangle(
            (6, 11, size - 6, size - 2), outline=colour, width=1)
        return image

    bounds = (3, 3, size - 3, size - 3)
    if action == "undo":
        draw.arc(bounds, start=35, end=315, fill=colour, width=2)
        draw.polygon(((3, 4), (3, 10), (8, 7)), fill=colour)
    else:
        draw.arc(bounds, start=225, end=505, fill=colour, width=2)
        draw.polygon(((size - 3, 4), (size - 3, 10),
                      (size - 8, 7)), fill=colour)
    return image


def create_current_frame_icon(size: int = 14) -> Image.Image:
    """Draw a compact target icon meaning use the current frame."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    centre = size // 2
    colour = DARK["foreground"]
    draw.ellipse((2, 2, size - 3, size - 3), outline=colour, width=1)
    draw.line((centre, 0, centre, 4), fill=colour, width=1)
    draw.line((centre, size - 5, centre, size - 1), fill=colour, width=1)
    draw.line((0, centre, 4, centre), fill=colour, width=1)
    draw.line((size - 5, centre, size - 1, centre), fill=colour, width=1)
    draw.ellipse(
        (centre - 1, centre - 1, centre + 1, centre + 1),
        fill=DARK["accent"])
    return image


def compute_track_centroids(
        labels: np.ndarray) -> dict[int, list[tuple[int, float, float]]]:
    """Build one ordered `(frame, y, x)` centroid path per identity."""
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("track centroids require an integer (T,Y,X) label stack")
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for frame_index, frame in enumerate(labels):
        y, x = np.nonzero(frame)
        if not len(y):
            continue
        identities, inverse, counts = np.unique(
            frame[y, x], return_inverse=True, return_counts=True)
        y_sums = np.bincount(inverse, weights=y)
        x_sums = np.bincount(inverse, weights=x)
        for index, identity_value in enumerate(identities):
            identity = int(identity_value)
            tracks.setdefault(identity, []).append((
                frame_index,
                float(y_sums[index] / counts[index]),
                float(x_sums[index] / counts[index]),
            ))
    return tracks


def _draw_dashed_segment(
        draw: ImageDraw.ImageDraw, start: tuple[float, float],
        end: tuple[float, float], fill: tuple[int, int, int],
        width: int) -> None:
    x0, y0 = start
    x1, y1 = end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        return
    dash = 3.0
    for distance in np.arange(0.0, length, dash * 2):
        finish = min(length, distance + dash)
        start_fraction = distance / length
        finish_fraction = finish / length
        draw.line((
            x0 + (x1 - x0) * start_fraction,
            y0 + (y1 - y0) * start_fraction,
            x0 + (x1 - x0) * finish_fraction,
            y0 + (y1 - y0) * finish_fraction,
        ), fill=fill, width=width)


def render_editor_frame(
        raw: np.ndarray, labels: np.ndarray, selected: set[int],
        show_identity_numbers: bool = True,
        tracks: dict[int, list[tuple[int, float, float]]] | None = None,
        frame_index: int = 0, show_tracks: bool = False) -> np.ndarray:
    """Render one aligned raw frame with outlines and selected identities."""
    if raw.ndim != 2 or labels.ndim != 2 or raw.shape != labels.shape:
        raise ValueError("editor raw and label frames must be same-shaped 2D arrays")
    rendered = outline_overlay(raw[None], labels[None], thick=2)[0]
    if selected:
        selected_labels = np.where(np.isin(labels, list(selected)), labels, 0)
        selected_edges = label_edges(selected_labels[None], thick=3)[0]
        rendered[selected_edges] = np.array([255, 220, 40], np.uint8)

    image = Image.fromarray(rendered)
    draw = ImageDraw.Draw(image)
    if show_tracks and tracks and selected:
        visible_identities = set(selected)
        for identity in sorted(visible_identities):
            points = tracks.get(identity, [])
            if not points:
                continue
            colour_array = OUTLINE_COLOURS[
                (identity - 1) % len(OUTLINE_COLOURS)]
            colour = tuple(int(value) for value in colour_array)
            future_colour = tuple(
                int(24 + value * 0.42) for value in colour_array)
            width = 2 if identity in selected else 1
            for first, second in zip(points, points[1:]):
                first_frame, first_y, first_x = first
                second_frame, second_y, second_x = second
                future = first_frame >= frame_index
                discontinuous = second_frame - first_frame > 1
                segment = ((first_x, first_y), (second_x, second_y))
                if future or discontinuous:
                    _draw_dashed_segment(
                        draw, segment[0], segment[1],
                        future_colour if future else colour, width)
                else:
                    draw.line(segment, fill=colour, width=width)
            current = next(
                (point for point in points if point[0] == frame_index), None)
            if current is not None:
                _time, current_y, current_x = current
                radius = 2 if identity in selected else 1
                draw.ellipse((
                    current_x - radius, current_y - radius,
                    current_x + radius, current_y + radius,
                ), fill=(255, 230, 60) if identity in selected else colour)

    if not show_identity_numbers:
        return np.asarray(image)

    for identity in np.unique(labels):
        identity = int(identity)
        if identity <= 0:
            continue
        y, x = np.nonzero(labels == identity)
        centre_x = int(round(float(x.mean())))
        centre_y = int(round(float(y.mean())))
        text = str(identity)
        box = draw.textbbox((centre_x, centre_y), text, anchor="mm")
        draw.rectangle(
            (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1),
            fill=(28, 28, 28))
        draw.text(
            (centre_x, centre_y), text, anchor="mm",
            fill=((255, 230, 60) if identity in selected else (255, 255, 255)))
    return np.asarray(image)


def identity_at(labels: np.ndarray, y: int, x: int) -> int:
    """Return the positive identity under one image coordinate, or zero."""
    if labels.ndim != 2:
        raise ValueError("identity lookup requires one 2D label frame")
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _default_output_path(controller: EditingHistoryController) -> Path:
    if controller.portable_source:
        source = controller.source_reference
        source_name = (
            source.stem if source.is_file()
            or source.suffix.lower() == ".motionpkg" else source.name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return source.parent / (
            f"{source_name}_manual_edit_{controller.position + 1}_{stamp}.motionpkg")
    source_dir = controller.source_manifest.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return source_dir.parent / (
        f"{source_dir.name}_manual_edit_{controller.position + 1}_{stamp}")


class ManualEditingApp:
    """Tk desktop workspace backed by immutable Motion editing sessions."""

    def __init__(self, root, source_path: Path):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        apply_dark_theme(root)
        self.controller = EditingHistoryController(source_path)
        self.labels, self.provenance, self.excluded = \
            self.controller.materialize()
        self.track_centroids = compute_track_centroids(self.labels)
        self.frame_index = 0
        self.selected: set[int] = set()
        self.zoom = tk.StringVar(value="200%")
        self.show_numbers = tk.BooleanVar(value=True)
        self.show_tracks = tk.BooleanVar(value=False)
        self.output_path = tk.StringVar(value=str(
            _default_output_path(self.controller)))
        self.status = tk.StringVar()
        self.frame_status = tk.StringVar()
        self.source_status = tk.StringVar()
        self.selection_status = tk.StringVar(value="No identities selected")
        self.filter_combine = tk.StringVar(value="all")
        self.filter_status = tk.StringVar(value="No filters added")
        self.selection_filter_rows: list[dict] = []
        self.radius = tk.StringVar(value="1")
        self.range_start = tk.StringVar(value="1")
        self.range_end = tk.StringVar(value=str(len(self.labels)))
        self.assignment_target = tk.StringVar()
        calibration = self.controller.bundle.manifest["spatial_calibration"]
        self.radius_unit = tk.StringVar(
            value="calibrated" if calibration["unit"] != "pixel" else "pixel")
        self.photo = None

        root.title("Motion manual track editor")
        root.geometry("1420x900")
        root.minsize(1050, 680)
        self._build()
        self._bind()
        self._refresh_all("Ready")

    def _build(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(
            header, text="Manual track editor",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side="left")
        ttk.Label(header, textvariable=self.source_status).pack(
            side="left", padx=(18, 0))
        ttk.Button(
            header, text="Apply batch JSON…", command=self._apply_batch_file
        ).pack(side="right")
        ttk.Button(
            header, text="Freeze package…", command=self._freeze_package
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            header, text="Import package…", command=self._import_package
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            header, text="History table…", command=self._open_history
        ).pack(side="right", padx=(0, 6))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 0))

        viewer = ttk.Frame(body)
        sidebar_host = ttk.Frame(body)
        body.add(viewer, weight=3)
        body.add(sidebar_host, weight=2)

        from PIL import ImageTk

        history_toolbar = ttk.Frame(viewer)
        history_toolbar.pack(fill="x", pady=(0, 5))
        ttk.Style(self.root).configure("EditorIcon.TButton", padding=(3, 2))
        self.history_icons = {
            action: ImageTk.PhotoImage(
                create_history_icon(action), master=self.root)
            for action in ("undo", "redo", "save")
        }
        self.current_frame_icon = ImageTk.PhotoImage(
            create_current_frame_icon(), master=self.root)
        self.undo_button = ttk.Button(
            history_toolbar, image=self.history_icons["undo"],
            command=self._undo, style="EditorIcon.TButton", cursor="hand2")
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(
            history_toolbar, image=self.history_icons["redo"],
            command=self._redo, style="EditorIcon.TButton", cursor="hand2")
        self.redo_button.pack(side="left", padx=(4, 0))
        self.save_button = ttk.Button(
            history_toolbar, image=self.history_icons["save"],
            command=self._save_checkpoint, style="EditorIcon.TButton",
            cursor="hand2")
        self.save_button.pack(side="left", padx=(4, 0))
        self._bind_status_hint(
            self.undo_button, "Undo last edit (Ctrl+Z)")
        self._bind_status_hint(
            self.redo_button, "Redo last undone edit (Ctrl+Y)")
        self._bind_status_hint(
            self.save_button, "Save this checkpoint as a new edit session")

        # REGRESSION GUARD: Fixed-height sidebars hid the lower controls on
        # shorter displays. Keep every action inside this vertical scroller.
        self.sidebar_canvas = self.tk.Canvas(
            sidebar_host, background=DARK["background"],
            highlightthickness=0, width=640)
        sidebar_scroll = ttk.Scrollbar(
            sidebar_host, orient="vertical",
            command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")
        sidebar = ttk.Frame(self.sidebar_canvas, padding=(10, 0, 6, 0))
        self.sidebar_frame = sidebar
        sidebar_window = self.sidebar_canvas.create_window(
            0, 0, window=sidebar, anchor="nw")
        sidebar.bind(
            "<Configure>",
            lambda _event: self.sidebar_canvas.configure(
                scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind(
            "<Configure>",
            lambda event: self.sidebar_canvas.itemconfigure(
                sidebar_window, width=event.width))

        canvas_frame = ttk.Frame(viewer)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = self.tk.Canvas(
            canvas_frame, background="#171717", highlightthickness=0,
            cursor="crosshair")
        x_scroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        navigation = ttk.Frame(viewer, padding=(0, 8, 0, 0))
        navigation.pack(fill="x")
        ttk.Button(
            navigation, text="◀ Previous", command=lambda: self._step_frame(-1)
        ).pack(side="left")
        ttk.Button(
            navigation, text="Next ▶", command=lambda: self._step_frame(1)
        ).pack(side="left", padx=(6, 10))
        self.frame_scale = ttk.Scale(
            navigation, from_=1, to=len(self.labels), orient="horizontal",
            command=self._slider_changed)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(navigation, textvariable=self.frame_status, width=24).pack(
            side="left", padx=(10, 0))

        view_controls = ttk.Frame(viewer, padding=(0, 6, 0, 0))
        view_controls.pack(fill="x")
        ttk.Label(view_controls, text="Zoom:").pack(side="left")
        ttk.Button(
            view_controls, text="−", width=3,
            command=lambda: self._zoom_step(-1),
        ).pack(side="left", padx=(6, 2))
        zoom_box = ttk.Combobox(
            view_controls, textvariable=self.zoom,
            values=tuple(f"{round(value * 100)}%" for value in ZOOM_LEVELS),
            state="readonly", width=7)
        zoom_box.pack(side="left")
        ttk.Button(
            view_controls, text="+", width=3,
            command=lambda: self._zoom_step(1),
        ).pack(side="left", padx=(2, 4))
        ttk.Button(
            view_controls, text="Fit", command=self._fit_zoom,
        ).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(
            view_controls, text="Show identity numbers",
            variable=self.show_numbers, command=self._render,
        ).pack(side="left")
        ttk.Checkbutton(
            view_controls, text="Show tracks",
            variable=self.show_tracks, command=self._render,
        ).pack(side="left", padx=(10, 0))
        ttk.Label(
            view_controls,
            text=("Click to select; Ctrl-click adds a selection; "
                  "mouse wheel zooms."),
        ).pack(side="right")

        selection = ttk.LabelFrame(sidebar, text="Identity selection", padding=6)
        selection.pack(fill="both", expand=True)
        selection.columnconfigure(0, weight=1, uniform="identity_selection")
        selection.columnconfigure(1, weight=1, uniform="identity_selection")
        selection.rowconfigure(0, weight=1)

        selected_panel = ttk.LabelFrame(
            selection, text="Selected identities", padding=6)
        selected_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.selection_list = self.tk.Listbox(
            selected_panel, height=10, exportselection=False)
        self.selection_list.pack(fill="both", expand=True)
        ttk.Label(selected_panel, textvariable=self.selection_status).pack(
            anchor="w", pady=(6, 0))
        ttk.Button(
            selected_panel, text="Clear selection", command=self._clear_selection
        ).pack(anchor="w", pady=(6, 0))

        filter_panel = ttk.LabelFrame(
            selection, text="Select by filters", padding=6)
        filter_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        filter_header = ttk.Frame(filter_panel)
        filter_header.pack(fill="x")
        ttk.Label(filter_header, text="Match").pack(side="left")
        ttk.Combobox(
            filter_header, textvariable=self.filter_combine,
            values=("all", "any"), state="readonly", width=5,
        ).pack(side="left", padx=(5, 0))
        ttk.Button(
            filter_header, text="Add filter",
            command=self._add_selection_filter,
        ).pack(side="right")

        filter_columns = ttk.Frame(filter_panel)
        filter_columns.pack(fill="x", pady=(6, 2))
        for column, (title, weight) in enumerate((
                ("Metric", 3), ("Condition", 1), ("Value", 1), ("", 0))):
            ttk.Label(filter_columns, text=title).grid(
                row=0, column=column, sticky="w")
            filter_columns.columnconfigure(column, weight=weight)

        self.filter_rows_frame = ttk.Frame(filter_panel)
        self.filter_rows_frame.pack(fill="x")
        for column, weight in enumerate((3, 1, 1, 0)):
            self.filter_rows_frame.columnconfigure(column, weight=weight)
        ttk.Label(
            filter_panel, textvariable=self.filter_status, wraplength=285,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Button(
            filter_panel, text="Apply filters to selection",
            command=self._apply_selection_filters,
        ).pack(fill="x", pady=(6, 0))

        actions = ttk.LabelFrame(sidebar, text="Edit selected", padding=8)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Label(actions, text="Frame range").grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(actions, text="Start frame").grid(
            row=1, column=0, sticky="w", pady=(5, 0))
        start_field = ttk.Frame(actions)
        start_field.grid(
            row=1, column=1, sticky="ew", padx=(6, 4), pady=(5, 0))
        ttk.Entry(
            start_field, textvariable=self.range_start, width=5
        ).pack(side="left", fill="x", expand=True)
        start_current_button = ttk.Button(
            start_field, image=self.current_frame_icon,
            command=lambda: self.range_start.set(str(self.frame_index + 1)),
            style="EditorIcon.TButton", cursor="hand2")
        start_current_button.pack(side="left", padx=(3, 0))
        self._bind_status_hint(
            start_current_button, "Set start to the current frame")
        ttk.Label(actions, text="End frame").grid(
            row=1, column=2, sticky="e", pady=(5, 0))
        end_field = ttk.Frame(actions)
        end_field.grid(
            row=1, column=3, sticky="ew", padx=(4, 0), pady=(5, 0))
        ttk.Entry(
            end_field, textvariable=self.range_end, width=5
        ).pack(side="left", fill="x", expand=True)
        end_current_button = ttk.Button(
            end_field, image=self.current_frame_icon,
            command=lambda: self.range_end.set(str(self.frame_index + 1)),
            style="EditorIcon.TButton", cursor="hand2")
        end_current_button.pack(side="left", padx=(3, 0))
        self._bind_status_hint(
            end_current_button, "Set end to the current frame")
        ttk.Button(
            actions, text="Delete outlines in range",
            command=self._delete_selected_interval,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(
            actions, text="Swap pair in range", command=self._swap_selected,
        ).grid(row=2, column=2, columnspan=2, sticky="ew", padx=(4, 0),
               pady=(6, 0))
        ttk.Label(actions, text="Target identity:").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.assignment_target_box = ttk.Combobox(
            actions, textvariable=self.assignment_target,
            state="readonly", width=7)
        self.assignment_target_box.grid(
            row=3, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Button(
            actions, text="Assign selected to target",
            command=self._reassign_selected,
        ).grid(row=3, column=2, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(actions, text="Expansion radius:").grid(
            row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(actions, textvariable=self.radius, width=7).grid(
            row=4, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Combobox(
            actions, textvariable=self.radius_unit,
            values=("calibrated", "pixel"), state="readonly", width=11,
        ).grid(row=4, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(
            actions, text="Expand range", command=self._expand_selected,
        ).grid(row=4, column=3, sticky="ew", padx=(4, 0), pady=(6, 0))
        ttk.Button(
            actions, text="Force split selected…",
            command=self._force_split_selected,
        ).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Button(
            actions, text="Add new cells…",
            command=self._add_new_cells,
        ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Button(
            actions, text="Retune masks after edit…",
            command=self._retune_masks_after_edit,
        ).grid(row=7, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Separator(actions, orient="horizontal").grid(
            row=8, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        ttk.Button(
            actions, text="Remove complete selected tracks",
            command=self._remove_selected,
        ).grid(row=9, column=0, columnspan=4, sticky="ew")
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(3, weight=1)

        self.output_frame = ttk.LabelFrame(
            sidebar, text="Next session", padding=8)
        self.output_frame.pack(fill="x", pady=(10, 0))
        ttk.Entry(self.output_frame, textvariable=self.output_path).pack(fill="x")
        self.output_browse_button = ttk.Button(
            self.output_frame, text="Choose folder…", command=self._browse_output
        )
        self.output_browse_button.pack(anchor="w", pady=(6, 0))

        ttk.Label(
            outer, textvariable=self.status, relief="sunken", anchor="w",
            padding=(6, 3),
        ).pack(fill="x", pady=(8, 0))
        self._bind_sidebar_mousewheel(sidebar)

    def _bind_sidebar_mousewheel(self, widget) -> None:
        widget.bind("<MouseWheel>", self._scroll_sidebar)
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def _bind_status_hint(self, widget, message: str) -> None:
        def show(_event) -> None:
            widget.previous_status = self.status.get()
            self.status.set(message)

        def restore(_event) -> None:
            self.status.set(getattr(widget, "previous_status", ""))

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", restore)

    def _scroll_sidebar(self, event):
        units = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(units * 3, "units")
        return "break"

    def _bind(self) -> None:
        self.canvas.bind("<Button-1>", self._canvas_clicked)
        self.canvas.bind("<MouseWheel>", self._canvas_zoomed)
        self.root.bind("<Left>", lambda _event: self._step_frame(-1))
        self.root.bind("<Right>", lambda _event: self._step_frame(1))
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Control-y>", lambda _event: self._redo())
        self.root.bind("<Control-minus>", lambda _event: self._zoom_step(-1))
        self.root.bind("<Control-plus>", lambda _event: self._zoom_step(1))
        self.root.bind("<Control-equal>", lambda _event: self._zoom_step(1))
        self.root.bind("<Control-Key-0>", lambda _event: self._fit_zoom())
        self.zoom.trace_add("write", lambda *_args: self._render())

    def _zoom_factor(self) -> float:
        try:
            return max(0.1, float(self.zoom.get().rstrip("%")) / 100.0)
        except ValueError:
            return 1.0

    def _set_zoom_factor(
            self, factor: float,
            anchor: tuple[float, float] | None = None) -> None:
        old_factor = self._zoom_factor()
        if anchor is None:
            anchor = (
                max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2,
            )
        anchor_x, anchor_y = anchor
        image_x = self.canvas.canvasx(anchor_x) / old_factor
        image_y = self.canvas.canvasy(anchor_y) / old_factor
        factor = min(8.0, max(0.1, float(factor)))
        self.zoom.set(f"{round(factor * 100)}%")
        self.root.update_idletasks()
        rendered_width = max(1, round(self.labels.shape[2] * factor))
        rendered_height = max(1, round(self.labels.shape[1] * factor))
        target_x = max(0.0, image_x * factor - anchor_x)
        target_y = max(0.0, image_y * factor - anchor_y)
        self.canvas.xview_moveto(target_x / rendered_width)
        self.canvas.yview_moveto(target_y / rendered_height)
        self.status.set(f"Image zoom: {round(factor * 100)}%")

    def _zoom_step(
            self, direction: int,
            anchor: tuple[float, float] | None = None) -> None:
        current = self._zoom_factor()
        if direction > 0:
            candidates = [value for value in ZOOM_LEVELS if value > current + 1e-9]
            target = candidates[0] if candidates else ZOOM_LEVELS[-1]
        else:
            candidates = [value for value in ZOOM_LEVELS if value < current - 1e-9]
            target = candidates[-1] if candidates else ZOOM_LEVELS[0]
        self._set_zoom_factor(target, anchor)

    def _fit_zoom(self) -> None:
        available_width = max(1, self.canvas.winfo_width() - 4)
        available_height = max(1, self.canvas.winfo_height() - 4)
        factor = min(
            available_width / self.labels.shape[2],
            available_height / self.labels.shape[1],
        )
        self._set_zoom_factor(factor, (0, 0))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _canvas_zoomed(self, event):
        self._zoom_step(
            1 if event.delta > 0 else -1,
            (float(event.x), float(event.y)))
        return "break"

    def _render(self) -> None:
        from PIL import ImageTk

        frame = render_editor_frame(
            self.controller.bundle.registered_raw[self.frame_index],
            self.labels[self.frame_index], self.selected,
            self.show_numbers.get(), self.track_centroids,
            self.frame_index, self.show_tracks.get())
        scale = self._zoom_factor()
        image = Image.fromarray(frame)
        if scale != 1:
            image = image.resize(
                (max(1, round(image.width * scale)),
                 max(1, round(image.height * scale))),
                Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
        present = np.unique(self.labels[self.frame_index])
        present_count = int(np.count_nonzero(present))
        self.frame_status.set(
            f"Frame {self.frame_index + 1} / {len(self.labels)}  "
            f"{present_count} identities")

    def _refresh_all(self, message: str) -> None:
        self.frame_scale.configure(to=len(self.labels))
        self.frame_scale.set(self.frame_index + 1)
        self.source_status.set(
            f"{(self.controller.source_reference.name if self.controller.portable_source else self.controller.source_manifest.parent.name)}  ·  "
            f"checkpoint {self.controller.position} of "
            f"{len(self.controller.operations)}")
        self.output_frame.configure(
            text=("Next immutable package" if self.controller.portable_source
                  else "Next session"))
        self.output_browse_button.configure(
            text=("Choose package file…" if self.controller.portable_source
                  else "Choose folder…"))
        self.undo_button.configure(
            state="normal" if self.controller.can_undo else "disabled")
        self.redo_button.configure(
            state="normal" if self.controller.can_redo else "disabled")
        active_identities = sorted(
            int(value) for value in np.unique(self.labels) if value > 0)
        self.assignment_target_box.configure(
            values=tuple(str(identity) for identity in active_identities))
        current_target = self.assignment_target.get()
        if current_target and int(current_target) not in active_identities:
            self.assignment_target.set("")
        self._refresh_selection()
        self._render()
        self.status.set(message)

    def _refresh_selection(self) -> None:
        self.selection_list.delete(0, "end")
        for identity in sorted(self.selected):
            frames = int(np.count_nonzero(
                np.any(self.labels == identity, axis=(1, 2))))
            pixels = int(np.count_nonzero(
                self.labels[self.frame_index] == identity))
            self.selection_list.insert(
                "end", f"Identity {identity}  ·  {frames} frames  ·  "
                f"{pixels} px here")
        count = len(self.selected)
        self.selection_status.set(
            "No identities selected" if not count
            else f"{count} identit{'y' if count == 1 else 'ies'} selected")

    def _slider_changed(self, value: str) -> None:
        index = min(len(self.labels) - 1, max(0, round(float(value)) - 1))
        if index != self.frame_index:
            self.frame_index = index
            self._refresh_selection()
            self._render()

    def _step_frame(self, step: int) -> None:
        index = min(len(self.labels) - 1, max(0, self.frame_index + step))
        if index != self.frame_index:
            self.frame_index = index
            self.frame_scale.set(index + 1)
            self._refresh_selection()
            self._render()

    def _canvas_clicked(self, event) -> None:
        scale = self._zoom_factor()
        x = int(self.canvas.canvasx(event.x) / scale)
        y = int(self.canvas.canvasy(event.y) / scale)
        identity = identity_at(self.labels[self.frame_index], y, x)
        additive = bool(event.state & 0x0004)
        if not additive:
            self.selected.clear()
        if identity > 0:
            if additive and identity in self.selected:
                self.selected.remove(identity)
            else:
                self.selected.add(identity)
            self.status.set(
                f"Identity {identity} selected at image coordinate x={x}, y={y}")
        else:
            self.status.set(f"Background at image coordinate x={x}, y={y}")
        self._refresh_selection()
        self._render()

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._refresh_selection()
        self._render()
        self.status.set("Selection cleared")

    def _add_selection_filter(self) -> None:
        metric = self.tk.StringVar(
            value=FILTER_METRIC_LABELS["observed_frames"])
        operator = self.tk.StringVar(value="<")
        value = self.tk.StringVar(value=str(len(self.labels)))
        row: dict = {
            "metric": metric, "operator": operator, "value": value,
            "widgets": [],
        }
        row_index = len(self.selection_filter_rows)
        metric_box = self.ttk.Combobox(
            self.filter_rows_frame, textvariable=metric,
            values=tuple(sorted(FILTER_METRIC_KEYS)), state="readonly",
            width=16)
        metric_box.grid(
            row=row_index, column=0, sticky="ew", padx=(0, 3), pady=2)
        operator_box = self.ttk.Combobox(
            self.filter_rows_frame, textvariable=operator,
            values=tuple(FILTER_OPERATOR_KEYS), state="readonly", width=7)
        operator_box.grid(
            row=row_index, column=1, sticky="ew", padx=(0, 3), pady=2)
        value_entry = self.ttk.Entry(
            self.filter_rows_frame, textvariable=value, width=8)
        value_entry.grid(
            row=row_index, column=2, sticky="ew", padx=(0, 3), pady=2)
        remove_button = self.ttk.Button(
            self.filter_rows_frame, text="×", width=2,
            command=lambda: self._remove_selection_filter(row),
            style="EditorIcon.TButton")
        remove_button.grid(row=row_index, column=3, pady=2)
        row["widgets"] = [metric_box, operator_box, value_entry, remove_button]
        self.selection_filter_rows.append(row)

        def describe_metric(_event=None) -> None:
            key = FILTER_METRIC_KEYS[metric.get()]
            description = FILTERABLE_METRICS[key]
            if key in PERCENT_FILTER_METRICS:
                description += "; enter 10 for 10%"
            self.filter_status.set(description)

        def adjust_value_for_operator(*_args) -> None:
            raw = value.get().strip()
            ranged = operator.get() in {"between", "outside"}
            if ranged and "," not in raw:
                value.set(f"{raw}, {raw}")
            elif not ranged and "," in raw:
                value.set(raw.split(",", 1)[0].strip())

        metric_box.bind("<<ComboboxSelected>>", describe_metric)
        operator.trace_add("write", adjust_value_for_operator)
        self._bind_sidebar_mousewheel(self.filter_rows_frame)
        describe_metric()

    def _remove_selection_filter(self, row: dict) -> None:
        if row not in self.selection_filter_rows:
            return
        for widget in row["widgets"]:
            widget.destroy()
        self.selection_filter_rows.remove(row)
        for row_index, remaining in enumerate(self.selection_filter_rows):
            for widget in remaining["widgets"]:
                widget.grid_configure(row=row_index)
        count = len(self.selection_filter_rows)
        self.filter_status.set(
            "No filters added" if not count
            else f"{count} filter{'s' if count != 1 else ''} ready")

    def _selection_filter_spec(self) -> dict:
        rows = []
        for row in self.selection_filter_rows:
            metric = FILTER_METRIC_KEYS[row["metric"].get()]
            operator = FILTER_OPERATOR_KEYS[row["operator"].get()]
            rows.append((metric, operator, row["value"].get()))
        return build_selection_filter_spec(rows, self.filter_combine.get())

    def _apply_selection_filters(self) -> None:
        from tkinter import messagebox

        try:
            spec = self._selection_filter_spec()
            self.root.configure(cursor="watch")
            self.status.set("Calculating track filters…")
            self.root.update_idletasks()
            metrics, _metadata = build_track_filter_metrics(
                self.controller.source_manifest, self.controller.position)
            matches = evaluate_track_filters(metrics, spec)
            self.selected = set(matches.loc[
                matches.selected, "identity"].astype(int).tolist())
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot apply filters", str(error), parent=self.root)
            return
        finally:
            self.root.configure(cursor="")

        matched = len(self.selected)
        total = len(matches)
        self.filter_status.set(
            f"{matched} of {total} active identities matched")
        self._refresh_selection()
        self._render()
        self.status.set(
            f"Filters selected {matched} of {total} active identities")

    def _target_path(self) -> Path:
        raw = self.output_path.get().strip()
        if not raw:
            raise ValueError("choose a new session folder")
        return Path(raw)

    def _confirm(self, title: str, message: str) -> bool:
        from tkinter import messagebox
        return bool(messagebox.askyesno(title, message, parent=self.root))

    def _run_saved_action(
            self, action: Callable[[Path], Path], progress: str) -> None:
        from tkinter import messagebox

        try:
            target = self._target_path()
            self.root.configure(cursor="watch")
            self.root.update_idletasks()
            saved = action(target)
            self._load_saved_source(saved)
            self._refresh_all(
                f"{progress}; validated session saved to {saved.parent}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot save edit", str(error), parent=self.root)
        finally:
            self.root.configure(cursor="")

    def _load_saved_source(self, path: Path) -> None:
        self.controller = EditingHistoryController(path)
        self.labels, self.provenance, self.excluded = \
            self.controller.materialize()
        self.track_centroids = compute_track_centroids(self.labels)
        self.frame_index = min(self.frame_index, len(self.labels) - 1)
        active = set(int(value) for value in np.unique(self.labels) if value > 0)
        self.selected.intersection_update(active)
        self.output_path.set(str(_default_output_path(self.controller)))

    def _remove_selected(self) -> None:
        if not self.selected:
            self.status.set("Select at least one identity before removing tracks")
            return
        identities = sorted(self.selected)
        if not self._confirm(
                "Remove selected tracks",
                "Remove complete tracks for identities "
                f"{', '.join(map(str, identities))}?\n\n"
                "The operation is reversible from history."):
            return
        self._run_saved_action(
            lambda target: self.controller.remove_identities(identities, target),
            f"Removed {len(identities)} selected track(s)")

    def _selected_frame_range(self) -> tuple[int, int] | None:
        try:
            return int(self.range_start.get()), int(self.range_end.get())
        except ValueError:
            self.status.set("Start and end frames must be whole numbers")
            return None

    def _delete_selected_interval(self) -> None:
        if not self.selected:
            self.status.set(
                "Select at least one identity before deleting a frame range")
            return
        frame_range = self._selected_frame_range()
        if frame_range is None:
            return
        start, end = frame_range
        identities = sorted(self.selected)
        if not self._confirm(
                "Delete selected identities in frame range",
                f"Delete identities {', '.join(map(str, identities))} from "
                f"ImageJ frame {start} through {end}, inclusive?\n\n"
                "Their outlines before and after this interval remain intact."):
            return
        self._run_saved_action(
            lambda target: self.controller.delete_identities_between_frames(
                identities, start, end, target),
            f"Deleted {len(identities)} selected identity outline(s) from "
            f"frames {start}–{end}")

    def _reassign_selected(self) -> None:
        if not self.selected:
            self.status.set(
                "Select at least one source identity before assigning it")
            return
        raw_target = self.assignment_target.get().strip()
        if not raw_target:
            self.status.set("Choose a target identity before assigning")
            return
        target_identity = int(raw_target)
        source_identities = sorted(self.selected - {target_identity})
        if not source_identities:
            self.status.set(
                "Select at least one source identity different from the target")
            return
        frame_range = self._selected_frame_range()
        if frame_range is None:
            return
        start, end = frame_range
        joined = ", ".join(map(str, source_identities))
        if not self._confirm(
                "Assign selected identities to target",
                f"Relabel identities {joined} as identity {target_identity} from "
                f"ImageJ frame {start} through {end}, inclusive?\n\n"
                "Existing target outlines will remain and combine with them."):
            return
        self._run_saved_action(
            lambda output: self.controller.reassign_identities(
                source_identities, target_identity, start, end, output),
            f"Assigned identities {joined} to identity {target_identity} from "
            f"frames {start}–{end}")

    def _expand_selected(self) -> None:
        if not self.selected:
            self.status.set("Select at least one identity before expanding outlines")
            return
        frame_range = self._selected_frame_range()
        if frame_range is None:
            return
        start, end = frame_range
        try:
            radius = float(self.radius.get())
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError
        except ValueError:
            self.status.set("Expansion radius must be a positive number")
            return
        identities = sorted(self.selected)
        unit = self.radius_unit.get()
        if not self._confirm(
                "Expand selected outlines",
                f"Expand identities {', '.join(map(str, identities))} by "
                f"{radius:g} {unit} from ImageJ frame {start} through {end}, "
                "inclusive?\n\nExisting identities will not be overwritten."):
            return
        self._run_saved_action(
            lambda target: self.controller.expand_identities(
                identities, radius, target, unit, start, end),
            f"Expanded {len(identities)} selected identity outline(s) from "
            f"frames {start}–{end}")

    def _swap_selected(self) -> None:
        if len(self.selected) != 2:
            self.status.set("Select exactly two identities before swapping them")
            return
        frame_range = self._selected_frame_range()
        if frame_range is None:
            return
        start, end = frame_range
        first, second = sorted(self.selected)
        if not self._confirm(
                "Swap selected identities",
                f"Swap identities {first} and {second} from ImageJ frame "
                f"{start} through {end}, inclusive?"):
            return
        self._run_saved_action(
            lambda target: self.controller.swap_identities(
                first, second, start, target, end),
            f"Swapped identities {first} and {second}")

    def _force_split_selected(self) -> None:
        if not self.selected:
            self.status.set("Select at least one host identity before splitting")
            return
        frame_range = self._selected_frame_range()
        if frame_range is None:
            return
        start, end = frame_range
        if start < 1 or end < start or end > len(self.labels):
            self.status.set(
                f"Split frame range must be within 1–{len(self.labels)}")
            return
        absent = [
            identity for identity in sorted(self.selected)
            if any(not np.any(self.labels[frame - 1] == identity)
                   for frame in range(start, end + 1))
        ]
        if absent:
            self.status.set(
                "Each split host must exist in every selected frame; absent: "
                + ", ".join(map(str, absent)))
            return
        try:
            target = self._target_path()
            from manual_splitting_ui import open_force_split_workspace

            open_force_split_workspace(
                self.root, self.controller, self.labels,
                sorted(self.selected), start, end, target,
                self._forced_split_saved)
        except (OSError, ValueError) as error:
            from tkinter import messagebox
            messagebox.showerror(
                "Cannot open forced splitting", str(error), parent=self.root)
            return
        self.status.set(
            f"Opened {len(self.selected)} forced-split job(s) for frames "
            f"{start}–{end}")

    def _forced_split_saved(self, path: Path) -> None:
        self._load_saved_source(path)
        self._refresh_all(
            f"Forced splits saved and reimported from {path.parent}")

    def _add_new_cells(self) -> None:
        try:
            target = self._target_path()
            from manual_new_cells_ui import open_new_cell_workspace

            open_new_cell_workspace(
                self.root, self.controller, self.labels, target,
                self._new_cells_saved)
        except (OSError, ValueError) as error:
            from tkinter import messagebox
            messagebox.showerror(
                "Cannot open new-cell workspace", str(error), parent=self.root)
            return
        self.status.set(
            "Opened the new-cell workspace; click missed-cell centroids to add jobs")

    def _new_cells_saved(self, path: Path) -> None:
        self._load_saved_source(path)
        self._refresh_all(
            f"New identities saved and reimported from {path.parent}")

    def _retune_masks_after_edit(self) -> None:
        try:
            target = self._target_path()
            from manual_retuning_ui import open_mask_retuning_workspace

            open_mask_retuning_workspace(
                self.root, self.controller, self.labels,
                sorted(self.selected), target, self._retuning_saved)
        except (OSError, ValueError) as error:
            from tkinter import messagebox
            messagebox.showerror(
                "Cannot open mask retuning", str(error), parent=self.root)
            return
        self.status.set(
            "Opened mask retuning; choose a deletion and recipient identities")

    def _retuning_saved(self, path: Path) -> None:
        self._load_saved_source(path)
        self._refresh_all(
            f"Mask retuning saved and reimported from {path.parent}")

    def _undo(self) -> None:
        if not self.controller.can_undo:
            return
        checkpoint = self.controller.undo()
        self.labels, self.provenance, self.excluded = \
            self.controller.materialize()
        self.track_centroids = compute_track_centroids(self.labels)
        self.output_path.set(str(_default_output_path(self.controller)))
        self._refresh_all(f"Restored: {checkpoint.action}")

    def _redo(self) -> None:
        if not self.controller.can_redo:
            return
        checkpoint = self.controller.redo()
        self.labels, self.provenance, self.excluded = \
            self.controller.materialize()
        self.track_centroids = compute_track_centroids(self.labels)
        self.output_path.set(str(_default_output_path(self.controller)))
        self._refresh_all(f"Restored: {checkpoint.action}")

    def _save_checkpoint(self) -> None:
        if self.controller.position == len(self.controller.operations):
            self.status.set("This checkpoint is already the saved session head")
            return
        self._run_saved_action(
            lambda target: self.controller.continue_from_here(target),
            f"Saved checkpoint {self.controller.position}")

    def _browse_output(self) -> None:
        from tkinter import filedialog

        if self.controller.portable_source:
            selected = filedialog.asksaveasfilename(
                title="Choose the next immutable Motion package",
                initialdir=str(self.controller.source_reference.parent),
                initialfile=_default_output_path(self.controller).name,
                defaultextension=".motionpkg",
                filetypes=(("Motion editing package", "*.motionpkg"),
                           ("All files", "*.*")))
            if selected:
                self.output_path.set(str(Path(selected)))
            return
        parent = filedialog.askdirectory(
            title="Choose where the next session folder will be created",
            initialdir=str(self.controller.source_manifest.parent.parent))
        if parent:
            self.output_path.set(str(Path(parent) / _default_output_path(
                self.controller).name))

    def _freeze_package(self) -> None:
        from tkinter import filedialog, messagebox

        initial_source = self.controller.source_reference
        initial_name = (
            initial_source.stem if initial_source.is_file()
            else initial_source.name)
        selected = filedialog.asksaveasfilename(
            title="Freeze the active checkpoint as a portable package",
            initialdir=str(initial_source.parent),
            initialfile=f"{initial_name}_checkpoint_{self.controller.position}.motionpkg",
            defaultextension=".motionpkg",
            filetypes=(("Motion editing package", "*.motionpkg"),
                       ("All files", "*.*")))
        if not selected:
            return
        target = Path(selected)
        if not messagebox.askyesno(
                "Freeze portable package",
                "Copy the canonical labels, aligned raw and motion evidence, complete "
                "edit history, and software snapshot into one immutable package?\n\n"
                "This can be a large file.", parent=self.root):
            return
        try:
            from manual_portable_package import freeze_portable_package

            self.root.configure(cursor="watch")
            self.status.set("Freezing and replay-checking the portable package…")
            self.root.update_idletasks()
            path = freeze_portable_package(
                self.controller.source_reference, target,
                operation_count=self.controller.position)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot freeze portable package", str(error), parent=self.root)
            return
        finally:
            self.root.configure(cursor="")
        self.status.set(f"Portable package verified and saved to {path}")

    def _import_package(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askopenfilename(
            title="Import a portable Motion editing package",
            filetypes=(("Motion editing package", "*.motionpkg"),
                       ("All files", "*.*")))
        if not selected:
            return
        try:
            self.root.configure(cursor="watch")
            self.status.set("Verifying and importing the portable package…")
            self.root.update_idletasks()
            self._load_saved_source(Path(selected))
            self.selected.clear()
            self._refresh_all(
                f"Portable package verified and imported from {selected}")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot import portable package", str(error), parent=self.root)
        finally:
            self.root.configure(cursor="")

    def _apply_batch_file(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose a Motion edit-batch JSON file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if not selected:
            return
        try:
            operations = load_edit_batch(Path(selected))
        except (OSError, ValueError) as error:
            from tkinter import messagebox
            messagebox.showerror(
                "Cannot load edit batch", str(error), parent=self.root)
            return
        if not self._confirm(
                "Apply edit batch",
                f"Apply all {len(operations)} operation(s) from "
                f"{Path(selected).name}?\n\nThe batch writes only if every operation succeeds."):
            return
        self._run_saved_action(
            lambda target: self.controller.apply_batch(operations, target),
            f"Applied {len(operations)} batch operation(s)")

    def _open_history(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        window = tk.Toplevel(self.root)
        window.title("Edit history")
        window.geometry("900x420")
        window.transient(self.root)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        columns = ("action", "scope", "user", "pixels")
        tree = ttk.Treeview(
            frame, columns=columns, show="tree headings", selectmode="browse")
        tree.heading("#0", text="Step")
        for column, title in (
                ("action", "Edit"), ("scope", "Scope"),
                ("user", "User"), ("pixels", "Changed pixels")):
            tree.heading(column, text=title)
        tree.column("#0", width=55, stretch=False)
        tree.column("action", width=300)
        tree.column("scope", width=180)
        tree.column("user", width=120)
        tree.column("pixels", width=110, anchor="e")
        for checkpoint in self.controller.checkpoints:
            tree.insert(
                "", "end", iid=str(checkpoint.operation_count),
                text=str(checkpoint.operation_count), values=(
                    checkpoint.action, checkpoint.scope, checkpoint.user,
                    f"{checkpoint.changed_pixels:,}"))
        tree.selection_set(str(self.controller.position))
        tree.pack(fill="both", expand=True)

        def restore() -> None:
            selected = tree.selection()
            if not selected:
                return
            checkpoint = self.controller.restore(int(selected[0]))
            self.labels, self.provenance, self.excluded = \
                self.controller.materialize()
            self.track_centroids = compute_track_centroids(self.labels)
            self.output_path.set(str(_default_output_path(self.controller)))
            self._refresh_all(f"Restored: {checkpoint.action}")
            window.destroy()

        ttk.Button(
            frame, text="Restore selected", command=restore
        ).pack(side="left", pady=(8, 0))
        ttk.Button(
            frame, text="Close", command=window.destroy
        ).pack(side="right", pady=(8, 0))
        tree.bind("<Double-1>", lambda _event: restore())


def launch_manual_editor(source_path: Path) -> Path:
    """Open the manual editing workspace and return its final source manifest."""
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "the manual editor requires a graphical desktop session") from error
    app = ManualEditingApp(root, source_path)
    root.mainloop()
    return app.controller.source_manifest
