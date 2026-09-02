from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from manual_editing import EditingHistoryController
from manual_editing_theme import DARK, apply_dark_theme
from manual_editing_ui import (compute_track_centroids, render_editor_frame)
from manual_new_cells import (NEW_CELL_METHODS, NewCellProposal,
                              NewCellRequest, allocate_new_identity,
                              apply_add_identity_operation,
                              build_add_identity_operation,
                              default_crop_side_px, editing_state_sha256,
                              interpolate_manual_outlines,
                              load_new_cell_draft, paint_new_identity,
                              proposal_from_draft_job,
                              proposal_to_draft_job, propose_new_identity,
                              promote_labels_for_identity,
                              revalidate_new_identity_proposal,
                              save_new_cell_draft)


METHOD_LABELS = {
    "Soma-seed expansion": "soma_seed_expansion",
    "Fixed crop threshold": "fixed_local_threshold",
    "Automatically adjusting threshold": "adaptive_local_threshold",
    "Manual outline": "manual_outline",
}
METHOD_KEYS = {value: key for key, value in METHOD_LABELS.items()}
ZOOM_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def proposal_warning_frames(proposal: NewCellProposal) -> set[int]:
    result: set[int] = set()
    for warning in proposal.warnings:
        prefix = "ImageJ frame "
        if warning.startswith(prefix):
            token = warning[len(prefix):].split(":", 1)[0]
            try:
                result.add(int(token))
            except ValueError:
                pass
    return result


def batch_collision_frames(jobs: list[dict[str, Any]]) -> dict[int, set[int]]:
    """Return frame collisions between proposed new-cell jobs."""
    collisions: dict[int, set[int]] = {}
    for left_index, left in enumerate(jobs):
        left_proposal = left.get("proposal")
        if left_proposal is None:
            continue
        for right_index in range(left_index + 1, len(jobs)):
            right = jobs[right_index]
            right_proposal = right.get("proposal")
            if right_proposal is None:
                continue
            common_start = max(
                left_proposal.request.start_imagej_frame,
                right_proposal.request.start_imagej_frame)
            common_end = min(
                left_proposal.request.end_imagej_frame,
                right_proposal.request.end_imagej_frame)
            for imagej_frame in range(common_start, common_end + 1):
                left_mask = left_proposal.masks[
                    imagej_frame - left_proposal.request.start_imagej_frame]
                right_mask = right_proposal.masks[
                    imagej_frame - right_proposal.request.start_imagej_frame]
                if np.any(left_mask & right_mask):
                    collisions.setdefault(left_index, set()).add(imagej_frame)
                    collisions.setdefault(right_index, set()).add(imagej_frame)
    return collisions


def new_cell_commit_summary(jobs: list[dict[str, Any]]) -> str:
    lines = []
    for job in jobs:
        proposal = job.get("proposal")
        changed = 0 if proposal is None else int(np.count_nonzero(proposal.masks))
        warning_frames = (set() if proposal is None
                          else proposal_warning_frames(proposal))
        lines.append(
            f"Identity {job['identity']}: frames {job['start']}-{job['end']}, "
            f"{job['method'].replace('_', ' ')}, {len(job['anchors'])} anchor(s), "
            f"{len(job['manual_masks'])} manually corrected frame(s), "
            f"{len(warning_frames)} warning frame(s), "
            f"{len(job['linked_frames'])} linked split frame(s), "
            f"{changed} proposed pixel(s)")
    return "\n".join(lines)


def _outline(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))


def extract_padded_moving_crop(
        image: np.ndarray, centre: tuple[float, float], side: int
        ) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a fixed-size display crop and its possibly negative image origin."""
    side = max(1, int(side))
    top = int(round(float(centre[0]) - (side - 1) / 2.0))
    left = int(round(float(centre[1]) - (side - 1) / 2.0))
    bottom, right = top + side, left + side
    source_y0, source_y1 = max(0, top), min(image.shape[0], bottom)
    source_x0, source_x1 = max(0, left), min(image.shape[1], right)
    result = np.zeros((side, side, *image.shape[2:]), dtype=image.dtype)
    target_y0, target_x0 = source_y0 - top, source_x0 - left
    if source_y1 > source_y0 and source_x1 > source_x0:
        result[
            target_y0:target_y0 + source_y1 - source_y0,
            target_x0:target_x0 + source_x1 - source_x0,
        ] = image[source_y0:source_y1, source_x0:source_x1]
    return result, (top, left)


def render_new_cell_full_field(
        raw: np.ndarray, labels: np.ndarray,
        proposal_mask: np.ndarray | None,
        crop_bounds: list[int] | tuple[int, int, int, int] | None,
        anchors: list[tuple[int, int]], identity: int | None,
        show_numbers: bool, show_tracks: bool,
        track_centroids: dict[int, list[tuple[int, float, float]]],
        frame_index: int, polygon_points: list[tuple[int, int]] | None = None,
        crop_centres: list[dict[str, Any]] | None = None,
        show_crop_path: bool = False
        ) -> np.ndarray:
    image = render_editor_frame(
        raw, labels, set(), show_numbers, track_centroids,
        frame_index, show_tracks).copy()
    if proposal_mask is not None and np.any(proposal_mask):
        edge = _outline(np.asarray(proposal_mask, bool))
        image[edge] = np.array([40, 235, 220], np.uint8)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    if show_crop_path and crop_centres:
        path = [(float(row["x"]), float(row["y"]))
                for row in crop_centres
                if row.get("x") is not None and row.get("y") is not None]
        if len(path) > 1:
            draw.line(path, fill=(255, 176, 54), width=1)
    if crop_bounds is not None:
        y0, y1, x0, x1 = map(int, crop_bounds)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 176, 54), width=1)
        margin_y = max(1, int(round((y1 - y0) * 0.15)))
        margin_x = max(1, int(round((x1 - x0) * 0.15)))
        if y1 - y0 > 2 * margin_y and x1 - x0 > 2 * margin_x:
            draw.rectangle(
                (x0 + margin_x, y0 + margin_y,
                 x1 - margin_x - 1, y1 - margin_y - 1),
                outline=(155, 111, 42), width=1)
    for y, x in anchors:
        draw.line((x - 4, y, x + 4, y), fill=(255, 255, 255), width=1)
        draw.line((x, y - 4, x, y + 4), fill=(255, 255, 255), width=1)
    points = polygon_points or []
    if points:
        xy = [(x, y) for y, x in points]
        if len(xy) > 1:
            draw.line(xy, fill=(255, 85, 190), width=1)
        for x, y in xy:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=(255, 85, 190))
    if identity is not None and proposal_mask is not None and np.any(proposal_mask):
        yy, xx = np.nonzero(proposal_mask)
        draw.text((int(xx.mean()), int(yy.mean())), str(identity), anchor="mm",
                  fill=(0, 0, 0), stroke_width=2, stroke_fill=(40, 235, 220))
    return np.asarray(pil)


class NewCellWorkspace:
    """Batch moving-crop, segmentation, correction, and review workspace."""

    def __init__(
            self, parent, controller: EditingHistoryController,
            labels: np.ndarray, output_dir: Path,
            on_saved: Callable[[Path], None],
            initial_anchor: tuple[int, int, int] | None = None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.controller = controller
        self.source_labels = np.asarray(labels).copy()
        self.excluded_identities = set(controller.materialize()[2])
        self.raw = controller.bundle.registered_raw
        self.unclaimed = controller.bundle.unclaimed_labels
        self.raw_frame_indices = controller.bundle.registered_raw_indices
        self.lag = None
        lag_path = controller.bundle.evidence_paths.get("lag_float")
        if lag_path is not None:
            source = np.asarray(tifffile.imread(lag_path))
            indices = np.asarray(controller.bundle.manifest["assets"][
                "lag_float"]["label_frame_to_source"], dtype=np.intp)
            self.lag = source[indices]
            if self.lag.shape != self.source_labels.shape:
                raise ValueError(
                    "aligned lag evidence and current labels differ in shape")
        self.track_centroids = compute_track_centroids(self.source_labels)
        self.output_dir = Path(output_dir)
        self.on_saved = on_saved
        self.parent_fingerprint = editing_state_sha256(self.source_labels)

        self.window = tk.Toplevel(parent)
        self.window.title("Add new cells")
        self.window.geometry("1450x900")
        self.window.minsize(1080, 680)
        self.window.transient(parent)
        apply_dark_theme(self.window)

        self.jobs: list[dict[str, Any]] = []
        self.job_index = -1
        self.frame = int(initial_anchor[0]) if initial_anchor else 1
        self.zoom = tk.StringVar(value="200%")
        self.view_mode = tk.StringVar(value="Full field")
        self.tool = tk.StringVar(value="Add cell anchor")
        self.method = tk.StringVar(value="Soma-seed expansion")
        calibration = controller.bundle.manifest["spatial_calibration"]
        self.spatial_unit = str(calibration["unit"])
        self.pixel_size_y = float(calibration["pixel_size_y"])
        self.pixel_size_x = float(calibration["pixel_size_x"])
        self.crop_unit_options = (("px", self.spatial_unit)
                                  if self.spatial_unit.lower() != "pixel"
                                  else ("px",))
        self.crop_unit = tk.StringVar(value="px")
        self._last_crop_unit = "px"
        self.crop_side = tk.StringVar(value=str(default_crop_side_px(
            self.source_labels)))
        self.threshold = tk.StringVar(value="2.0")
        self.smoothing = tk.StringVar(value="0.55")
        self.minimum_area = tk.StringVar(value="4")
        self.start_frame = tk.StringVar(value="1")
        self.end_frame = tk.StringVar(value=str(len(self.source_labels)))
        self.start_reason = tk.StringVar(value="movie_start")
        self.end_reason = tk.StringVar(value="movie_end")
        self.brush_radius = tk.IntVar(value=2)
        self.show_numbers = tk.BooleanVar(value=True)
        self.show_tracks = tk.BooleanVar(value=False)
        self.show_crop_path = tk.BooleanVar(value=True)
        self.review_start = tk.StringVar(value="1")
        self.review_end = tk.StringVar(value=str(len(self.source_labels)))
        self.status = tk.StringVar(value=(
            "Click a missed-cell centroid in the full-field view"))
        self.frame_status = tk.StringVar()
        self.job_status = tk.StringVar(value="No new-cell jobs")
        self.review_status = tk.StringVar(value="No proposal")
        self.timeline_status = tk.StringVar(value="")
        self.threshold_trace_status = tk.StringVar(value="")
        self.crop_calibration_status = tk.StringVar(value="")
        default_side = int(self.crop_side.get())
        self.crop_calibration_status.set(
            f"{default_side} px = {default_side * self.pixel_size_y:.5g} by "
            f"{default_side * self.pixel_size_x:.5g} {self.spatial_unit}")
        self.photo = None
        self._painting = False
        self._playing = False
        self._polygon_points: list[tuple[int, int]] = []
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []

        self._build()
        self._bind()
        if initial_anchor is not None:
            self._add_job(initial_anchor[0], initial_anchor[1], initial_anchor[2])
        else:
            self._refresh_all()

    @property
    def current_job(self) -> dict[str, Any] | None:
        return self.jobs[self.job_index] if 0 <= self.job_index < len(self.jobs) \
            else None

    def _local_snapshot(self) -> dict[str, Any]:
        return {
            "jobs": copy.deepcopy(self.jobs),
            "job_index": int(self.job_index),
            "frame": int(self.frame),
            "polygon_points": list(self._polygon_points),
        }

    def _record_local_state(self) -> None:
        self._undo_stack.append(self._local_snapshot())
        if len(self._undo_stack) > 80:
            del self._undo_stack[0]
        self._redo_stack.clear()

    def _restore_local_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.jobs = snapshot["jobs"]
        self.job_index = min(
            int(snapshot["job_index"]), len(self.jobs) - 1)
        self.frame = int(snapshot["frame"])
        if self.current_job is not None:
            self._load_job(self.job_index)
        else:
            self._refresh_all()
        self._polygon_points = list(snapshot["polygon_points"])
        self._refresh_all()

    def _local_undo(self) -> None:
        if not self._undo_stack:
            self.status.set("No local new-cell action to undo")
            return
        self._redo_stack.append(self._local_snapshot())
        self._restore_local_snapshot(self._undo_stack.pop())
        self.status.set("Undid the last local new-cell action")

    def _local_redo(self) -> None:
        if not self._redo_stack:
            self.status.set("No local new-cell action to redo")
            return
        self._undo_stack.append(self._local_snapshot())
        self._restore_local_snapshot(self._redo_stack.pop())
        self.status.set("Redid the last local new-cell action")

    def _build(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="Add new cells",
                  font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Label(
            header,
            text="Moving crops and outlines remain proposals until Commit",
        ).pack(side="left", padx=(16, 0))
        ttk.Button(header, text="Load draft...", command=self._load_draft).pack(
            side="right")
        ttk.Button(header, text="Save draft...", command=self._save_draft).pack(
            side="right", padx=(0, 6))
        ttk.Button(header, text="Redo", command=self._local_redo).pack(
            side="right", padx=(0, 6))
        ttk.Button(header, text="Undo", command=self._local_undo).pack(
            side="right", padx=(0, 6))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 0))
        viewer = ttk.Frame(body)
        side_host = ttk.Frame(body)
        body.add(viewer, weight=3)
        body.add(side_host, weight=2)

        canvas_frame = ttk.Frame(viewer)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = self.tk.Canvas(
            canvas_frame, background=DARK["field"], cursor="crosshair",
            highlightthickness=0)
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
        ttk.Button(navigation, text="Previous issue",
                   command=lambda: self._step_issue(-1)).pack(side="left")
        ttk.Button(navigation, text="Previous",
                   command=lambda: self._step_frame(-1)).pack(
            side="left", padx=(5, 0))
        self.play_button = ttk.Button(
            navigation, text="Play", command=self._toggle_play)
        self.play_button.pack(side="left", padx=(5, 0))
        ttk.Button(navigation, text="Next",
                   command=lambda: self._step_frame(1)).pack(
            side="left", padx=(5, 0))
        ttk.Button(navigation, text="Next issue",
                   command=lambda: self._step_issue(1)).pack(
            side="left", padx=(5, 10))
        self.frame_scale = ttk.Scale(
            navigation, from_=1, to=len(self.source_labels),
            orient="horizontal", command=self._slider_changed)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(navigation, textvariable=self.frame_status, width=18).pack(
            side="left", padx=(8, 0))

        view = ttk.Frame(viewer, padding=(0, 6, 0, 0))
        view.pack(fill="x")
        ttk.Label(view, text="Zoom").pack(side="left")
        ttk.Button(view, text="-", width=3,
                   command=lambda: self._zoom_step(-1)).pack(
            side="left", padx=(5, 2))
        ttk.Combobox(
            view, textvariable=self.zoom,
            values=tuple(f"{int(value * 100)}%" for value in ZOOM_LEVELS),
            state="readonly", width=7).pack(side="left")
        ttk.Button(view, text="+", width=3,
                   command=lambda: self._zoom_step(1)).pack(
            side="left", padx=(2, 8))
        ttk.Combobox(
            view, textvariable=self.view_mode,
            values=("Full field", "Moving crop"), state="readonly",
            width=13).pack(side="left")
        ttk.Checkbutton(view, text="Numbers", variable=self.show_numbers,
                        command=self._render).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(view, text="Tracks", variable=self.show_tracks,
                        command=self._render).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(view, text="Crop path", variable=self.show_crop_path,
                        command=self._render).pack(side="left", padx=(6, 0))
        ttk.Label(
            view, text="cyan: proposed outline | orange: crop | white: anchors",
        ).pack(side="right")
        ttk.Label(viewer, textvariable=self.timeline_status,
                  wraplength=900).pack(fill="x", pady=(5, 0))
        ttk.Label(viewer, textvariable=self.threshold_trace_status,
                  wraplength=900).pack(fill="x", pady=(2, 0))

        self.side_canvas = self.tk.Canvas(
            side_host, background=DARK["background"], highlightthickness=0,
            width=520)
        side_scroll = ttk.Scrollbar(
            side_host, orient="vertical", command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        self.side_canvas.pack(side="left", fill="both", expand=True)
        side_scroll.pack(side="right", fill="y")
        side = ttk.Frame(self.side_canvas, padding=(10, 0, 5, 0))
        side_window = self.side_canvas.create_window(0, 0, window=side, anchor="nw")
        side.bind("<Configure>", lambda _event: self.side_canvas.configure(
            scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind("<Configure>", lambda event:
            self.side_canvas.itemconfigure(side_window, width=event.width))

        jobs = ttk.LabelFrame(side, text="New-cell jobs", padding=8)
        jobs.pack(fill="x")
        self.job_list = self.tk.Listbox(jobs, height=6, exportselection=False)
        self.job_list.pack(fill="x")
        self.job_list.bind("<<ListboxSelect>>", self._job_selected)
        job_buttons = ttk.Frame(jobs)
        job_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(job_buttons, text="Add cell (click)",
                   command=self._begin_add_job).pack(side="left")
        ttk.Button(job_buttons, text="Remove job",
                   command=self._remove_job).pack(side="left", padx=(5, 0))
        ttk.Label(jobs, textvariable=self.job_status).pack(anchor="w", pady=(5, 0))

        config = ttk.LabelFrame(side, text="Configure current job", padding=8)
        config.pack(fill="x", pady=(8, 0))
        ttk.Label(config, text="Method").grid(row=0, column=0, sticky="w")
        method_box = ttk.Combobox(
            config, textvariable=self.method, values=tuple(METHOD_LABELS),
            state="readonly")
        method_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(6, 0))
        ttk.Label(config, text="Crop side").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        crop_input = ttk.Frame(config)
        crop_input.grid(row=1, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Entry(crop_input, textvariable=self.crop_side, width=7).pack(
            side="left", fill="x", expand=True)
        crop_units = ttk.Combobox(
            crop_input, textvariable=self.crop_unit,
            values=self.crop_unit_options, state="readonly", width=7)
        crop_units.pack(side="left", padx=(4, 0))
        crop_units.bind("<<ComboboxSelected>>", self._crop_unit_changed)
        ttk.Label(config, text="Threshold (z)").grid(
            row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Entry(config, textvariable=self.threshold, width=8).grid(
            row=1, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(config, text="Smoothing").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(config, textvariable=self.smoothing, width=8).grid(
            row=2, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Label(config, text="Minimum area (px)").grid(
            row=2, column=2, sticky="e", pady=(6, 0))
        ttk.Entry(config, textvariable=self.minimum_area, width=8).grid(
            row=2, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(config, text="Start").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(config, textvariable=self.start_frame, width=7).grid(
            row=3, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Combobox(config, textvariable=self.start_reason,
                     values=("movie_start", "birth", "border_entry"),
                     state="readonly", width=13).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(config, text="End").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(config, textvariable=self.end_frame, width=7).grid(
            row=4, column=1, sticky="ew", padx=(6, 4), pady=(6, 0))
        ttk.Combobox(config, textvariable=self.end_reason,
                     values=("movie_end", "border_exit"), state="readonly",
                     width=13).grid(row=4, column=2, columnspan=2,
                                    sticky="ew", pady=(6, 0))
        ttk.Button(config, text="Recenter on current frame (click)",
                   command=self._begin_recenter).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(config, text="Set threshold keyframe",
                   command=self._set_threshold_keyframe).grid(
            row=5, column=2, columnspan=2, sticky="ew", padx=(5, 0), pady=(7, 0))
        ttk.Button(config, text="Generate current",
                   command=self._generate_current).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(config, text="Generate all ready",
                   command=self._generate_all).grid(
            row=6, column=2, columnspan=2, sticky="ew", padx=(5, 0), pady=(5, 0))
        ttk.Button(config, text="Copy method settings to all jobs",
                   command=self._apply_shared_defaults).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Label(config, textvariable=self.crop_calibration_status).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(5, 0))
        config.columnconfigure(1, weight=1)
        config.columnconfigure(3, weight=1)

        manual = ttk.LabelFrame(side, text="Manual correction", padding=8)
        manual.pack(fill="x", pady=(8, 0))
        ttk.Label(manual, text="Tool").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            manual, textvariable=self.tool,
            values=("Inspect", "Add cell anchor", "Recenter anchor",
                    "Paint outline", "Erase outline", "Polygon outline"),
            state="readonly").grid(row=0, column=1, columnspan=3,
                                    sticky="ew", padx=(6, 0))
        ttk.Label(manual, text="Brush radius").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(manual, from_=0, to=30, textvariable=self.brush_radius,
                    width=6).grid(row=1, column=1, sticky="w", padx=(6, 4), pady=(6, 0))
        ttk.Button(manual, text="Close polygon",
                   command=self._close_polygon).grid(
            row=1, column=2, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(manual, text="Copy current to next",
                   command=self._copy_to_next).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(manual, text="Interpolate manual keyframes",
                   command=self._interpolate_keyframes).grid(
            row=2, column=2, columnspan=2, sticky="ew", padx=(5, 0), pady=(5, 0))
        manual.columnconfigure(3, weight=1)

        review = ttk.LabelFrame(side, text="Review current frame", padding=8)
        review.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(review, textvariable=self.review_status, wraplength=480,
                  justify="left").pack(anchor="w")
        self.metrics_list = self.tk.Listbox(review, height=8)
        self.metrics_list.pack(fill="both", expand=True, pady=(5, 0))
        frame_buttons = ttk.Frame(review)
        frame_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(frame_buttons, text="Accept frame",
                   command=self._accept_frame).pack(side="left")
        ttk.Button(frame_buttons, text="Needs correction",
                   command=self._reject_frame).pack(side="left", padx=(5, 0))
        range_row = ttk.Frame(review)
        range_row.pack(fill="x", pady=(6, 0))
        ttk.Label(range_row, text="Range").pack(side="left")
        ttk.Entry(range_row, textvariable=self.review_start, width=6).pack(
            side="left", padx=(5, 2))
        ttk.Label(range_row, text="to").pack(side="left")
        ttk.Entry(range_row, textvariable=self.review_end, width=6).pack(
            side="left", padx=(2, 5))
        ttk.Button(range_row, text="Accept clean range",
                   command=self._accept_clean_range).pack(side="left")
        ttk.Button(review, text="Apply threshold to review range",
                   command=self._apply_threshold_range).pack(fill="x", pady=(6, 0))
        ttk.Button(review, text="Resolve current collision as forced split...",
                   command=self._resolve_collision).pack(fill="x", pady=(6, 0))

        commit = ttk.Frame(side, padding=(0, 8, 0, 0))
        commit.pack(fill="x")
        self.commit_button = ttk.Button(
            commit, text="Commit reviewed new-cell jobs",
            command=self._commit, state="disabled")
        self.commit_button.pack(side="right")
        ttk.Button(commit, text="Cancel", command=self._cancel).pack(
            side="right", padx=6)

        ttk.Label(outer, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(6, 3)).pack(fill="x", pady=(8, 0))
        self._bind_sidebar_mousewheel(side)

    def _bind_sidebar_mousewheel(self, widget) -> None:
        widget.bind("<MouseWheel>", self._scroll_sidebar)
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def _scroll_sidebar(self, event):
        self.side_canvas.yview_scroll((-1 if event.delta > 0 else 1) * 3, "units")
        return "break"

    def _bind(self) -> None:
        self.canvas.bind("<Button-1>", self._canvas_pressed)
        self.canvas.bind("<B1-Motion>", self._canvas_dragged)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_released)
        self.canvas.bind("<MouseWheel>", self._canvas_zoomed)
        self.window.bind("<Left>", lambda _event: self._step_frame(-1))
        self.window.bind("<Right>", lambda _event: self._step_frame(1))
        self.window.bind("<space>", lambda _event: self._toggle_play())
        self.zoom.trace_add("write", lambda *_args: self._render())
        self.view_mode.trace_add("write", lambda *_args: self._render())
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

    def _new_job(self, frame: int, y: int, x: int) -> dict[str, Any]:
        identity = allocate_new_identity(
            self.source_labels,
            self.excluded_identities
            | {job["identity"] for job in self.jobs})
        return {
            "identity": identity, "start": 1, "end": len(self.source_labels),
            "start_reason": "movie_start", "end_reason": "movie_end",
            "anchors": {int(frame): (int(y), int(x))},
            "method": "soma_seed_expansion",
            "crop_side": default_crop_side_px(self.source_labels),
            "threshold": 2.0, "smoothing": 0.55,
            "threshold_keyframes": {}, "minimum_area": 4,
            "manual_masks": {}, "manual_keyframes": set(),
            "proposal": None, "accepted": set(),
            "linked_frames": set(), "linked_operations": [],
            "message": "Ready to generate",
        }

    def _add_job(self, frame: int, y: int, x: int) -> None:
        if not (1 <= frame <= len(self.source_labels)):
            raise ValueError("new-cell anchor frame is outside the movie")
        if y < 0 or x < 0 or y >= self.source_labels.shape[1] \
                or x >= self.source_labels.shape[2]:
            raise ValueError("new-cell anchor is outside the image")
        existing = int(self.source_labels[frame - 1, y, x])
        if existing > 0:
            raise ValueError(
                f"clicked identity {existing}; use forced splitting instead")
        self.jobs.append(self._new_job(frame, y, x))
        self.job_index = len(self.jobs) - 1
        self.frame = int(frame)
        self.tool.set("Inspect")
        self._load_job(self.job_index)
        self.status.set(
            f"Created new identity {self.current_job['identity']} from "
            f"ImageJ frame {frame} at y={y}, x={x}")

    def _begin_add_job(self) -> None:
        self.tool.set("Add cell anchor")
        self.view_mode.set("Full field")
        self.status.set("Click the centroid of another missed cell")

    def _remove_job(self) -> None:
        if self.current_job is None:
            return
        self._record_local_state()
        identity = self.current_job["identity"]
        del self.jobs[self.job_index]
        self.job_index = min(self.job_index, len(self.jobs) - 1)
        if self.current_job is not None:
            self._load_job(self.job_index)
        else:
            self._refresh_all()
        self.status.set(f"Removed unsaved new-cell job {identity}")

    def _load_job(self, index: int) -> None:
        if not 0 <= index < len(self.jobs):
            return
        self.job_index = index
        job = self.current_job
        self.method.set(METHOD_KEYS[job["method"]])
        self._set_crop_display(job["crop_side"])
        self.threshold.set(f"{job['threshold']:g}")
        self.smoothing.set(f"{job['smoothing']:g}")
        self.minimum_area.set(str(job["minimum_area"]))
        self.start_frame.set(str(job["start"]))
        self.end_frame.set(str(job["end"]))
        self.start_reason.set(job["start_reason"])
        self.end_reason.set(job["end_reason"])
        self.review_start.set(str(job["start"]))
        self.review_end.set(str(job["end"]))
        self._polygon_points.clear()
        self._refresh_all()
        self.job_list.selection_clear(0, "end")
        self.job_list.selection_set(index)

    def _calibrated_crop_scale(self) -> float:
        return float(np.sqrt(self.pixel_size_y * self.pixel_size_x))

    def _crop_pixels_from_display(self, unit: str | None = None) -> int:
        value = float(self.crop_side.get())
        active_unit = self.crop_unit.get() if unit is None else unit
        if not np.isfinite(value) or value <= 0:
            raise ValueError("crop side must be a positive finite number")
        pixels = value if active_unit == "px" \
            else value / self._calibrated_crop_scale()
        return int(round(pixels))

    def _set_crop_display(self, pixels: int) -> None:
        pixels = int(pixels)
        if self.crop_unit.get() == "px":
            self.crop_side.set(str(pixels))
        else:
            self.crop_side.set(f"{pixels * self._calibrated_crop_scale():.5g}")
        self.crop_calibration_status.set(
            f"{pixels} px = {pixels * self.pixel_size_y:.5g} by "
            f"{pixels * self.pixel_size_x:.5g} {self.spatial_unit}")

    def _crop_unit_changed(self, _event=None) -> None:
        try:
            pixels = self._crop_pixels_from_display(self._last_crop_unit)
        except ValueError:
            pixels = (self.current_job["crop_side"] if self.current_job is not None
                      else default_crop_side_px(self.source_labels))
        self._last_crop_unit = self.crop_unit.get()
        self._set_crop_display(pixels)

    def _job_selected(self, _event=None) -> None:
        selection = self.job_list.curselection()
        if not selection:
            return
        next_index = int(selection[0])
        if next_index == self.job_index:
            return
        try:
            self._save_configuration()
        except ValueError as error:
            from tkinter import messagebox
            messagebox.showerror("Cannot leave new-cell job", str(error),
                                 parent=self.window)
            self.job_list.selection_clear(0, "end")
            self.job_list.selection_set(self.job_index)
            return
        self._load_job(next_index)

    def _configuration(self) -> tuple[Any, ...]:
        return (
            METHOD_LABELS[self.method.get()], self._crop_pixels_from_display(),
            float(self.threshold.get()), float(self.smoothing.get()),
            int(self.minimum_area.get()), int(self.start_frame.get()),
            int(self.end_frame.get()), self.start_reason.get(),
            self.end_reason.get())

    def _save_configuration(self) -> None:
        job = self.current_job
        if job is None:
            return
        method, crop, threshold, smoothing, minimum, start, end, \
            start_reason, end_reason = self._configuration()
        if crop < 9 or crop > max(self.source_labels.shape[1:]) \
                or minimum < 1 or not 0 <= smoothing <= 1:
            raise ValueError(
                "crop must be 9 px through the larger image dimension, minimum "
                "area positive, and smoothing between 0 and 1")
        if start < 1 or end < start or end > len(self.source_labels):
            raise ValueError(
                f"lifetime must be within ImageJ frames 1-{len(self.source_labels)}")
        if any(frame < start or frame > end for frame in job["anchors"]):
            raise ValueError("one or more anchor clicks lie outside the lifetime")
        old = (job["method"], job["crop_side"], job["threshold"],
               job["smoothing"], job["minimum_area"], job["start"], job["end"],
               job["start_reason"], job["end_reason"])
        new = (method, crop, threshold, smoothing, minimum, start, end,
               start_reason, end_reason)
        if new != old:
            job["proposal"] = None
            job["accepted"].clear()
            job["linked_frames"].clear()
            job["linked_operations"].clear()
            job["message"] = "Configuration changed; regenerate"
        (job["method"], job["crop_side"], job["threshold"],
         job["smoothing"], job["minimum_area"], job["start"], job["end"],
         job["start_reason"], job["end_reason"]) = new

    def _request(self, job: dict[str, Any]) -> NewCellRequest:
        return NewCellRequest(
            identity=job["identity"], start_imagej_frame=job["start"],
            end_imagej_frame=job["end"], anchor_points=dict(job["anchors"]),
            method=job["method"], crop_side_px=job["crop_side"],
            threshold_z=job["threshold"],
            threshold_smoothing=job["smoothing"],
            threshold_keyframes=dict(job["threshold_keyframes"]),
            minimum_area_px=job["minimum_area"],
            start_reason=job["start_reason"], end_reason=job["end_reason"],
            manual_masks={frame: mask.copy()
                          for frame, mask in job["manual_masks"].items()})

    def _generate_job(self, job: dict[str, Any],
                      invalidated_frames: set[int] | None = None) -> None:
        retained_acceptance = set(job["accepted"])
        request = self._request(job)
        proposal = propose_new_identity(
            self.source_labels, self.raw, request, self.unclaimed,
            self.lag, self.raw_frame_indices)
        job["proposal"] = proposal
        if invalidated_frames is None:
            job["accepted"].clear()
            job["linked_frames"].clear()
            job["linked_operations"].clear()
        else:
            job["accepted"] = retained_acceptance - invalidated_frames
            job["linked_frames"].difference_update(invalidated_frames)
            job["linked_operations"] = [
                operation for operation in job["linked_operations"]
                if not any(operation["start_imagej_frame"] <= frame
                           <= operation["end_imagej_frame"]
                           for frame in invalidated_frames)]
        failures = sum(not row["valid"] for row in proposal.per_frame)
        job["message"] = (f"{failures} frame(s) need correction"
                          if failures else "Needs review")

    def _generate_current(self) -> None:
        from tkinter import messagebox

        if self.current_job is None:
            self.status.set("Add a missed-cell anchor before generating")
            return
        try:
            self._save_configuration()
            self._record_local_state()
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            self._generate_job(self.current_job)
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror("Cannot generate new cell", str(error),
                                 parent=self.window)
            return
        finally:
            self.window.configure(cursor="")
        self.status.set(
            f"Generated {self.current_job['method'].replace('_', ' ')} for "
            f"identity {self.current_job['identity']}")
        self._refresh_all()

    def _generate_all(self) -> None:
        from tkinter import messagebox

        if not self.jobs:
            return
        failures = []
        try:
            self._save_configuration()
            self._record_local_state()
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            for job in self.jobs:
                try:
                    self._generate_job(job)
                except ValueError as error:
                    job["message"] = str(error)
                    failures.append(job["identity"])
        finally:
            self.window.configure(cursor="")
        if failures:
            messagebox.showwarning(
                "Some new-cell jobs failed",
                "No edits were saved. Failed identities: "
                + ", ".join(map(str, failures)), parent=self.window)
        self.status.set(
            f"Generated {len(self.jobs) - len(failures)} of {len(self.jobs)} jobs")
        self._refresh_all()

    def _apply_shared_defaults(self) -> None:
        if self.current_job is None:
            self.status.set("Add at least one new-cell job first")
            return
        try:
            self._save_configuration()
        except ValueError as error:
            self.status.set(str(error))
            return
        self._record_local_state()
        source = self.current_job
        shared_keys = (
            "method", "crop_side", "threshold", "smoothing", "minimum_area")
        changed = 0
        for job in self.jobs:
            if job is source:
                continue
            if any(job[key] != source[key] for key in shared_keys):
                for key in shared_keys:
                    job[key] = source[key]
                job["proposal"] = None
                job["accepted"].clear()
                job["linked_frames"].clear()
                job["linked_operations"].clear()
                job["message"] = "Shared method settings changed; regenerate"
                changed += 1
        self.status.set(f"Copied method settings to {changed} other job(s)")
        self._refresh_all()

    def _begin_recenter(self) -> None:
        if self.current_job is None:
            return
        self.tool.set("Recenter anchor")
        self.view_mode.set("Full field")
        self.status.set(f"Click identity {self.current_job['identity']} on frame {self.frame}")

    def _set_threshold_keyframe(self) -> None:
        if self.current_job is None:
            return
        try:
            value = float(self.threshold.get())
            if not np.isfinite(value):
                raise ValueError
        except ValueError:
            self.status.set("Threshold keyframe must be a finite number")
            return
        self._record_local_state()
        self.current_job["threshold_keyframes"][self.frame] = value
        self.current_job["proposal"] = None
        self.current_job["accepted"].clear()
        self.current_job["linked_frames"].clear()
        self.current_job["linked_operations"].clear()
        self.current_job["message"] = "Threshold keyframe changed; regenerate"
        self.status.set(f"Set threshold {value:g} on ImageJ frame {self.frame}")
        self._refresh_all()

    def _zoom_factor(self) -> float:
        try:
            return max(0.1, float(self.zoom.get().rstrip("%")) / 100.0)
        except ValueError:
            return 1.0

    def _zoom_step(self, direction: int) -> None:
        current = self._zoom_factor()
        values = ([value for value in ZOOM_LEVELS if value > current]
                  if direction > 0 else
                  [value for value in ZOOM_LEVELS if value < current])
        target = ((values[0] if direction > 0 else values[-1]) if values
                  else (ZOOM_LEVELS[-1] if direction > 0 else ZOOM_LEVELS[0]))
        self.zoom.set(f"{int(target * 100)}%")

    def _canvas_zoomed(self, event):
        self._zoom_step(1 if event.delta > 0 else -1)
        return "break"

    def _active_centre_row(self) -> dict[str, Any] | None:
        job = self.current_job
        if job is None:
            return None
        proposal = job.get("proposal")
        if proposal is not None and job["start"] <= self.frame <= job["end"]:
            return proposal.crop_centres[self.frame - job["start"]]
        anchors = job["anchors"]
        if not anchors:
            return None
        nearest = min(anchors, key=lambda value: abs(value - self.frame))
        y, x = anchors[nearest]
        side = job["crop_side"]
        half = side / 2
        return {"y": float(y), "x": float(x), "bounds_yx": [
            max(0, int(y - half)), min(self.source_labels.shape[1], int(y + half + 1)),
            max(0, int(x - half)), min(self.source_labels.shape[2], int(x + half + 1))]}

    def _view_origin(self) -> tuple[int, int, int, int]:
        if self.view_mode.get() != "Moving crop":
            return 0, 0, self.source_labels.shape[1], self.source_labels.shape[2]
        centre = self._active_centre_row()
        if centre is None or centre.get("y") is None or centre.get("x") is None:
            return 0, 0, self.source_labels.shape[1], self.source_labels.shape[2]
        side = int(self.current_job["crop_side"])
        top = int(round(float(centre["y"]) - (side - 1) / 2.0))
        left = int(round(float(centre["x"]) - (side - 1) / 2.0))
        return top, left, top + side, left + side

    def _canvas_point(self, event) -> tuple[int, int]:
        scale = self._zoom_factor()
        y0, x0, _y1, _x1 = self._view_origin()
        return (int(self.canvas.canvasy(event.y) / scale) + y0,
                int(self.canvas.canvasx(event.x) / scale) + x0)

    def _canvas_pressed(self, event) -> None:
        if self.tool.get() in (
                "Add cell anchor", "Recenter anchor", "Paint outline",
                "Erase outline", "Polygon outline"):
            self._record_local_state()
        self._painting = True
        self._apply_canvas_tool(self._canvas_point(event))

    def _canvas_dragged(self, event) -> None:
        if self._painting and self.tool.get() in ("Paint outline", "Erase outline"):
            self._apply_canvas_tool(self._canvas_point(event))

    def _canvas_released(self, _event) -> None:
        self._painting = False

    def _apply_canvas_tool(self, point: tuple[int, int]) -> None:
        from tkinter import messagebox

        y, x = point
        try:
            if self.tool.get() == "Add cell anchor":
                self._add_job(self.frame, y, x)
                return
            job = self.current_job
            if job is None:
                raise ValueError("add a new-cell job first")
            if y < 0 or x < 0 or y >= self.source_labels.shape[1] \
                    or x >= self.source_labels.shape[2]:
                raise ValueError("click lies outside the image")
            if self.tool.get() == "Recenter anchor":
                if self.source_labels[self.frame - 1, y, x] > 0:
                    raise ValueError("recenter click touches an existing identity")
                job["anchors"][self.frame] = (y, x)
                job["proposal"] = None
                job["accepted"].clear()
                job["linked_frames"].clear()
                job["linked_operations"].clear()
                job["message"] = "Anchor changed; regenerate"
                self.tool.set("Inspect")
                self.status.set(
                    f"Recentered identity {job['identity']} on frame {self.frame}")
            elif self.tool.get() in ("Paint outline", "Erase outline"):
                if job["proposal"] is None:
                    self._save_configuration()
                    self._generate_job(job)
                paint_new_identity(
                    job["proposal"], self.source_labels, self.raw, self.frame,
                    [point], int(self.brush_radius.get()),
                    add=self.tool.get() == "Paint outline")
                job["manual_masks"][self.frame] = job["proposal"].masks[
                    self.frame - job["start"]].copy()
                job["accepted"].discard(self.frame)
                job["linked_frames"].discard(self.frame)
                job["linked_operations"] = [
                    operation for operation in job["linked_operations"]
                    if not (operation["start_imagej_frame"] <= self.frame
                            <= operation["end_imagej_frame"])]
                job["manual_keyframes"].add(self.frame)
                job["message"] = "Manual correction needs review"
            elif self.tool.get() == "Polygon outline":
                self._polygon_points.append(point)
                self.status.set(
                    f"Polygon has {len(self._polygon_points)} point(s); "
                    "click Close polygon when complete")
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror("Cannot edit new cell", str(error),
                                 parent=self.window)
        self._refresh_all()

    def _close_polygon(self) -> None:
        job = self.current_job
        if job is None or len(self._polygon_points) < 3:
            self.status.set("A polygon outline requires at least three points")
            return
        self._record_local_state()
        if job["proposal"] is None:
            try:
                self._save_configuration()
                self._generate_job(job)
            except ValueError as error:
                self.status.set(str(error))
                return
        mask_image = Image.new(
            "1", (self.source_labels.shape[2], self.source_labels.shape[1]))
        ImageDraw.Draw(mask_image).polygon(
            [(x, y) for y, x in self._polygon_points], fill=1)
        mask = np.asarray(mask_image, bool)
        offset = self.frame - job["start"]
        if offset < 0 or offset >= len(job["proposal"].masks):
            self.status.set("Polygon frame lies outside the declared lifetime")
            return
        job["proposal"].masks[offset] = mask
        job["proposal"].provenance[offset] = np.where(mask, 1, 0).astype(np.uint8)
        job["manual_masks"][self.frame] = mask.copy()
        revalidate_new_identity_proposal(
            self.source_labels, self.raw, job["proposal"])
        job["accepted"].discard(self.frame)
        job["manual_keyframes"].add(self.frame)
        job["message"] = "Manual polygon needs review"
        self._polygon_points.clear()
        self.status.set(f"Closed manual polygon on ImageJ frame {self.frame}")
        self._refresh_all()

    def _copy_to_next(self) -> None:
        job = self.current_job
        if job is None or job["proposal"] is None or self.frame >= job["end"]:
            self.status.set("Generate a proposal and choose a frame before the end")
            return
        self._record_local_state()
        source = self.frame - job["start"]
        target = source + 1
        job["proposal"].masks[target] = job["proposal"].masks[source]
        job["proposal"].provenance[target] = np.where(
            job["proposal"].masks[target], 1, 0).astype(np.uint8)
        target_frame = self.frame + 1
        job["manual_masks"][target_frame] = job["proposal"].masks[target].copy()
        job["manual_keyframes"].add(target_frame)
        job["accepted"].discard(target_frame)
        revalidate_new_identity_proposal(self.source_labels, self.raw,
                                         job["proposal"])
        self.status.set(f"Copied outline to ImageJ frame {target_frame}")
        self._refresh_all()

    def _interpolate_keyframes(self) -> None:
        job = self.current_job
        if job is None or job["proposal"] is None \
                or len(job["manual_keyframes"]) < 2:
            self.status.set("Create at least two manual outline keyframes first")
            return
        self._record_local_state()
        frames = sorted(job["manual_keyframes"])
        changed = []
        for left, right in zip(frames, frames[1:]):
            if right <= left + 1:
                continue
            first = job["proposal"].masks[left - job["start"]]
            second = job["proposal"].masks[right - job["start"]]
            for frame, mask in zip(
                    range(left + 1, right),
                    interpolate_manual_outlines(first, second, right - left - 1)):
                offset = frame - job["start"]
                job["proposal"].masks[offset] = mask
                job["proposal"].provenance[offset] = np.where(
                    mask, 1, 0).astype(np.uint8)
                job["manual_masks"][frame] = mask.copy()
                job["accepted"].discard(frame)
                changed.append(frame)
        revalidate_new_identity_proposal(self.source_labels, self.raw,
                                         job["proposal"])
        self.status.set(
            f"Interpolated {len(changed)} frame(s); every result needs review")
        self._refresh_all()

    def _slider_changed(self, value: str) -> None:
        frame = min(len(self.source_labels), max(1, int(round(float(value)))))
        if frame != self.frame:
            self.frame = frame
            self._refresh_all()

    def _step_frame(self, amount: int) -> None:
        self.frame = min(len(self.source_labels), max(1, self.frame + amount))
        self.frame_scale.set(self.frame)
        self._refresh_all()

    def _issue_frames(self) -> list[int]:
        job = self.current_job
        if job is None or job["proposal"] is None:
            return []
        warning = proposal_warning_frames(job["proposal"])
        return sorted(set(
            row["imagej_frame"] for row in job["proposal"].per_frame
            if not row["valid"]) | warning)

    def _step_issue(self, direction: int) -> None:
        issues = self._issue_frames()
        if not issues:
            return
        ordered = issues if direction > 0 else list(reversed(issues))
        candidate = [frame for frame in ordered
                     if (frame > self.frame if direction > 0 else frame < self.frame)]
        self.frame = candidate[0] if candidate else ordered[0]
        self.frame_scale.set(self.frame)
        self._refresh_all()

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.configure(text="Pause" if self._playing else "Play")
        if self._playing:
            self._play_step()

    def _play_step(self) -> None:
        if not self._playing:
            return
        job = self.current_job
        end = len(self.source_labels) if job is None else job["end"]
        start = 1 if job is None else job["start"]
        self.frame = start if self.frame >= end else self.frame + 1
        self.frame_scale.set(self.frame)
        self._refresh_all()
        self.window.after(140, self._play_step)

    def _accept_frame(self) -> None:
        job = self.current_job
        if job is None or job["proposal"] is None:
            self.status.set("Generate a proposal before review")
            return
        if self.frame < job["start"] or self.frame > job["end"]:
            self.status.set("Current frame is outside this identity's lifetime")
            return
        row = job["proposal"].per_frame[self.frame - job["start"]]
        if not row["valid"] and self.frame not in job["linked_frames"]:
            self.status.set("Correct the hard errors or resolve the collision first")
            return
        self._record_local_state()
        job["accepted"].add(self.frame)
        self.status.set(f"Accepted new identity on ImageJ frame {self.frame}")
        self._refresh_all()

    def _reject_frame(self) -> None:
        if self.current_job is None:
            return
        self._record_local_state()
        self.current_job["accepted"].discard(self.frame)
        self.current_job["message"] = "Needs correction"
        self.status.set(f"ImageJ frame {self.frame} marked for correction")
        self._refresh_all()

    def _accept_clean_range(self) -> None:
        job = self.current_job
        if job is None or job["proposal"] is None:
            return
        try:
            start, end = int(self.review_start.get()), int(self.review_end.get())
        except ValueError:
            self.status.set("Review range must contain integer ImageJ frames")
            return
        start, end = max(job["start"], start), min(job["end"], end)
        self._record_local_state()
        warnings = proposal_warning_frames(job["proposal"])
        accepted = []
        skipped = []
        for frame in range(start, end + 1):
            row = job["proposal"].per_frame[frame - job["start"]]
            if ((row["valid"] or frame in job["linked_frames"])
                    and frame not in warnings):
                job["accepted"].add(frame); accepted.append(frame)
            else:
                skipped.append(frame)
        self.status.set(
            f"Accepted {len(accepted)} clean frame(s); "
            f"{len(skipped)} warning or blocked frame(s) remain")
        self._refresh_all()

    def _apply_threshold_range(self) -> None:
        job = self.current_job
        if job is None:
            return
        try:
            start = max(job["start"], int(self.review_start.get()))
            end = min(job["end"], int(self.review_end.get()))
            value = float(self.threshold.get())
            if end < start or not np.isfinite(value):
                raise ValueError
        except ValueError:
            self.status.set(
                "Threshold range needs valid ImageJ frames and a finite z value")
            return
        self._record_local_state()
        baseline = float(job["threshold"])
        if start > job["start"]:
            job["threshold_keyframes"].setdefault(start - 1, baseline)
        for frame in range(start, end + 1):
            job["threshold_keyframes"][frame] = value
        if end < job["end"]:
            job["threshold_keyframes"].setdefault(end + 1, baseline)
        invalidated = set(range(start, end + 1))
        if job["method"] == "adaptive_local_threshold":
            invalidated = set(range(job["start"], job["end"] + 1))
        try:
            self._generate_job(job, invalidated)
        except ValueError as error:
            self.status.set(str(error))
            return
        job["message"] = (
            f"Threshold {value:g} applied to frames {start}-{end}; review changed "
            "proposals")
        self.status.set(job["message"])
        self._refresh_all()

    def _resolve_collision(self) -> None:
        from tkinter import messagebox

        job = self.current_job
        if job is None or job["proposal"] is None \
                or not (job["start"] <= self.frame <= job["end"]):
            self.status.set("Choose a proposed collision frame first")
            return
        row = job["proposal"].per_frame[self.frame - job["start"]]
        collisions = row["collision_identities"]
        if len(collisions) != 1:
            self.status.set(
                "Collision resolution requires exactly one existing host on this frame")
            return
        host = int(collisions[0])
        collision_frames = {
            value["imagej_frame"] for value in job["proposal"].per_frame
            if host in value["collision_identities"]}
        start = end = self.frame
        while start - 1 in collision_frames:
            start -= 1
        while end + 1 in collision_frames:
            end += 1
        if not messagebox.askyesno(
                "Resolve as forced split",
                f"Partition existing identity {host} between ImageJ frames "
                f"{start} and {end}, using new identity {job['identity']} as a child?",
                parent=self.window):
            return
        temporary = promote_labels_for_identity(
            self.source_labels, job["identity"]).copy()
        temporary_provenance = np.zeros(temporary.shape, np.uint8)
        for offset, imagej_frame in enumerate(range(job["start"], job["end"] + 1)):
            mask = job["proposal"].masks[offset] & (temporary[imagej_frame - 1] == 0)
            temporary[imagej_frame - 1][mask] = job["identity"]
            temporary_provenance[imagej_frame - 1][mask] = 2
        try:
            from manual_splitting import apply_force_split_operation
            for operation in job["linked_operations"]:
                apply_force_split_operation(
                    temporary, temporary_provenance, operation, set())
            from manual_splitting_ui import open_force_split_workspace

            def receive(operations: list[dict]) -> None:
                self._record_local_state()
                job["linked_operations"].extend(operations)
                job["linked_frames"].update(range(start, end + 1))
                job["accepted"].update(range(start, end + 1))
                job["message"] = "Linked split reviewed; continue review"
                self.status.set(
                    f"Linked forced split for identity {host}, frames {start}-{end}")
                self._refresh_all()

            open_force_split_workspace(
                self.window, self.controller, temporary, [host], start, end,
                self.output_dir, None,
                suggested_children={host: [host, job["identity"]]},
                operation_callback=receive)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open linked forced split", str(error),
                                 parent=self.window)

    def _all_ready(self) -> bool:
        if not self.jobs:
            return False
        collisions = batch_collision_frames(self.jobs)
        if collisions:
            return False
        for job in self.jobs:
            proposal = job["proposal"]
            required = set(range(job["start"], job["end"] + 1))
            if proposal is None or job["accepted"] != required:
                return False
            if any(not row["valid"] and row["imagej_frame"] not in job["linked_frames"]
                   for row in proposal.per_frame):
                return False
            if job["linked_frames"] and not job["linked_operations"]:
                return False
        return True

    def _refresh_all(self) -> None:
        self.frame_scale.set(self.frame)
        self.frame_status.set(f"Frame {self.frame} / {len(self.source_labels)}")
        self.job_list.delete(0, "end")
        collisions = batch_collision_frames(self.jobs)
        for index, job in enumerate(self.jobs):
            proposal = job["proposal"]
            total = job["end"] - job["start"] + 1
            accepted = len(job["accepted"])
            state = job["message"]
            if index in collisions:
                state = f"New-job collision on {len(collisions[index])} frame(s)"
            elif proposal is not None and accepted == total:
                state = "Accepted"
            elif proposal is not None:
                state = f"Needs review {accepted}/{total}"
            self.job_list.insert("end", f"Identity {job['identity']} | {state}")
        if self.current_job is not None:
            self.job_list.selection_set(self.job_index)
            self.job_status.set(
                f"Job {self.job_index + 1} of {len(self.jobs)} | "
                f"new identity {self.current_job['identity']}")
        else:
            self.job_status.set("No new-cell jobs")
        self._refresh_metrics()
        self._refresh_timeline()
        self._render()
        self.commit_button.configure(
            state="normal" if self._all_ready() else "disabled")

    def _refresh_timeline(self) -> None:
        job = self.current_job
        if job is None or job["proposal"] is None:
            self.timeline_status.set("")
            self.threshold_trace_status.set("")
            return
        warnings = proposal_warning_frames(job["proposal"])
        tokens = []
        for row in job["proposal"].per_frame:
            frame = row["imagej_frame"]
            suffix = ("R" if frame in job["accepted"] else
                      "!" if not row["valid"] and frame not in job["linked_frames"] else
                      "?" if frame in warnings else ".")
            tokens.append(f"{frame}{suffix}")
        self.timeline_status.set(
            "Frames (R reviewed, ! blocked, ? warning): " + " ".join(tokens))
        thresholds = [
            f"{row['imagej_frame']}:{row['threshold_z']:.2f}"
            for row in job["proposal"].per_frame
            if row.get("threshold_z") is not None]
        self.threshold_trace_status.set(
            "Threshold trace (frame:z): " + " ".join(thresholds)
            if thresholds else "Threshold trace: manual outlines")

    def _refresh_metrics(self) -> None:
        self.metrics_list.delete(0, "end")
        job = self.current_job
        if job is None or job["proposal"] is None \
                or not (job["start"] <= self.frame <= job["end"]):
            self.review_status.set("No proposal on the current frame")
            return
        row = job["proposal"].per_frame[self.frame - job["start"]]
        state = ("Accepted" if self.frame in job["accepted"] else
                 "Linked split" if self.frame in job["linked_frames"] else
                 "Needs review" if row["valid"] else "Needs correction")
        threshold = row["threshold_z"]
        threshold_text = "manual" if threshold is None else f"z={threshold:.2f}"
        self.review_status.set(
            f"{state} | {threshold_text} | tracking confidence "
            f"{row['tracking_confidence']:.3f}")
        self.metrics_list.insert("end", f"Area: {row['area_px']} px")
        self.metrics_list.insert(
            "end", f"Connected components: {row['connected_components']}")
        self.metrics_list.insert(
            "end", f"Mean intensity: {row['mean_intensity']}")
        self.metrics_list.insert(
            "end", f"Manual / assisted pixels: {row['manual_pixels']} / "
            f"{row['assisted_pixels']}")
        self.metrics_list.insert(
            "end", "Source raw frame index: "
            f"{row.get('source_raw_frame_index', self.frame - 1)}")
        motion = row.get("motion_support")
        if motion is not None:
            self.metrics_list.insert("end", f"Motion support: {motion:.3f}")
        if row["collision_identities"]:
            self.metrics_list.insert(
                "end", "COLLISION: existing identities "
                + ", ".join(map(str, row["collision_identities"])))
        for error in row["errors"]:
            self.metrics_list.insert("end", "BLOCKED: " + error)
        for warning in job["proposal"].warnings:
            if warning.startswith(f"ImageJ frame {self.frame}:"):
                self.metrics_list.insert("end", "WARNING: " + warning)
        for error in job["proposal"].lifetime_errors:
            self.metrics_list.insert("end", "LIFETIME: " + error)

    def _render(self) -> None:
        from PIL import ImageTk

        job = self.current_job
        proposal_mask = None
        crop_bounds = None
        anchors: list[tuple[int, int]] = []
        identity = None
        if job is not None:
            identity = job["identity"]
            anchors = [point for frame, point in job["anchors"].items()
                       if frame == self.frame]
            if job["proposal"] is not None and job["start"] <= self.frame <= job["end"]:
                offset = self.frame - job["start"]
                proposal_mask = job["proposal"].masks[offset]
                crop_bounds = job["proposal"].crop_centres[offset].get("bounds_yx")
            else:
                centre = self._active_centre_row()
                crop_bounds = None if centre is None else centre.get("bounds_yx")
        rendered = render_new_cell_full_field(
            self.raw[self.frame - 1], self.source_labels[self.frame - 1],
            proposal_mask, crop_bounds, anchors, identity,
            self.show_numbers.get(), self.show_tracks.get(),
            self.track_centroids, self.frame - 1, self._polygon_points,
            None if job is None or job.get("proposal") is None
            else job["proposal"].crop_centres,
            self.show_crop_path.get())
        y0, x0, y1, x1 = self._view_origin()
        if self.view_mode.get() == "Moving crop":
            centre = self._active_centre_row()
            if centre is not None and centre.get("y") is not None:
                rendered, _origin = extract_padded_moving_crop(
                    rendered, (centre["y"], centre["x"]),
                    int(job["crop_side"]))
        image = Image.fromarray(rendered)
        scale = self._zoom_factor()
        if scale != 1:
            image = image.resize(
                (max(1, round(image.width * scale)),
                 max(1, round(image.height * scale))),
                Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image, master=self.window)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def _save_draft(self) -> None:
        from tkinter import filedialog, messagebox

        if not self.jobs:
            self.status.set("There are no new-cell jobs to save")
            return
        selected = filedialog.asksaveasfilename(
            title="Save non-authoritative new-cell draft",
            initialdir=str(self.output_dir.parent),
            initialfile=f"{self.output_dir.name}_new_cell_draft.json",
            defaultextension=".json", filetypes=(("JSON files", "*.json"),))
        if not selected:
            return
        try:
            self._save_configuration()
            jobs = [proposal_to_draft_job(
                job["proposal"], self._request(job), job["accepted"],
                job["linked_frames"], job["linked_operations"])
                for job in self.jobs]
            path = save_new_cell_draft(
                Path(selected), self.parent_fingerprint, jobs)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot save draft", str(error), parent=self.window)
            return
        self.status.set(f"Saved non-authoritative draft: {path.name}")

    def _load_draft(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askopenfilename(
            title="Open new-cell draft", filetypes=(("JSON files", "*.json"),))
        if not selected:
            return
        try:
            document = load_new_cell_draft(Path(selected), self.parent_fingerprint)
            jobs = []
            for row in document["jobs"]:
                request, proposal, accepted, linked, operations = \
                    proposal_from_draft_job(self.source_labels, self.raw, row)
                jobs.append({
                    "identity": request.identity,
                    "start": request.start_imagej_frame,
                    "end": request.end_imagej_frame,
                    "start_reason": request.start_reason,
                    "end_reason": request.end_reason,
                    "anchors": dict(request.anchor_points),
                    "method": request.method, "crop_side": request.crop_side_px,
                    "threshold": request.threshold_z,
                    "smoothing": request.threshold_smoothing,
                    "threshold_keyframes": dict(request.threshold_keyframes),
                    "minimum_area": request.minimum_area_px,
                    "manual_masks": {frame: mask.copy()
                                     for frame, mask in request.manual_masks.items()},
                    "manual_keyframes": set(request.manual_masks),
                    "proposal": proposal, "accepted": accepted,
                    "linked_frames": linked,
                    "linked_operations": operations,
                    "message": "Draft restored",
                })
        except (OSError, KeyError, ValueError) as error:
            messagebox.showerror("Cannot load draft", str(error), parent=self.window)
            return
        self._record_local_state()
        self.jobs = jobs
        self.job_index = 0 if jobs else -1
        if jobs:
            self._load_job(0)
        else:
            self._refresh_all()
        self.status.set(f"Restored {len(jobs)} job(s) from draft")

    def _commit(self) -> None:
        from tkinter import messagebox

        if not self._all_ready():
            self.status.set(
                "Review every lifetime frame and resolve all collisions before commit")
            return
        if not messagebox.askyesno(
                "Commit new identities",
                f"Save {len(self.jobs)} reviewed new-cell job(s) as one atomic "
                "editing session?\n\n" + new_cell_commit_summary(self.jobs),
                parent=self.window):
            return
        try:
            operations: list[dict[str, Any]] = []
            for job in self.jobs:
                operations.append(build_add_identity_operation(
                    job["proposal"], job["accepted"], job["linked_frames"],
                    parent_labels=self.source_labels))
                operations.extend(job["linked_operations"])
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            saved = self.controller.apply_batch(operations, self.output_dir)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot commit new identities", str(error),
                                 parent=self.window)
            return
        finally:
            self.window.configure(cursor="")
        self.on_saved(saved)
        self.window.destroy()

    def _cancel(self) -> None:
        from tkinter import messagebox

        if self.jobs and not messagebox.askyesno(
                "Discard new-cell work",
                "Discard the unsaved jobs and close? Use Save draft first to "
                "retain the workspace.", parent=self.window):
            return
        self._playing = False
        self.window.destroy()


def open_new_cell_workspace(
        parent, controller: EditingHistoryController, labels: np.ndarray,
        output_dir: Path, on_saved: Callable[[Path], None],
        initial_anchor: tuple[int, int, int] | None = None
        ) -> NewCellWorkspace:
    return NewCellWorkspace(
        parent, controller, labels, output_dir, on_saved, initial_anchor)
