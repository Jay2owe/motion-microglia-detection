from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from common import OUTLINE_COLOURS, label_edges, outline_overlay
from manual_editing import EditingHistoryController
from manual_editing_ui import compute_track_centroids
from manual_editing_theme import DARK, apply_dark_theme
from manual_splitting import (SPLIT_METHODS, SplitProposal, SplitRequest,
                              build_force_split_operation,
                              paint_split_ownership,
                              propose_forced_split,
                              suggest_child_identities)


METHOD_LABELS = {
    "Marker watershed": "marker_controlled_watershed",
    "Soma-seed expansion": "soma_seed_expansion",
    "Manual drawing": "manual_drawing",
    "Temporal persistence": "temporal_persistence",
}
METHOD_KEYS = {value: key for key, value in METHOD_LABELS.items()}
SPLIT_ZOOM_LEVELS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def render_split_frame(
        raw: np.ndarray, source_labels: np.ndarray, host_identity: int,
        proposal_labels: np.ndarray | None = None,
        seeds: list[tuple[int, int]] | None = None,
        active_child_index: int = 0,
        show_numbers: bool = False,
        show_tracks: bool = False,
        tracks: dict[int, list[tuple[int, float, float]]] | None = None,
        frame_index: int = 0,
        visible_identities: tuple[int, ...] = ()) -> np.ndarray:
    """Render the strict host edge, proposed children, and editable seeds."""
    if raw.ndim != 2 or source_labels.shape != raw.shape:
        raise ValueError("split viewer requires same-shaped two-dimensional frames")
    host = source_labels == host_identity
    base_labels = np.where(host, host_identity, 0).astype(source_labels.dtype)
    rendered = outline_overlay(raw[None], base_labels[None], thick=2)[0]
    host_edge = label_edges(base_labels[None], thick=2)[0]
    rendered[host_edge] = np.array([245, 245, 245], np.uint8)
    if proposal_labels is not None:
        proposal = np.where(host, proposal_labels, 0)
        child_edges = label_edges(proposal[None], thick=1)[0]
        coloured = outline_overlay(raw[None], proposal[None], thick=1)[0]
        rendered[child_edges] = coloured[child_edges]

        # Make the inferred internal boundary explicit in yellow.
        internal = np.zeros(host.shape, bool)
        for child in np.unique(proposal):
            child = int(child)
            if child <= 0:
                continue
            mask = proposal == child
            neighbours = np.zeros(mask.shape, bool)
            neighbours[:-1] |= (proposal[1:] > 0) & (proposal[1:] != child)
            neighbours[1:] |= (proposal[:-1] > 0) & (proposal[:-1] != child)
            neighbours[:, :-1] |= ((proposal[:, 1:] > 0)
                                    & (proposal[:, 1:] != child))
            neighbours[:, 1:] |= ((proposal[:, :-1] > 0)
                                   & (proposal[:, :-1] != child))
            internal |= mask & neighbours & host
        rendered[internal] = np.array([255, 210, 45], np.uint8)

    image = Image.fromarray(rendered)
    draw = ImageDraw.Draw(image)
    if show_tracks and tracks:
        for identity in visible_identities:
            points = tracks.get(identity, [])
            colour = tuple(map(int, OUTLINE_COLOURS[
                (identity - 1) % len(OUTLINE_COLOURS)]))
            for first, second in zip(points, points[1:]):
                _first_frame, first_y, first_x = first
                _second_frame, second_y, second_x = second
                draw.line(
                    (first_x, first_y, second_x, second_y),
                    fill=colour, width=1)
            current = next(
                (point for point in points if point[0] == frame_index), None)
            if current is not None:
                _frame, y, x = current
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colour)
    if show_numbers:
        numbered = proposal_labels if proposal_labels is not None else base_labels
        for identity in visible_identities:
            points = np.column_stack(np.nonzero(numbered == identity))
            if not len(points):
                continue
            y, x = points.mean(axis=0)
            text = str(identity)
            bounds = draw.textbbox((x, y), text, anchor="mm")
            draw.rectangle(bounds, fill=(20, 20, 20))
            draw.text((x, y), text, anchor="mm", fill=(255, 255, 255))
    for index, (y, x) in enumerate(seeds or []):
        if y < 0 or x < 0:
            continue
        colour_array = OUTLINE_COLOURS[index % len(OUTLINE_COLOURS)]
        colour = tuple(map(int, colour_array))
        radius = 4 if index == active_child_index else 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255), fill=colour, width=1)
        draw.text((x, y), str(index + 1), anchor="mm", fill=(0, 0, 0))
    return np.asarray(image)


class ForceSplitWorkspace:
    """Dedicated proposal, correction, review, and transactional split window."""

    def __init__(
            self, parent, controller: EditingHistoryController,
            labels: np.ndarray, host_identities: list[int],
            start_imagej_frame: int, end_imagej_frame: int,
            output_dir: Path, on_saved: Callable[[Path], None] | None,
            suggested_children: dict[int, list[int]] | None = None,
            operation_callback: Callable[[list[dict]], None] | None = None):
        import tkinter as tk
        from tkinter import ttk

        if not host_identities:
            raise ValueError("forced splitting requires at least one host identity")
        if start_imagej_frame < 1 or end_imagej_frame < start_imagej_frame \
                or end_imagej_frame > len(labels):
            raise ValueError(
                f"split frame range must be within 1-{len(labels)}")
        self.tk = tk
        self.ttk = ttk
        self.controller = controller
        self.source_labels = np.asarray(labels).copy()
        self.raw = controller.bundle.registered_raw
        self.track_centroids = compute_track_centroids(self.source_labels)
        self.start = int(start_imagej_frame)
        self.end = int(end_imagej_frame)
        self.output_dir = Path(output_dir)
        self.on_saved = on_saved
        self.operation_callback = operation_callback
        self.window = tk.Toplevel(parent)
        self.window.title("Force cell splitting")
        self.window.geometry("1380x880")
        self.window.minsize(1050, 680)
        self.window.transient(parent)
        apply_dark_theme(self.window)

        reserved: set[int] = set()
        self.jobs: list[dict] = []
        for host in host_identities:
            supplied = None if suggested_children is None \
                else suggested_children.get(int(host))
            children = (tuple(map(int, supplied)) if supplied is not None
                        else suggest_child_identities(
                            self.source_labels, int(host), self.start,
                            self.end, 2, reserved))
            if len(children) < 2 or len(children) > 4 \
                    or len(set(children)) != len(children) \
                    or int(host) not in children:
                raise ValueError(
                    f"suggested children for host {host} must contain 2-4 "
                    "unique identities including the host")
            reserved.update(children[1:])
            self.jobs.append({
                "host": int(host),
                "children": list(children),
                "method": "marker_controlled_watershed",
                "seeds": {},
                "manual_assignments": {},
                "proposal": None,
                "accepted": set(),
                "message": "Not configured",
            })
        self.job_index = 0
        self.frame = self.start
        self.zoom = tk.StringVar(value="200%")
        self.method = tk.StringVar(value="Marker watershed")
        self.child_count = tk.IntVar(value=2)
        self.child_identities = tk.StringVar()
        self.active_child = tk.StringVar()
        self.tool = tk.StringVar(value="Place seed")
        self.brush_radius = tk.IntVar(value=2)
        self.minimum_area = tk.IntVar(value=3)
        self.show_numbers = tk.BooleanVar(value=True)
        self.show_tracks = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Configure the first split job")
        self.frame_status = tk.StringVar()
        self.review_status = tk.StringVar()
        self.job_status = tk.StringVar()
        self.photo = None
        self._painting = False

        self._build()
        self._load_job(0)
        self._bind()

    @property
    def current_job(self) -> dict:
        return self.jobs[self.job_index]

    def _build(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(
            header, text="Force cell splitting",
            font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Label(
            header,
            text=(f"ImageJ frames {self.start} to {self.end} · "
                  "proposals do not change the saved session"),
        ).pack(side="left", padx=(16, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 0))
        viewer = ttk.Frame(body)
        side_host = ttk.Frame(body)
        body.add(viewer, weight=3)
        body.add(side_host, weight=2)

        self.side_canvas = self.tk.Canvas(
            side_host, background=DARK["background"], highlightthickness=0,
            width=470)
        side_scroll = ttk.Scrollbar(
            side_host, orient="vertical", command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        self.side_canvas.pack(side="left", fill="both", expand=True)
        side_scroll.pack(side="right", fill="y")
        side = ttk.Frame(self.side_canvas, padding=(10, 0, 5, 0))
        side_window = self.side_canvas.create_window(
            0, 0, window=side, anchor="nw")
        side.bind(
            "<Configure>",
            lambda _event: self.side_canvas.configure(
                scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind(
            "<Configure>",
            lambda event: self.side_canvas.itemconfigure(
                side_window, width=event.width))

        canvas_frame = ttk.Frame(viewer)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = self.tk.Canvas(
            canvas_frame, background="#15171a", cursor="crosshair",
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
        ttk.Button(
            navigation, text="Previous warning",
            command=lambda: self._step_needing_review(-1)).pack(side="left")
        ttk.Button(
            navigation, text="Previous", command=lambda: self._step_frame(-1)
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            navigation, text="Next", command=lambda: self._step_frame(1)
        ).pack(side="left", padx=(4, 0))
        ttk.Button(
            navigation, text="Next warning",
            command=lambda: self._step_needing_review(1)).pack(
                side="left", padx=(4, 10))
        self.frame_scale = ttk.Scale(
            navigation, from_=self.start, to=self.end, orient="horizontal",
            command=self._slider_changed)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(navigation, textvariable=self.frame_status, width=18).pack(
            side="left", padx=(8, 0))

        view = ttk.Frame(viewer, padding=(0, 6, 0, 0))
        view.pack(fill="x")
        ttk.Label(view, text="Zoom").pack(side="left")
        ttk.Button(
            view, text="−", width=3, command=lambda: self._zoom_step(-1)
        ).pack(side="left", padx=(5, 2))
        ttk.Combobox(
            view, textvariable=self.zoom,
            values=tuple(f"{int(value * 100)}%" for value in SPLIT_ZOOM_LEVELS),
            state="readonly", width=7).pack(side="left")
        ttk.Button(
            view, text="+", width=3, command=lambda: self._zoom_step(1)
        ).pack(side="left", padx=(2, 8))
        ttk.Label(
            view,
            text="White: original host edge · yellow: inferred split boundary",
        ).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(
            view, text="Numbers", variable=self.show_numbers,
            command=self._render).pack(side="left")
        ttk.Checkbutton(
            view, text="Tracks", variable=self.show_tracks,
            command=self._render).pack(side="left", padx=(6, 0))

        jobs = ttk.LabelFrame(side, text="Split jobs", padding=8)
        jobs.pack(fill="x")
        self.job_list = self.tk.Listbox(jobs, height=min(6, len(self.jobs) + 1))
        self.job_list.pack(fill="x")
        self.job_list.bind("<<ListboxSelect>>", self._job_selected)
        ttk.Label(jobs, textvariable=self.job_status).pack(anchor="w", pady=(5, 0))

        configure = ttk.LabelFrame(side, text="Configure current job", padding=8)
        configure.pack(fill="x", pady=(8, 0))
        ttk.Label(configure, text="Method").grid(row=0, column=0, sticky="w")
        method_box = ttk.Combobox(
            configure, textvariable=self.method,
            values=tuple(METHOD_LABELS), state="readonly")
        method_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(6, 0))
        method_box.bind("<<ComboboxSelected>>", lambda _event: self._invalidate())
        ttk.Label(configure, text="Children").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        child_spin = ttk.Spinbox(
            configure, from_=2, to=4, textvariable=self.child_count, width=5,
            command=self._resize_children)
        child_spin.grid(row=1, column=1, sticky="w", padx=(6, 4), pady=(6, 0))
        ttk.Label(configure, text="Identity values").grid(
            row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Entry(
            configure, textvariable=self.child_identities, width=18
        ).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(configure, text="Active child").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        self.active_child_box = ttk.Combobox(
            configure, textvariable=self.active_child,
            state="readonly", width=8)
        self.active_child_box.grid(
            row=2, column=1, sticky="w", padx=(6, 4), pady=(6, 0))
        ttk.Label(configure, text="Tool").grid(
            row=2, column=2, sticky="e", pady=(6, 0))
        ttk.Combobox(
            configure, textvariable=self.tool,
            values=("Place seed", "Paint ownership"),
            state="readonly", width=18).grid(
                row=2, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Label(configure, text="Brush radius").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            configure, from_=0, to=20, textvariable=self.brush_radius,
            width=5).grid(row=3, column=1, sticky="w", padx=(6, 4), pady=(6, 0))
        ttk.Label(configure, text="Minimum area").grid(
            row=3, column=2, sticky="e", pady=(6, 0))
        ttk.Spinbox(
            configure, from_=1, to=10000, textvariable=self.minimum_area,
            width=8).grid(row=3, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
        ttk.Button(
            configure, text="Clear seeds on this frame",
            command=self._clear_frame_seeds).grid(
                row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(
            configure, text="Generate current proposal",
            command=self._generate_current).grid(
                row=4, column=2, columnspan=2, sticky="ew",
                padx=(5, 0), pady=(8, 0))
        ttk.Button(
            configure, text="Generate all jobs",
            command=self._generate_all).grid(
                row=5, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        configure.columnconfigure(3, weight=1)

        review = ttk.LabelFrame(side, text="Review current frame", padding=8)
        review.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(
            review, textvariable=self.review_status, wraplength=440,
            justify="left").pack(anchor="w")
        self.metrics_list = self.tk.Listbox(review, height=9)
        self.metrics_list.pack(fill="both", expand=True, pady=(6, 0))
        buttons = ttk.Frame(review)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(
            buttons, text="Accept this frame",
            command=self._accept_frame).pack(side="left")
        ttk.Button(
            buttons, text="Mark for correction",
            command=self._reject_frame).pack(side="left", padx=(5, 0))

        commit = ttk.Frame(side, padding=(0, 8, 0, 0))
        commit.pack(fill="x")
        self.commit_button = ttk.Button(
            commit, text="Commit reviewed split jobs",
            command=self._commit, state="disabled")
        self.commit_button.pack(side="right")
        ttk.Button(commit, text="Cancel", command=self._cancel).pack(side="right", padx=6)

        ttk.Label(
            outer, textvariable=self.status, relief="sunken", anchor="w",
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
        self.zoom.trace_add("write", lambda *_args: self._render())
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

    def _load_job(self, index: int) -> None:
        if not 0 <= index < len(self.jobs):
            return
        self.job_index = index
        job = self.current_job
        self.method.set(METHOD_KEYS[job["method"]])
        self.child_count.set(len(job["children"]))
        self.child_identities.set(", ".join(map(str, job["children"])))
        self._refresh_child_choices()
        self._refresh_all()
        self.job_list.selection_clear(0, "end")
        self.job_list.selection_set(index)

    def _job_selected(self, _event=None) -> None:
        selected = self.job_list.curselection()
        if not selected:
            return
        next_index = int(selected[0])
        if next_index == self.job_index:
            return
        try:
            self._save_configuration()
        except ValueError as error:
            from tkinter import messagebox
            messagebox.showerror(
                "Cannot leave split job", str(error), parent=self.window)
            self.job_list.selection_clear(0, "end")
            self.job_list.selection_set(self.job_index)
            return
        self._load_job(next_index)

    def _parse_children(self) -> list[int]:
        try:
            children = [
                int(value.strip()) for value in self.child_identities.get().split(",")
                if value.strip()]
        except ValueError as error:
            raise ValueError("child identities must be comma-separated integers") from error
        count = int(self.child_count.get())
        if len(children) != count:
            raise ValueError(f"enter exactly {count} child identities")
        if len(set(children)) != len(children) or any(value <= 0 for value in children):
            raise ValueError("child identities must be unique positive integers")
        if self.current_job["host"] not in children:
            raise ValueError("one child must keep the host identity")
        maximum = int(np.iinfo(self.source_labels.dtype).max)
        if max(children) > maximum:
            raise ValueError(
                f"child identities must fit {self.source_labels.dtype.name}")
        return children

    def _resize_children(self) -> None:
        count = int(self.child_count.get())
        host = self.current_job["host"]
        current = self.current_job["children"]
        if len(current) == count:
            return
        reserved = {
            value for job in self.jobs for value in job["children"]
            if value != host
        }
        children = list(suggest_child_identities(
            self.source_labels, host, self.start, self.end, count,
            reserved - set(current)))
        self.current_job["children"] = children
        self.child_identities.set(", ".join(map(str, children)))
        self._refresh_child_choices()
        self._invalidate()

    def _refresh_child_choices(self) -> None:
        children = [
            value.strip() for value in self.child_identities.get().split(",")
            if value.strip()]
        self.active_child_box.configure(values=children)
        if self.active_child.get() not in children:
            self.active_child.set(children[0] if children else "")

    def _save_configuration(self) -> None:
        children = self._parse_children()
        method = METHOD_LABELS[self.method.get()]
        if method not in SPLIT_METHODS:
            raise ValueError("choose a supported split method")
        job = self.current_job
        if job["children"] != children or job["method"] != method:
            job["proposal"] = None
            job["accepted"].clear()
        job["children"] = children
        job["method"] = method
        self._refresh_child_choices()

    def _invalidate(self) -> None:
        job = self.current_job
        job["proposal"] = None
        job["accepted"].clear()
        job["message"] = "Configuration changed; regenerate"
        self._refresh_all()

    def _seeds_for_current_frame(self) -> list[tuple[int, int]]:
        job = self.current_job
        points = job["seeds"].get(self.frame, {})
        return [points.get(child, (-100, -100)) for child in job["children"]]

    def _clear_frame_seeds(self) -> None:
        self.current_job["seeds"].pop(self.frame, None)
        self._invalidate()
        self.status.set(f"Cleared seeds on ImageJ frame {self.frame}")

    def _request_for_job(self, job: dict) -> SplitRequest:
        seeds = {}
        for frame, mapping in job["seeds"].items():
            if all(child in mapping for child in job["children"]):
                seeds[int(frame)] = tuple(mapping[child] for child in job["children"])
        return SplitRequest(
            host_identity=job["host"],
            start_imagej_frame=self.start,
            end_imagej_frame=self.end,
            child_identities=tuple(job["children"]),
            method=job["method"],
            seeds=seeds,
            manual_assignments=job["manual_assignments"],
            minimum_child_area_px=int(self.minimum_area.get()),
        )

    def _generate_job(self, job: dict) -> None:
        request = self._request_for_job(job)
        proposal = propose_forced_split(self.source_labels, self.raw, request)
        job["proposal"] = proposal
        job["accepted"].clear()
        failed = [row for row in proposal.per_frame if not row["valid"]]
        if proposal.boundary_errors:
            job["message"] = "Boundary continuity needs correction"
        elif failed:
            job["message"] = f"{len(failed)} frame(s) need correction"
        else:
            job["message"] = "Needs review"

    def _generate_current(self) -> None:
        from tkinter import messagebox

        try:
            self._save_configuration()
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            self._generate_job(self.current_job)
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot generate split", str(error), parent=self.window)
            return
        finally:
            self.window.configure(cursor="")
        self.status.set(
            f"Generated {self.current_job['method'].replace('_', ' ')} "
            f"proposal for identity {self.current_job['host']}")
        self._refresh_all()

    def _generate_all(self) -> None:
        from tkinter import messagebox

        failures = []
        try:
            self._save_configuration()
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            for job in self.jobs:
                try:
                    self._generate_job(job)
                except ValueError as error:
                    job["proposal"] = None
                    job["message"] = str(error)
                    failures.append(job["host"])
        finally:
            self.window.configure(cursor="")
        self._refresh_all()
        if failures:
            messagebox.showwarning(
                "Some proposals failed",
                "No edits were saved. Failed host identities: "
                + ", ".join(map(str, failures)), parent=self.window)
        self.status.set(
            f"Generated {len(self.jobs) - len(failures)} of "
            f"{len(self.jobs)} split proposal(s)")

    def _zoom_factor(self) -> float:
        try:
            return max(0.1, float(self.zoom.get().rstrip("%")) / 100.0)
        except ValueError:
            return 1.0

    def _zoom_step(self, direction: int) -> None:
        current = self._zoom_factor()
        if direction > 0:
            values = [value for value in SPLIT_ZOOM_LEVELS if value > current]
            target = values[0] if values else SPLIT_ZOOM_LEVELS[-1]
        else:
            values = [value for value in SPLIT_ZOOM_LEVELS if value < current]
            target = values[-1] if values else SPLIT_ZOOM_LEVELS[0]
        self.zoom.set(f"{int(target * 100)}%")

    def _canvas_zoomed(self, event):
        self._zoom_step(1 if event.delta > 0 else -1)
        return "break"

    def _canvas_point(self, event) -> tuple[int, int]:
        scale = self._zoom_factor()
        return (
            int(self.canvas.canvasy(event.y) / scale),
            int(self.canvas.canvasx(event.x) / scale),
        )

    def _canvas_pressed(self, event) -> None:
        self._painting = True
        self._apply_canvas_tool(self._canvas_point(event))

    def _canvas_dragged(self, event) -> None:
        if self._painting and self.tool.get() == "Paint ownership":
            self._apply_canvas_tool(self._canvas_point(event))

    def _canvas_released(self, _event) -> None:
        self._painting = False

    def _apply_canvas_tool(self, point: tuple[int, int]) -> None:
        from tkinter import messagebox

        try:
            self._save_configuration()
            child = int(self.active_child.get())
            host = self.current_job["host"]
            y, x = point
            if y < 0 or x < 0 or y >= self.source_labels.shape[1] \
                    or x >= self.source_labels.shape[2] \
                    or self.source_labels[self.frame - 1, y, x] != host:
                self.status.set("Seeds and paint must stay inside the host identity")
                return
            if self.tool.get() == "Place seed":
                self.current_job["seeds"].setdefault(self.frame, {})[child] = point
                self.current_job["proposal"] = None
                self.current_job["accepted"].clear()
                self.current_job["message"] = "Seed changed; regenerate"
                self.status.set(
                    f"Placed child {child} seed at y={y}, x={x} on frame {self.frame}")
            else:
                proposal = self.current_job["proposal"]
                if proposal is None:
                    raise ValueError("generate a proposal before painting ownership")
                paint_split_ownership(
                    self.source_labels, proposal, self.frame, child, [point],
                    max(0, int(self.brush_radius.get())))
                self.current_job["accepted"].discard(self.frame)
                self.current_job["message"] = "Manual correction needs review"
                self.status.set(
                    f"Painted child {child} ownership on frame {self.frame}")
        except (KeyError, ValueError) as error:
            messagebox.showerror(
                "Cannot edit split", str(error), parent=self.window)
        self._refresh_all()

    def _slider_changed(self, value: str) -> None:
        frame = min(self.end, max(self.start, int(round(float(value)))))
        if frame != self.frame:
            self.frame = frame
            self._refresh_all()

    def _step_frame(self, amount: int) -> None:
        self.frame = min(self.end, max(self.start, self.frame + amount))
        self.frame_scale.set(self.frame)
        self._refresh_all()

    def _step_needing_review(self, direction: int) -> None:
        accepted = self.current_job["accepted"]
        frames = list(range(self.start, self.end + 1))
        ordered = frames if direction > 0 else list(reversed(frames))
        candidates = [
            frame for frame in ordered
            if frame not in accepted and (
                (direction > 0 and frame > self.frame)
                or (direction < 0 and frame < self.frame))]
        if not candidates:
            candidates = [frame for frame in ordered if frame not in accepted]
        if candidates:
            self.frame = candidates[0]
            self.frame_scale.set(self.frame)
            self._refresh_all()

    def _accept_frame(self) -> None:
        proposal: SplitProposal | None = self.current_job["proposal"]
        if proposal is None:
            self.status.set("Generate a proposal before reviewing it")
            return
        row = proposal.per_frame[self.frame - self.start]
        if not row["valid"]:
            self.status.set("Correct the hard split errors before accepting this frame")
            return
        self.current_job["accepted"].add(self.frame)
        self.current_job["message"] = "Needs review"
        if len(self.current_job["accepted"]) == self.end - self.start + 1:
            self.current_job["message"] = "Accepted"
        self.status.set(f"Accepted split on ImageJ frame {self.frame}")
        self._step_needing_review(1)
        self._refresh_all()

    def _reject_frame(self) -> None:
        self.current_job["accepted"].discard(self.frame)
        self.current_job["message"] = "Needs correction"
        self.status.set(f"ImageJ frame {self.frame} marked for correction")
        self._refresh_all()

    def _all_jobs_accepted(self) -> bool:
        required = set(range(self.start, self.end + 1))
        return all(job["proposal"] is not None
                   and job["proposal"].valid
                   and job["accepted"] == required for job in self.jobs)

    def _refresh_all(self) -> None:
        self.job_list.delete(0, "end")
        for job in self.jobs:
            proposal = job["proposal"]
            accepted = len(job["accepted"])
            total = self.end - self.start + 1
            state = job["message"]
            if proposal is not None and proposal.valid and accepted < total:
                state = f"Needs review {accepted}/{total}"
            elif proposal is not None and accepted == total:
                state = "Accepted"
            self.job_list.insert("end", f"Identity {job['host']} · {state}")
        self.job_list.selection_set(self.job_index)
        self.job_status.set(
            f"Job {self.job_index + 1} of {len(self.jobs)} · "
            f"host identity {self.current_job['host']}")
        self.frame_scale.set(self.frame)
        self.frame_status.set(f"Frame {self.frame} / {self.end}")
        self._refresh_metrics()
        self._render()
        self.commit_button.configure(
            state="normal" if self._all_jobs_accepted() else "disabled")

    def _refresh_metrics(self) -> None:
        self.metrics_list.delete(0, "end")
        proposal = self.current_job["proposal"]
        if proposal is None:
            self.review_status.set(
                "No proposal. Place optional seeds, then generate. Automatic seeds "
                "are used when none are supplied.")
            return
        row = proposal.per_frame[self.frame - self.start]
        state = "Accepted" if self.frame in self.current_job["accepted"] \
            else "Needs review" if row["valid"] else "Needs correction"
        self.review_status.set(
            f"{state} · boundary confidence {row['boundary_confidence']:.3f} · "
            f"{row['manual_pixels']} manually corrected pixels")
        self.metrics_list.insert(
            "end", f"Host union: {row['assigned_pixels']} / {row['host_pixels']} px")
        self.metrics_list.insert(
            "end", f"Outside host: {row['outside_pixels']} px")
        if row.get("runner_up_margin") is not None:
            self.metrics_list.insert(
                "end", f"Best-versus-runner-up margin: "
                f"{row['runner_up_margin']:.3f}")
        for child in row["children"]:
            self.metrics_list.insert(
                "end", f"Identity {child['identity']}: {child['area_px']} px · "
                f"{child['connected_components']} component(s)")
        for error in row["errors"]:
            self.metrics_list.insert("end", "BLOCKED: " + error)
        for error in proposal.boundary_errors:
            self.metrics_list.insert("end", "BOUNDARY BLOCK: " + error)
        for warning in proposal.warnings:
            if f"frame {self.frame} " in warning:
                self.metrics_list.insert("end", "WARNING: " + warning)

    def _render(self) -> None:
        from PIL import ImageTk

        job = self.current_job
        proposal = job["proposal"]
        proposal_frame = None if proposal is None else proposal.labels[
            self.frame - self.start]
        points = self._seeds_for_current_frame()
        try:
            active_index = job["children"].index(int(self.active_child.get()))
        except (ValueError, TypeError):
            active_index = 0
        rendered = render_split_frame(
            self.raw[self.frame - 1], self.source_labels[self.frame - 1],
            job["host"], proposal_frame, points, active_index,
            self.show_numbers.get(), self.show_tracks.get(),
            self.track_centroids, self.frame - 1,
            tuple(job["children"]))
        scale = self._zoom_factor()
        image = Image.fromarray(rendered)
        if scale != 1:
            image = image.resize(
                (max(1, round(image.width * scale)),
                 max(1, round(image.height * scale))),
                Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image, master=self.window)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def _commit(self) -> None:
        from tkinter import messagebox

        if not self._all_jobs_accepted():
            self.status.set("Review and accept every valid frame before committing")
            return
        if not messagebox.askyesno(
                "Commit forced splits",
                f"Save {len(self.jobs)} reviewed split job(s) as one atomic "
                "editing session?", parent=self.window):
            return
        try:
            operations = [
                build_force_split_operation(job["proposal"])
                for job in self.jobs]
            if self.operation_callback is not None:
                self.operation_callback(operations)
                self.window.destroy()
                return
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            saved = self.controller.apply_batch(operations, self.output_dir)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot commit forced splits", str(error), parent=self.window)
            return
        finally:
            self.window.configure(cursor="")
        if self.on_saved is not None:
            self.on_saved(saved)
        self.window.destroy()

    def _cancel(self) -> None:
        from tkinter import messagebox

        has_work = any(job["proposal"] is not None or job["seeds"]
                       for job in self.jobs)
        if has_work and not messagebox.askyesno(
                "Discard split proposals",
                "Discard the unsaved split proposals and close this workspace?",
                parent=self.window):
            return
        self.window.destroy()


def open_force_split_workspace(
        parent, controller: EditingHistoryController, labels: np.ndarray,
        host_identities: list[int], start_imagej_frame: int,
        end_imagej_frame: int, output_dir: Path,
        on_saved: Callable[[Path], None] | None,
        suggested_children: dict[int, list[int]] | None = None,
        operation_callback: Callable[[list[dict]], None] | None = None
        ) -> ForceSplitWorkspace:
    return ForceSplitWorkspace(
        parent, controller, labels, host_identities, start_imagej_frame,
        end_imagej_frame, output_dir, on_saved,
        suggested_children, operation_callback)
