from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from manual_editing import EditingHistoryController
from manual_editing_theme import DARK, apply_dark_theme
from manual_retuning import (RETUNE_DOMAIN_POLICIES, RetuneProposal,
                             RetuneRequest, build_eligible_domain,
                             build_retune_operation,
                             eligible_retune_source_operations,
                             propose_mask_retuning,
                             released_background_from_history,
                             source_operation_frame_range,
                             suggest_recipient_identities)


ZOOM_LEVELS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DOMAIN_LABELS = {
    "Vacated pixels only": "vacated_only",
    "Vacated pixels + halo": "vacated_plus_halo",
}
DOMAIN_KEYS = {value: key for key, value in DOMAIN_LABELS.items()}


def _unit_image(frame: np.ndarray) -> np.ndarray:
    values = frame[np.isfinite(frame)]
    if not values.size:
        return np.zeros(frame.shape, np.uint8)
    lo, hi = map(float, np.percentile(values, [1, 99.5]))
    if hi <= lo:
        return np.zeros(frame.shape, np.uint8)
    return np.rint(np.clip((frame.astype(float) - lo) / (hi - lo), 0, 1) * 255
                    ).astype(np.uint8)


def _edges(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))


def render_retune_comparison(
        raw: np.ndarray, current_labels: np.ndarray, assignments: np.ndarray,
        eligible: np.ndarray, released: np.ndarray, contested: np.ndarray,
        imagej_frame: int) -> np.ndarray:
    """Render current state and proposed ownership with explicit vacancy states."""
    grey = _unit_image(raw)
    before = np.repeat(grey[..., None], 3, axis=2)
    after = before.copy()
    current_edges = _edges(current_labels > 0)
    before[current_edges] = np.array([0, 220, 235], np.uint8)
    after[current_edges] = np.array([0, 150, 170], np.uint8)

    # Released pixels are yellow; halo-only eligibility is orange.
    before[eligible] = np.rint(
        0.45 * before[eligible] + 0.55 * np.array([230, 145, 35])
    ).astype(np.uint8)
    before[released] = np.rint(
        0.35 * before[released] + 0.65 * np.array([250, 215, 45])
    ).astype(np.uint8)
    unresolved = eligible & (assignments == 0)
    after[unresolved] = np.rint(
        0.45 * after[unresolved] + 0.55 * np.array([235, 155, 35])
    ).astype(np.uint8)
    after[contested] = np.array([235, 55, 210], np.uint8)

    identities = sorted(int(value) for value in np.unique(assignments) if value > 0)
    colours = (
        (80, 235, 120), (60, 170, 255), (255, 105, 80),
        (190, 110, 255), (255, 220, 70), (75, 235, 225),
    )
    for index, identity in enumerate(identities):
        mask = assignments == identity
        colour = np.asarray(colours[index % len(colours)], np.uint8)
        after[mask] = np.rint(0.20 * after[mask] + 0.80 * colour).astype(np.uint8)
        after[_edges(mask)] = colour

    gap = np.full((raw.shape[0], 8, 3), 22, np.uint8)
    combined = np.concatenate((before, gap, after), axis=1)
    header = np.full((28, combined.shape[1], 3), 28, np.uint8)
    image = Image.fromarray(np.concatenate((header, combined), axis=0))
    draw = ImageDraw.Draw(image)
    draw.text((7, 7), f"Current after edit - ImageJ frame {imagej_frame}",
              fill=(235, 235, 235))
    draw.text((raw.shape[1] + 16, 7), "Proposed reclaimed ownership",
              fill=(235, 235, 235))
    return np.asarray(image)


def describe_source_operation(operation: dict) -> str:
    operation_id = int(operation["operation_id"])
    if operation["type"] == "exclude_identity":
        return f"Operation {operation_id}: removed identity {operation['identity']}"
    identities = ", ".join(map(str, operation["identities"]))
    return (
        f"Operation {operation_id}: deleted {identities}, frames "
        f"{operation['start_imagej_frame']}-{operation['end_imagej_frame']}")


class MaskRetuningWorkspace:
    """Review and commit one batch post-edit mask-retuning proposal."""

    def __init__(
            self, parent, controller: EditingHistoryController,
            labels: np.ndarray, initial_recipients: list[int], output_dir: Path,
            on_saved: Callable[[Path], None]):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.controller = controller
        self.labels = np.asarray(labels)
        self.raw = controller.bundle.registered_raw
        self.output_dir = Path(output_dir)
        self.on_saved = on_saved
        self.source_operations = eligible_retune_source_operations(
            controller.operations, controller.position)
        if not self.source_operations:
            raise ValueError(
                "the active history has no full-track removal or frame-range deletion")
        self.source_by_label = {
            describe_source_operation(operation): operation
            for operation in self.source_operations}
        self.initial_recipients = sorted(set(map(int, initial_recipients)))
        self.proposal: RetuneProposal | None = None
        self.reviewed: set[int] = set()
        self.frame_index = 0
        self.photo = None

        self.window = tk.Toplevel(parent)
        apply_dark_theme(self.window)
        self.window.title("Retune masks after an edit")
        self.window.geometry("1320x860")
        self.window.minsize(980, 680)

        self.source = tk.StringVar(value=list(self.source_by_label)[-1])
        self.recipients = tk.StringVar()
        self.start_frame = tk.StringVar(value="1")
        self.end_frame = tk.StringVar(value=str(len(labels)))
        self.domain = tk.StringVar(value=DOMAIN_KEYS["vacated_only"])
        self.halo = tk.StringVar(value="2")
        self.maximum_reach = tk.StringVar(value="12")
        self.sensitivity = tk.StringVar(value="0.60")
        self.margin = tk.StringVar(value="0.08")
        self.zoom = tk.StringVar(value="200%")
        self.frame_status = tk.StringVar()
        self.review_status = tk.StringVar(value="Generate a proposal to begin review")
        self.status = tk.StringVar(value="Ready")

        self._build()
        self._source_changed()
        self._bind()

    def _build(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="Retune masks after an edit",
            font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text=("Yellow = released pixels; orange = unresolved; magenta = "
                  "contested. Current positive owners are protected."),
        ).pack(anchor="w", pady=(3, 8))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        viewer = ttk.Frame(body)
        sidebar_host = ttk.Frame(body)
        body.add(viewer, weight=3)
        body.add(sidebar_host, weight=2)

        canvas_host = ttk.Frame(viewer)
        canvas_host.pack(fill="both", expand=True)
        self.canvas = self.tk.Canvas(
            canvas_host, background="#171717", highlightthickness=0)
        x_scroll = ttk.Scrollbar(
            canvas_host, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(
            canvas_host, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_host.rowconfigure(0, weight=1)
        canvas_host.columnconfigure(0, weight=1)

        navigation = ttk.Frame(viewer, padding=(0, 7, 0, 0))
        navigation.pack(fill="x")
        ttk.Button(
            navigation, text="Previous", command=lambda: self._step(-1)
        ).pack(side="left")
        ttk.Button(
            navigation, text="Next", command=lambda: self._step(1)
        ).pack(side="left", padx=(5, 8))
        self.frame_scale = ttk.Scale(
            navigation, from_=1, to=len(self.labels), orient="horizontal",
            command=self._slider_changed)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(navigation, textvariable=self.frame_status, width=30).pack(
            side="left", padx=(8, 0))

        view_controls = ttk.Frame(viewer, padding=(0, 6, 0, 0))
        view_controls.pack(fill="x")
        ttk.Label(view_controls, text="Zoom").pack(side="left")
        ttk.Button(
            view_controls, text="-", width=3,
            command=lambda: self._zoom_step(-1)).pack(side="left", padx=(5, 2))
        zoom_box = ttk.Combobox(
            view_controls, textvariable=self.zoom,
            values=tuple(f"{round(value * 100)}%" for value in ZOOM_LEVELS),
            state="readonly", width=7)
        zoom_box.pack(side="left")
        zoom_box.bind("<<ComboboxSelected>>", lambda _event: self._render())
        ttk.Button(
            view_controls, text="+", width=3,
            command=lambda: self._zoom_step(1)).pack(side="left", padx=(2, 8))
        ttk.Button(
            view_controls, text="Accept frame", command=self._accept_frame
        ).pack(side="left")
        ttk.Button(
            view_controls, text="Accept all frames", command=self._accept_all
        ).pack(side="left", padx=(5, 0))

        self.sidebar_canvas = self.tk.Canvas(
            sidebar_host, background=DARK["background"],
            highlightthickness=0, width=410)
        sidebar_scroll = ttk.Scrollbar(
            sidebar_host, orient="vertical",
            command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")
        sidebar = ttk.Frame(self.sidebar_canvas, padding=(8, 0, 5, 0))
        sidebar_window = self.sidebar_canvas.create_window(
            0, 0, window=sidebar, anchor="nw")
        sidebar.bind(
            "<Configure>", lambda _event: self.sidebar_canvas.configure(
                scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind(
            "<Configure>", lambda event: self.sidebar_canvas.itemconfigure(
                sidebar_window, width=event.width))

        source_box = ttk.LabelFrame(sidebar, text="1. Source edit", padding=8)
        source_box.pack(fill="x")
        source_combo = ttk.Combobox(
            source_box, textvariable=self.source,
            values=tuple(self.source_by_label), state="readonly")
        source_combo.pack(fill="x")
        source_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._source_changed())

        scope = ttk.LabelFrame(sidebar, text="2. Recipients and frames", padding=8)
        scope.pack(fill="x", pady=(8, 0))
        ttk.Label(scope, text="Recipient identity numbers (comma separated)").grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(scope, textvariable=self.recipients).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 7))
        ttk.Label(scope, text="Start frame").grid(row=2, column=0, sticky="w")
        ttk.Label(scope, text="End frame").grid(row=2, column=1, sticky="w")
        ttk.Entry(scope, textvariable=self.start_frame, width=8).grid(
            row=3, column=0, sticky="ew", padx=(0, 4))
        ttk.Entry(scope, textvariable=self.end_frame, width=8).grid(
            row=3, column=1, sticky="ew", padx=(4, 0))
        scope.columnconfigure(0, weight=1)
        scope.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(sidebar, text="3. Proposal settings", padding=8)
        settings.pack(fill="x", pady=(8, 0))
        ttk.Label(settings, text="Eligible background").grid(
            row=0, column=0, sticky="w")
        domain_combo = ttk.Combobox(
            settings, textvariable=self.domain,
            values=tuple(DOMAIN_LABELS), state="readonly")
        domain_combo.grid(row=0, column=1, sticky="ew", padx=(7, 0))
        domain_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._domain_changed())
        for row, (label, variable) in enumerate((
                ("Halo (pixels)", self.halo),
                ("Maximum reach (pixels)", self.maximum_reach),
                ("Sensitivity (0-1)", self.sensitivity),
                ("Competition margin (0-1)", self.margin)), start=1):
            ttk.Label(settings, text=label).grid(
                row=row, column=0, sticky="w", pady=(5, 0))
            ttk.Entry(settings, textvariable=variable, width=9).grid(
                row=row, column=1, sticky="ew", padx=(7, 0), pady=(5, 0))
        settings.columnconfigure(1, weight=1)
        ttk.Button(
            settings, text="Generate proposal", command=self._generate
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(9, 0))

        review = ttk.LabelFrame(sidebar, text="4. Review and commit", padding=8)
        review.pack(fill="x", pady=(8, 0))
        ttk.Label(
            review, textvariable=self.review_status, wraplength=360,
            justify="left").pack(anchor="w")
        ttk.Button(
            review, text="Commit reviewed retuning", command=self._commit
        ).pack(fill="x", pady=(8, 0))
        ttk.Button(
            review, text="Cancel", command=self.window.destroy
        ).pack(fill="x", pady=(5, 0))

        ttk.Label(
            outer, textvariable=self.status, relief="sunken", anchor="w",
            padding=(6, 3)).pack(fill="x", pady=(8, 0))
        self._bind_sidebar_mousewheel(sidebar)

    def _bind(self) -> None:
        self.window.bind("<Left>", lambda _event: self._step(-1))
        self.window.bind("<Right>", lambda _event: self._step(1))
        self.canvas.bind("<MouseWheel>", self._mousewheel_zoom)

    def _bind_sidebar_mousewheel(self, widget) -> None:
        widget.bind(
            "<MouseWheel>", lambda event: self.sidebar_canvas.yview_scroll(
                (-1 if event.delta > 0 else 1) * 3, "units"))
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def _current_source(self) -> dict:
        return self.source_by_label[self.source.get()]

    def _source_changed(self) -> None:
        operation = self._current_source()
        source_id = int(operation["operation_id"])
        start, end = source_operation_frame_range(
            self.controller.operations[:self.controller.position], [source_id],
            len(self.labels))
        self.start_frame.set(str(start))
        self.end_frame.set(str(end))
        released = released_background_from_history(
            self.controller.bundle,
            self.controller.operations[:self.controller.position], [source_id])
        _released, eligible = build_eligible_domain(
            self.labels, released, start, end, "vacated_only", 0)
        active_initial = [identity for identity in self.initial_recipients
                          if np.any(self.labels[start - 1:end] == identity)]
        recipients = (tuple(active_initial) if active_initial
                      else suggest_recipient_identities(
                          self.labels, eligible, start, end))
        self.recipients.set(", ".join(map(str, recipients)))
        self.frame_index = start - 1
        self.frame_scale.configure(from_=start, to=end)
        self.frame_scale.set(start)
        self.proposal = None
        self.reviewed.clear()
        self._render()
        self.status.set(
            f"Recovered the exact vacancy from operation {source_id}; choose recipients")

    def _domain_changed(self) -> None:
        if DOMAIN_LABELS[self.domain.get()] == "vacated_only":
            self.halo.set("0")
        elif self.halo.get().strip() in {"", "0"}:
            self.halo.set("2")
        self.proposal = None
        self.reviewed.clear()
        self._render()

    def _parse_recipients(self) -> tuple[int, ...]:
        tokens = [value.strip() for value in self.recipients.get().split(",")]
        try:
            identities = tuple(sorted(set(int(value) for value in tokens if value)))
        except ValueError as error:
            raise ValueError("recipient identities must be comma-separated integers") from error
        if not identities or any(identity <= 0 for identity in identities):
            raise ValueError("choose at least one positive recipient identity")
        return identities

    def _generate(self) -> None:
        from tkinter import messagebox

        try:
            source_id = int(self._current_source()["operation_id"])
            start = int(self.start_frame.get())
            end = int(self.end_frame.get())
            policy = DOMAIN_LABELS[self.domain.get()]
            request = RetuneRequest(
                source_operation_ids=(source_id,),
                recipient_identities=self._parse_recipients(),
                start_imagej_frame=start, end_imagej_frame=end,
                domain_policy=policy,
                halo_px=(0 if policy == "vacated_only" else int(self.halo.get())),
                maximum_reach_px=float(self.maximum_reach.get()),
                sensitivity=float(self.sensitivity.get()),
                competition_margin=float(self.margin.get()))
            released = released_background_from_history(
                self.controller.bundle,
                self.controller.operations[:self.controller.position], [source_id])
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            self.proposal = propose_mask_retuning(
                self.labels, self.raw, released, request)
            self.reviewed.clear()
            self.frame_index = start - 1
            self.frame_scale.configure(from_=start, to=end)
            self.frame_scale.set(start)
            reclaimed = int(np.count_nonzero(self.proposal.assignments))
            contested = int(np.count_nonzero(self.proposal.contested))
            unresolved = int(np.count_nonzero(
                self.proposal.eligible_domain & (self.proposal.assignments == 0)))
            self.review_status.set(
                f"{reclaimed} pixels reclaimed; {unresolved} unresolved; "
                f"{contested} contested. Review {end - start + 1} frame(s).")
            self.status.set("Proposal generated in memory; no session has been saved")
            self._render()
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot generate retuning proposal", str(error), parent=self.window)
        finally:
            self.window.configure(cursor="")

    def _interval_bounds(self) -> tuple[int, int]:
        if self.proposal is not None:
            request = self.proposal.request
            return request.start_imagej_frame, request.end_imagej_frame
        return int(self.start_frame.get()), int(self.end_frame.get())

    def _slider_changed(self, value: str) -> None:
        start, end = self._interval_bounds()
        index = min(end - 1, max(start - 1, round(float(value)) - 1))
        if index != self.frame_index:
            self.frame_index = index
            self._render()

    def _step(self, step: int) -> None:
        try:
            start, end = self._interval_bounds()
        except ValueError:
            return
        index = min(end - 1, max(start - 1, self.frame_index + step))
        if index != self.frame_index:
            self.frame_index = index
            self.frame_scale.set(index + 1)
            self._render()

    def _zoom_factor(self) -> float:
        return float(self.zoom.get().rstrip("%")) / 100.0

    def _zoom_step(self, direction: int) -> None:
        current = self._zoom_factor()
        index = min(range(len(ZOOM_LEVELS)),
                    key=lambda value: abs(ZOOM_LEVELS[value] - current))
        index = min(len(ZOOM_LEVELS) - 1, max(0, index + direction))
        self.zoom.set(f"{round(ZOOM_LEVELS[index] * 100)}%")
        self._render()

    def _mousewheel_zoom(self, event):
        self._zoom_step(1 if event.delta > 0 else -1)
        return "break"

    def _render(self) -> None:
        from PIL import ImageTk

        frame = self.frame_index
        if self.proposal is None:
            grey = _unit_image(self.raw[frame])
            rgb = np.repeat(grey[..., None], 3, axis=2)
            rgb[_edges(self.labels[frame] > 0)] = np.array(
                [0, 220, 235], np.uint8)
            image = Image.fromarray(rgb)
            self.frame_status.set(f"ImageJ frame {frame + 1}; no proposal")
        else:
            request = self.proposal.request
            offset = frame - (request.start_imagej_frame - 1)
            image = Image.fromarray(render_retune_comparison(
                self.raw[frame], self.labels[frame],
                self.proposal.assignments[offset],
                self.proposal.eligible_domain[offset],
                self.proposal.released_domain[offset],
                self.proposal.contested[offset], frame + 1))
            row = self.proposal.per_frame[offset]
            accepted = "accepted" if frame + 1 in self.reviewed else "not accepted"
            self.frame_status.set(
                f"Frame {frame + 1}: {row['reclaimed_pixels']} reclaimed; {accepted}")
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

    def _accept_frame(self) -> None:
        if self.proposal is None:
            self.status.set("Generate a proposal before accepting frames")
            return
        self.reviewed.add(self.frame_index + 1)
        self.status.set(f"Accepted ImageJ frame {self.frame_index + 1}")
        self._render()

    def _accept_all(self) -> None:
        if self.proposal is None:
            self.status.set("Generate a proposal before accepting frames")
            return
        request = self.proposal.request
        self.reviewed = set(range(
            request.start_imagej_frame, request.end_imagej_frame + 1))
        self.status.set(f"Accepted all {len(self.reviewed)} proposal frames")
        self._render()

    def _commit(self) -> None:
        from tkinter import messagebox

        if self.proposal is None:
            self.status.set("Generate and review a proposal before committing")
            return
        try:
            operation = build_retune_operation(self.proposal, self.reviewed)
            reclaimed = int(np.count_nonzero(self.proposal.assignments))
            recipients = ", ".join(map(
                str, self.proposal.request.recipient_identities))
            if not messagebox.askyesno(
                    "Commit mask retuning",
                    f"Assign {reclaimed} background pixels to identities "
                    f"{recipients}?\n\nThis creates a separate reversible history step.",
                    parent=self.window):
                return
            self.window.configure(cursor="watch")
            self.window.update_idletasks()
            path = self.controller.retune_masks_after_edit(
                operation, self.output_dir)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot commit mask retuning", str(error), parent=self.window)
            return
        finally:
            self.window.configure(cursor="")
        self.window.destroy()
        self.on_saved(path)


def open_mask_retuning_workspace(
        parent, controller: EditingHistoryController, labels: np.ndarray,
        initial_recipients: list[int], output_dir: Path,
        on_saved: Callable[[Path], None]) -> MaskRetuningWorkspace:
    return MaskRetuningWorkspace(
        parent, controller, labels, initial_recipients, output_dir, on_saved)
