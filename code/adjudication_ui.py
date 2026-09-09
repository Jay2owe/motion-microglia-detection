"""Click-through adjudication of flagged events for ISSUE-005.

One window, one flagged event at a time: a zoomable view of the raw movie with
the accepted identity outlines and numbers drawn on top, a frame slider that
spans the event, a seven-frame strip to jump around it, and three verdict keys.
Every verdict is written straight back into the adjudication sheet
(``review_cases.csv``) so nothing has to be typed into a spreadsheet.

Launch from the Motion root::

    python code/adjudication_ui.py
    python code/adjudication_ui.py --sheet <review_cases.csv> --inputs <inputs.csv>

Keys: F failure, C control, U undecidable, X clear, Enter save and next
unlabelled, PageDown/PageUp next/previous case, Left/Right frame,
Shift+Left/Right five frames, Home/End event start/end, Space play, 1-7 jump
to the strip frames, +/- or wheel zoom, R recentre, O outlines, I fill,
N numbers, B unclaimed bodies, M marker, Ctrl+O open the full-field review
TIFF. Click a cell to make it the expected identity; drag to pan.

The sheet is the only file this program writes. It never touches a label
stack, a catalogue or a ledger.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import datetime as dt
import os
import re
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
import tifffile
from PIL import Image, ImageTk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_editing_theme import DARK, apply_dark_theme  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ISSUE_DIR = PROJECT / "issues/04_95_A4/ISSUE-005-learned-event-adjudication"
DEFAULT_SHEET = ISSUE_DIR / "review_cases.csv"
DEFAULT_INPUTS = (ISSUE_DIR / "analysis/learned_event_adjudication_tuning"
                  / "inputs/inputs.csv")

UNDECIDED = "unadjudicated"
VERDICTS = ("failure", "control", "undecidable")
VERDICT_COLOUR = {"failure": "#c0392b", "control": "#0e8f8f",
                  "undecidable": "#8a7d3a", UNDECIDED: DARK["panel"]}
ZOOM_LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_ZOOM = 4.0
CONTRAST_PERCENTILES = (0.5, 99.8)     # whole movie, fixed per well
SOURCE_FRAME_OFFSET = 2                # ImageJ frame = review frame + 2
MARKER_RADIUS = 10                     # raw pixels
MARKER = "#ffe100"
UNCLAIMED_RGB = (235, 235, 235)
THUMB_ROLES = ("before", "start", "quarter", "middle", "three-quarter",
               "end", "after")
THUMB_HALF = 40                        # raw pixels each side, as the montage
PLAY_MS = 160
PAD_FRAMES = 3
STATUS_FILTERS = ("unlabelled", "all", "failure", "control", "undecidable")
PRIORITY_FILTERS = ("high", "all")


# ----------------------------------------------------------------- sheet I/O

def read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    return columns, rows, newline


def write_sheet(path: Path, columns: list[str], rows: list[dict[str, str]],
                newline: str = "\r\n") -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore",
                                lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def backup_sheet(path: Path, stamp: str | None = None) -> Path:
    stamp = stamp or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = path.parent / "backups"
    folder.mkdir(exist_ok=True)
    target = folder / f"{path.stem}_{stamp}.csv"
    shutil.copy2(path, target)
    return target


def resolve_cases(rows: list[dict[str, str]], spec: str) -> list[int]:
    """Row indices for a comma-separated list of case ids or event ids.

    A token is a case id (``I005-95_A4-D017``), an event id (``D017``) or a
    well-qualified event id (``95_A4:D017`` or ``95_A4 D017``). A bare event id
    must be unique across wells.
    """
    picked: list[int] = []
    for token in [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]:
        parts = token.replace(":", " ").split()
        well = parts[0] if len(parts) == 2 else None
        key = parts[-1].lower()
        hits = [i for i, row in enumerate(rows)
                if row["case_id"].lower() == key
                or (row["event_id"].lower() == key and (well is None or row["well"] == well))]
        if not hits:
            raise SystemExit(f"no sheet row matches {token!r}")
        if len(hits) > 1:
            wells = ", ".join(rows[i]["well"] for i in hits)
            raise SystemExit(f"{token!r} matches several wells ({wells}); qualify it as <well>:{parts[-1]}")
        if hits[0] not in picked:
            picked.append(hits[0])
    return picked


def build_queue(rows: list[dict[str, str]], well: str = "All",
                priority: str = "high", status: str = "unlabelled",
                only: list[int] | None = None) -> list[int]:
    """Row indices in sheet order (or in ``only`` order) that pass the filters."""
    picked = []
    order = list(range(len(rows))) if only is None else list(only)
    for index in order:
        row = rows[index]
        if well != "All" and row.get("well") != well:
            continue
        if priority == "high" and row.get("priority") != "high":
            continue
        verdict = row.get("case_type", UNDECIDED) or UNDECIDED
        if status == "unlabelled" and verdict != UNDECIDED:
            continue
        if status in VERDICTS and verdict != status:
            continue
        picked.append(index)
    return picked


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {UNDECIDED: 0, **{v: 0 for v in VERDICTS}}
    for row in rows:
        counts[row.get("case_type") or UNDECIDED] = \
            counts.get(row.get("case_type") or UNDECIDED, 0) + 1
    return counts


def parse_identities(text: str) -> list[int]:
    return [int(t) for t in re.findall(r"\d+", text or "")]


def format_evidence(text: str) -> str:
    parts = [p.strip() for p in (text or "").split(" ; ") if p.strip()]
    return "\n".join(f"- {p}" for p in parts) if parts else "(no evidence text)"


# ------------------------------------------------------------- pixel helpers

def identity_colour(identity: int) -> tuple[int, int, int]:
    """Same deterministic colour per identity as the navigation montage."""
    hue = (identity * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def colour_lut(max_identity: int) -> np.ndarray:
    lut = np.zeros((max(int(max_identity), 1) + 1, 3), dtype=np.uint8)
    for identity in range(1, lut.shape[0]):
        lut[identity] = identity_colour(identity)
    return lut


def to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def label_edges(lab: np.ndarray) -> np.ndarray:
    """Pixels of a labelled object whose 4-neighbour has a different label."""
    edge = np.zeros(lab.shape, dtype=bool)
    edge[1:] |= lab[1:] != lab[:-1]
    edge[:-1] |= lab[:-1] != lab[1:]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return edge & (lab > 0)


def thumb_frames(first: int, last: int, n_frames: int) -> list[int]:
    span = max(last - first, 0)
    inner = [first + int(round(span * q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    frames = [first - 1] + inner + [last + 1]
    return [int(min(max(f, 0), n_frames - 1)) for f in frames]


def crop_origin(centre: float, half: int, size: int) -> int:
    origin = int(round(centre)) - half
    return int(min(max(origin, 0), max(size - (2 * half + 1), 0)))


@dataclass
class WellStacks:
    stem: str
    raw: np.ndarray
    labels: np.ndarray
    unclaimed: np.ndarray | None
    low: float
    high: float
    lut: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.raw.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.raw.shape[1]), int(self.raw.shape[2])


def compose_frame(well: WellStacks, frame: int, brightness: float = 1.0,
                  outlines: bool = True, fill: bool = False,
                  unclaimed: bool = False,
                  focus: tuple[int, ...] = ()) -> np.ndarray:
    """RGB uint8 picture of one frame with the requested overlays."""
    raw = well.raw[frame].astype(np.float32)
    unit = np.clip((raw - well.low) / max(well.high - well.low, 1e-6)
                   * brightness, 0.0, 1.0)
    grey = (unit * 255.0).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    lab = well.labels[frame]
    if fill:
        mask = lab > 0
        rgb[mask] = (0.65 * rgb[mask] + 0.35 * well.lut[lab[mask]]).astype(np.uint8)
    if unclaimed and well.unclaimed is not None:
        rgb[label_edges(well.unclaimed[frame])] = UNCLAIMED_RGB
    if outlines:
        edge = label_edges(lab)
        colour = well.lut[lab[edge]].astype(np.float32)
        if focus:
            dim = ~np.isin(lab[edge], np.asarray(focus))
            colour[dim] = 0.45 * colour[dim] + 0.55 * 96.0
        rgb[edge] = colour.astype(np.uint8)
    return rgb


class WellStore:
    """Lazy loader for the frozen raw, label and unclaimed stacks per well."""

    def __init__(self, inputs_csv: Path):
        self.paths: dict[str, dict[str, Path]] = {}
        with open(inputs_csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                for kind in ("raw", "labels", "unclaimed"):
                    suffix = f"_{kind}"
                    if row["name"].endswith(suffix):
                        stem = row["name"][:-len(suffix)]
                        self.paths.setdefault(stem, {})[kind] = Path(row["source_path"])
        self.loaded: dict[str, WellStacks] = {}

    def wells(self) -> list[str]:
        return sorted(self.paths)

    def get(self, stem: str, report=None) -> WellStacks:
        if stem in self.loaded:
            return self.loaded[stem]
        paths = self.paths[stem]
        if report:
            report(f"loading {stem} raw movie and accepted labels ...")
        raw = tifffile.imread(paths["raw"])
        labels = tifffile.imread(paths["labels"])
        unclaimed = tifffile.imread(paths["unclaimed"]) if "unclaimed" in paths else None
        for kind, stack in (("labels", labels), ("unclaimed", unclaimed)):
            if stack is not None and stack.shape != raw.shape:
                raise SystemExit(f"{stem}: {kind} stack {stack.shape} does not match "
                                 f"the raw movie {raw.shape}")
        sample = raw[::5, ::3, ::3].ravel()
        low, high = np.percentile(sample, CONTRAST_PERCENTILES)
        well = WellStacks(stem, raw, labels, unclaimed, float(low), float(high),
                          colour_lut(int(labels.max())))
        self.loaded[stem] = well
        return well


# ------------------------------------------------------------------- the app

class AdjudicationApp:
    def __init__(self, root: tk.Tk, sheet: Path, inputs: Path,
                 backup: bool = True, only: str | None = None):
        self.root = root
        self.sheet_path = sheet
        self.columns, self.rows, self.newline = read_sheet(sheet)
        for needed in ("case_type", "expected_identity", "notes", "adjudicated_on"):
            if needed not in self.columns:
                raise SystemExit(f"sheet lacks column {needed!r}: {sheet}")
        self.only = resolve_cases(self.rows, only) if only else None
        self.backup_path = backup_sheet(sheet) if backup else None
        self.store = WellStore(inputs)
        self.queue: list[int] = []
        self.position = -1
        self.row: dict[str, str] | None = None
        self.well: WellStacks | None = None
        self.frame = 0
        self.zoom = DEFAULT_ZOOM
        self.centre = (0.0, 0.0)
        self.focus: tuple[int, ...] = ()
        self.thumb_map: list[int] = []
        self.playing = False
        self._play_job = None
        self._slider_guard = False
        self._drag = None
        self._compose_cache: dict[tuple, np.ndarray] = {}
        self._photo = None
        self._thumb_photos: list[ImageTk.PhotoImage] = []
        self._thumb_boxes: list[tuple[int, int, int, int]] = []
        self._build()
        self._apply_filters(keep_position=False)

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        root = self.root
        root.title("Motion - event adjudication (ISSUE-005)")
        apply_dark_theme(root)
        style = ttk.Style(root)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Big.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Muted.TLabel", foreground=DARK["muted"])
        style.configure("Mono.TLabel", font=("Consolas", 10))
        for name, colour in VERDICT_COLOUR.items():
            style.configure(f"{name}.TButton", background=DARK["panel"],
                            foreground=DARK["foreground"], padding=(10, 8),
                            font=("Segoe UI", 11, "bold"))
            style.map(f"{name}.TButton",
                      background=[("pressed", colour), ("active", DARK["border"])],
                      foreground=[("!disabled", DARK["foreground"])])
            style.configure(f"{name}On.TButton", background=colour,
                            foreground="#ffffff", padding=(10, 8),
                            font=("Segoe UI", 11, "bold"))
            style.map(f"{name}On.TButton",
                      background=[("pressed", colour), ("active", colour)],
                      foreground=[("!disabled", "#ffffff")])
        root.geometry("1560x980")
        root.minsize(1100, 700)

        chosen = self.only is not None
        self.var_well = tk.StringVar(value="All")
        self.var_priority = tk.StringVar(value="all" if chosen else "high")
        self.var_status = tk.StringVar(value="all" if chosen else "unlabelled")
        self.var_jump = tk.StringVar()
        self.var_outlines = tk.BooleanVar(value=True)
        self.var_fill = tk.BooleanVar(value=False)
        self.var_numbers = tk.BooleanVar(value=True)
        self.var_unclaimed = tk.BooleanVar(value=False)
        self.var_marker = tk.BooleanVar(value=True)
        self.var_brightness = tk.DoubleVar(value=1.0)
        self.var_advance = tk.BooleanVar(value=True)
        self.var_identity = tk.StringVar()
        self.var_notes = tk.StringVar()
        self.var_progress = tk.StringVar()
        self.var_frame_text = tk.StringVar()
        self.var_status_text = tk.StringVar(value="ready")

        # ---- top bar
        top = ttk.Frame(root, padding=(8, 6))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Well").pack(side="left")
        wells = ["All"] + self.store.wells()
        ttk.Combobox(top, textvariable=self.var_well, values=wells, width=8,
                     state="readonly").pack(side="left", padx=(4, 12))
        ttk.Label(top, text="Priority").pack(side="left")
        ttk.Combobox(top, textvariable=self.var_priority, values=PRIORITY_FILTERS,
                     width=6, state="readonly").pack(side="left", padx=(4, 12))
        ttk.Label(top, text="Show").pack(side="left")
        ttk.Combobox(top, textvariable=self.var_status, values=STATUS_FILTERS,
                     width=11, state="readonly").pack(side="left", padx=(4, 12))
        ttk.Label(top, text="Jump to").pack(side="left")
        jump = ttk.Entry(top, textvariable=self.var_jump, width=18)
        jump.pack(side="left", padx=(4, 4))
        jump.bind("<Return>", lambda e: self._jump())
        ttk.Button(top, text="Go", command=self._jump).pack(side="left", padx=(0, 12))
        ttk.Button(top, text="Open review TIFF  (Ctrl+O)",
                   command=self._open_review_tiff).pack(side="left")
        ttk.Button(top, text="Open sheet folder",
                   command=lambda: self._open_path(self.sheet_path.parent)).pack(side="left", padx=6)
        for var in (self.var_well, self.var_priority, self.var_status):
            var.trace_add("write", lambda *_: self._apply_filters(keep_position=True))

        # ---- status bar
        status = ttk.Frame(root, padding=(8, 3))
        status.pack(side="bottom", fill="x")
        ttk.Label(status, textvariable=self.var_status_text, style="Muted.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.var_progress).pack(side="right")

        # ---- main split
        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.pack(side="top", fill="both", expand=True)
        viewer = ttk.Frame(paned)
        sidebar = ttk.Frame(paned, width=430)
        paned.add(viewer, weight=4)
        paned.add(sidebar, weight=0)

        # viewer: canvas / frame controls / thumbnails
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewer, background="#000000", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", self._wheel)

        controls = ttk.Frame(viewer, padding=(6, 4))
        controls.grid(row=1, column=0, sticky="ew")
        ttk.Button(controls, text="<", width=3, command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(controls, text=">", width=3, command=lambda: self._step(1)).pack(side="left", padx=(2, 6))
        self.play_button = ttk.Button(controls, text="Play", width=6, command=self._toggle_play)
        self.play_button.pack(side="left", padx=(0, 8))
        slider_box = ttk.Frame(controls)
        slider_box.pack(side="left", fill="x", expand=True)
        self.slider = ttk.Scale(slider_box, from_=0, to=98, orient="horizontal",
                                command=self._slider_changed)
        self.slider.pack(side="top", fill="x")
        self.span_bar = tk.Canvas(slider_box, height=6, background=DARK["background"],
                                  highlightthickness=0)
        self.span_bar.pack(side="top", fill="x", padx=6)
        self.span_bar.bind("<Configure>", lambda e: self._draw_span_bar())
        ttk.Label(controls, textvariable=self.var_frame_text,
                  style="Mono.TLabel").pack(side="left", padx=(8, 0))

        toggles = ttk.Frame(viewer, padding=(6, 2))
        toggles.grid(row=3, column=0, sticky="ew")
        for text, var in (("Outlines (O)", self.var_outlines), ("Fill (I)", self.var_fill),
                          ("Numbers (N)", self.var_numbers),
                          ("Unclaimed bodies (B)", self.var_unclaimed),
                          ("Marker (M)", self.var_marker)):
            ttk.Checkbutton(toggles, text=text, variable=var,
                            command=self._overlay_changed).pack(side="left", padx=(0, 10))
        ttk.Label(toggles, text="Brightness").pack(side="left", padx=(12, 4))
        ttk.Scale(toggles, from_=0.3, to=3.0, variable=self.var_brightness,
                  orient="horizontal", length=140,
                  command=lambda v: self._overlay_changed()).pack(side="left")
        ttk.Label(toggles, text="Zoom").pack(side="left", padx=(12, 4))
        self.zoom_label = ttk.Label(toggles, text="4.0x", width=5)
        self.zoom_label.pack(side="left")
        ttk.Button(toggles, text="-", width=2, command=lambda: self._zoom_step(-1)).pack(side="left")
        ttk.Button(toggles, text="+", width=2, command=lambda: self._zoom_step(1)).pack(side="left", padx=(2, 6))
        ttk.Button(toggles, text="Recentre (R)", command=self._recentre).pack(side="left")

        self.thumbs = tk.Canvas(viewer, height=150, background=DARK["background"],
                                highlightthickness=0)
        self.thumbs.grid(row=4, column=0, sticky="ew", pady=(4, 2))
        self.thumbs.bind("<Configure>", lambda e: self._render_thumbs())
        self.thumbs.bind("<ButtonPress-1>", self._thumb_click)

        # sidebar
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)
        sidebar.rowconfigure(5, weight=1)
        head = ttk.Frame(sidebar, padding=(10, 8, 10, 4))
        head.grid(row=0, column=0, sticky="ew")
        self.lbl_case = ttk.Label(head, text="", style="Big.TLabel")
        self.lbl_case.pack(anchor="w")
        self.lbl_family = ttk.Label(head, text="", style="Muted.TLabel")
        self.lbl_family.pack(anchor="w")
        self.info = tk.Text(sidebar, height=7, width=46, wrap="word",
                            background=DARK["field"], foreground=DARK["foreground"],
                            relief="flat", font=("Consolas", 10), padx=8, pady=6)
        self.info.grid(row=1, column=0, sticky="ew", padx=10)
        self.info.configure(state="disabled")
        evidence_box = ttk.Labelframe(sidebar, text="Evidence", padding=4)
        evidence_box.grid(row=2, column=0, sticky="nsew", padx=10, pady=(6, 4))
        self.evidence = tk.Text(evidence_box, wrap="word", background=DARK["field"],
                                foreground=DARK["muted"], relief="flat",
                                font=("Segoe UI", 9), padx=6, pady=4, height=7,
                                width=46)
        scroll = ttk.Scrollbar(evidence_box, command=self.evidence.yview)
        self.evidence.configure(yscrollcommand=scroll.set, state="disabled")
        self.evidence.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        verdict = ttk.Labelframe(sidebar, text="Verdict", padding=8)
        verdict.grid(row=3, column=0, sticky="ew", padx=10, pady=4)
        verdict.columnconfigure((0, 1, 2), weight=1)
        self.verdict_buttons: dict[str, ttk.Button] = {}
        for column, (name, text) in enumerate((("failure", "Failure  (F)"),
                                               ("control", "Control  (C)"),
                                               ("undecidable", "Undecidable  (U)"))):
            button = ttk.Button(verdict, text=text, style=f"{name}.TButton",
                                command=lambda n=name: self._set_verdict(n))
            button.grid(row=0, column=column, sticky="ew", padx=2)
            self.verdict_buttons[name] = button
        ttk.Label(verdict, text="Expected identity (click a cell, or type)").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.entry_identity = ttk.Entry(verdict, textvariable=self.var_identity, width=12)
        self.entry_identity.grid(row=1, column=2, sticky="ew", pady=(8, 0))
        ttk.Label(verdict, text="Notes").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.entry_notes = ttk.Entry(verdict, textvariable=self.var_notes)
        self.entry_notes.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        self.lbl_saved = ttk.Label(verdict, text="", style="Muted.TLabel")
        self.lbl_saved.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        nav = ttk.Frame(verdict)
        nav.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(nav, text="Back (PgUp)", command=lambda: self._move(-1)).pack(side="left")
        ttk.Button(nav, text="Skip (PgDn)", command=lambda: self._move(1)).pack(side="left", padx=4)
        ttk.Button(nav, text="Clear (X)", command=lambda: self._set_verdict(UNDECIDED)).pack(side="left", padx=4)
        ttk.Button(nav, text="Save & next  (Enter)", command=self._save_and_next).pack(side="right")
        ttk.Checkbutton(verdict, text="Control / undecidable also advance",
                        variable=self.var_advance).grid(row=5, column=0, columnspan=3,
                                                        sticky="w", pady=(6, 0))

        queue_box = ttk.Labelframe(sidebar, text="Queue", padding=4)
        queue_box.grid(row=5, column=0, sticky="nsew", padx=10, pady=(4, 8))
        self.tree = ttk.Treeview(queue_box, columns=("case", "family", "score", "verdict"),
                                 show="headings", selectmode="browse", height=8)
        for column, text, width in (("case", "case", 150), ("family", "family", 140),
                                    ("score", "score", 48), ("verdict", "verdict", 84)):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, anchor="w", stretch=column == "family")
        for name, colour in VERDICT_COLOUR.items():
            if name != UNDECIDED:
                self.tree.tag_configure(name, foreground=colour)
        tscroll = ttk.Scrollbar(queue_box, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tscroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)

        help_text = ("F/C/U verdict, X clear, Enter save+next, PgUp/PgDn move, "
                     "Left/Right frame (Shift: 5), Home/End event span, Space play, "
                     "1-7 strip frames, +/- zoom, R recentre, drag to pan, click a cell "
                     "to pick its identity")
        ttk.Label(sidebar, text=help_text, style="Muted.TLabel", wraplength=400,
                  font=("Segoe UI", 8)).grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 6))

        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------- filters
    def _apply_filters(self, keep_position: bool) -> None:
        current = self.queue[self.position] if 0 <= self.position < len(self.queue) else None
        self.queue = build_queue(self.rows, self.var_well.get(), self.var_priority.get(),
                                 self.var_status.get(), self.only)
        self._fill_tree()
        if not self.queue:
            self.position = -1
            self.row = None
            self._set_status("no cases match these filters")
            self._update_progress()
            self.canvas.delete("all")
            return
        if keep_position and current in self.queue:
            self._show(self.queue.index(current))
        else:
            self._show(0)

    def _fill_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index in self.queue:
            row = self.rows[index]
            verdict = row.get("case_type") or UNDECIDED
            self.tree.insert("", "end", iid=str(index),
                             values=(row["case_id"], row["mechanism"],
                                     f"{float(row['disruption_score'] or 0):.0f}",
                                     "" if verdict == UNDECIDED else verdict),
                             tags=(verdict,))

    def _refresh_tree_row(self, index: int) -> None:
        if not self.tree.exists(str(index)):
            return
        row = self.rows[index]
        verdict = row.get("case_type") or UNDECIDED
        self.tree.item(str(index), values=(row["case_id"], row["mechanism"],
                                            f"{float(row['disruption_score'] or 0):.0f}",
                                            "" if verdict == UNDECIDED else verdict),
                       tags=(verdict,))

    def _tree_selected(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        index = int(selected[0])
        if index in self.queue and self.queue.index(index) != self.position:
            self._show(self.queue.index(index))

    def _jump(self) -> None:
        text = self.var_jump.get().strip()
        if not text:
            return
        tokens = text.replace(",", " ").split()
        hit = None
        for index, row in enumerate(self.rows):
            if row["case_id"].lower() == text.lower():
                hit = index
                break
            if row["event_id"].lower() == tokens[-1].lower() and (
                    len(tokens) == 1 or row["well"].lower() == tokens[0].lower()):
                hit = index if hit is None else hit
        if hit is None:
            self._set_status(f"no case matches {text!r}")
            return
        if hit not in self.queue:
            self.var_status.set("all")
            self.var_priority.set("all")
            if self.var_well.get() != "All" and self.rows[hit]["well"] != self.var_well.get():
                self.var_well.set("All")
            self._apply_filters(keep_position=False)
        self._show(self.queue.index(hit))

    # ------------------------------------------------------------- showing
    def _show(self, position: int) -> None:
        self._stop_play()
        self.position = position
        index = self.queue[position]
        self.row = row = self.rows[index]
        self.well = self.store.get(row["well"], self._set_status)
        self._compose_cache.clear()
        n = self.well.n_frames
        first = int(row["review_frame_start"]) - 1
        last = int(row["review_frame_end"]) - 1
        self.event_span = (max(first, 0), min(last, n - 1))
        self.event_xy = (float(row["x"]), float(row["y"]))
        self.focus = tuple(sorted(set(parse_identities(row["identities"])
                                      + parse_identities(row["accepted_before"])
                                      + parse_identities(row["accepted_after"]))))
        self.thumb_map = thumb_frames(first, last, n)
        self.slider.configure(to=n - 1)
        self.centre = (self.event_xy[0] + 0.5, self.event_xy[1] + 0.5)
        self.var_identity.set(row.get("expected_identity", ""))
        self.var_notes.set(row.get("notes", ""))
        self._paint_verdict_buttons()
        self._fill_info()
        self._set_frame(self.event_span[0])
        self._render_thumbs()
        self._draw_span_bar()
        self._update_progress()
        if self.tree.exists(str(index)):
            self.tree.selection_set(str(index))
            self.tree.see(str(index))
        self._set_status(f"{row['case_id']}: {row['mechanism']} "
                         f"score {float(row['disruption_score']):.1f}")
        self.canvas.focus_set()

    def _fill_info(self) -> None:
        row = self.row
        self.lbl_case.configure(text=row["case_id"])
        self.lbl_family.configure(
            text=f"{row['well']}  ·  {row['event_id']}  ·  {row['mechanism']}  ·  "
                 f"score {float(row['disruption_score']):.1f}  ·  {row['priority']} priority")
        r0, r1 = int(row["review_frame_start"]), int(row["review_frame_end"])
        lines = [
            f"frames    review {r0}-{r1}   ImageJ {r0 + SOURCE_FRAME_OFFSET}-{r1 + SOURCE_FRAME_OFFSET}",
            f"location  x {float(row['x']):.1f}  y {float(row['y']):.1f}",
            f"before    {row['accepted_before'] or '-'}",
            f"after     {row['accepted_after'] or '-'}",
            f"identities {row['identities'] or '-'}",
            f"tracks    {row['track_ref'] or '-'}",
        ]
        if row.get("adjudicated_on"):
            lines.append(f"decided   {row['case_type']} on {row['adjudicated_on']}")
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        self.evidence.configure(state="normal")
        self.evidence.delete("1.0", "end")
        self.evidence.insert("1.0", format_evidence(row.get("evidence", "")))
        self.evidence.configure(state="disabled")

    def _paint_verdict_buttons(self) -> None:
        current = (self.row or {}).get("case_type") or UNDECIDED
        for name, button in self.verdict_buttons.items():
            button.configure(style=f"{name}On.TButton" if name == current else f"{name}.TButton")
        when = (self.row or {}).get("adjudicated_on") or ""
        self.lbl_saved.configure(text=f"saved {current} at {when}" if when else "not yet decided")

    def _update_progress(self) -> None:
        counts = label_counts(self.rows)
        high = [r for r in self.rows if r["priority"] == "high"]
        high_done = sum(1 for r in high if (r["case_type"] or UNDECIDED) != UNDECIDED)
        total_done = len(self.rows) - counts[UNDECIDED]
        where = f"queue {self.position + 1}/{len(self.queue)}" if self.queue else "queue empty"
        if self.only is not None:
            where = f"chosen cases: {where}"
        self.var_progress.set(
            f"{where}   ·   high {high_done}/{len(high)}   all {total_done}/{len(self.rows)}"
            f"   ·   failure {counts['failure']}  control {counts['control']}  "
            f"undecidable {counts['undecidable']}")

    def _set_status(self, text: str) -> None:
        self.var_status_text.set(text)
        self.root.update_idletasks()

    # ------------------------------------------------------------- verdicts
    def _set_verdict(self, verdict: str) -> None:
        if self.row is None:
            return
        self._commit(verdict)
        if verdict == "failure":
            self.entry_identity.focus_set()
            self.entry_identity.selection_range(0, "end")
            self._set_status("failure saved; click the correct cell or type its identity, then Enter")
        elif verdict in ("control", "undecidable") and self.var_advance.get():
            self._next_unlabelled()
        elif verdict == UNDECIDED:
            self._set_status("verdict cleared")

    def _commit(self, verdict: str | None = None) -> None:
        row = self.row
        if row is None:
            return
        if verdict is None:
            verdict = row.get("case_type") or UNDECIDED
        row["case_type"] = verdict
        row["expected_identity"] = self.var_identity.get().strip()
        row["notes"] = self.var_notes.get().strip()
        row["adjudicated_on"] = ("" if verdict == UNDECIDED
                                 else dt.datetime.now().isoformat(timespec="seconds"))
        write_sheet(self.sheet_path, self.columns, self.rows, self.newline)
        self._paint_verdict_buttons()
        self._refresh_tree_row(self.queue[self.position])
        self._update_progress()
        self._fill_info()
        self._set_status(f"saved {row['case_id']} as {verdict}"
                         + (f" (expected {row['expected_identity']})" if row["expected_identity"] else ""))

    def _save_and_next(self) -> None:
        if self.row is None:
            return
        verdict = self.row.get("case_type") or UNDECIDED
        if verdict == UNDECIDED:
            self._set_status("choose a verdict first: F failure, C control, U undecidable")
            return
        self._commit(verdict)
        self._next_unlabelled()

    def _next_unlabelled(self) -> None:
        n = len(self.queue)
        for step in range(1, n + 1):
            position = (self.position + step) % n
            row = self.rows[self.queue[position]]
            if (row.get("case_type") or UNDECIDED) == UNDECIDED:
                self._show(position)
                return
        if self.var_status.get() == "unlabelled":
            self._apply_filters(keep_position=False)
            if self.queue:
                return
        self._set_status("every case in this queue has a verdict")

    def _move(self, delta: int) -> None:
        if not self.queue:
            return
        self._show((self.position + delta) % len(self.queue))

    # ------------------------------------------------------------- frames
    def _set_frame(self, frame: int) -> None:
        if self.well is None:
            return
        self.frame = int(min(max(frame, 0), self.well.n_frames - 1))
        self._slider_guard = True
        self.slider.set(self.frame)
        self._slider_guard = False
        review = self.frame + 1
        inside = self.event_span[0] <= self.frame <= self.event_span[1]
        self.var_frame_text.set(
            f"review {review:3d}  ImageJ {review + SOURCE_FRAME_OFFSET:3d}  "
            f"{'in event' if inside else '        '}")
        self._render()
        self._highlight_thumb()

    def _slider_changed(self, value: str) -> None:
        if self._slider_guard:
            return
        self._set_frame(int(round(float(value))))

    def _step(self, delta: int) -> None:
        self._set_frame(self.frame + delta)

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_play()
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            self._play_tick()

    def _stop_play(self) -> None:
        self.playing = False
        if self._play_job is not None:
            self.root.after_cancel(self._play_job)
            self._play_job = None
        self.play_button.configure(text="Play")

    def _play_tick(self) -> None:
        if not self.playing or self.well is None:
            return
        lo = max(self.event_span[0] - PAD_FRAMES, 0)
        hi = min(self.event_span[1] + PAD_FRAMES, self.well.n_frames - 1)
        nxt = self.frame + 1 if lo <= self.frame < hi else lo
        self._set_frame(nxt)
        self._play_job = self.root.after(PLAY_MS, self._play_tick)

    def _draw_span_bar(self) -> None:
        self.span_bar.delete("all")
        if self.well is None:
            return
        width = max(self.span_bar.winfo_width(), 10)
        n = max(self.well.n_frames - 1, 1)
        x0 = width * self.event_span[0] / n
        x1 = width * self.event_span[1] / n
        self.span_bar.create_rectangle(0, 2, width, 4, fill=DARK["border"], width=0)
        self.span_bar.create_rectangle(x0, 0, max(x1, x0 + 2), 6, fill=MARKER, width=0)

    # ------------------------------------------------------------- render
    def _composed(self, frame: int) -> np.ndarray:
        key = (frame, round(self.var_brightness.get(), 3), self.var_outlines.get(),
               self.var_fill.get(), self.var_unclaimed.get(), self.focus)
        picture = self._compose_cache.get(key)
        if picture is None:
            picture = compose_frame(self.well, frame, key[1], key[2], key[3], key[4], self.focus)
            if len(self._compose_cache) > 48:
                self._compose_cache.clear()
            self._compose_cache[key] = picture
        return picture

    def _to_canvas(self, u: float, v: float) -> tuple[float, float]:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        return ((u - self.centre[0]) * self.zoom + cw / 2,
                (v - self.centre[1]) * self.zoom + ch / 2)

    def _to_raw(self, px: float, py: float) -> tuple[float, float]:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        return ((px - cw / 2) / self.zoom + self.centre[0],
                (py - ch / 2) / self.zoom + self.centre[1])

    def _render(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        if self.well is None:
            return
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 4 or ch < 4:
            return
        height, width = self.well.shape
        z = self.zoom
        u0, v0 = self._to_raw(0, 0)
        u1, v1 = self._to_raw(cw, ch)
        x0, y0 = int(max(np.floor(u0), 0)), int(max(np.floor(v0), 0))
        x1, y1 = int(min(np.ceil(u1) + 1, width)), int(min(np.ceil(v1) + 1, height))
        if x1 <= x0 or y1 <= y0:
            return
        region = self._composed(self.frame)[y0:y1, x0:x1]
        image = Image.fromarray(region).resize(
            (max(int(round((x1 - x0) * z)), 1), max(int(round((y1 - y0) * z)), 1)),
            Image.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        ox, oy = self._to_canvas(x0, y0)
        canvas.create_image(ox, oy, image=self._photo, anchor="nw")
        if self.var_numbers.get():
            self._draw_numbers(x0, y0, x1, y1)
        if self.var_marker.get():
            mx, my = self._to_canvas(self.event_xy[0] + 0.5, self.event_xy[1] + 0.5)
            r = MARKER_RADIUS * z
            canvas.create_oval(mx - r, my - r, mx + r, my + r, outline=MARKER, width=2)
            canvas.create_line(mx - r * 1.4, my, mx - r * 1.1, my, fill=MARKER, width=2)
            canvas.create_line(mx + r * 1.1, my, mx + r * 1.4, my, fill=MARKER, width=2)
        canvas.create_text(8, 8, anchor="nw", fill=DARK["muted"],
                           font=("Segoe UI", 9),
                           text=f"{self.well.stem}   review frame {self.frame + 1}   "
                                f"ImageJ {self.frame + 1 + SOURCE_FRAME_OFFSET}   zoom {z:g}x")

    def _draw_numbers(self, x0: int, y0: int, x1: int, y1: int) -> None:
        lab = self.well.labels[self.frame][y0:y1, x0:x1]
        ids = np.unique(lab)
        ids = ids[ids > 0]
        if ids.size == 0:
            return
        centres = ndi.center_of_mass(np.ones(lab.shape, dtype=np.uint8), lab, ids)
        size = 9 if self.zoom < 2 else 11 if self.zoom < 4 else 13
        for identity, (cy, cx) in zip(ids.tolist(), centres):
            px, py = self._to_canvas(cx + 0.5 + x0, cy + 0.5 + y0)
            bold = identity in self.focus
            font = ("Segoe UI", size + (2 if bold else 0), "bold" if bold else "normal")
            self.canvas.create_text(px + 1, py + 1, text=str(identity), fill="#000000",
                                    font=font)
            self.canvas.create_text(px, py, text=str(identity),
                                    fill=to_hex(self.well.lut[identity]), font=font)

    def _overlay_changed(self, *_) -> None:
        self._render()
        self._render_thumbs()

    def _zoom_step(self, direction: int) -> None:
        levels = list(ZOOM_LEVELS)
        nearest = min(range(len(levels)), key=lambda i: abs(levels[i] - self.zoom))
        self.zoom = levels[min(max(nearest + direction, 0), len(levels) - 1)]
        self.zoom_label.configure(text=f"{self.zoom:g}x")
        self._render()

    def _recentre(self) -> None:
        self.zoom = DEFAULT_ZOOM
        self.zoom_label.configure(text=f"{self.zoom:g}x")
        self.centre = (self.event_xy[0] + 0.5, self.event_xy[1] + 0.5)
        self._render()

    def _wheel(self, event) -> None:
        self._zoom_step(1 if event.delta > 0 else -1)

    def _press(self, event) -> None:
        self._drag = (event.x, event.y, self.centre, 0.0)
        self.canvas.focus_set()

    def _drag_motion(self, event) -> None:
        if self._drag is None:
            return
        sx, sy, centre, moved = self._drag
        dx, dy = event.x - sx, event.y - sy
        self._drag = (sx, sy, centre, max(moved, abs(dx) + abs(dy)))
        self.centre = (centre[0] - dx / self.zoom, centre[1] - dy / self.zoom)
        self._render()

    def _release(self, event) -> None:
        if self._drag is None:
            return
        moved = self._drag[3]
        self._drag = None
        if moved > 3 or self.well is None:
            return
        u, v = self._to_raw(event.x, event.y)
        col, row_ = int(np.floor(u)), int(np.floor(v))
        height, width = self.well.shape
        if not (0 <= col < width and 0 <= row_ < height):
            return
        identity = int(self.well.labels[self.frame][row_, col])
        if identity > 0:
            self.var_identity.set(str(identity))
            self._set_status(f"expected identity set to {identity} "
                             f"(frame {self.frame + 1}); press F then Enter, or Enter if already a failure")
            return
        if self.well.unclaimed is not None:
            body = int(self.well.unclaimed[self.frame][row_, col])
            if body > 0:
                self._set_status(f"unclaimed body {body} here (no accepted identity)")
                return
        self._set_status("background")

    # ------------------------------------------------------------- thumbs
    def _render_thumbs(self) -> None:
        canvas = self.thumbs
        canvas.delete("all")
        self._thumb_photos = []
        self._thumb_boxes = []
        if self.well is None:
            return
        width = canvas.winfo_width()
        n = len(THUMB_ROLES)
        gap = 8
        tile = int(min(160, max((width - gap * (n + 1)) // n, 40)))
        canvas.configure(height=tile + 44)
        height, wide = self.well.shape
        x0 = crop_origin(self.event_xy[0], THUMB_HALF, wide)
        y0 = crop_origin(self.event_xy[1], THUMB_HALF, height)
        size = 2 * THUMB_HALF + 1
        for i, (role, frame) in enumerate(zip(THUMB_ROLES, self.thumb_map)):
            crop = self._composed(frame)[y0:y0 + size, x0:x0 + size]
            image = Image.fromarray(crop).resize((tile, tile), Image.NEAREST)
            photo = ImageTk.PhotoImage(image)
            self._thumb_photos.append(photo)
            left = gap + i * (tile + gap)
            top = 20
            canvas.create_image(left, top, image=photo, anchor="nw")
            self._thumb_boxes.append((left, top, left + tile, top + tile))
            inside = self.event_span[0] <= frame <= self.event_span[1]
            colour = DARK["foreground"] if inside else DARK["muted"]
            weight = "bold" if inside else "normal"
            canvas.create_text(left + tile / 2, 10, text=f"{i + 1}  {role}",
                               fill=colour, font=("Segoe UI", 9, weight))
            canvas.create_text(left + tile / 2, top + tile + 11,
                               text=f"{frame + 1} · ImageJ {frame + 1 + SOURCE_FRAME_OFFSET}",
                               fill=colour, font=("Segoe UI", 8, weight))
            if self.var_marker.get():
                mx = left + (self.event_xy[0] + 0.5 - x0) * tile / size
                my = top + (self.event_xy[1] + 0.5 - y0) * tile / size
                r = MARKER_RADIUS * tile / size
                canvas.create_oval(mx - r, my - r, mx + r, my + r, outline=MARKER, width=1)
        self._highlight_thumb()

    def _highlight_thumb(self) -> None:
        self.thumbs.delete("highlight")
        for (left, top, right, bottom), frame in zip(self._thumb_boxes, self.thumb_map):
            if frame == self.frame:
                self.thumbs.create_rectangle(left - 2, top - 2, right + 2, bottom + 2,
                                             outline=MARKER, width=2, tags="highlight")

    def _thumb_click(self, event) -> None:
        for (left, top, right, bottom), frame in zip(self._thumb_boxes, self.thumb_map):
            if left <= event.x <= right and top <= event.y <= bottom:
                self._set_frame(frame)
                return

    # ------------------------------------------------------------- keys
    def _on_key(self, event) -> str | None:
        widget = self.root.focus_get()
        typing = isinstance(widget, (tk.Entry, tk.Text))
        key = event.keysym
        if key == "Return":
            if widget is getattr(self, "entry_notes", None) or widget is self.entry_identity \
                    or not typing:
                self._save_and_next()
                return "break"
            return None
        if key == "Escape":
            self.canvas.focus_set()
            return "break"
        if typing:
            return None
        ctrl = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        lower = key.lower()
        if ctrl and lower == "o":
            self._open_review_tiff()
        elif lower == "f":
            self._set_verdict("failure")
        elif lower == "c":
            self._set_verdict("control")
        elif lower == "u":
            self._set_verdict("undecidable")
        elif lower == "x":
            self._set_verdict(UNDECIDED)
        elif key == "Next":
            self._move(1)
        elif key == "Prior":
            self._move(-1)
        elif key == "Left":
            self._step(-5 if shift else -1)
        elif key == "Right":
            self._step(5 if shift else 1)
        elif key == "Home":
            self._set_frame(self.event_span[0])
        elif key == "End":
            self._set_frame(self.event_span[1])
        elif key == "space":
            self._toggle_play()
        elif key in ("plus", "equal", "KP_Add"):
            self._zoom_step(1)
        elif key in ("minus", "underscore", "KP_Subtract"):
            self._zoom_step(-1)
        elif lower == "r":
            self._recentre()
        elif lower == "o":
            self.var_outlines.set(not self.var_outlines.get()); self._overlay_changed()
        elif lower == "i":
            self.var_fill.set(not self.var_fill.get()); self._overlay_changed()
        elif lower == "n":
            self.var_numbers.set(not self.var_numbers.get()); self._render()
        elif lower == "b":
            self.var_unclaimed.set(not self.var_unclaimed.get()); self._overlay_changed()
        elif lower == "m":
            self.var_marker.set(not self.var_marker.get()); self._overlay_changed()
        elif key in tuple("1234567") and self.thumb_map:
            self._set_frame(self.thumb_map[int(key) - 1])
        else:
            return None
        return "break"

    # ------------------------------------------------------------- misc
    def _open_review_tiff(self) -> None:
        if self.row is None:
            return
        path = PROJECT / self.row.get("review_tiff", "")
        if not path.is_file():
            self._set_status(f"review TIFF not found: {path}")
            return
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
            self._set_status(f"opened {path.name}")
        except OSError as error:
            self._set_status(f"could not open {path}: {error}")

    def _close(self) -> None:
        self._stop_play()
        self.root.destroy()

    # ------------------------------------------------------------- smoke
    def screenshot(self, target: Path) -> Path:
        from PIL import ImageGrab
        self.root.update()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(target)
        return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--only", default=None,
                        help="comma-separated case ids or <well>:<event id> tokens; "
                             "the queue holds just these, in this order, all filters off")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the one session-start copy under backups/")
    parser.add_argument("--smoke", type=Path, default=None,
                        help="render the first case, save a screenshot here, exit")
    args = parser.parse_args(argv)
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - only matters on Windows
        pass
    root = tk.Tk()
    app = AdjudicationApp(root, args.sheet, args.inputs,
                          backup=not (args.smoke or args.no_backup), only=args.only)
    if args.smoke:
        root.update()
        app._zoom_step(0)
        app._render()
        app._render_thumbs()
        root.after(400, lambda: (app.screenshot(args.smoke), root.destroy()))
        root.mainloop()
        print(f"smoke screenshot: {args.smoke}")
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
