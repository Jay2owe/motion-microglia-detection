"""Shared plumbing for the figure builders.

Each figure in this folder is a standalone script that reads only the CSV
tables written by ``python -m analysis run`` and writes a plot-that bundle: the
SVG, the exact plotted table, copies of every source with its SHA256, and a
README. Nothing here reads an image stack except the cell report card, which
needs the outlines to draw them.

    python analysis/figures/build_all.py outputs/<run>

A figure is never the primary artefact. The tables are; a figure is a rendering
of a few of their columns, and the bundle exists so that claim can be checked.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.theme import Theme, load_theme  # noqa: E402

from _metrics import METRICS  # noqa: E402
from _options import skip_tokens, switch  # noqa: E402


def drafting() -> bool:
    """Whether this build is a look-at-it round rather than an audited one.

    Deciding what a plot should be takes many rounds and none of them is a
    result. A draft writes the figure and nothing else, over the top of the last
    one, so the folder holds the current answer rather than a numbered history of
    abandoned ones. What it skips - the copied sources, their hashes, the plotted
    table, the README - is what makes a figure checkable later, so a figure that
    is going to be shown to anybody is built without ``--draft``.
    """
    return switch("draft")


#: Where a draft's derived tables go. A builder writes them whichever kind of
#: build it is in - it should not have to know - but in a draft nobody is going
#: to read them, and leaving them beside the figures would put a folder of CSVs
#: in the one place that is meant to hold nothing but pictures.
DRAFT_SCRATCH = Path(tempfile.gettempdir()) / "motion-figure-drafts"


def draft_root(run: Path) -> Path:
    """The flat folder a draft's figures are written to.

    Beside the run folders rather than inside one, because a draft does not
    belong to a run: the tables have not changed, only the drawing has. Writing
    into a run folder would make an immutable folder mutable.
    """
    return Path(run).parent / "figures"


def bundle_for(run: Path, slug: str) -> Path:
    """The folder this build works in: the run's audited bundle, or scratch."""
    if drafting():
        return DRAFT_SCRATCH / slug
    return Path(run) / "figures" / slug


def figure_path(run: Path, bundle: Path, name: str) -> Path:
    """The file the figure is saved to.

    A draft is flat and named for the figure - ``outputs/figures/<slug>.svg``,
    overwritten each round - so every figure sits side by side and can be flipped
    through. A bundle keeps it in ``fig/``, where the rest of the bundle is what
    gives the file its meaning.
    """
    if drafting():
        return draft_root(run) / name
    return Path(bundle) / "fig" / name


def save_figure(theme: Theme, figure: Any, run: Path, bundle: Path, name: str) -> Path:
    """Write the figure and its PNG preview where this kind of build wants them.

    A bundle keeps ``fig/<slug>.svg`` beside ``fig/preview.png``, which is the
    name plot-that looks for. A draft has ten figures in one flat folder, so a
    file called ``preview.png`` would be whichever figure was drawn last; the
    preview is named after its own figure instead.
    """
    target = figure_path(run, bundle, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = theme.save(figure, target, preview=False)
    figure.savefig(
        target.with_suffix(".png") if drafting() else target.parent / "preview.png",
        format="png", dpi=200, transparent=False, facecolor="white", bbox_inches="tight",
    )
    return saved


def theme_for(run: Path) -> Theme:
    """The look this run was made with, or the house default.

    ``python -m analysis run`` drops the theme block into the run folder, so a
    figure rebuilt a year later matches the figures built beside the tables
    rather than whatever the configuration happens to say now.
    """
    stored = Path(run) / "theme.json"
    theme = load_theme(stored if stored.exists() else None)
    theme.apply()
    return theme


def design_for(run: Path) -> dict:
    """The experimental design this run was measured under, or an empty one.

    A figure that splits anything by condition asks this first. The
    ``synthetic`` flag is the one a comparison panel must act on: groups that
    were invented to exercise the pipeline have to say so on the figure, not
    only in the folder they came from.
    """
    stored = Path(run) / "conditions.json"
    if not stored.exists():
        return {"conditions": [], "assignments": [], "synthetic": False, "warnings": []}
    return json.loads(stored.read_text(encoding="utf-8"))


def summary_for(run: Path, stem: str | None = None) -> dict:
    """One movie's summary from the run manifest.

    The place to get an acquisition fact - the frame interval, the number of
    frames, whether a spatial scale was resolved - rather than retyping it into
    a builder where it becomes wrong the first time a different movie is drawn.
    """
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    movies = manifest["movies"]
    if stem is not None:
        movies = [m for m in movies if m["stem"] == stem]
    if len(movies) != 1:
        names = ", ".join(m["stem"] for m in manifest["movies"]) or "none"
        raise SystemExit(f"pass a stem: this run holds {names}")
    return movies[0]["summary"]


def inputs_for(run: Path, stem: str | None = None) -> dict:
    """The image stacks this run was measured from, with their hashes.

    A builder that needs pixels asks the run which files they were, rather than
    repeating a path. A path typed into a builder is correct for exactly one run
    and silently draws the wrong movie for the next one, while the hash beside it
    here is what proves the tiles and the tables came from the same stack.
    """
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    movies = manifest["movies"]
    if stem is not None:
        movies = [m for m in movies if m["stem"] == stem]
    if len(movies) != 1:
        names = ", ".join(m["stem"] for m in manifest["movies"]) or "none"
        raise SystemExit(f"pass a stem: this run holds {names}")
    return dict(movies[0]["provenance"]["inputs"])


def field_for(run: Path, stem: str | None = None) -> dict:
    """The pixel dimensions of the movie, from the run manifest.

    A figure drawn in image coordinates - a trajectory map, an outline over a
    frame - must be bounded by the field, not by where the cells happen to be.
    Otherwise the axes silently rescale between two runs and the reader compares
    two maps at different magnifications without being told.
    """
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    movies = manifest["movies"]
    if stem is not None:
        movies = [m for m in movies if m["stem"] == stem]
    if len(movies) != 1:
        names = ", ".join(m["stem"] for m in manifest["movies"]) or "none"
        raise SystemExit(f"pass a stem: this run holds {names}")
    return dict(movies[0]["provenance"]["field"])


def wrap_title(theme, figure, text: str) -> str:
    """A title broken to fit the canvas it is drawn on.

    Pure layout, and the reason it exists is that a title is now the user's to
    write: one longer than the builder's default would otherwise push the
    figure wider than the panel inside it. Breaks at a semicolon first, then on
    whitespace. Pass a title containing newlines to control the break yourself.
    """
    if "\n" in text:
        return text
    import textwrap

    inches = float(figure.get_size_inches()[0])
    # 0.5 em per character is close for a bold sans-serif at these sizes; the
    # margin below keeps a slightly wide line inside the canvas anyway.
    budget = max(28, int((inches * 72 * 0.94) / (0.52 * theme.size("title"))))
    if len(text) <= budget:
        return text
    if "; " in text:
        head, _, tail = text.partition("; ")
        if max(len(head) + 1, len(tail)) <= budget:
            return f"{head};\n{tail}"
    return "\n".join(textwrap.wrap(text, budget))


def module_params(run: Path, module: str, stem: str | None = None) -> dict:
    """The parameters one module actually ran with.

    A figure that quotes a threshold - "the 24 frames a fit needs", "16-32 h
    search" - must read it from the run rather than repeat it, or the caption
    keeps describing a setting somebody has since changed.
    """
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    movies = manifest["movies"]
    if stem is not None:
        movies = [m for m in movies if m["stem"] == stem]
    for record in movies[0]["modules"]:
        if record["module"] == module:
            return dict(record.get("parameters", {}))
    raise KeyError(f"{module!r} did not run in {Path(run).name}")


def units_note(summary: dict) -> str:
    """The sentence about spatial units, matching what the run actually resolved.

    Another fact that must not be typed into a builder: a figure that says
    "uncalibrated dataset, so distances are pixels" keeps saying it after
    somebody sets ``microns_per_pixel``.
    """
    scale = summary.get("scale", {})
    if scale.get("calibrated"):
        return (
            f"Distances are microns at {scale['microns_per_pixel']:.4g} um/px, "
            f"resolved from {scale.get('source', 'the configuration')}."
        )
    return (
        "Uncalibrated dataset, so distances are pixels; setting "
        "microns_per_pixel in the configuration converts every number here "
        "without touching the code."
    )


def provenance_note(summary: dict) -> str:
    """One line on how much of the measured footprint was attributed by inference.

    Beside ``units_note`` and for the same reason: a fact about the run that
    several figures must state identically, and that no builder should type out
    for itself. Empty when the run carries no provenance, so a figure built
    from an older run is unchanged rather than carrying a hedge it cannot
    support.

    The claim is about the *name*, not the picture, and the second sentence
    is what keeps a reader from hearing the first one as "invented". Almost
    every flagged pixel was shown by the microscope; what the tracker
    supplied is which cell it belongs to. The handful where the microscope
    showed nothing at all is counted separately and stated as a count, not a
    share, because rounding it to a percentage prints 0.1% and reads as
    though it had been glossed over.

    It states the shares and stops. A reconstructed attribution is not a wrong
    one, and what a reader should conclude from the number is theirs to decide,
    not this package's - see the ``figures`` block of the configuration.
    """
    record = summary.get("provenance") or {}
    share = record.get("inferred_fraction_of_footprint")
    if share is None:
        return ""
    note = (
        f"For {share:.1%} of the outline pixels behind these numbers, which "
        f"cell owns the pixel was decided by the tracker's reconstruction "
        f"rather than by observing that cell there."
    )
    added = record.get("added_px")
    if added is None:
        return note
    # Its own line, not appended: the first sentence already fills the
    # footnote width, and a figure whose caption runs off the canvas is a
    # figure nobody reads the qualification on.
    if added == 0:
        return note + "\nEvery one of those pixels was shown by the microscope."
    entire = record.get("cell_frames_entirely_supplied")
    touched = record.get("cell_frames_with_added_px")
    total = record.get("cell_frames")
    # Cell-frames when every supplied pixel is a whole missed outline, which
    # is what a reader can act on. Pixels when it is not, because then the
    # cell-frame count would overstate what the tracker supplied.
    if entire and touched and total and entire == touched:
        return note + (
            f"\nAll but {added:,} of them were shown by the microscope. Those "
            f"{added:,} are {entire:,} whole cell-frames, of {total:,}, that "
            f"the detection did not see at all."
        )
    return note + (
        f"\nAll but {added:,} of them were shown by the microscope; those "
        f"{added:,} are outline the tracker supplied where the detection saw "
        f"nothing."
    )


def provenance_table(summary: dict) -> pd.DataFrame:
    """The same numbers as a one-row table, so the bundle stays self-contained.

    Empty when the run has no provenance, which a caller checks with
    ``.empty`` before writing it.
    """
    record = (summary.get("provenance") or {}).copy()
    if not record:
        return pd.DataFrame()
    record["stem"] = summary.get("stem")
    return pd.DataFrame([record])


def write_readme(bundle: Path, title: str, text, body: str) -> Path:
    """The bundle README.

    *text* is the figure's :class:`FigureText`. Its ``claim`` is whatever the
    user wrote and is reproduced verbatim; there is no default, because the
    package does not interpret figures. An unset claim is recorded as unset
    rather than filled in.
    """
    if drafting():
        return bundle / "README.md"
    claim = getattr(text, "claim", "") or ""
    lines = [f"# {title}", ""]
    if claim:
        lines += [f"**Claim.** {claim}", ""]
    else:
        lines += [
            "**Claim.** None recorded. This figure has not been interpreted; the "
            "title describes what is plotted. Set `claim` for this figure in the "
            "`figures` block of the analysis configuration, or pass `--claim`, to "
            "state what it shows.",
            "",
        ]
    lines += [body.strip(), "", "## Wording", ""]
    for slot, origin in sorted((getattr(text, "source", None) or {}).items()):
        value = getattr(text, slot, "")
        shown = (value[:96] + "...") if len(value) > 99 else (value or "(unset)")
        lines.append(f"- `{slot}` from **{origin}**: {shown}")
    lines += [
        "",
        "`default` means the builder's own descriptive wording. `config` means the "
        "`figures` block of the run's `figures.json`; `flag` means a command-line "
        "option on this build.",
        "",
    ]
    path = bundle / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (bundle / "data" / "der" / "figure_text.json").write_text(
        json.dumps(text.as_dict(), indent=2), encoding="utf-8"
    )
    return path


def run_folder(default: str | None = None) -> Path:
    """The analysis run to draw from: first positional argument, or the default.

    Skips ``--option value`` pairs, so a builder can be given wording on the
    command line without the value being read as a path.
    """
    argv = list(sys.argv[1:])
    index = 0
    while index < len(argv):
        if argv[index].startswith("--"):
            index += skip_tokens(argv[index])
            continue
        return Path(argv[index]).resolve()
    if default:
        return Path(default).resolve()
    raise SystemExit("usage: python <figure script> <path to an analysis run folder>")


def tables_for(run: Path, stem: str | None = None) -> Path:
    """The tables folder for a stem, or the only stem if there is just one."""
    if stem:
        return run / stem / "tables"
    candidates = sorted(p for p in run.iterdir() if (p / "tables").is_dir())
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise SystemExit(f"pass a stem: this run holds {names}")
    return candidates[0] / "tables"


def require_table(tables: Path, name: str, module: str) -> Path:
    """A table's path, or a message naming the module that would have written it.

    A builder that opens a missing CSV fails with a traceback out of pandas
    naming a path, which sends the reader looking for a lost file. The file was
    never written: the module is switched off in ``enabled_modules``, or was
    skipped because the movie lacks an input it needs. Say that instead.

    Two folders are tried, in this order: the measured tables, then the tables
    copied out of the tracking run. ``tables/`` first is what keeps runs written
    before the split working - every file is already there, so the second
    candidate is never reached and the old layout needs no version flag and no
    detection.
    """
    for candidate in (Path(tables) / name,                      # measured here
                      Path(tables).parent / "tracker" / name):  # copied verbatim
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"{name} is not in {tables}: the {module!r} module did not run for this "
        f"movie. Add {module!r} to enabled_modules in the analysis configuration "
        f"and re-run, or check `python -m analysis run` for a skip - a module is "
        f"skipped rather than failed when a movie lacks an input it needs."
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bundle(root: Path, sources: dict[str, Path]) -> pd.DataFrame:
    """Create the bundle skeleton, copy every source in, and hash all of them.

    A draft build skips all of it. The builder still writes its derived tables,
    so they go to a scratch folder beside the figure rather than being routed
    around - a builder should not have to know which kind of build it is in.
    """
    root = Path(root)
    if drafting():
        (root / "data" / "der").mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=["short_name", "original_path", "sha256"])

    for sub in ("data/src", "data/der", "fig", "code"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for short_name, original in sources.items():
        original = Path(original)
        copied = root / "data" / "src" / short_name
        shutil.copy2(original, copied)
        stat = original.stat()
        rows.append(
            {
                "short_name": short_name,
                "original_path": str(original),
                "copied_path": copied.relative_to(root).as_posix(),
                "file_name": original.name,
                "modification_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "byte_size": stat.st_size,
                "sha256": sha256_of(original),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(root / "data" / "sources.csv", index=False)

    lines = ["# Sources", ""]
    for row in rows:
        lines += [
            f"## {row['short_name']}",
            "",
            f"- original: `{row['original_path']}`",
            f"- copied to: `{row['copied_path']}`",
            f"- modified (UTC): {row['modification_time']}",
            f"- bytes: {row['byte_size']:,}",
            f"- sha256: `{row['sha256']}`",
            "",
        ]
    (root / "data" / "sources.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def stack_for(tables: Path, name: str, module: str) -> Path:
    """A module image stack beside ``tables/``, with the same readable refusal."""
    candidate = Path(tables).parent / "stacks" / name
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"{name} is not beside {tables}: the {module!r} module did not write its "
        "image outputs. Enable that module and rebuild the analysis run."
    )


def write_standalone_producer(bundle: Path, title: str, grammar: str) -> Path:
    """Write a bundle-relative producer for the exact ``figure_data.csv``.

    Planned figures are multi-panel pages, so this compact producer renders the
    primary table according to its columns.  The full page builder remains the
    project script; the embedded producer is deliberately dependency-light and
    travels with the data when the bundle is moved.
    """
    path = Path(bundle) / "code" / "plot.py"
    if drafting():
        return path
    # Sorted, because the vocabulary now comes from the module registry: without
    # this, renaming a module reshuffles this dict and every bundle's plot.py
    # differs from the one before it without a single label having changed.
    public_labels = {column: METRICS[column].label for column in sorted(METRICS)}
    public_labels.update({"row_fraction": "Share of transitions", "count": "Transitions"})
    source = '''from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[1]
DATA = pd.read_csv(BUNDLE / "data" / "der" / "figure_data.csv")
TITLE = %r
GRAMMAR = %r
LABELS = %r

fig, ax = plt.subplots(figsize=(8, 5.5))
numeric = [c for c in DATA.columns if pd.api.types.is_numeric_dtype(DATA[c])]
plottable = [c for c in numeric if c in LABELS]
if {"from_regime", "to_regime"}.issubset(DATA.columns):
    value = "row_fraction" if "row_fraction" in DATA else ("count" if "count" in DATA else numeric[-1])
    matrix = DATA.pivot(index="from_regime", columns="to_regime", values=value)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label=LABELS.get(value, "Transitions"))
    ax.set_xlabel("To state"); ax.set_ylabel("From state")
elif "hours" in DATA.columns and plottable:
    y = plottable[0]
    group = DATA.groupby("hours", as_index=False)[y].mean()
    ax.plot(group["hours"], group[y], color="#4878A8", linewidth=2.4)
    ax.set_xlabel("Hours from start of recording"); ax.set_ylabel(LABELS[y])
elif len(plottable) >= 2:
    ax.scatter(DATA[plottable[0]], DATA[plottable[1]], color="#4878A8", alpha=0.6)
    ax.set_xlabel(LABELS[plottable[0]]); ax.set_ylabel(LABELS[plottable[1]])
elif plottable:
    ax.hist(DATA[plottable[0]].dropna(), bins=24, color="#4878A8")
    ax.set_xlabel(LABELS[plottable[0]]); ax.set_ylabel("Observations")
else:
    ax.text(0.5, 0.5, f"{len(DATA):,} plotted rows", ha="center", va="center")
    ax.set_axis_off()
ax.set_title(TITLE, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(BUNDLE / "fig" / "reproduced.svg", transparent=True, bbox_inches="tight")
''' % (title, grammar, public_labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def save_reprofig_figure(
    theme: Theme,
    figure: Any,
    run: Path,
    bundle: Path,
    name: str,
    *,
    claim: str,
    grammar: str,
    producer: str = "code/plot.py",
) -> Path:
    """Save a planned figure as a master-profile ReproFig SVG."""
    if drafting():
        return save_figure(theme, figure, run, bundle, name)
    scripts = Path.home() / ".claude" / "skills" / "plot-that" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from reprofig_bundle import save_matplotlib_svg

    target = figure_path(run, bundle, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = save_matplotlib_svg(
        figure,
        target,
        claim=claim or None,
        grammar=grammar,
        producer=producer,
        statistics_status="not_applicable",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    figure.savefig(
        target.parent / "preview.png", format="png", dpi=200, transparent=False,
        facecolor="white", bbox_inches="tight",
    )
    return saved
