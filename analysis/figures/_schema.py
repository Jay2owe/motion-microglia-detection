"""What a figure is, declared rather than assumed.

A figure is a slug, a set of panels, the tables it reads and the options it
accepts. Writing that down is what lets the package refuse an option this page
cannot honour, print a ``--help`` that is true, and carry a user's choices into
the run folder so a rebuild draws the figure that was drawn rather than the
default.

Deliberately the same shape as ``analysis/registry.py``: one file per figure,
self-registering on import, no central list to keep in step. "Which module wrote
this number" and "which file draws it" already have one answer; this gives
"which options does it take" one too.

    @figure(
        number=9,
        slug="step-size-distribution",
        summary="how far a cell moves between one frame and the next",
        title="{metric} between consecutive frames, {interval} apart",
        grammar="histogram",
        reads=(Table("cell_frame.csv", module="motility"),),
        panels=(Panel("distribution", motility_panels.step_histogram),),
        options=(Option("bins", default=45),),
    )
    def build(ctx: FigureContext) -> FigureResult:
        ...

    if __name__ == "__main__":
        run_figure("step-size-distribution")

The builder returns a result; it does not save, write a README, place a title or
close a figure. Those are the same for all thirty-six pages and are done once,
in ``_finish``.

A figure's **slug** and its **filename** are allowed to differ. The slug is what
the configuration block, the bundle folder and the catalogue use; the filename
is a command line. ``python -m analysis figures`` prints both.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402

from _bundle import (bundle_for, field_for, inputs_for, make_bundle, module_params,  # noqa: E402
                     provenance_note,
                     provenance_table, require_table, run_folder,
                     save_reprofig_figure, stack_for, summary_for, tables_for,
                     theme_for, units_note, wrap_title, write_readme,
                     write_standalone_producer)
from _options import (OPTIONS, PLACEMENTS, SWITCHES, Length, Vocabulary,  # noqa: E402
                      as_length, commas, length, option, raw_flag, skip_tokens)
from _text import OPTIONS_KEY, SLOTS, _from_run, figure_text  # noqa: E402
from analysis.theme import parse_hours_per_tick  # noqa: E402
from analysis.units import Scale  # noqa: E402
from panels import common  # noqa: E402
from panels._contract import PanelResult  # noqa: E402

__all__ = ["FIGURES", "FigureContext", "FigureResult", "FigureSpec", "Option",
           "Input", "Panel", "PanelList", "PanelResult", "Stack", "Table",
           "catalogue", "figure", "get_figure", "load_all", "run_figure"]

HERE = Path(__file__).resolve().parent

#: Options every figure accepts without declaring them, because they are about
#: which figure is being drawn, or about the sheet, rather than about what it
#: draws. ``stem`` picks the movie in a run that holds several; ``panels`` names
#: the parts of a page to keep, and is checked against the figure's own panel
#: list rather than against this one. The four in ``PLACEMENTS`` move the words
#: on the page and are read by ``_finish``, not by any builder, which is why
#: no figure declares them and every figure honours them.
UNIVERSAL: tuple[str, ...] = ("stem", "panels", *PLACEMENTS)

#: The one switch every figure honours. The others change what a particular page
#: draws and are declared on it, the same as an option: ``--overlay`` on a figure
#: with nothing to overlay is the same silence as ``--bins`` on a figure with no
#: histogram.
UNIVERSAL_SWITCHES: tuple[str, ...] = ("draft",)

#: Named because a footnote is built by joining sentences and the joining
#: character is easy to lose in an f-string.
NEWLINE = "\n"


# --------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Option:
    """One option this figure accepts, and what it means when unset.

    The name has to be one of ``_options.OPTIONS``, so a flag means the same
    thing on every page that takes it. ``cast`` and ``help`` are overrides for
    the case where it does not quite: a page drawing one metric takes a string
    where the shared cast gives a list, and should say so in its own words.
    """

    name: str
    default: Any = None
    #: ``None`` uses the shared vocabulary's cast.
    cast: Callable[[str], Any] | None = None
    #: ``""`` uses the shared vocabulary's help line.
    help: str = ""
    #: ``""`` uses the shared vocabulary's. A figure that narrows the cast
    #: usually has to narrow this too - one column is COL, not COL,COL.
    metavar: str = ""

    def vocabulary(self) -> Vocabulary:
        return OPTIONS[self.name]

    def caster(self) -> Callable[[str], Any]:
        return self.cast or OPTIONS[self.name].cast

    def helpline(self) -> str:
        return self.help or OPTIONS[self.name].help

    def placeholder(self) -> str:
        return self.metavar or OPTIONS[self.name].metavar

    def shown_default(self) -> str:
        """The declared default as a user would type it, not as Python spells it."""
        if self.default is None:
            return "unset"
        if isinstance(self.default, (list, tuple)):
            return ",".join(str(item) for item in self.default) or "none"
        if self.default == "":
            return "empty"
        return str(self.default)

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")


@dataclass(frozen=True)
class Table:
    """A CSV under ``<run>/<stem>/tables`` this figure reads.

    ``module`` is not decoration. A missing table is almost never a lost file -
    the module that writes it was switched off, or skipped because the movie
    lacks an input - and naming it turns a pandas traceback into one sentence a
    user can act on.
    """

    name: str
    module: str
    #: Missing: skip whatever needs it, rather than stop the build.
    optional: bool = False


@dataclass(frozen=True)
class Stack:
    """An image stack under ``<run>/<stem>/tables`` this figure reads."""

    name: str
    module: str
    optional: bool = False


@dataclass(frozen=True)
class Input:
    """One of the movie's own image stacks, read through the run's manifest.

    Not a table and not a module's output: the labels, the raw signal, the
    unclaimed foreground. A page that draws pixels rather than numbers reads
    these, and declaring them is what puts a copy and a hash of the stack in the
    bundle - without which the picture is unfalsifiable.
    """

    #: ``labels``, ``raw``, ``unclaimed``, ``evidence`` or ``provenance``.
    kind: str
    optional: bool = True

    @property
    def name(self) -> str:
        return self.kind

    @property
    def module(self) -> str:
        return "the movie"


@dataclass(frozen=True)
class Panel:
    """One part of a page that ``--panels`` can leave out.

    ``draw`` is the function in ``panels/`` that makes the marks. It is held
    here rather than only called in the build function so the coverage test can
    check the placement rule - a panel takes an axes, numbers and the theme, and
    never opens a file - without running anything.
    """

    key: str
    #: The function in ``panels/`` that makes this panel's marks. ``None`` says
    #: the build function draws it inline, which is a debt rather than a
    #: choice: it is a panel no other page can reuse. Declaring it as ``None``
    #: rather than leaving it out is what makes the debt countable.
    draw: Callable | None = None
    #: ``""`` becomes ``key.replace("_", " ").capitalize()``.
    title: str = ""
    #: Takes ``(figure, rect, ...)`` rather than ``(ax, ...)``: a scatter with
    #: its own marginal axes cannot be given one rectangle to draw in.
    block: bool = False
    polar: bool = False
    #: ``Table``/``Stack``/``Input`` names without which this panel is skipped,
    #: and the skip is reported, rather than drawn empty. A name may narrow to
    #: one column - ``presence_frame.csv:unclaimed_px`` - for a panel that needs
    #: not a file but one measurement inside it: the presence module writes the
    #: same table with and without an unclaimed stack, and only the column says
    #: which run this is.
    needs: tuple[str, ...] = ()

    def heading(self) -> str:
        return self.title or self.key.replace("_", " ").capitalize()


class PanelList(list):
    """The panels being drawn, which answers to a key as well as to a Panel.

    ``"tiles" in panels`` is how a build function asks whether it has to read an
    image stack at all, and it reads better than the alternatives. A plain list
    of Panel objects would answer no to that and quietly skip the read.
    """

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return any(panel.key == item for panel in self)
        return super().__contains__(item)

    @property
    def keys(self) -> list[str]:
        return [panel.key for panel in self]


@dataclass(frozen=True)
class FigureSpec:
    """One figure's declaration: everything true of it before any data is read."""

    slug: str
    number: int
    title: str
    grammar: str
    build: Callable[["FigureContext"], "FigureResult"]
    panels: tuple[Panel, ...] = ()
    reads: tuple[Table | Stack | Input, ...] = ()
    options: tuple[Option, ...] = ()
    #: On-or-off options this figure honours, beyond ``--draft``.
    switches: tuple[str, ...] = ()
    #: One line for the catalogue. ``""`` takes the first line of ``build``'s
    #: docstring, or of the module's.
    summary: str = ""
    source: Path | None = None

    def option(self, name: str) -> Option:
        for declared in self.options:
            if declared.name == name:
                return declared
        accepted = ", ".join(o.name for o in self.options) or "none"
        raise KeyError(
            f"{self.slug} reads option {name!r}, which it does not declare; "
            f"it declares {accepted}. Add it to the @figure options, or the "
            f"flag it honours will not appear in --help"
        )

    def panel(self, key: str) -> Panel:
        for declared in self.panels:
            if declared.key == key:
                return declared
        raise KeyError(f"{self.slug} has no panel {key!r}; it draws "
                       + ", ".join(p.key for p in self.panels))

    def reads_named(self, name: str) -> Table | Stack | Input | None:
        for source in self.reads:
            if source.name == name:
                return source
        return None

    def command(self) -> str:
        if self.source is None:
            return f"python analysis/figures/{self.slug}.py"
        return f"python analysis/figures/{self.source.name}"

    def usage(self) -> str:
        """This figure's ``--help``: what it draws, reads and accepts.

        One figure's list rather than the whole vocabulary, which is the point.
        A help page listing thirty-one flags of which three are honoured is a
        help page that has to be checked against the source before it is
        believed.
        """
        lines = [f"usage: {self.command()} <run> [options]", ""]
        lines.append(f"figure {self.number}   {self.slug}")
        if self.summary:
            lines.append(f"  {self.summary}")
        lines.append("")

        if self.reads:
            lines.append("reads")
            width = max(len(s.name) for s in self.reads)
            for source in self.reads:
                tail = f"written by {source.module}"
                if source.optional:
                    tail += "; optional"
                lines.append(f"  {source.name:<{max(width, 30)}}  {tail}")
            lines.append("")

        if self.panels:
            keys = ", ".join(p.key for p in self.panels)
            lines.append(f"panels                          --panels {keys}")
            width = max(len(p.key) for p in self.panels)
            for panel in self.panels:
                tail = panel.heading()
                if panel.needs:
                    tail += f"   (needs {', '.join(panel.needs)})"
                lines.append(f"  {panel.key:<{max(width, 30)}}  {tail}")
            lines.append("")

        lines.append("options")
        if self.options:
            shown = [(f"{o.flag} {o.placeholder()}", o.helpline(), o.shown_default())
                     for o in self.options]
            width = max(len(head) for head, _, _ in shown)
            for head, helpline, default in shown:
                lines.append(f"  {head:<{width}}  {helpline}   [{default}]")
        else:
            lines.append("  this figure declares none")
        lines.append("")

        lines += [
            "always accepted",
            f"  --stem NAME                     {OPTIONS['stem'].help}",
            "",
            "text, on every figure",
            "  " + " ".join(f"--{slot}" for slot in SLOTS),
            '  set any to "" to take that line off the figure',
            "",
            "placement, on every figure",
        ]
        width = max(len(OPTIONS[name].flag) for name in PLACEMENTS)
        for name in PLACEMENTS:
            entry = OPTIONS[name]
            lines.append(f"  {entry.flag:<{width}} {entry.metavar:<8} {entry.help}")
        lines += [
            "  a share of the page as 2%, or inches as 0.25in; the unit is required",
            "",
            "switches",
        ]
        names = list(UNIVERSAL_SWITCHES) + [name for name in self.switches
                                            if name not in UNIVERSAL_SWITCHES]
        width = max(len(name) for name in names)
        for name in names:
            lines.append(f"  --{name:<{width}}  {SWITCHES[name]}")
        return NEWLINE.join(lines)


#: slug -> declaration. Populated by importing a builder, never by editing a
#: list here; see ``analysis/registry.py`` for the same arrangement on the
#: measurement side.
FIGURES: dict[str, FigureSpec] = {}


def figure(
    *,
    slug: str,
    number: int,
    title: str,
    grammar: str,
    panels: tuple[Panel, ...] = (),
    reads: tuple[Table | Stack | Input, ...] = (),
    options: tuple[Option, ...] = (),
    switches: tuple[str, ...] = (),
    summary: str = "",
) -> Callable:
    """Declare one figure. Same contract as ``analysis.registry.register``."""

    def decorator(build: Callable) -> Callable:
        source = Path(inspect.getfile(build)).resolve()
        existing = FIGURES.get(slug)
        if existing is not None and existing.source != source:
            raise KeyError(
                f"two figures claim the slug {slug!r}: {existing.source} and {source}"
            )
        for declared in options:
            if declared.name not in OPTIONS:
                raise KeyError(
                    f"{slug} declares option {declared.name!r}, which is not in the "
                    f"shared vocabulary; add it to _options.OPTIONS with a "
                    f"one-line meaning, or fix the spelling"
                )
        for name in switches:
            if name not in SWITCHES:
                raise KeyError(
                    f"{slug} declares switch {name!r}, which is not in "
                    f"_options.SWITCHES; declared switches are "
                    f"{', '.join(sorted(SWITCHES))}")
        keys = [panel.key for panel in panels]
        if len(set(keys)) != len(keys):
            raise KeyError(f"{slug} declares the same panel twice: {', '.join(keys)}")
        described = summary or _first_line(build)
        FIGURES[slug] = FigureSpec(
            slug=slug, number=number, title=title, grammar=grammar, build=build,
            panels=tuple(panels), reads=tuple(reads), options=tuple(options),
            switches=tuple(switches), summary=described, source=source,
        )
        return build

    return decorator


def _first_line(build: Callable) -> str:
    """The one-line summary a builder did not write out separately."""
    for text in (build.__doc__, sys.modules.get(build.__module__, None) and
                 sys.modules[build.__module__].__doc__):
        if text:
            first = text.strip().splitlines()[0].strip()
            if first:
                return first.rstrip(".")
    return ""


def get_figure(slug: str) -> FigureSpec:
    if slug not in FIGURES:
        known = ", ".join(sorted(FIGURES)) or "none registered"
        raise KeyError(f"unknown figure {slug!r}; registered figures: {known}")
    return FIGURES[slug]


def builder_files() -> list[Path]:
    """Every numbered builder, in number order."""
    return sorted(HERE.glob("[0-9][0-9]_*.py"))


def load_all() -> dict[str, FigureSpec]:
    """Import every builder that is on the schema, and return the catalogue.

    Only the ones on the schema. A builder not yet converted runs its whole
    figure at import - that is what being a top-to-bottom script means - so
    importing it to ask what it is would draw it. The declaration is read out of
    the source text instead, which costs nothing and cannot have side effects.
    """
    for path in builder_files():
        if "@figure(" not in path.read_text(encoding="utf-8"):
            continue
        if path.stem in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    return dict(FIGURES)


def catalogue() -> list[tuple[int, Path, FigureSpec | None]]:
    """One row per numbered builder: its number, its file, and its declaration.

    A file with no declaration still gets a row. The numbered set is what a user
    sees in the folder, and a catalogue that listed only the converted half
    would read as though the others had been deleted.
    """
    load_all()
    by_source = {spec.source: spec for spec in FIGURES.values()}
    rows = []
    for path in builder_files():
        rows.append((int(path.name[:2]), path, by_source.get(path.resolve())))
    return sorted(rows, key=lambda row: row[0])


# ------------------------------------------------------------------- context


@dataclass
class FigureContext:
    """Everything a build function is given, and the only way it reads a file.

    A builder never opens a path itself: every table and stack it touches goes
    through here, which is what puts a copy and a SHA256 of it in the bundle
    without the builder having to remember.
    """

    spec: FigureSpec
    run: Path
    tables: Path
    theme: Any
    bundle: Path
    summary: dict
    field: dict
    stem: str | None
    sources: dict[str, Path] = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    #: Where each resolved option came from: ``flag``, ``config`` or ``default``.
    option_source: dict[str, str] = field(default_factory=dict)
    #: What each resolved option came out as, so the closing line can say.
    option_value: dict[str, Any] = field(default_factory=dict)
    #: What each panel drew, by key, so a figure that does not build its own
    #: audit table gets its leading panel's.
    panel_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Panels left out because the run has no data for them, and what is missing.
    skipped_panels: list[tuple[str, list[str]]] = field(default_factory=list)
    _drawn: list[str] | None = None
    _run_options: dict[str, Any] | None = None
    _run_placements: dict[str, Any] | None = None

    # -------------------------------------------------------------- the movie

    @property
    def interval(self) -> float:
        return float(self.summary["minutes_per_frame"])

    @property
    def scale(self) -> Scale:
        details = self.summary.get("scale", {})
        return Scale(
            self.interval,
            details.get("microns_per_pixel"),
            details.get("source", "uncalibrated"),
        )

    @property
    def hour_ticks(self) -> float:
        """Hours between ticks on a time axis.

        A declared option like any other, so it appears in ``--help`` and can be
        set in the configuration. A figure with a time axis that has not
        declared it is a figure whose ``--hour-ticks`` would be silently
        ignored, so reading this without declaring it is refused.
        """
        return parse_hours_per_tick(self.option("hour_ticks"))

    @property
    def length_label(self) -> str:
        return "µm" if self.scale.calibrated else "px"

    @property
    def area_label(self) -> str:
        return "µm²" if self.scale.calibrated else "px²"

    # -------------------------------------------------------------- the inputs

    def table_path(self, name: str, module: str | None = None) -> Path:
        path = require_table(self.tables, name, module or self._module_for(name))
        self.sources.setdefault(name, path)
        return path

    def table(self, name: str, module: str | None = None) -> pd.DataFrame:
        return pd.read_csv(self.table_path(name, module))

    def optional_table(self, name: str) -> pd.DataFrame | None:
        """A table if the run has it, ``None`` if it does not.

        For a table a figure can improve on but does not need. ``table`` stops
        the build with an explanation, which is right for an input the figure
        cannot draw without and wrong for one that only adds to it.

        It goes through the same resolver ``table`` does rather than building a
        path of its own. Building its own is how it would silently return
        ``None`` for a table that is present but in the copied-tables folder -
        and ``None`` means "the run does not have it", which is the one failure
        that draws a wrong figure instead of raising.
        """
        try:
            path = require_table(self.tables, name, self._module_for(name))
        except SystemExit:
            return None
        self.sources.setdefault(name, path)
        return pd.read_csv(path)

    def stack_path(self, name: str, module: str | None = None) -> Path:
        path = stack_for(self.tables, name, module or self._module_for(name))
        self.sources.setdefault(name, path)
        return path

    def stack(self, name: str, module: str | None = None) -> np.ndarray:
        return tifffile.imread(self.stack_path(name, module))

    def input_path(self, kind: str) -> Path | None:
        """One of the movie's own image stacks, or ``None`` if it has none.

        Called before the bundle is made, so the path is recorded even when the
        panel that wanted it turns out to have nothing to draw: a bundle that
        quietly lost a source is a bundle that cannot be checked.
        """
        self._declared(kind)
        record = inputs_for(self.run, self.stem).get(kind)
        if not record:
            return None
        path = Path(record["path"])
        if path.exists():
            key = f"{kind}{path.suffix.lower()}"
            self.sources.setdefault(key, path)
            return path
        return None

    def has(self, name: str) -> bool:
        """Whether the run holds one of the inputs this figure declared.

        A name may name a column too - ``presence_frame.csv:unclaimed_px`` -
        because a table can be present and still not answer the question a
        panel asks of it. Only the header is read: the file is opened again in
        full by whichever panel survives this.
        """
        name, _, column = name.partition(":")
        declared = self.spec.reads_named(name)
        if isinstance(declared, Input):
            return self.input_path(name) is not None
        path = Path(self.tables) / name
        if not path.exists():
            return False
        return not column or column in pd.read_csv(path, nrows=0).columns

    def switch(self, name: str) -> bool:
        """Whether an on-or-off option this figure declared was given.

        Refuses one it did not declare, for the reason ``option`` does: a page
        that honours a switch its ``--help`` does not list is a page whose
        behaviour cannot be read off it.
        """
        if name not in UNIVERSAL_SWITCHES and name not in self.spec.switches:
            declared = ", ".join(self.spec.switches) or "none"
            raise KeyError(
                f"{self.spec.slug} reads switch {name!r}, which it does not "
                f"declare; it declares {declared}")
        return bool({f"--{name}", f"--{name.replace('_', '-')}"} & set(self.argv))

    def _declared(self, name: str) -> Table | Stack | Input:
        declared = self.spec.reads_named(name)
        if declared is None:
            raise KeyError(
                f"{self.spec.slug} reads {name!r}, which it does not declare; add "
                f"it to the @figure reads, or the bundle will not record where "
                f"the picture came from"
            )
        return declared

    def _module_for(self, name: str) -> str:
        return self._declared(name).module

    def module_params(self, module: str) -> dict:
        """What one module was configured with when this run was measured."""
        return module_params(self.run, module, self.stem)

    def prepare(self) -> None:
        make_bundle(self.bundle, self.sources)

    # ------------------------------------------------------------- the choices

    def option(self, name: str) -> Any:
        """This figure's value for one option: a flag, else the spec's default.

        Refuses a name this figure did not declare. A builder that reads an
        option it never advertised would honour a flag ``--help`` does not list,
        which is the same silence this schema exists to remove, from the other
        side.
        """
        declared = self.spec.option(name)
        raw = raw_flag(name, self.argv)
        if raw is not None:
            try:
                value = declared.caster()(raw)
            except (TypeError, ValueError) as error:
                raise SystemExit(f"{declared.flag} {raw!r}: {error}") from None
            self.option_source[name] = "flag"
            self.option_value[name] = value
            return value
        recorded = self._from_run_options()
        if name in recorded:
            self.option_source[name] = "config"
            self.option_value[name] = recorded[name]
            return recorded[name]
        self.option_source[name] = "default"
        self.option_value[name] = declared.default
        return declared.default

    def option_or(self, name: str, unset: Any) -> Any:
        """The option's value, or this panel's own default when nobody set it.

        For a page where one flag drives two panels that want different numbers
        by default: a marginal strip beside a scatter is not a distribution and
        does not want as many bands as one. The flag still drives both, which is
        what a user means by ``--bins``; only the unset case differs, and
        ``--help`` reports the figure's own default rather than this one.
        """
        value = self.option(name)
        return unset if self.option_source[name] == "default" else value

    def _from_run_placements(self) -> dict[str, Any]:
        """The placements this run recorded, read once for all four of them.

        Kept apart from ``_from_run_options`` because these belong to every
        figure rather than to this one, so the check that refuses an option the
        figure does not declare must not see them.
        """
        if self._run_placements is None:
            recorded = _from_run(Path(self.run), self.spec.slug)[1]
            self._run_placements = {
                name: value for name, value in recorded.items()
                if name in PLACEMENTS and value is not None
            }
        return self._run_placements

    def _from_run_options(self) -> dict[str, Any]:
        """The options this run recorded for this figure, read once and checked.

        An option in the block that this figure does not declare is refused
        here rather than ignored. It is the same failure the flag check
        prevents, arriving by the other route: a misspelled option draws
        successfully with the defaults, which is the failure that looks like
        success.
        """
        if self._run_options is None:
            _, recorded = _from_run(Path(self.run), self.spec.slug)
            declared = {o.name: o for o in self.spec.options}
            # A placement in the block is for every figure, so it is not this
            # figure's to accept or refuse; ``_placement`` reads it separately.
            recorded = {name: value for name, value in recorded.items()
                        if name not in PLACEMENTS}
            unknown = [name for name in recorded if name not in declared]
            if unknown:
                accepted = ", ".join(declared) or "none"
                raise SystemExit(
                    f"figures.{self.spec.slug}.options: "
                    f"{', '.join(repr(name) for name in sorted(unknown))} "
                    f"{'is' if len(unknown) == 1 else 'are'} not "
                    f"{'an option' if len(unknown) == 1 else 'options'} of this "
                    f"figure; it accepts {accepted}"
                )
            self._run_options = {
                name: _from_json(declared[name], value)
                for name, value in recorded.items() if value is not None
            }
        return dict(self._run_options)

    def panels(self) -> list[Panel]:
        """The panels to draw, in declaration order.

        Availability is settled here rather than inside the build function for
        the same reason ``AnalysisModule.available`` settles it before a module
        runs: a page that quietly loses a panel is indistinguishable from a page
        that never had one, and the user should be told which of the two
        happened.

        Note the asymmetry. Naming a panel this figure does not have is an
        error - it is a typo, and drawing the rest would hide it. A panel that
        exists but has no data behind it is a skip, and is reported.
        """
        available = [p.key for p in self.spec.panels]
        wanted = raw_flag("panels", self.argv)
        keys = commas(wanted) if wanted is not None else list(available)
        unknown = [key for key in keys if key not in available]
        if unknown:
            raise SystemExit(
                f"--panels names {', '.join(unknown)}; this figure draws "
                f"{', '.join(available)}"
            )
        drawn: list[Panel] = []
        skipped: list[tuple[str, list[str]]] = []
        for panel in self.spec.panels:
            if panel.key not in keys:
                continue
            missing = [need for need in panel.needs if not self.has(need)]
            if missing:
                skipped.append((panel.key, missing))
                continue
            drawn.append(panel)
        if not drawn:
            asked = ", ".join(keys) or "none"
            reasons = "; ".join(f"{key} needs {_named(missing)}"
                                for key, missing in skipped)
            raise SystemExit(
                f"--panels {asked} selected no panels this run can draw"
                + (f": {reasons}" if reasons else "")
            )
        self.skipped_panels = skipped
        self._drawn = [panel.key for panel in drawn]
        for key, missing in skipped:
            print(f"  {self.spec.slug}: panel {key} not drawn - this run has no "
                  f"{_named(missing)}")
        return PanelList(drawn)

    @property
    def drawn(self) -> list[str]:
        """The panel keys actually drawn, for the README and the closing line."""
        if self._drawn is None:
            return [p.key for p in self.spec.panels]
        return list(self._drawn)

    # -------------------------------------------------------------- the words

    def footnote(self, *extra: str) -> str:
        """This figure's own sentences, above the units line every figure carries.

        A builder that wants to say something under the plot has to repeat the
        units sentence or lose it, because ``_finish`` falls back to the units
        line only when the footnote is empty. One place to add to it instead.
        """
        return NEWLINE.join(part for part in (*extra, units_note(self.summary))
                            if part)

    def provenance_footnote(self, *extra: str) -> str:
        """As ``footnote``, plus the provenance sentence when the movie has one.

        Empty-safe at both ends: a run with nothing to declare gets exactly the
        footnote it would have got without this.
        """
        return NEWLINE.join(part for part in (*extra, units_note(self.summary),
                                              provenance_note(self.summary)) if part)

    def provenance_auxiliary(self) -> dict[str, pd.DataFrame]:
        """The movie-level provenance numbers, so the bundle can be read alone."""
        table = provenance_table(self.summary)
        return {"provenance.csv": table} if not table.empty else {}

    # -------------------------------------------------------------- the canvas

    def drew(self, key: str, result: PanelResult) -> PanelResult:
        """Record what one panel drew, and hand it straight back.

        A builder calls its panels itself, so the schema only learns what went
        on the page if it is told. Telling it is one word at the call site and
        it buys the audit table for free: a figure that does not build its own
        gets its leading panel's, which is the table a reader would check the
        figure against.
        """
        self.panel_data[key] = result.data
        return result

    def leading_table(self) -> pd.DataFrame | None:
        """The table of the first panel drawn, for a build that named none."""
        for key in self.drawn:
            if key in self.panel_data:
                return self.panel_data[key]
        return None

    def block(self, figure_: Any, ax: Any, key: str, *args: Any,
              **kwargs: Any) -> Any:
        """Draw a panel that needs the sheet and a rectangle, not one axes.

        A scatter with its own marginal histograms has to make three axes, so it
        cannot be given one. The slot the layout allotted is measured, removed,
        and handed over as a rectangle - which keeps the decision about where
        the panel goes with the layout, where it belongs.

        The drawing function comes off the spec rather than being named here, so
        ``--help`` and the coverage test see the same function the page draws.
        """
        panel = self.spec.panel(key)
        if panel.draw is None:
            raise KeyError(f"{self.spec.slug}.{key} declares no drawing function")
        rect = tuple(ax.get_position().bounds)
        ax.remove()
        drawn = panel.draw(figure_, rect, *args, **kwargs)
        return self.drew(key, drawn) if isinstance(drawn, PanelResult) else drawn

    def sheet(self, width: float, height: float) -> Any:
        """A blank canvas of this many inches, sized through the theme.

        For a page whose proportions are its own - a report card that grows per
        metric, a histogram laid out for a three-line footnote. A figure that is
        just a stack of panels should use ``layout`` instead and inherit the
        spacing every other stacked page has.
        """
        return plt.figure(figsize=self.theme.canvas(width, height))

    def layout(self, panels: list[Panel] | PanelList, bottom_inches: float = 1.55
               ) -> tuple[Any, dict[str, Any]]:
        """A stack of panels on one sheet, with room for the header and footnote.

        Physical inches rather than fixed fractions for the title and the
        provenance line, so a large house-theme label stays clear on both a
        two-panel and a five-panel canvas.
        """
        if not panels:
            raise SystemExit("--panels selected no panels")
        height = 2.4 + 3.5 * len(panels)
        figure_ = plt.figure(figsize=self.theme.canvas(13.8, height))
        grid = figure_.add_gridspec(
            len(panels), 1, left=0.14, right=0.94, bottom=bottom_inches / height,
            top=1 - 1.65 / height, hspace=0.88,
        )
        axes = {}
        for index, panel in enumerate(panels):
            axes[panel.key] = figure_.add_subplot(
                grid[index, 0], projection="polar" if panel.polar else None)
            axes[panel.key].set_title(panel.heading(), loc="left",
                                      fontsize=self.theme.size("panel"),
                                      fontweight="bold")
        return figure_, axes


@dataclass
class FigureResult:
    """What a build function hands back: the drawing and the words beside it.

    Everything that happens to these - saving, the README, the copied sources,
    the closing line - is the same for all thirty-six pages, so it happens once
    in ``_finish`` rather than at the bottom of every builder.
    """

    figure: Any
    axes: list[Any]
    #: The exact table the leading panel drew. Left unset it is taken from
    #: whatever ``ctx.drew`` recorded, which is right whenever one panel leads
    #: the page; a figure combining two panels' tables still builds its own.
    figure_data: pd.DataFrame | None = None
    subtitle: str = ""
    footnote: str = ""
    note: str = ""
    #: Where a note goes on a page with room for one beside the drawing rather
    #: than under it: keyword arguments to ``figure.text`` with ``x`` and ``y``
    #: as figure fractions, and the words left out. ``None`` draws no note.
    #:
    #: The position is here and the wording is not, because the wording is a
    #: slot like the title: a builder that placed *and* resolved its own note
    #: would be a page whose ``--note`` did nothing.
    note_at: dict[str, Any] | None = None
    #: Named sibling tables written beside ``figure_data.csv``.
    auxiliary: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Fills the ``{placeholders}`` in the spec's default title. Formatted
    #: before the user's own title is considered, so a configured title is
    #: never format-checked and may contain a literal brace.
    title_fields: dict[str, Any] = field(default_factory=dict)
    #: This figure's README body. Empty takes the generic one.
    readme: str = ""
    #: The bundle README's heading, if the title is not the right name for the
    #: page. Empty becomes "<title>, <stem>", which is right for most.
    heading: str = ""
    #: A line the builder wants printed when it finishes - what it measured,
    #: which is not always visible on the figure.
    console: str = ""
    #: Where the words go. Each is a ``Length``, so a builder says whether it
    #: means inches or a share of the page, and a bare number still means a
    #: share - which is what these were before there was a choice.
    #:
    #: The defaults are not all in one unit, and that is the point. The title
    #: and subtitle sit a fixed number of *inches* below the top edge so that
    #: two figures of different heights carry their titles at the same distance
    #: from the paper's edge; a share would put them further down the taller
    #: page. The left edge and the footnote are *shares*, because they run with
    #: the plot rather than with the paper - the footnote in particular has to
    #: stay clear of the bottom axis, which is placed as a share too.
    #:
    #: All four are overridable per run: ``--title-y 0.4in``, ``--header-x 6%``,
    #: or an ``options`` block in the run's ``figures.json``.
    #:
    #: ``header_x`` is the left edge of the title, subtitle and footnote. The
    #: house value puts them near the paper's edge; a page whose plot starts
    #: further in - because it carries a colour bar and a category strip on the
    #: right - lines its words up with its plot instead.
    header_x: Length | float = 0.02
    #: The title, measured down from the top edge.
    title_y: Length | float = Length(0.20, "in")
    #: The subtitle, measured down from the top edge. Half an inch under the
    #: title, which is a gap and not a coincidence: closer and the two read as
    #: one paragraph, further and the subtitle detaches from what it qualifies.
    subtitle_y: Length | float = Length(0.70, "in")
    #: The footnote, measured up from the bottom edge.
    footnote_y: Length | float = 0.008
    #: Spines the house style would drop that this page keeps. A map of the
    #: field is bounded by the field, and a map with two of its four edges
    #: missing reads as a plot whose axes happen to end there.
    keep_spines: tuple[str, ...] = ()


# -------------------------------------------------------------------- running


def _from_json(declared: Option, value: Any) -> Any:
    """One configured option as the value a builder works with.

    A configuration is typed JSON, so ``"bins": 60`` is already a number and
    putting it through the cast a flag needs would be a round trip through
    text. A string still goes through the cast, because ``"metrics": "a,b"``
    and ``["a", "b"]`` are both reasonable things to write and should mean the
    same thing.

    ``null`` never reaches here: it means "use the default", which is different
    from ``""`` meaning "the empty value" - and the difference matters for
    ``--fit=``, where empty is a real choice.
    """
    if isinstance(value, str):
        try:
            return declared.caster()(value)
        except (TypeError, ValueError) as error:
            raise SystemExit(
                f"figures option {declared.name}: {value!r}: {error}") from None
    return value


def _check_argv(spec: FigureSpec, argv: list[str]) -> None:
    """Stop on a flag this figure does not accept, naming the ones it does.

    The failure this prevents is the quiet one. ``--bins 60`` on a page with no
    histogram is currently accepted and ignored: the user gets a figure that is
    not the one they asked for and no message saying so.
    """
    accepted = ({o.name for o in spec.options} | set(UNIVERSAL) | set(SLOTS)
                | set(UNIVERSAL_SWITCHES) | set(spec.switches))
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token.partition("=")[0][2:].replace("-", "_")
        if name not in accepted:
            declared = " ".join(o.flag for o in spec.options) or "(none)"
            raise SystemExit(
                f"--{token.partition('=')[0][2:]} is not an option of {spec.slug}.\n"
                f"This figure accepts: {declared}\n"
                + (f"This figure also accepts: "
                   + " ".join(f"--{name}" for name in spec.switches) + NEWLINE
                   if spec.switches else "")
                + f"Every figure also accepts: --stem --panels "
                + " ".join(f"--{slot}" for slot in SLOTS) + " "
                + " ".join(f"--{name}" for name in UNIVERSAL_SWITCHES)
            )
        index += skip_tokens(token)


def _named(missing: list[str]) -> str:
    """What a skipped panel wanted, in words rather than in the needs spelling.

    ``presence_frame.csv:unclaimed_px`` is a fine thing to declare and a poor
    thing to read, and the sentence it lands in is the one telling a user why
    their figure is a panel short.
    """
    parts = []
    for need in missing:
        table, _, column = need.partition(":")
        parts.append(f"{column} in {table}" if column else table)
    return ", ".join(parts)


def _place_note(figure_: Any, theme: Any, words: str, where: dict[str, Any] | None,
                width: float, height: float) -> Any:
    """The note beside the drawing, on a page that declared room for one.

    Two things have to be true before a note appears: the figure has somewhere
    to put it, which is the builder's decision and comes from the result, and
    there are words to put there, which is the user's and comes through the
    same resolution as the title. Either one missing draws nothing - a page
    with no room silently ignores ``--note`` rather than writing over its own
    plot, and a page with room and no words leaves the space empty.

    ``x`` and ``y`` may each be a ``Length``, so a page can pin its note a fixed
    number of inches in from the left and up from the bottom - matplotlib's own
    origin - rather than at a share of the sheet. A bare number still means a
    share, which is what every existing page's note is written in.

    Returns the artist, so the footnote's growth pass can put it back where it
    was in inches, or ``None`` when there is nothing to draw.
    """
    if not words or not where:
        return None
    placement = dict(where)
    return figure_.text(
        as_length(placement.pop("x")).fraction(width),
        as_length(placement.pop("y")).fraction(height), words,
        fontsize=placement.pop("fontsize", theme.size("caption")),
        color=placement.pop("color", theme.colour("caption")),
        va=placement.pop("va", "top"), **placement,
    )


def _clear_footnote(figure_: Any, footnote: Any, footnote_at: Length,
                    header: list[Any]) -> None:
    """Give the canvas more paper if the footnote grew under the plot.

    The footnote is anchored to the bottom and grows upward, so a run with an
    extra sentence to declare pushes it into whatever the lowest axis put there
    - normally its x-label. A builder lays its axes out for the footnote it
    expects, and how many lines that footnote will have is the one thing it
    cannot know: the provenance sentences appear only when the movie has a
    sidecar.

    So nothing is squashed to make room. The sheet gets taller at the bottom by
    exactly the overlap, and every axis and every line of header text is put
    back where it was in inches, which leaves the drawing identical and the
    footnote clear. A figure whose footnote already fits is untouched.

    Held in inches whichever unit the placement was written in, because this is
    not a resize anybody asked for - it is paper added at the bottom, and a
    title that slid down the page because the footnote ran to three lines would
    be the growth pass causing the problem it exists to avoid. The one thing the
    unit does change is the footnote itself: an inch-specified one is put back
    at its inches above the bottom edge, which can only ever move it further
    from the drawing.
    """
    figure_.canvas.draw()
    renderer = figure_.canvas.get_renderer()
    to_figure = figure_.transFigure.inverted()
    # A line of clear air between the footnote and whatever sits above it.
    ceiling = footnote.get_window_extent(renderer).transformed(to_figure).y1 + 0.008

    axes = [ax for ax in figure_.axes if ax.get_position().height > 0]
    if not axes:
        return
    try:
        floor = min(ax.get_tightbbox(renderer).transformed(to_figure).y0
                    for ax in axes)
    except (AttributeError, ValueError):
        return
    overlap = ceiling - floor
    if overlap <= 0:
        return

    height = float(figure_.get_figheight())
    grow = overlap * height
    # A guard, not a policy: an overlap this large is a builder's layout problem
    # and growing the page would only hide it behind a differently wrong figure.
    if grow > 0.35 * height:
        return
    taller = height + grow
    figure_.set_figheight(taller)
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, (box.y0 * height + grow) / taller,
                         box.width, box.height * height / taller])
    # The title and subtitle sit a fixed number of inches below the top edge,
    # which is a moving fraction once the sheet changes size.
    for artist in header:
        y = artist.get_position()[1]
        artist.set_y(1.0 - (1.0 - y) * height / taller)
    if footnote_at.in_inches:
        footnote.set_y(footnote.get_position()[1] * height / taller)


GENERIC_README = """## What the figure shows

{title}. Panels drawn: {panels}.

## Data

`data/der/figure_data.csv` is the exact leading-panel table. Named sibling
tables contain the other panels' returned values. Source tables and image
stacks are copied under `data/src/` and fingerprinted in `data/sources.csv`.

## Caveat

{caveat}
"""


def _default_title(spec: FigureSpec, result: FigureResult) -> str:
    """The spec's title with its placeholders filled from the build's numbers."""
    if not result.title_fields:
        return spec.title
    try:
        return spec.title.format(**result.title_fields)
    except KeyError as error:
        raise SystemExit(
            f"{spec.slug}: the declared title needs {error} and the build did "
            f"not supply it; title_fields holds "
            f"{', '.join(sorted(result.title_fields)) or 'nothing'}"
        ) from None


def _placement(ctx: FigureContext, name: str, declared: Length | float) -> Length:
    """Where one line of the header goes: a flag, else the run, else the builder.

    The same three-level resolution a declared option gets, for the four
    placements no figure declares because every figure has them. Where the
    answer came from is recorded in ``ctx.option_source`` alongside the rest, so
    anything asking whether a value was chosen or defaulted gets one answer for
    every option on the page rather than one answer for most of them.
    """
    raw = raw_flag(name, ctx.argv)
    if raw is not None:
        try:
            value = length(raw)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"--{name.replace('_', '-')} {raw!r}: {error}") from None
        source = "flag"
    else:
        recorded = ctx._from_run_placements()
        if name in recorded:
            try:
                value = length(str(recorded[name]))
            except (TypeError, ValueError) as error:
                raise SystemExit(
                    f"figures.{ctx.spec.slug}.options.{name}: {error}") from None
            source = "config"
        else:
            value, source = as_length(declared), "default"
    ctx.option_source[name] = source
    ctx.option_value[name] = value
    return value


def _header_positions(width: float, height: float, x: Length, title: Length,
                      subtitle: Length, footnote: Length) -> tuple[float, ...]:
    """The four placements as matplotlib figure coordinates.

    Its own function because this is where the two units and the two directions
    meet, and both are easy to get quietly wrong: the title and subtitle are
    measured *down from the top* and so are subtracted from 1, the footnote is
    measured *up from the bottom* and is not, and each of the four may be in
    either unit. Wrong by a sign is a title off the top of the page; wrong by a
    unit is a title an inch from where it was. Neither raises.
    """
    return (x.fraction(width), 1.0 - title.fraction(height),
            1.0 - subtitle.fraction(height), footnote.fraction(height))


def _finish(ctx: FigureContext, result: FigureResult) -> Path:
    """Save the figure, the table it was drawn from, and the words beside it.

    Every line of this is the same for all thirty-six figures, which is why it
    is here and not at the bottom of each one. A builder that did its own saving
    would be a builder whose bundle could differ from its neighbours' without
    anybody noticing.
    """
    ctx.prepare()
    der = ctx.bundle / "data" / "der"
    der.mkdir(parents=True, exist_ok=True)
    table = result.figure_data
    if table is None:
        table = ctx.leading_table()
    if table is None:
        raise SystemExit(
            f"{ctx.spec.slug}: no figure_data. Either set it on the FigureResult "
            f"or pass the leading panel through ctx.drew(<key>, ...), or the "
            f"bundle has a picture with no table behind it")
    table.to_csv(der / "figure_data.csv", index=False)
    for name, table in result.auxiliary.items():
        table.to_csv(der / name, index=False)

    text = figure_text(
        ctx.run, ctx.spec.slug, argv=ctx.argv,
        title=_default_title(ctx.spec, result),
        subtitle=result.subtitle,
        footnote=result.footnote or units_note(ctx.summary),
        note=result.note,
    )
    figure_height = float(result.figure.get_figheight())
    figure_width = float(result.figure.get_figwidth())
    under = _placement(ctx, "footnote_y", result.footnote_y)
    left, title_y, subtitle_y, footnote_y = _header_positions(
        figure_width, figure_height,
        _placement(ctx, "header_x", result.header_x),
        _placement(ctx, "title_y", result.title_y),
        _placement(ctx, "subtitle_y", result.subtitle_y),
        under,
    )
    header = [result.figure.text(
        left, title_y, wrap_title(ctx.theme, result.figure, text.title),
        fontsize=ctx.theme.size("title"), fontweight="bold",
        color=ctx.theme.colour("ink"), va="top",
    ), result.figure.text(
        left, subtitle_y, text.subtitle, fontsize=ctx.theme.size("subtitle"),
        color=ctx.theme.colour("caption"), va="top",
    )]
    note = _place_note(result.figure, ctx.theme, text.note, result.note_at,
                       figure_width, figure_height)
    if note is not None:
        header.append(note)
    footnote = None
    if text.footnote:
        footnote = result.figure.text(
            left, footnote_y, text.footnote, fontsize=ctx.theme.size("note"),
            color=ctx.theme.colour("caption"), va="bottom",
        )
    for ax in result.axes:
        try:
            if ax.get_legend() is None and not getattr(ax, "_semantic_legend_handled", False):
                common.semantic_legend(ax, ctx.theme, location="inside")
            if getattr(ax, "name", "") != "polar":
                ctx.theme.finish(ax, keep_spines=result.keep_spines)
        except (KeyError, AttributeError):
            pass
    if footnote is not None:
        _clear_footnote(result.figure, footnote, under, header)

    body = result.readme or GENERIC_README.format(
        title=_default_title(ctx.spec, result),
        panels=", ".join(ctx.drawn),
        caveat=result.footnote or units_note(ctx.summary),
    )
    if ctx.skipped_panels:
        body += "\n\n## Panels not drawn\n\n" + "\n".join(
            f"- `{key}`: this run has no {_named(missing)}"
            for key, missing in ctx.skipped_panels
        ) + "\n"
    heading = result.heading or _default_title(ctx.spec, result)
    write_readme(ctx.bundle, f"{heading}, {ctx.summary['stem']}", text, body)
    write_standalone_producer(ctx.bundle, text.title, ctx.spec.grammar)
    save_reprofig_figure(
        ctx.theme, result.figure, ctx.run, ctx.bundle, f"{ctx.spec.slug}.svg",
        claim=text.claim, grammar=ctx.spec.grammar, producer="code/plot.py",
    )
    target = ctx.bundle / "fig" / f"{ctx.spec.slug}.svg"
    plt.close(result.figure)
    if result.console:
        print(result.console)
    print(f"{ctx.spec.slug}: panels {ctx.drawn}; rows {len(table):,}; {target}")
    print(f"title ({text.source['title']}): {text.title}")
    for name, origin in sorted(ctx.option_source.items()):
        if origin != "default":
            print(f"{name} ({origin}): {ctx.option_value[name]}")
    return target


def run_figure(slug: str, default_run: str | None = None,
               argv: list[str] | None = None) -> Path:
    """Draw one figure from the command line, start to finish."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    spec = get_figure(slug)
    if {"--help", "-h"} & set(tokens):
        print(spec.usage())
        raise SystemExit(0)
    _check_argv(spec, tokens)
    run = run_folder(default_run)
    stem = option("stem", default=None, argv=tokens)
    ctx = FigureContext(
        spec=spec,
        run=run,
        tables=tables_for(run, stem),
        theme=theme_for(run),
        bundle=bundle_for(run, slug),
        summary=summary_for(run, stem),
        field=field_for(run, stem),
        stem=stem,
        argv=tokens,
    )
    return _finish(ctx, spec.build(ctx))
