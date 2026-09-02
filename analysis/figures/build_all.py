"""Build every figure for one analysis run.

    python analysis/figures/build_all.py outputs/a03_my_run

Two kinds of build. The default is the audited one: each builder writes a
plot-that bundle to ``<run>/figures/<slug>/`` - the figure, the exact table it
was drawn from, copies of every source with its hash, and a README.

    python analysis/figures/build_all.py outputs/a03_my_run --draft

A draft writes the figures and nothing else, flat, into ``outputs/figures/``,
over the top of the last round. That is the loop for deciding what a plot should
look like: the tables have not changed, only the drawing has, so there is no new
run folder and nothing to keep. Build without ``--draft`` once a figure is
settled - a draft is not checkable and is not meant to leave the machine.

**Only the arguments every figure shares are passed through** - ``--stem`` and
the switches. A figure's own options belong to that figure: ``--bins`` means
something on six of the thirty-six and nothing on the rest, and passing it to
all of them would be refused by thirty of them and silently wrong on none, which
is worse than saying so here. Set a figure's options in the ``figures`` block of
the analysis configuration, where they survive into the run folder, or run that
one builder directly:

    python analysis/figures/09_step_size_distribution.py <run> --bins 60

Registering a bundle is a separate, deliberate step and is not done here:

    python ~/.claude/skills/plot-that/scripts/register.py add <bundle> --claim "..."
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDERS = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _options import skip_tokens  # noqa: E402

#: What means the same thing to every figure, and so may be given to all of them:
#: which movie to draw, and the switches every figure honours. Written out rather
#: than imported so that building does not pull matplotlib into this launcher;
#: `test_figure_schema` checks it still matches `_schema.UNIVERSAL_SWITCHES`.
SHARED = {"stem", "draft"}


def partition(arguments: list[str]) -> tuple[list[str], list[str]]:
    """Split the command line into what every figure takes and what it does not."""
    passthrough: list[str] = []
    personal: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            index += 1
            continue
        width = skip_tokens(token)
        name = token.partition("=")[0][2:].replace("-", "_")
        (passthrough if name in SHARED else personal).extend(
            arguments[index: index + width])
        index += width
    return passthrough, personal


def main() -> int:
    arguments = list(sys.argv[1:])
    positional = [a for a in arguments if not a.startswith("--")]
    if not positional:
        raise SystemExit(
            "usage: python build_all.py <path to an analysis run folder> [--draft]"
        )
    run = Path(positional[0]).resolve()
    if not run.is_dir():
        raise SystemExit(f"not a run folder: {run}")

    passthrough, personal = partition([a for a in arguments if a != positional[0]])
    if personal:
        named = " ".join(a for a in personal if a.startswith("--"))
        raise SystemExit(
            f"{named} belongs to one figure, not to all of them.\n"
            "Every figure takes: "
            + " ".join(f"--{name}" for name in sorted(SHARED)) + "\n"
            "For a figure's own options, either set them in the `figures` block "
            "of the analysis\nconfiguration, where they survive into the run "
            "folder, or run that builder alone:\n"
            "    python analysis/figures/09_step_size_distribution.py <run> --bins 60\n"
            "    python -m analysis figures --slug <slug>   to see what one accepts"
        )
    draft = "--draft" in passthrough

    failures = []
    for builder in BUILDERS:
        print(f"--- {builder.name}")
        result = subprocess.run([sys.executable, str(builder), str(run), *passthrough])
        if result.returncode != 0:
            failures.append(builder.name)

    if failures:
        print()
        print(f"{len(failures)} builder(s) failed: {', '.join(failures)}")
        return 1
    where = run.parent / "figures" if draft else run / "figures"
    kind = "draft figure(s)" if draft else "bundle(s)"
    print()
    print(f"{len(BUILDERS)} {kind} written to {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
