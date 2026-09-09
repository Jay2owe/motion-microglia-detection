"""Build every figure for one analysis run.

    python analysis/figures/build_all.py outputs/a03_my_run

Two kinds of output. The default is the audited one: each builder writes a
plot-that bundle to ``<run>/figures/<slug>/`` - the figure, the exact table it
was drawn from, copies of every source with its hash, and a README.

    python analysis/figures/build_all.py outputs/a03_my_run --draft
    python analysis/figures/build_all.py outputs/a03_my_run --results-only
    python analysis/figures/build_all.py outputs/a03_my_run --review-only

A draft writes the figures and nothing else, flat, into ``outputs/figures/``,
over the top of the last round. That is the loop for deciding what a plot should
look like: the tables have not changed, only the drawing has, so there is no new
run folder and nothing to keep. Build without ``--draft`` once a figure is
settled - a draft is not checkable and is not meant to leave the machine.

**Only the arguments every figure shares are passed through** - ``--stem`` and
the switches. A figure's own options belong to that figure: ``--bins`` means
something on histogram figures and nothing on the rest, and passing it to
all of them would be refused wherever it has no meaning. Set a figure's options in the ``figures`` block of
the analysis configuration, where they survive into the run folder, or run that
one builder directly:

    python analysis/figures/09_step_size_distribution.py <run> --bins 60

Registering a bundle is a separate, deliberate step and is not done here:

    python ~/.claude/skills/plot-that/scripts/register.py add <bundle> --claim "..."

By default both scientific result figures and audit/review figures are built.
``--results-only`` and ``--review-only`` select one source module without
passing that launcher-only choice into the individual builders.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _builders import builder_files, is_review_builder  # noqa: E402
from _options import skip_tokens  # noqa: E402

BUILDERS = builder_files()

#: What means the same thing to every figure, and so may be given to all of them:
#: which movie to draw, and the switches every figure honours. Written out rather
#: than imported so that building does not pull matplotlib into this launcher;
#: `test_figure_schema` checks it still matches `_schema.UNIVERSAL_SWITCHES`.
SHARED = {"stem", "draft"}
SCOPE_SWITCHES = {"results_only", "review_only"}


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
            "usage: python build_all.py <path to an analysis run folder> "
            "[--draft] [--results-only | --review-only]"
        )
    run = Path(positional[0]).resolve()
    if not run.is_dir():
        raise SystemExit(f"not a run folder: {run}")

    selected_scopes = {token[2:].replace("-", "_") for token in arguments
                       if token.startswith("--") and
                       token[2:].replace("-", "_") in SCOPE_SWITCHES}
    if len(selected_scopes) > 1:
        raise SystemExit("--results-only and --review-only cannot be used together")
    figure_arguments = [a for a in arguments
                        if a != positional[0] and
                        a[2:].replace("-", "_") not in SCOPE_SWITCHES]
    passthrough, personal = partition(figure_arguments)
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

    builders = BUILDERS
    if "results_only" in selected_scopes:
        builders = [path for path in builders if not is_review_builder(path)]
    elif "review_only" in selected_scopes:
        builders = [path for path in builders if is_review_builder(path)]

    failures = []
    for builder in builders:
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
    print(f"{len(builders)} {kind} written to {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
