"""Flat, registered plot-that bundles for builders with exact producers."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def finish(ctx, result):
    import matplotlib.pyplot as plt
    scripts = Path.home() / ".claude/skills/plot-that/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_style import save

    bundle = ctx.bundle
    # Never mingle an old nested bundle with a new flat one.
    if bundle.exists() and any(p.is_dir() for p in bundle.iterdir()):
        raise ValueError(f"{bundle} contains subdirectories; choose a new plot item name")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "README.md").write_text(result.readme, encoding="utf-8")
    result.figure_data.to_csv(bundle / "figure_data.csv", index=False)
    for name, table in result.auxiliary.items():
        name = name if name == "statistics.csv" else "der_" + name.removeprefix("der_")
        table.to_csv(bundle / name, index=False)
    (bundle / "plot.py").write_text(result.standalone_producer, encoding="utf-8")
    sources = {**ctx.sources, **result.producer_sources}
    rows = []
    for short, source in sources.items():
        source = Path(source).resolve()
        name = "src_" + Path(short).name.removeprefix("src_")
        target = bundle / name
        if source != target.resolve():
            shutil.copy2(source, target)
        stat = source.stat()
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise IOError(f"copied source hash mismatch: {target}")
        origin = getattr(ctx, "source_origins", {}).get(short, str(source))
        rows.append(dict(original_path=origin, copied_path=name, file_name=Path(origin).name,
                         modification_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                         byte_size=stat.st_size, sha256=digest))
    pd.DataFrame(rows).to_csv(bundle / "sources.csv", index=False)
    (bundle / "sources.md").write_text("# Sources\n\n" + "\n".join(
        f"- `{r['copied_path']}`: {r['sha256']}" for r in rows) + "\n", encoding="utf-8")
    status = "complete" if "statistics.csv" in result.auxiliary else "not_applicable"
    target = bundle / (ctx.name.replace("/", "-").replace("\\", "-") + ".svg")
    claim = result.heading or ctx.spec.summary
    # Dropbox may lock hidden candidate files during atomic SVG/PNG embedding.
    # Finish the carriers off the synced drive and copy only completed files.
    with tempfile.TemporaryDirectory(prefix="motion-spatial-export-") as temporary:
        staged = Path(temporary)
        for source in bundle.iterdir():
            if source.is_file() and not source.name.startswith("."):
                shutil.copy2(source, staged / source.name)
        save(result.figure, staged / target.name, claim=claim, grammar=ctx.spec.grammar,
             producer="plot.py", statistics_status=status)
        for name in (target.name, "preview.png"):
            for attempt in range(6):
                try:
                    shutil.copy2(staged / name, bundle / name)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(.25 * (attempt + 1))
    plt.close(result.figure)
    command = [sys.executable, str(scripts / "register.py"), "add", str(bundle),
                    "--claim", claim, "--grammar", ctx.spec.grammar, "--producer", "plot.py",
                    "--statistics-status", status]
    for attempt in range(4):
        registered = subprocess.run(command, capture_output=True, text=True)
        if registered.returncode == 0:
            print(registered.stdout)
            break
        if "PermissionError" not in registered.stderr or attempt == 3:
            raise RuntimeError(registered.stdout + registered.stderr)
        time.sleep(.5 * (attempt + 1))
    # Preserve the user's flat image-review workflow without replacing any old plot.
    gallery = ctx.run / "figures" / "_all-images"
    gallery.mkdir(exist_ok=True)
    shutil.copy2(bundle / "preview.png", gallery / (target.stem + ".png"))
    print(f"{ctx.name}: {len(result.figure_data):,} encoded rows; {target}")
    return target
