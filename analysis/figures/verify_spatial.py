"""Read-only audit of saved spatial bundles, including their full producers.

Producer save calls are intercepted in memory: no extra figures, caches or
verification subdirectories are written into the flat bundles.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import pandas as pd
import reprofig
from refresh_spatial import SLUGS


def verify(run):
    for slug in SLUGS:
        bundle = Path(run) / "figures" / slug
        assert not any(path.is_dir() for path in bundle.iterdir()), bundle
        for row in pd.read_csv(bundle / "sources.csv").itertuples():
            with (bundle / row.copied_path).open("rb") as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == row.sha256
        master = bundle / f"{slug}.svg"
        record = reprofig.extract_record(master)
        captured = []
        original_save = reprofig.save_figure
        saved_argv = sys.argv
        try:
            sys.path.insert(0, str(bundle))
            sys.modules.pop("src_renderer", None)
            spec = importlib.util.spec_from_file_location("spatial_producer_verification", bundle / "plot.py")
            module = importlib.util.module_from_spec(spec)
            # Audit the known generated producer before executing its renderer.
            tree = ast.parse((bundle / "plot.py").read_text(encoding="utf-8"))
            imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            assert imports <= {"pathlib"}
            spec.loader.exec_module(module)
            reprofig.save_figure = lambda figure, *a, **k: captured.append(figure)
            sys.argv = [str(bundle / "plot.py")]
            module.main()
            assert len(captured) == 1
            captured[0].canvas.draw()
            assert len(captured[0].axes) >= 1
            print(f"{slug}: source hashes and full producer passed; {record.figure_id}", flush=True)
        finally:
            reprofig.save_figure = original_save
            sys.argv = saved_argv
            sys.path.remove(str(bundle))
            for figure in captured:
                plt.close(figure)


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
