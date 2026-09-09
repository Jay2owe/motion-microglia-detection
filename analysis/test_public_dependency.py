"""Motion Analysis uses only Circadian Workbench's stable public boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from analysis import circadian, contrasts


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_circadian_gateway_imports_only_the_package_root():
    path = Path(circadian.__file__)

    assert [name for name in _imports(path) if name.startswith("circadian_workbench")] == [
        "circadian_workbench"
    ]


def test_contrasts_use_public_statistics_without_copied_formulae():
    path = Path(contrasts.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "circadian_workbench" in " ".join(_imports(path))
    assert "hedges_g" not in definitions
    assert "_variance_noise_floor" not in definitions
    assert "workbench_statistics.is_degenerate(values)" in source
    assert "workbench_statistics.adjust_pvalues" in source
