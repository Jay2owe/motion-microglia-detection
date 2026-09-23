# Motion analysis compatibility commands

Measurements, cell states, clustering, figures, cell videos and the six saved
analysis workflows now live in **PyMicroglia**. Motion retains detection,
tracking, accepted identity history, feedback marks and review images.

Install the published analysis package and state-learning dependencies:

```powershell
python -m pip install "PyMicroglia[states]>=0.3,<0.4"
```

The compatible release range is `PyMicroglia[states]>=0.3,<0.4`.

## Command mapping

| Existing command | PyMicroglia action |
| --- | --- |
| `analysis run --config ... --out ...` | `measure`, with the complete `analysis_config` |
| `analysis pool`, `window`, `contrasts` | `pool`, `window`, `contrasts` |
| `analysis states`, `cluster` | `states`, `cluster` |
| `analysis videos` | `follow` |
| `analysis plots` | The figure named by each saved plot-plan item |
| `analysis pipeline` | The requested workflow's public action |
| `analysis doctor`, `modules`, `figures` | `doctor`, `discover`, `describe` |

Each compatibility command prints its equivalent installed command before
execution. Configuration paths, calibration, selected movies, scientific options
and saved plans are retained. Expanded plot plans retain their separate names.

## Result layout

New measurement runs contain `measure/<movie>/`, `tracker/<movie>/`,
`stacks/<movie>/`, `windows/<movie>/` and `pooled/`. Shared records use one
`artefacts.json` ledger per folder. Read saved documents through
`pymicroglia._results.read_document`; physical bookkeeping lives in the hidden
`.auto-organotypic/` directory.

Figure actions expose named `view` choices. Scientific calculations go through
`pymicroglia.workbench` into Circadian Workbench. Period estimation, significance
and sufficient observation remain separate; no daily rhythm is assumed.

Existing analysis configuration files remain valid inputs. Historical
`analysis/*.example.json` option paths resolve to their unchanged copies
inside the installed PyMicroglia package. Historical issue
records keep their original commands. Tracking has not moved; its PyMicroglia
entry point remains explicitly pending.
