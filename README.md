# Motion Microglia Detection Pipeline

A motion-first detector and longitudinal tracker for registered microglial imaging data. Cell identity is anchored to the soma in the registered signal, while motion evidence supports links through movement, merges, splits, and temporary loss of visibility.

## Public repository scope

This repository contains the reusable Python pipeline only. Raw microscopy data, experiment-specific configuration, validation records, tests, and generated outputs remain private and are excluded from Git history.

## Requirements

- Python 3.12
- Packages listed in `requirements.txt`
- Registered raw image stacks and the corresponding motion-evidence stacks

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Configure

Copy `config.example.json` to the ignored local file `config.json`:

```powershell
Copy-Item config.example.json config.json
```

Then replace `YOUR_DATASET`, `YOUR_STEM`, the input directories, and every `<replace-with-sha256>` value. In PowerShell, calculate a file fingerprint with:

```powershell
(Get-FileHash .\path\to\input.tif -Algorithm SHA256).Hash.ToLower()
```

The pipeline verifies these fingerprints before analysis and writes each run to new immutable stage folders.

## Run

Run the complete pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Run NEW_RUN -Stem YOUR_STEM
```

Rerun only the review layer from an unchanged identity run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 `
  -Run NEW_REVIEW -Stem YOUR_STEM -ReviewOnlyFrom NEW_RUN
```

## Output

The accepted label stack is written under `m20_oscillatory_lineage/<run>/out/`. The complete-field review stack is written under `m6_review/<run>/out/`. All run folders are ignored by Git.

## Licence

No open-source licence is currently granted. The source is public for inspection; contact the maintainer before reuse.

Report problems through [GitHub Issues](https://github.com/Jay2owe/motion-microglia-detection/issues).
