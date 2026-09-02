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

The final label stack is written under `m21_stationary_reconciliation/<run>/out/`.
This last stage automatically repairs strongly evidenced stationary identity takeovers
across the whole field. It is enabled by default and can be disabled with
`accepted_postprocessing.field_wide_stationary_reconciliation.enabled` in the local
configuration. The complete-field review stack is written under `m6_review/<run>/out/`.
All run folders are ignored by Git.

Beside the accepted label stack, `m22_accepted_history/<run>/out/` also holds
`<stem>_provenance.tif`: one `uint8` frame per label frame, carrying three bit
flags.

The current accepted history ends with Issue 089 A022, applied after the clean
Issue 090 parent. It audits every overlapping raw-reference pair and preserves
both established owners when a moving body temporarily merges with a recurrent
stationary seat, including a second resident-owned merge. Raw two-core frames
use a watershed; a dim single-core bridge preserves a field-derived
resident-sized core and assigns the connected remainder to the moving lineage.
It introduces no identity, preserves the foreground-ledger union, and receives
no identity, track, frame, event, coordinate, region, or review-case targets.
Issues 065, 074, 081, 084, and 086 remain excluded from accepted production.

The accepted disruption-confidence catalogue retains the score-only Issue 083
alternating-reference calibration on the Issue 089 labels. It removes a merge alert only when two
same-birth references share one owner, alternate visibility inside an uncensored
lifetime, and have no two-core raw-signal evidence. It changes no label or
unclaimed-pixel TIFF and accepts no identity, track, frame, event, coordinate,
region, or review-case target.

| bit | value | meaning |
| --- | --- | --- |
| 0 | `0b001` | the owning identity was decided by reconstruction, not by observing that cell there |
| 1 | `0b010` | foreground the tracker never resolved to anyone |
| 2 | `0b100` | an outline pixel the segmentation never saw: the tracker supplied the pixel, not just the name |

It is saved from arrays the run already holds, so it costs no detection time.

Bits 0 and 2 are the ones that are easy to confuse, and the difference is the
whole reason bit 2 exists. Bit 0 alone is a statement about the *name*; bit 2 is
a statement about the *picture*. On the accepted `95_A3`, of the 404,337 flagged
pixels lying inside an outline, 403,765 were shown by the microscope with only
the owner supplied by inference, and 572 are outline the tracker supplied where
the detection saw nothing at all. On `95_B4` the same split is 220,121 against
384. Bit 2 has never yet appeared without bit 0, which is checked on every build
and reported as `provenance_added_unflagged_px`.

The 572 are not a rim of invented pixels around real cells. They are 17 whole
cell-frames, of 6,073, belonging to four identities, where the cell is on screen
and the detection saw nothing there at all: a held position carried through a
frame the microscope did not support. That is the sharp, checkable version of
"the tracker drew this cell", and it is 0.28% of the cell-frames.

So a high reconstructed share still matters to any per-cell number, since every
one of them depends on which cell owns the pixel, but it is not the same as
saying the outline was invented - and now that is measured rather than asserted.

For a run made before the sidecar existed,
`code/build_provenance_sidecar.py` rebuilds it from the per-stage records still
on disk and writes it to its own immutable `m23_provenance/<run>/` folder rather
than touching the run it describes.

## Prepare an output for manual editing

Create a validated editing bundle from the locally accepted result:

```powershell
python .\code\manual_editing.py prepare-accepted `
  --output .\manual_editing\NEW_BUNDLE `
  --accepted-base .\accepted_base.json `
  --config .\config.json `
  --stem YOUR_STEM
```

Prove that the bundle survives an unchanged save and reimport before editing it:

```powershell
python .\code\manual_editing.py zero-edit `
  --bundle .\manual_editing\NEW_BUNDLE `
  --output .\manual_editing\NEW_ZERO_EDIT_SESSION
```

Open the graphical editor on a prepared bundle or saved session:

```powershell
python .\code\manual_editing.py edit `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE
```

Click an outlined cell to select it; Ctrl-click selects a batch. The canvas provides
frame navigation, `−`, `+`, and `Fit` controls, mouse-wheel zoom around the cursor,
identity numbers, an optional centroid-track overlay, removal, expansion, frame-range
swaps, directional identity assignment, inline track-filter selection, JSON batches,
and the complete Undo/Redo history. Each
applied operation is saved and reimported as a new immutable session while the window
stays open.

Freeze the active bundle or session into one self-contained file before moving it or
archiving it:

```powershell
python .\code\manual_editing.py freeze-package `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\CURRENT_CHECKPOINT.motionpkg

python .\code\manual_editing.py validate-package `
  --package .\manual_editing\CURRENT_CHECKPOINT.motionpkg
```

The graphical editor also provides `Freeze package...` and `Import package...`.
Opening a package keeps the same Undo, Redo, review, and editing workflows, but every
new save is another complete `.motionpkg` file rather than a folder that refers back to
the original computer.

Create a new immutable session that swaps two identities over an inclusive ImageJ
frame range:

```powershell
python .\code\manual_editing.py swap-identities `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_SWAP_SESSION `
  --identity-a FIRST_IDENTITY `
  --identity-b SECOND_IDENTITY `
  --start-frame FIRST_IMAGEJ_FRAME `
  --end-frame LAST_IMAGEJ_FRAME
```

Relabel one or more source identities as an existing target without swapping the
target back into the sources:

```powershell
python .\code\manual_editing.py assign-identities `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_ASSIGNMENT_SESSION `
  --source-identity SOURCE_IDENTITY `
  --target-identity TARGET_IDENTITY `
  --start-frame FIRST_IMAGEJ_FRAME `
  --end-frame LAST_IMAGEJ_FRAME
```

Remove several cell identities in one atomic session by repeating `--identity`:

```powershell
python .\code\manual_editing.py remove-identity `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_REMOVAL_SESSION `
  --identity FIRST_IDENTITY_TO_REMOVE `
  --identity SECOND_IDENTITY_TO_REMOVE
```

Delete selected identities only between inclusive ImageJ frame numbers while
preserving their earlier and later track segments:

```powershell
python .\code\manual_editing.py delete-between-frames `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_INTERVAL_DELETION_SESSION `
  --identity FIRST_IDENTITY `
  --identity SECOND_IDENTITY `
  --start-frame FIRST_IMAGEJ_FRAME `
  --end-frame LAST_IMAGEJ_FRAME
```

Expand one cell or several cells simultaneously into unassigned background. Existing
cell labels are protected, and `calibrated` uses the bundle's physical spatial unit:

```powershell
python .\code\manual_editing.py expand-identities `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_EXPANSION_SESSION `
  --identity FIRST_IDENTITY `
  --identity SECOND_IDENTITY `
  --radius 2 `
  --radius-unit calibrated `
  --start-frame FIRST_IMAGEJ_FRAME `
  --end-frame LAST_IMAGEJ_FRAME
```

Apply an ordered mixture of swaps and removals from the versioned example batch:

```powershell
python .\code\manual_editing.py apply-batch `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --output .\manual_editing\NEW_BATCH_SESSION `
  --batch .\docs\manual-track-editing\batch-example.json
```

Open the graphical checkpoint history. Undo and Redo move the active checkpoint;
continuing from an earlier state creates a new immutable branch and retains the
original later edits:

```powershell
python .\code\manual_editing.py history-controls `
  --source .\manual_editing\CURRENT_SESSION
```

The same window accepts comma-separated identities or a batch JSON file and saves the
result as one session.

Preview track filters before removing anything:

```powershell
python .\code\manual_editing.py preview-filters `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --filters .\docs\manual-track-editing\filter-example.json `
  --output .\manual_editing\NEW_FILTER_PREVIEW
```

After reviewing `filter_matches.csv`, apply the same filters as one reversible batch:

```powershell
python .\code\manual_editing.py remove-by-filters `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --filters .\docs\manual-track-editing\filter-example.json `
  --output .\manual_editing\NEW_FILTERED_SESSION
```

The editor splits identity selection into a manual list and an inline filter builder.
Add editable rules, combine them with `all` or `any`, then apply them to fill the
selected-identity list before choosing any edit operation.

Apply the same filter set to expansion from the command line:

```powershell
python .\code\manual_editing.py apply-by-filters `
  --operation expand `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE `
  --filters .\docs\manual-track-editing\filter-example.json `
  --output .\manual_editing\NEW_FILTERED_EXPANSION_SESSION `
  --radius 2 `
  --radius-unit calibrated
```

The same history is available without a graphical desktop:

```powershell
python .\code\manual_editing.py list-history `
  --source .\manual_editing\CURRENT_SESSION
```

Generate full-field, focused, and representative before/after/change images:

```powershell
python .\code\manual_editing.py build-review `
  --session .\manual_editing\NEW_SWAP_SESSION `
  --output .\manual_editing\NEW_SWAP_SESSION_REVIEW
```

Add missed cells from moving crops, automatic local proposals, or manual outlines by
opening the editor and clicking `Add new cells...`:

```powershell
python .\code\manual_editing.py edit `
  --source .\manual_editing\CURRENT_SESSION_OR_BUNDLE
```

The new-cell batch is reviewed before one atomic save and remains replayable through
the same Undo, Redo, branch, reimport, and review commands.

After a complete-track removal or frame-range deletion, click
`Retune masks after edit...` to let selected or suggested neighbouring identities
compete for the released background. The default changes only the exact vacated pixels;
existing owners are protected, ambiguous pixels remain background, and the reviewed
result is saved as its own replayable Undo step.

## Licence

No open-source licence is currently granted. The source is public for inspection; contact the maintainer before reuse.

Report problems through [GitHub Issues](https://github.com/Jay2owe/motion-microglia-detection/issues).
