# Additional spatial time-series plots

These six figures are additions alongside the canonical six-panel
`tissue-tectonics` summary.

- `tissue-expansion-sequence`: actual-frame maps of a chosen metric, with optional
  consecutive-frame soma/centroid displacement arrows.
- `spatial-rhythm-maps`: broad-search period map and a peak-position map on each
  significant cell's own 0-100% cycle scale. Solid markers meet period-support
  requirements; outlined markers retain exploratory estimates. Explicitly
  selected period groups can instead be shown in hours on their named group cycles.
- `spatial-rhythm-progression`: actual traces ordered by physical X or Y position,
  with a position map. Ordering is not inferred from the trace's phase.
- `reporter-shape-timing`: two freely chosen metrics, with signed peak delay at a
  common recording hour. Unsupported or incompatible rhythms are not assigned delays.
- `neighbour-coordination`: simultaneous Pearson correlation on a symmetric
  nearest-neighbour graph, plus a whole-trace spatial permutation comparison.
- `tissue-coverage-gaps`: tracked-label occupancy history and the occupied fraction
  of the tissue occupied at least once during the recording.

## Main analysis configuration

Use the ordinary `plots` registry. For example:

```json
{
  "plots": [
    {
      "figure": "tissue-expansion-sequence",
      "options": {
        "spatial_metric": "radial_occupancy",
        "snapshot_hours": [1, 13, 25, 37],
        "display": "standardized",
        "spatial_arrows": true
      }
    },
    {
      "figure": "spatial-rhythm-maps",
      "options": {
        "spatial_metric": "area_px",
        "fit_method": "fft_nlls",
        "significance_method": "lomb",
        "period_min_hours": 2,
        "period_max_hours": 48,
        "phase_group_hours": [6, 12, 24],
        "detrend": "poly6",
        "min_cycles": 3
      }
    }
  ]
}
```

Each numbered builder also accepts the analysis-run folder on its command line.
`--help` lists its applicable options. The same declarations drive plot-plan
validation; parameters are not silently accepted by unrelated plots.

## Shared controls

Time-point plots (`tissue-expansion-sequence`, `tissue-coverage-gaps`, and
`spatial-rhythm-progression`) show the **final recorded frame by default**:

- `snapshot_count: 1` selects the final frame. A larger positive integer selects
  that many approximately evenly spaced actual frames, including start and end.
- `snapshot_hours: [1, 13, 25, 49]` selects named recording hours and overrides the
  count. Nearest actual frames are used, ordered chronologically, with duplicates
  removed. Out-of-range times and counts exceeding the frame count are rejected.
- `show_history: true` restores the complete progression matrix or the complete
  coverage trace; false is the default. Coverage snapshot maps still follow the
  selected hours/count. Selected matrix columns are separate snapshots, not
  interpolated intervals.

These options work in main-analysis `plots[].options`, individual builder flags
(`--snapshot-count`, `--snapshot-hours`, `--show-history`), and cached redraws.
Generic `snapshot_indices(hours, requested=None, count=1)` resolves the same
selection. The generic renderer accepts the resulting `snapshots` metadata;
`spatial_matrix(..., hours=chosen_hours, discrete=True)` draws selected columns.
Missing values at the final frame remain missing: earlier cell observations
are not carried forward. Empty final frames remain visible as empty fields.

The other three figures are already single **whole-recording summaries**, not
sequences of snapshots: period/phase maps, reporter-to-shape timing, and neighbour
coordination. Their inference continues to use full traces. With no
`phase_group_hours`, the peak map shows every significant cell with an available
estimate as a percentage through its own fitted cycle. Outlines distinguish
exploratory periods from supported periods; the normalization does not establish
comparable timing or synchrony. A blue-teal-gold-red cyclic palette gives 0%
and 100% the same colour because they are the same point in a cycle. Explicit
phase groups are different fitted
periods, not recording snapshots; `phase_group_hours` chooses those groups.
`timing_reference_hour` chooses the hour for the single timing map.
Snapshot flags are deliberately not accepted where they would have no effect.

The detected-period panel of `spatial-rhythm-maps` overlays cell tracks by
default. `spatial_track_cells: "rhythmic"` shows only cells with a significant
saved rhythm-test verdict; `"all"` includes every tracked cell. This choice does
not change which cell markers or statistical results are displayed. Tracks use
the period colour scale, with grey for cells lacking a displayed period.
Exploratory periods remain distinguished by their outlined cell markers.
`spatial_tracks: false` disables the overlay, and `spatial_track_width` sets its
line width in points (default 1.6). `spatial_centre` selects soma or centroid
positions. Only consecutive recorded frames with valid coordinates are joined.
The phase-group panels remain unchanged.

These controls work in main-analysis plot options, the individual builder, and
`refresh_spatial.py <run> --plots spatial-rhythm-maps --spatial-track-cells all`.
The cached redraw reads copied positions and existing test verdicts; no period
search is repeated. `der_tracks.csv` records the exact displayed segments.
The generic `panels.spatial.track_map` accepts preselected segments and their
scalar colours without knowing anything about rhythmicity.

`spatial_metric` accepts a numeric cell-frame column or `radial_occupancy`.
The latter is the mean occupancy across equal relative-distance rings, requiring
every ring to have sufficient geometric support (`annulus_support`, default 0.5).
It measures filling of a cell's current reach, **not absolute extension**. Use
`reach_p95` or `area_px` to ask about physical reach or footprint size.

Where a baseline is relevant, every Circadian Workbench detrending control is
inherited from the main rhythm analysis and can be overridden per figure:
method, window, polynomial degree, minimum valid fraction, kernel bandwidth,
frequency cutoffs and filter order. This includes polynomial degree six and
bicubic options. `display` is `raw`, `residual`, or `standardized` (residual divided
by its within-cell standard deviation). Missing observations are not interpolated.
Binary tissue occupancy is not detrended and does not advertise baseline controls.

Period/timing figures expose the estimator, separate significance test, search
range, `period_config` for additional Workbench settings, minimum observations,
minimum cycles, corrected significance threshold and multiple-testing correction.
The correction family contains all selected cell-metric traces in that figure.
Non-significant, untested, unavailable and exploratory fits remain distinct in
the saved table. A short or search-edge period is excluded from default timing
comparisons even if its trace passes the test.

For a selected phase group, the colour is `phase / individual_period * group_period`:
hours on the labelled group's cycle, **not universal clock hours**. The relative
period tolerance defaults to 0.1. Timing differences use a common recording hour
(default midpoint); the signed drift implied by differing fitted periods is saved.

The neighbour comparison uses a symmetric union of each cell's nearest neighbours.
All traces retain their time structure under permutation and are reassigned to
positions within fixed coverage-fraction quartiles. The comparison is one
within-recording association test, not evidence of contact, communication or an
across-animal treatment effect. Its exact null draws and histogram bins are saved.

Coverage is based on tracked labels, not proof of observed membrane movement.
Time since last occupancy bounds rather than exactly measures a vacancy duration.
The generic `coverage_gaps` calculation accepts an explicit observation mask;
the default figure uses the full analysed label field. Tracker losses can mimic gaps.

## Generic drawing and outputs

`panels.spatial.field_map`, `cell_map`, `spatial_matrix` and `connection_map` accept
already-computed numbers. They do not fit rhythms. Their data-only `render` function
composes the six layouts. Numeric derivations live in `analysis.spatial`, while
registry/source handling lives in `analysis/figures/_spatial.py`.

Each figure has its own flat, registered ReproFig bundle under the run's `figures`
folder: SVG master, PNG preview, exact `figure_data.csv`, copied sources and hashes,
derived tables, `statistics.csv` where applicable, and an exact `plot.py` producer.
The producer and copied renderer reproduce the complete display, not a substitute
summary plot. Preview copies also appear together in `figures/_all-images`.

Individual map width defaults to six inches and cannot be set below 5.5 inches.
The canvas grows to accommodate extra phase groups; maps are not shrunk to fit.

For layout-only review, `refresh_spatial.py <run> --plots <slug> ...` redraws these
bundles from saved encodings with the current renderer. It does not repeat any
period searches or permutation tests. Rerun a numbered builder when changing
metrics, detrending, periods, eligibility or other analysis settings.

For time-point plots the same refresh command accepts `--snapshot-count 4`,
`--snapshot-hours 1,13,25,49`, or `--show-history true`. Only select applicable
time-point slugs when using these options. It reuses cached display values,
positions and label movies; selecting different times does not redo detrending.
`figure_data.csv`, the literal producer settings and `der_snapshots.csv` always
record the newly displayed selection, while full measurements remain cached.

## Canonical tissue tectonics

`tissue-tectonics` fixes six panels in this order: first coverage time, total
cumulative occupancy, unique cells, median or mean speed, area covered by cells
with a significant corrected-intensity period, and split-event areas. The first panel hues
every eventually occupied pixel by elapsed time of first coverage. Cumulative
occupancy integrates all visits by all cells, assigning half each observed
interval to each endpoint and never extrapolating beyond the recording.

The intensity-period panel analyses each cell's `corrected_mean` intensity and
runs Circadian Workbench with Lomb-Scargle as the explicit default estimator and
significance test. It keeps the full shared period,
detrending, significance, correction and data-sufficiency options. Shared-pixel
period colours summarize contributing cell estimates rather than detecting a
pixel rhythm.

Significance is uncorrected by default. `--multiple-testing bh`,
`--multiple-testing bonferroni`, or `--multiple-testing sidak` applies a
correction; `--multiple-testing none` selects raw probabilities.

The final panel maps `contact_separate` events from
`history_merge_split_events.csv`. One event area combines the named cells' last
connected and first separated footprints; it is a tracker event, not evidence of
biological division.

`tissue_review.py --run <run>` writes a flat registered
`tissue-tectonics-current` bundle. Its captured producer reconstructs the six
panels from copied sources and verifies every exported pixel value.
