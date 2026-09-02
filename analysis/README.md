# `analysis` — what the cells did

The tracking pipeline in `code/` answers *which pixels are which cell*. This
package answers *what those cells did*. It reads an accepted set of outlines and
the registered signal they were built from, and writes tables, then figures.

It never recomputes tracking, never writes into a pipeline run folder, and never
needs the tracker to have been run at all — outlines from anywhere will do, as
long as they are a labelled `(frames, y, x)` stack.

## Run it

```powershell
Copy-Item analysis_config.example.json analysis_config.json   # then fill it in

python -m analysis modules                                    # what can be measured
python -m analysis theme --list                               # how figures will look
python -m analysis conditions --config analysis_config.json   # which movie is in which group
python -m analysis doctor --config analysis_config.json       # are the inputs readable and pinned
python -m analysis run    --config analysis_config.json --out outputs/a02_my_run
python analysis\figures\build_all.py outputs\a02_my_run       # figure bundles
```

`doctor` is the one to run first on a new dataset. It reports every input's
fingerprint, whether it matches the pin, and — importantly — whether a spatial
scale could be resolved or whether the run will be in pixels.

## What comes out

```
outputs/<run>/
  manifest.json              inputs + hashes + parameters + versions + timings
  conditions.json            which movie was which group, and how that was decided
  theme.json                 the look, and the condition colours
  figures.json               your wording per figure, if you set any
  <stem>/
    movie_summary.json       the numbers you would quote in a sentence
    tables/                  measured here
      cell_frame.csv         one row per cell per frame — the raw material
      cell_summary.csv       one row per cell
      frame_summary.csv      one row per frame (the population over time)
      cell_summary_windowed.csv    one row per cell per declared window
      frame_summary_windowed.csv   one row per frame per declared window
      window_change.csv      one row per cell per window per metric — against its baseline
      rhythms.csv            cosinor + Lomb-Scargle per cell per measurement
      rhythms_null.csv       the same tests on matched noise — the false-positive floor
      rhythms_population.csv do the cells agree on a phase (Rayleigh)
      recurrence.csv         one row per cell per time lag — how often it returns to a state
      recurrence_quantification.csv  one row per cell — determinism and laminarity
      sequence_distance.csv  one row per pair of cells — how alike, allowing one to run late
      channels.csv           one row per cell per frame per extra channel
      channel_frame.csv      the whole field, per frame per channel
      channel_tracks.csv     one row per cell per channel — level, trend, wobble
      objects.csv            one row per reference shape per frame per set
      cell_objects.csv       one row per cell per frame per set — how far, how much overlap
      cell_object_tracks.csv one row per cell per set — hours in contact, closest approach
      ...                    one file per module table that is not a fold
    tracker/                 copied out of the tracking run, unaltered
      history_*.csv          the tracker's own decision tables
    stacks/                  per-pixel images this package measured
  pooled/                    every movie's tables stacked, so a comparison is a filter
    manifest.json            who contributed what, and which columns one movie lacked
    tables/<name>.csv        the same files, all movies, stamped with stem/condition/subject
    tracker/<name>.csv       the copied tables, pooled but kept apart
  figures/<slug>/            plot-that bundles: SVG + plotted CSV + sources + README
```

Two folders, one distinction: `tables/` is what this package measured and is
answerable for, `tracker/` is somebody else's CSV reproduced without comment. A
module says which by declaring `origin` (see **Adding a module**); nothing in
the writer knows any module's name.

Anything a side table supplied is **joined into the roll-ups**, never written
as a file of its own: the user's spreadsheet already exists on disk and is
pinned by hash, and a copy would be a second place for it to disagree with
itself.

### `pooled/`

Every row a run writes already carries the movie it came from, the condition it
belongs to and the subject it was taken from, so pooling is a concatenation
rather than a join. `pooled/` does it once so that nobody does it by hand in a
spreadsheet, and a run with one movie produces a `pooled/` folder of the same
shape as a run with six — nothing downstream should have a one-movie case.

`pooled/manifest.json` is the part that makes the pooled tables trustworthy. For
each table it records which movies contributed, how many rows each gave,
**which columns a movie did not have**, and which movies did not produce the
table at all. Those last two are the same failure wearing two hats: a movie that
declared no object set has no object columns, and a pooled table that fills them
with blanks looks exactly like a movie whose cells touched no objects. Read the
manifest before testing any column.

```
python -m analysis pool outputs/<run>
```

pools a run that already exists, so runs measured before this feature existed do
not need re-measuring. `run` does it automatically at the end. Pooling refuses to
overwrite an existing `pooled/` folder, in line with immutable runs; delete it
deliberately or pool into a new run. Identity numbers are per movie — cell 12 of
one movie is not cell 12 of another — so the key in a pooled table is always
`(stem, identity)`, never `identity` alone.

Three tables are **roll-ups** rather than any one module's output —
`cell_frame`, `cell_summary` and `frame_summary`, one row per cell-frame, per
cell and per frame respectively. A module whose table shares a roll-up's grain
can declare `fold=True` and its columns are merged in rather than written to a
file of their own, so a per-cell comparison needs no join and the same numbers
are never in two files at once. All three roll-ups accept a fold: `neighbours`
and `walk` both measure something about a whole frame, and those columns land in
`frame_summary`.

Every metric on `summarise.SUMMARY_METRICS` is rolled up **four ways** in
`cell_summary` - `_median` with `_iqr`, `_mean` with `_sd` - and **two ways** in
`frame_summary`, `_median` and `_mean`. Which statistic to use is a judgement,
and a table offering only one makes that judgement for the reader without saying
so. It is not cosmetic: across those metrics, mean and median rank the cells
differently in 48 of 50, and in 11 they disagree by more than a quarter of the
spread between cells.

Numbers are written to **nine significant figures**. A mean of 16-bit pixel
values carried to seventeen digits records the arithmetic rather than the
measurement, and the extra digits were a quarter of a run's size. The exponent
is kept, so a p-value of `1.45517642e-15` survives intact.

A run folder is written once and refuses to overwrite itself. Change something,
choose a new run name; the manifest is what tells two runs apart. Each table's
entry in the manifest carries its grain, its origin and its folder, so a run
folder describes its own shape without the package that made it.

## Architecture

Two kinds of module, both self-registering. Adding a readout is one file plus
one import line in `modules/__init__.py`; the command line, the manifest and the
summaries pick it up with no other change.

```
   labels.tif ────┐
   raw.tif ───────┤
   evidence.tif ──┤
   unclaimed.tif ─┤
   valid_mask ────┼──> MeasurementContext ──> measurement modules ──> per-module tables
   any channel ───┤     (aligned in time and    │
   any shapes ────┤      cropped to the field)  ▼
   any table ─────┘                       joined cell_frame ──> derived modules ──> more tables
                                                 ▲                  (rhythms, …)
             side tables joined on ──────────────┘                        │
             frame_index or identity                                      ▼
                                    cell_summary · frame_summary · movie_summary
                                                 │
                                                 ├──> windowed summaries (per declared window)
                                                 ├──> pooled/  (once, across every movie)
                                                 ▼
                                        figure bundles (plot-that)
```

**Measurement modules** read pixels and return one row per cell per frame.

| Module | Needs | Measures |
|---|---|---|
| `morphology` | labels | area, perimeter, circularity, solidity, skeleton length, and every branch of the stick figure |
| `intensity` | labels, raw | signal in the outline, local background ring, dF/F, punctateness |
| `channels` | labels, channels | the same measurements again, in every extra imaging channel the configuration declares |
| `motility` | labels | centroid and soma steps, speed, path straightness, MSD exponent, territory |
| `surveillance` | labels | pixels held, gained and lost between frames — processes extending and retracting |
| `motion_evidence` | labels, evidence | the tracker's own evidence channels attributed to each cell, counted and totalled, plus how long ago the ground under each cell was disturbed |
| `presence` | labels | which frames each name is on screen for, and foreground carrying no name |
| `provenance` | labels, inferred | how much of each outline the tracker reconstructed rather than saw |
| `territory` | labels | per-pixel occupancy history: eventual territory, its core, its fringe and its shape |
| `neighbours` | labels | distance to the nearest cell, local density, Voronoi-style domains, field tiling |
| `sholl` | labels | radial occupancy and skeleton crossings, ring by ring, under two ring widths |
| `contacts` | labels | which pairs of cells share a boundary, frame by frame |
| `objects` | labels, objects | reference shapes — vessels, plaques, a wound edge — described as shapes |
| `object_geometry` | labels, objects | how far each cell is from those shapes, how much it overlaps, and for how long |

**Derived modules** read the joined table, not the pixels, so they re-run in
seconds.

| Module | Produces |
|---|---|
| `rhythms` | cosinor and Lomb-Scargle per cell, a matched-noise false-positive floor, a population phase test, and the non-parametric block — L5, M10, relative amplitude, onset, offset and the two stability measures |
| `coupling` | lag profiles between two measurements of one cell, and phase agreement between cells |
| `regimes` | a behavioural state per cell per frame, its dwell times and its transitions |
| `recurrence` | whether each cell returns to states it has been in before, and whether it holds them |
| `sequence_distance` | how alike two cells' state sequences are once one is allowed to run late |
| `territory_shape` | the eventual territory against the cell that made it, per frame and per cell |
| `walk` | step direction, turning angles, directional persistence and run lengths |
| `history` | the tracker's own record of why a name went missing, copied through rather than paraphrased |

`python -m analysis modules` prints this list from the registry, with every
column each one writes.

**Derived modules run in the order `enabled_modules` lists them**, and each is
handed a `cell_frame` carrying what the ones before it folded in. That is how
`sequence_distance` reads the state number `regimes` writes. Put a module before
the one whose column it needs and the run skips it, naming the column, rather
than failing — move it down the list.

### Extra imaging channels

The package was built around one signal stack because the outlines were drawn
from one. A microscope rarely records only one. Declare the others in the
movie's `channels` block and the `channels` module measures each of them
through the same outlines, in the same frames, with the same statistics
`intensity` uses on the stack the outlines came from.

Nothing in the package knows what a channel is, and that is the design. A
module that knew would need editing for every new experiment, and the first
dataset it had not been edited for would be measured wrongly or skipped. What
a channel stains is yours to write in `description`, and it travels into the
manifest untouched.

**The tables are long, not wide.** `channel` is a column, not a suffix on
twenty column names, so declaring a second channel adds rows and never adds
columns. The alternative reads better for one dataset and makes the column list
a function of the configuration — and then every figure, test and roll-up that
names a column breaks the first time somebody images a third dye.

That has one consequence worth knowing before you go looking: **`cell_summary`
carries no channel columns.** The per-cell channel numbers are in
`channel_tracks.csv`, one row per cell per channel, because a roll-up is keyed
on a cell and these are keyed on a cell *and* a channel.

Everything except `name` and `path` is a way of saying *this file is not
already in the same space as the labels*, and they are separate settings
because they are separate mistakes:

| Setting | Gets it wrong and you measure |
|---|---|
| `channel_index` | the wrong dye |
| `frame_offset` | the right dye at the wrong time |
| `crop_origin` | the right dye somewhere else in the dish |
| `shifts` | the right dye a few pixels from where the cell was |

The whole alignment is one line, and it is worth reading twice:

```
analysed[i, y, x] == source[frame_offset + i,
                            crop_origin[0] + y - shift_y[i],
                            crop_origin[1] + x - shift_x[i]]
```

`shifts` is a CSV with one row per source frame, read from the two columns
named in `shift_columns` (the y one first) and multiplied by `shift_scale`
before rounding to whole pixels. A registration log that records the drift it
*measured* rather than the correction it *applied* is the common case, and
`shift_scale: -1` turns one into the other.

A pixel the alignment cannot reach is left blank rather than wrapped round from
the far edge of the frame, so a cell that drifted half off the source is
visibly short of pixels instead of quietly measured against the other side of
the field. `channel_px` and `channel_missing_px` say how much of each cell was
actually there.

### Side tables

Everything a microscope does not record ends up in a spreadsheet: which frames
the stage jolted on, what the focus score was, which cell is which genotype,
when a drug went in. Declare it in the movie's `side_tables` block and its
columns are attached to the tables that share its key.

The file is **joined, never copied**. It stays where you put it, it is
fingerprinted like every other input, and `manifest.json` records what was read
and what matched under `side:<name>`. A copy in the run folder would be a
second place for it to disagree with itself.

There are two keys and no others:

| `keyed_on` | one row per | reaches |
|---|---|---|
| `frame_index` | timepoint | `frame_summary.csv`, and every row of `cell_frame.csv` |
| `identity` | cell | `cell_summary.csv`, and every row of that cell in `cell_frame.csv` |

A per-cell fact repeating down a cell's whole track is convenient and slightly
dishonest about being one fact. It is the default because it is what people
reach for; `cell_summary` is where it belongs.

**`key_space` is the setting to get right.** Label frame 0 is source ImageJ
frame 3 on the pinned movie, because the first two source frames were dropped
when the outlines were built. A log written in the microscope's own numbering
is 1-based and counts those frames, so it is converted:

```
frame_index = key - 1 - source_frame_offset
```

The default is `"label"`, meaning the key is already an index into the analysed
movie. Getting this wrong shifts every row by two frames and produces a
complete, plausible, wrong table. It means nothing for a per-cell table and is
refused there rather than ignored.

Three more things the loader does, each because the alternative fails quietly:

- **A repeated key is refused.** A merge against a repeated key multiplies rows
  instead of failing, so a hundred-row table silently becomes four hundred.
- **Every column is prefixed with the table's own name**, and a prefixed name
  that collides with something this package writes is refused. A user column
  called `area_px` landing on top of the measured one is not hypothetical.
- **Coverage is reported, not enforced.** `keys_matched`, `keys_unmatched` and
  `movie_rows_covered` go into the manifest. A log covering 90 of 99 frames is
  ordinary; a log covering none joins perfectly and produces a column of
  blanks, and these counts are the only thing that tells the two apart.

Rows the table does not cover keep their measurements and get a blank. The join
is a left join, always: an inner one would delete a measured cell because
somebody's notes were incomplete.

Nothing in the package derives anything from a side table on its own — no trend
fitting, no normalising, no "if this column looks like a flag then". The columns
are your words; what to do with them is yours too.

#### Computing with one

A side column can be one half of a calculation, but **only where you name it for
the module that does the calculating**:

```json
"modules": {
  "coupling": {
    "metric_pairs": [["schedule_dose", "turnover_index"]],
    "side_columns": ["schedule_dose"]
  }
}
```

`side_columns` is an allow-list, and it is enforced by a filter rather than by a
convention: each derived module is handed a table with every side column removed
*except* the ones named for it, so nothing can pick one up by scanning. The join
used to run last for the same reason; the protection is now explicit rather than
positional, which is what lets a dosing schedule be a lag partner at all.

Three things stop the run rather than being ignored, because each of them would
otherwise finish and quietly leave out the result you asked for:

| what | why it stops |
|---|---|
| a `side_columns` entry no side table supplied | a typo, and the run would otherwise complete with nothing computed |
| a `metric_pairs` entry naming a side column the module was not granted | the message names the column, the module and how to grant it |
| a granted side column that is not numeric | a genotype call is a string, and a lag profile is a correlation |

A *measured* column that is absent is still skipped rather than refused — that
means a module was switched off, and a run with fewer modules should produce
fewer results rather than stopping.

`manifest.json` records, per derived module, `side_column_use` — which columns it
was offered and which it actually reached for. **A column offered and unused is a
misconfiguration**, not a curiosity: the calculation you wanted did not happen.
The prefix travels all the way into the output, so `schedule_dose` on an axis
still says the number came from a spreadsheet.

Attaching data stays free. A run that declares no side tables, or declares one
and names it for no module, is bit-identical to a run made before any of this
existed — including the column order of `cell_frame.csv`, where the user's
columns still come last.

### Reference objects

`neighbours` and `contacts` measure a cell against another cell, and until now
that was the whole of what a cell could be related to. Distance to a blood
vessel, hours spent on a plaque, cells sitting at a wound edge — none of it had
a column to be written in.

Declare any number of labelled image stacks in the movie's `objects` block and
two modules pick them up. **`objects`** describes the shapes themselves;
**`object_geometry`** measures every cell against them. They are separate on
purpose: one asks *what are these shapes*, the other asks *how do the cells sit
against them*, and a user who wants only the first should not pay for the
second.

The alignment settings are the same words as `channels`, so lining up a second
input is the thing you already know how to do. One is new:

```json
"objects": [
  { "name": "vessels", "path": "…_vessels.tif", "static": true }
]
```

`static: true` is one map applied to every frame — a vessel tree drawn once. It
is **held fixed** rather than drifting with the registration, because such a map
is drawn in the analysed field's own space, and `frame_offset` and `shifts` are
refused alongside it rather than ignored.

Three tables, long rather than wide for the same reason the channel tables are —
`object_set` is a column, so a second set adds rows and never adds columns:

| Table | One row per |
|---|---|
| `objects.csv` | set, object, frame — area, perimeter, solidity, where it is |
| `cell_objects.csv` | cell, frame, set — how far, how much overlap, touching or not |
| `cell_object_tracks.csv` | cell, set — hours in contact, closest approach, longest bout |

Because those grains all carry `object_set`, none of them folds: **`cell_summary`
has no object columns**, and the per-cell numbers are in
`cell_object_tracks.csv`.

Four things to know before reading a number out of these:

- **An unreachable pixel reads as empty ground.** A label image has no spare
  value for "could not look", so the count is in `manifest.json` under
  `objects:<name>` as `unreachable_px` instead of in the array.
- **Distance 0 and overlap are different claims.** A cell brushing an object and
  a cell half buried in one both read 0 outline-to-outline;
  `object_overlap_share` is what separates them. Read them together.
- **This package does not track objects.** An object's number is whatever the
  input file says it is, frame by frame. `object_frames_present` is the check:
  if every object shows 1, the file renumbers itself and nothing that depends on
  an object keeping its number — `object_longest_contact_frames` above all —
  means anything.
- **Touching is a tolerance, not an observation.** `dilation_px` is on every row
  of `cell_objects.csv`, the same setting name and default `contacts` uses, so
  two tables built at different tolerances cannot be mistaken for one.

### Where measuring is trustworthy

Several numbers divide by the size of the imaged field, and the field is not all
measurable. Registration leaves a margin that is blank in some frames and not in
others; vignetting darkens the corners; a partial field may be imaged at all. On
the pinned movie the registration log records a valid margin of 6, 2, 18 and 2
pixels on the four sides, so about 6% of the 367 × 425 field is not present in
every frame — and every density and spacing figure divided by all of it.

Point the movie's `valid_mask` at a boolean or 0/1 TIFF and those denominators
use it instead:

```json
"valid_mask": "path/to/YOUR_STEM_valid.tif"
```

One path, because there is one answer to "can this pixel be measured". Either a
single `(y, x)` frame applied to every label frame, or a full `(frames, y, x)`
stack used frame by frame. Non-zero means valid.

**It is in label space**, like `unclaimed` and `provenance` and unlike `raw` and
`evidence` — it describes the analysed field, not the acquisition, so no
`source_frame_offset` is applied to it.

The four denominators it changes, which is the whole list:

| Column | was divided by | now divided by |
|---|---|---|
| `neighbour_ground_px`, and so `nearest_neighbour_expected_px` and `nearest_neighbour_index` | the whole frame | the measurable frame |
| `local_density`, `local_density_area_px` | occupiable ground within the radius | occupiable **and measurable** ground |
| `object_cover_within_radius` | field within the radius | measurable field within the radius |
| `object_field_share` | the whole frame | the measurable frame |

`territory` is deliberately not on that list: every share it writes divides by a
cell's own eventual footprint, not by a field, so a mask has nothing to correct
there.

**With no mask declared, every number is exactly what it was before this
existed.** That is not a hope — `valid_px` computes the undeclared case as the
product of the two side lengths rather than by summing an array of `True`, so
the division is the same operation on the same value it always was.

Two things a mask does not do:

- **It never drops a row.** Whether a cell that spends half its life in a
  vignetted corner should be excluded is a scientific decision, and this package
  does not make those. The mask changes what things are divided by; the cell
  keeps its rows and its measurements.
- **It never rescales an intensity.** A flat-field correction is a different job
  with different arithmetic. This says where not to trust, not how to fix.

`frame_summary.csv` always carries `valid_px` and `valid_share`, mask or no mask,
so a reader never has to work out which denominator was used.

### Adding a module

```python
# analysis/modules/my_readout.py
from analysis.registry import Column, MeasurementContext, Output, register

DEFAULTS = {"some_threshold": 3.0}

PRODUCES = (
    Column("branchiness", "Branchiness", "count", "morphology"),
    Column("branch_span_px", "Branch span", "px", "morphology"),
)

WRITES = (
    Output("my_readout", grain=("identity", "frame_index"), fold=True),
)

@register(
    name="my_readout",
    description="one line that appears in `python -m analysis modules`",
    requires=("labels", "raw"),      # skipped, not failed, if a movie lacks these
    defaults=DEFAULTS,
    produces=PRODUCES,               # every column this returns, except the index
    writes=WRITES,                   # every table this returns, and what a row is
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("my_readout")}
    ...
    return {"my_readout": table}     # columns identity, frame_index, <your columns>
```

Then add it to the import line in `modules/__init__.py`.

**`writes=` is what makes a new table land in the right shape first time.**
`produces=` says what each number means; `writes=` says what each *table* is,
and between them a module describes its output completely. Nothing in `run.py`
knows any module's name.

Each `Output` answers four questions:

| field | question | how to answer it |
|---|---|---|
| `grain` | one row per what? | the columns whose combination is unique in the table. Check it, do not assume it — `analysis/test_table_declarations.py` will |
| `fold` | are these the same rows a roll-up already has? | `True` merges the columns into the roll-up at that grain instead of writing a file. `False` writes a file |
| `origin` | did this package measure it, or copy it? | `"measured"` (the default) puts it in `tables/`; `"tracker"` puts it in `tracker/` and is the only case allowed to leave `grain` empty |
| `optional` | is it written for every movie? | `True` if the inputs it needs are not part of every chain, so an absent file is a skip rather than a failure |

`fold` is separate from `grain` on purpose. `presence` has exactly
`cell_frame`'s grain and must **not** fold into it: it carries a row for every
frame a cell was *absent* as well, so folding it would either lose the absences
or fill `cell_frame` with mostly blank rows. Grain says what a row is; `fold`
says whether these are the same rows.

The three roll-ups declare their own grain in `analysis/summarise.py:ROLLUPS`.
A fold at a grain no roll-up shares is refused, because it would be a
declaration that quietly meant nothing.

**`produces=` is what makes a new measurement draw correctly the first time.**
It is the wording every figure will use for that column: the words on the axis,
the unit, and the colour family, declared next to the code that computes the
number. Leave a column out and it still draws, with a label invented from its
name and the default colour - `branch_span_px` as "Branch span (px)" in the
morphology colour, on every figure that touches it, with nothing raised
anywhere. `analysis/test_column_declarations.py` runs each module on a synthetic
movie and fails when something comes back undeclared, which is the only thing
that notices.

Declare every column the module returns across all its tables, not just the
cell-frame one, and skip only `registry.SHARED_COLUMNS` - `identity`,
`frame_index`, `hours` and the rest of the index, which say which row this is
rather than what was measured. Two modules may declare the same column (both
`morphology` and `motility` write a centroid) as long as they say the same
thing about it; disagreeing is refused, because the join keeps one copy and the
label would otherwise depend on which module sorted first.

`python -m analysis modules` prints each module's columns with their units.

## Windows

Every summary above covers the whole recording. If a drug goes in at hour 20,
the before and the after are averaged into one number and the change partly
cancels itself out. A window is a named stretch of the recording, and a windowed
summary is the same roll-up run over that stretch — nothing is remeasured.

```json
"windows": [
  {"name": "baseline",  "from_hours": 0.0,  "to_hours": 20.0},
  {"name": "treatment", "from_hours": 20.0, "to_hours": 49.0, "baseline": "baseline"}
]
```

Declare in **hours or frames, never both**. Hours are hours since the recording
began — this package holds no recording start time, so a window is a position in
the movie and not a time of day. Windows are **half-open**, `from <= h < to`, so
the two above do not both claim hour 20; if they did, every paired difference
would be computed against a baseline holding the first frame of the response.
Windows may overlap and need not cover everything. A movie may carry its own
`windows` block, which *replaces* the shared one for that movie rather than
adding to it.

Three files come out, and a new window adds **rows, never columns**:

| file | one row per | what it holds |
|---|---|---|
| `cell_summary_windowed.csv` | cell, window | the per-cell roll-up, plus `window_frames`, `window_hours`, `window_coverage` |
| `frame_summary_windowed.csv` | frame, window | the per-frame roll-up over that stretch |
| `window_change.csv` | cell, window, metric, statistic | `value`, `baseline_value`, `change`, `ratio` |

`window_change.csv` is the number the whole feature exists for, and it is long
rather than wide on purpose: as columns it would have been four hundred of them.
Both a difference and a ratio, because a change of 40 camera units means nothing
without knowing whether the baseline was 50 or 5000, and a ratio of 1.8 means
nothing without knowing whether the difference is 4 units or 4000. A ratio
against a baseline of zero is blank rather than infinite, and a cell seen in only
one of the two windows gets no row at all — a blank in a table of differences
reads as "no change".

Two things do not carry over from the whole-recording summary. `coverage` still
means "of the whole recording", because it is the same formula the whole-
recording roll-up uses and one name with two meanings across two files is worse
than one extra column — divide `observed_frames` by `window_frames`, or read
`window_coverage`, for coverage of the window. And a module that folded a
per-cell table into `cell_summary` — a lifespan, a diffusion exponent — measured
the whole recording, so its columns are **absent** from the windowed summary
rather than repeated in every window's row.

A cell present for two frames of a twenty-frame window still gets a summary.
That is correct and it is a trap, which is why `window_frames`,
`window_coverage` and `observed_frames` travel in the same row; check them
before quoting anything.

```
python -m analysis window outputs/<run> --config analysis_config.json
```

adds windowed summaries to a run that already exists, so windows can be declared
after the measuring is done — which is the usual order, because you find out
that something changed at hour twenty by looking at the whole-recording result
first. `run` does it automatically when a `windows` block is present. Declaring
no windows writes no windowed files and changes nothing.

`python -m analysis doctor` lists every declared window with its range and how
many frames it catches, and counts a window that catches **no** frames as a
problem — a mistyped window is caught before a run rather than after.

## Conditions

`analysis/conditions.py`. A **condition** is an experimental group — a
treatment, a genotype, a light schedule. It is the one thing this package
cannot measure, and the one thing that, if it is wrong, makes every number
downstream wrong in a way no plot will reveal.

The short form is a name and a regular expression, searched in each movie's
stem:

```json
"conditions": {"treated": "_A\\d", "control": "_B\\d"}
```

Check it before running anything:

```powershell
python -m analysis conditions --config analysis_config.json --try 95_C4
```

```
factor: condition
  condition     label           colour    ref  patterns
  treated       treated         #0072B2        /_A\d/
  control       control         #D55E00   yes  /_B\d/

assignment
  95_A3                    -> treated              derived  /_A\d/
  95_B1                    -> control              derived  /_B\d/
  95_C4                    -> NOT RESOLVED         unassigned
                              nothing matched '95_C4' ...
```

**Three rules, and all three exist because the alternative is a mislabelled
experiment.** Two patterns matching one stem is an *error*, not a first-hit-wins
race — there is no ordering to remember, and the fix is a better pattern. A stem
no pattern matches stays unassigned, and `run` refuses to start. And a movie can
always override the derivation by naming its own `"condition"`, so one awkward
file is fixed by naming it rather than by contorting a regex to fit it.

Matching is case-insensitive, and `match_against` can search the `labels` or
`raw` file name instead of the stem.

**The long form** adds a display label, a colour, a reference group, and a
second factor. Give conditions different `factor` values and each is resolved
independently, then combined — a stem matching `_A2` and `18m` becomes
`HCQ_old`:

```json
"conditions": [
  {"name": "HCQ",     "label": "+HCQ",   "factor": "treatment", "match": "_A\\d", "colour": "red"},
  {"name": "vehicle", "label": "Vehicle","factor": "treatment", "match": "_B\\d", "control": true},
  {"name": "young",   "factor": "age",   "match": "^3m"},
  {"name": "old",     "factor": "age",   "match": "^18m"}
]
```

**Condition colours are project vocabulary, not house style.** Which colour a
genotype should be is a decision about the science, so a project names its own —
and the override reaches `theme.condition_colour(name)` only, never
`theme.colour(role)`. Declaring a condition called `ink` recolours that
condition and not the axes of every figure. Unnamed conditions get an Okabe-Ito
colour by declaration order, so adding a group later never recolours the ones
before it.

**Every table gains `stem`, `condition` and `subject` as its first three
columns**, so pooling ten movies is a concatenation rather than a join against
the configuration. `subject` — the animal, dish or slice — defaults to the stem,
because the opposite default silently turns pseudo-replication into
replication: 83 cells from one movie are one subject, not 83.

**Invented groups must say they are invented.** Testing a comparison before the
real groups exist means making some up, and a fabricated group that reads as a
measured one is the exact failure this module is built to prevent. So say so
once and it travels everywhere:

```json
"conditions": {
  "synthetic": true,
  "conditions": {"treated": "_A\\d", "control": "_B\\d"}
}
```

The flag is printed by `conditions` and `run`, written into `conditions.json`
and the manifest, and available to a figure through
`_bundle.design_for(run)["synthetic"]` so a comparison panel can label itself.
`analysis_config.demo.json` is a worked example: the one real movie replayed
under two invented labels, so the groups are identical by construction and any
difference a comparison reports is a bug in the comparison.

**A thin design gets a warning, not a refusal.** One movie per group, a single
group, or no control named are printed by `conditions` and `run` and recorded in
`conditions.json`, but they do not stop anything. Whether an experiment is worth
running is the user's call; whether a movie is in the right group is not.

## Contrasts

Every p-value elsewhere in this package asks whether *one cell* has a rhythm.
None of them asks whether treated differs from control. `statistics.csv` is
where that question is answered, one row per result, in a form a figure or a
paper can quote without recomputing anything.

```json
"contrasts": [
  {"name": "reporter_by_condition",
   "family": "primary",
   "table": "cell_summary",
   "metrics": ["corrected_mean_median", "area_px_median"],
   "group_by": "condition",
   "groups": ["control", "treated"],
   "unit": "subject",
   "aggregate": "median",
   "test": "mannwhitney",
   "correction": "benjamini_hochberg",
   "alpha": 0.05}
]
```

### Three rules, and they are the whole design

**Nothing is tested that was not named.** There is no "test every metric" mode
and there will not be one. The package declares 449 columns; a handful of
contrasts across all of them is thousands of tests, and a Benjamini–Hochberg
correction across all of those makes the twenty results anyone cared about
vanish. If a broad sweep is genuinely wanted, it goes in its own `family` with
its own correction and its own name — never as the default.

**`unit` has no default.** With `unit: "cell"`, 83 cells from one movie count as
83 independent samples and almost everything comes out significant. This is the
single most likely way for the package to produce a confidently wrong result, so
the setting is required and the run refuses without it.

| `unit` | one unit is | needs `aggregate` |
|---|---|---|
| `cell` | one cell, keyed `(stem, identity)` — cell 12 of one movie is not cell 12 of another | no |
| `movie` | one recording; its cells reduced to one value | yes |
| `subject` | one animal, dish or slice; two movies from one animal are **not** two samples | yes |

`subject` is the aggregation `CircadianWorkbench` cites as the Lazic 2010 /
Hughes 2017 unit-of-analysis guard, and it is why `subject` is stamped onto every
row this package writes. `aggregate` is `median` or `mean` and is also required
where it applies — which of the two is used changes the answer, so the package
does not choose it for you.

**A test that cannot run is a row, not a silence.** Too few units, a group that
is not there, a constant column, a metric one movie never measured — each is
written with the reason in `note` and every other column blank. No row is ever
blank on both counts.

### What is offered

| `test` | compares | effect written |
|---|---|---|
| `mannwhitney` | two groups, ranks | difference in medians |
| `kruskal` | three or more, ranks | epsilon-squared |
| `welch_t` | two groups, means | Hedges' g |
| `anova` | three or more, means | eta-squared |
| `wilcoxon` | one window against another, same cells | median paired difference |
| `paired_t` | the same, means | paired Hedges' g |

Which test runs is declared and **never chosen from the data**. Running a
normality test and picking accordingly is a garden of forking paths with a
friendly face.

Every result carries an effect size — a p-value alone is not a result — and
every two-group effect carries a 95% percentile interval from resampling the
units. An omnibus effect is a function of the statistic rather than of the
samples, so its interval is deliberately blank.

Corrections are `benjamini_hochberg` (the default), `bonferroni`, `sidak` and
`none`, and are applied **within a family and never across one**. Every contrast
in a family must agree about `correction` and `alpha`, or the run refuses —
whichever was read last would otherwise silently decide for both, and the
corrected p-value is the number anyone quotes.

Three floors are copied from `CircadianWorkbench` v0.5.0 with the source line
cited in `analysis/contrasts.py`, and `analysis/test_contrasts.py` runs Hedges' g
and both classical corrections through this module and through the workbench and
requires the same answer: fewer than **three units** in any group is a refusal;
zero within-group variance in *every* group is a refusal, because zero variance
is not infinite evidence; and a scale-relative noise floor stops an effect size
of 1e14 reaching a figure legend.

```
python -m analysis contrasts outputs/<run> --config analysis_config.json
```

tests a run that already exists — the question you want to ask is usually
decided after you have looked at the measurements. `run` does it automatically
when a `contrasts` block is present, after pooling. `python -m analysis doctor`
lists every declared contrast and, where the grouping is by condition, how many
units each group would have, so an impossible contrast is caught before a run
rather than after.

`statistics.csv` is `plot-that`'s `data/der/statistics.csv` without renaming a
column, and each row carries a `methods` sentence naming the test, the unit of
replication, the group sizes and the correction — so a reader holding only the
CSV never has to come back here to find out whether the p-value means anything.

## Figure text

`analysis/figures/_text.py`.

**The package never says what a figure means.** Every builder ships a plain
descriptive default - what is on the axes and how it was measured, nothing more
- and every word on the figure is a parameter you can replace. There are no
rules that inspect the data and pick a sentence.

Three places, most specific first:

```powershell
# 1. a flag, for a one-off
python analysis\figures\03_surveillance_not_translocation.py outputs\a02_my_run `
    --title "Microglia rebuild themselves without going anywhere"
```

```json
// 2. the figures block, which survives a rebuild
"figures": {
  "surveillance-not-translocation": {
    "title": "Microglia rebuild themselves without going anywhere",
    "subtitle": "95_A3, 80 cells with a measurable track.",
    "footnote": "...",
    "claim": "The median cell moves 0.72 px per 30 min while replacing 18% of its pixels."
  }
}
```

3. the builder's default, if you set neither.

The slots are `title`, `subtitle`, `footnote`, `note` (a second block some
layouts place beside the axes) and `claim`. A typo in any of them - as a flag or
as a configuration key - is refused rather than ignored, and each bundle records
which of the three sources every slot came from.

`claim` is the one-sentence statement for the bundle README and the figure
register. It has **no default**: unset, the README says the figure has not been
interpreted, because it has not been.

`run` copies the block into the run folder as `figures.json`, so rebuilding a
run's figures later reproduces the wording used at the time rather than whatever
the configuration says now.

### Options survive the same way

A figure's *options* — what it draws rather than what is written beside it —
resolve through the same three places in the same order, and live in an
`options` sub-block of the same entry:

```json
"figures": {
  "step-size-distribution": {
    "claim": "Half of all steps are under 0.31 px; the 95th percentile is 2.4 px.",
    "options": {
      "metrics": "soma_step_px_gapless",
      "bins": 60
    }
  }
}
```

This is the route that survives a rebuild. A figure tuned with six flags over an
afternoon and never written down cannot be redrawn from the run folder, because
the run folder does not know what was drawn.

Two differences from the wording, both because these are values rather than
sentences. They are typed JSON, so `60` is a number and `["a", "b"]` is a list;
a comma-separated string means the same as the list. And `null` means *use the
builder's default*, where a `null` text slot means the empty string and takes
that line off the figure — the difference matters for `--fit=`, where empty is a
real choice.

Which options a figure accepts is its own declaration, not this list:
`python -m analysis figures --slug <slug>`. A name it does not accept is refused
by `python -m analysis doctor`, before anything is measured, and named against
the ones it does — a misspelled option otherwise draws perfectly well with the
defaults, which is the failure that looks like success.

### Capitals

Titles and axis labels start with a capital, whether you wrote them or the
builder did. One character is raised and nothing else is touched, so a first
word carrying its own capitals - `pH`, `mCherry`, `CD68` - is left exactly as
typed, and a label that opens on a digit or a symbol (`24 h blocks`) is already
correct and is left alone too. Only the first line is affected: the second line
of a two-line label is a continuation, not a new label.

Body text is yours as written. Subtitles, footnotes, notes, legend entries and
tick labels are never re-cased.

### Builder options

Separate from wording: these change what a builder *draws*. The table below is
the shared **vocabulary** — what each flag is called and what it means — and
lives in one place, `analysis/figures/_options.py`, so a flag means the same
thing on every page that takes it.

Which of them a given figure accepts is declared on the figure itself, in its
`@figure` block. That is what makes `--bins 60` on a page with no histogram a
refusal rather than a silence, and what lets `python <builder> --help` print
that figure's own three or four options instead of all thirty-one. Run
`python -m analysis figures` for the whole set, or
`python -m analysis figures --slug <slug>` for one.

| flag | what it does |
| --- | --- |
| `--stem` | which movie in the run to draw, when the run holds more than one |
| `--identity` | which cell the report card draws |
| `--hour-ticks` | hours between ticks on a time axis |
| `--panels` | which panels of a multi-panel figure to draw, comma separated |
| `--metrics` | which measured columns the figure draws, comma separated |
| `--bins` | how many bands a histogram panel divides its range into |
| `--size` | which column sets point size on a scatter; empty for one size |
| `--cells` | identities to draw, or the count of longest-observed identities |
| `--quantile` | observed-data quantile used to define an event |
| `--window` | frames either side of an aligned event |
| `--max-lag` | largest lag retained on a lag or displacement panel |
| `--order` | row order for a raster or regime ribbon |
| `--normalise` | transition-matrix scaling: `row`, `column`, or `none` |
| `--shuffles` | shuffled-time replicates used for a displayed null |
| `--regime` | numbered regime used for a within-cell contrast |
| `--contour` | persistent-core contour as a share of observed frames |
| `--stages` | number of evenly spaced recording stages |
| `--thresholds` | primary and alternate core/transient cut-offs |
| `--rings` | number of Sholl rings retained for display |
| `--scaling` | which ring width to read: `cell` (each cell's own reach) or `global` (one width across the movie) |
| `--clusters` | number of sequence groups cut from a clustering tree |
| `--dilation` | one or two measured contact radii |
| `--min-hours` | minimum contact duration retained |
| `--min-contested` | minimum identities required to show a contested pixel |
| `--inferred-threshold` | share of an outline that must be reconstructed before a frame is drawn as reconstructed; `0` means any at all |
| `--images` | how many image tiles across the top; `0` for none |
| `--image-hours` | exact hours for the tiles, comma separated; beats `--images` |
| `--cell-lut` | the LUT the cell images are displayed through |
| `--trace-luts` | colour or LUT per trace, matched to `--metrics` in order |
| `--outline` | colour of the outline drawn over each tile |
| `--fit` | which traces carry the cosinor curve: a list, `all`, or empty for none |
| `--header-x` | left edge of the title, subtitle and footnote - see **Placement** |
| `--title-y` | title, measured down from the top edge |
| `--subtitle-y` | subtitle, measured down from the top edge |
| `--footnote-y` | footnote, measured up from the bottom edge |
| `--draft` | write only the figure, flat, into `outputs/figures/` - see **Drafts** |
| `--overlay` | draw the negative-space mask over the raw image |

`--metrics` means the same thing everywhere - which measured columns this page
draws - but what it does with them follows the figure: a panel each on the report
card, the two axes on a scatter, the rows on a ranked comparison, the channels on
the evidence budget. Each builder's docstring says which, and naming a column the
figure cannot use prints the ones it can.

`--panels` names the parts of a page a figure can leave out, so any panel can be
the whole figure. Asking for a panel a builder does not have prints the ones it
does.

A `--trace-luts` entry is either a flat colour - a theme role (`reporter`), a
palette name, or a hex value - or the name of a colour map (`viridis`), in which
case the series is coloured along the map. One option covers both, because "the
colour of this graph" and "the LUT for this graph" are one question.

### Placement

The four flags that move the words take a distance in either of two units, and
the unit is **required**:

```
--title-y 0.4in        four tenths of an inch below the top edge
--header-x 6%          six hundredths of the page width in from the left
--footnote-y 0.25in    a quarter inch up from the bottom edge
```

Both units exist because neither is right for everything. A title sits a fixed
number of **inches** below the top edge, so a stack of printed pages of
different heights carries its titles at one distance from the paper's edge; a
share would put the title further down the taller page. The left edge and the
footnote are **shares**, because they run with the plot rather than with the
paper - the footnote in particular has to stay clear of the bottom axis, which
is positioned as a share too. That is why `--title-y 0.2` is refused rather than
guessed at: 0.2 of a page and 0.2 inches are both plausible and the defaults are
not all in one unit.

Every figure accepts all four without declaring them, because they are about the
sheet rather than about what is drawn on it. Like any other option they resolve
flag → the run's `figures.json` → the builder's own value, so a project that
wants its titles lower everywhere sets it once:

```json
{"figures": {"identity-trajectories": {"options": {"title_y": "0.35in"}}}}
```

A page whose plot starts further in - because it carries a colour bar and a
category strip on the right - already lines its words up with its plot by
setting `header_x` on its `FigureResult`; the flag overrides that too.

One thing the units do not change: when a figure grows taller to clear a long
footnote, nothing above moves. The growth is paper added at the bottom, not a
resize anybody asked for, so the title stays where it was whichever unit placed
it there.

```powershell
python analysis\figures\05_cell_report_card.py outputs\a03_my_run `
  --identity 12 --metrics corrected_mean,area_px,solidity --images 5 `
  --cell-lut magma --trace-luts reporter,viridis,#4878A8 --fit all

python analysis\figures\01_cd68_reporter_rhythm.py outputs\a03_my_run --panels phase
python analysis\figures\03_surveillance_not_translocation.py outputs\a03_my_run `
  --metrics area_px_median,turnover_index_median --size coverage --bins 24
```

### Where the panels live

A builder draws nothing itself. Every panel is a function in
`analysis/figures/panels/`, taking an axes - or a figure and a rectangle, when it
has to make several axes of its own - then some numbers and the theme, and giving
back **the table it drew**:

```python
drawn = common.histogram(ax, values, theme, bins=40)
drawn.data      # bin_left, bin_right, count - exactly what is on the page
drawn.axes      # the axes it drew on, or the ones it made
drawn.extra     # what the caller cannot recompute: a colour bar's mappable,
                # the contrast a strip of tiles chose, a dendrogram's leaf order
```

One return shape for every panel, because the caller's first question is always
the same. Pass it through `ctx.drew("<panel key>", ...)` and a figure that sets no
`figure_data` gets its leading panel's table in the bundle - the table a reader
would check the figure against, rather than one the builder rebuilt from the
numbers it had just handed over.

Two kinds of function in these files are not panels, and say so by their return
type. A **key** - `colour_bar`, `inset_colour_bar`, `semantic_legend` - explains a
mapping something else already recorded. A **derivation** - `path_segments`,
`detrended_z`, `mechanism_colours`, `breakout_events` - takes no axes and draws
nothing; it lives beside the panel it feeds.


| file | panels |
| --- | --- |
| `panels/common.py` | the chart grammar - trace, cosinor curve, image tile and strip, raster, histogram, scatter with edge histograms, dumbbell, lollipop, reference lines, colour bar |
| `panels/rhythms.py` | detrended trace raster, peak-time histogram, noise-floor dumbbell |
| `panels/motility.py` | trajectory map, step histogram, and the gap-cutting both depend on |
| `panels/surveillance.py` | the tracker's motion-evidence channels and the colours they carry |
| `panels/presence.py` | names on screen, unclaimed foreground, persistence raster, and the three presence states |

A panel goes in a measurement module's file the moment it has to *know*
something about that measurement - which colours its states carry, what has to
be subtracted before it is honest, which points may be joined by a line.
Anything that would draw the same picture for any numbers stays in `common`.
The file names match `analysis/modules/`, so "which module wrote this number"
and "which file draws it" have one answer.

Wording and colour per column come from the module that writes the column, on
its `@register(produces=...)` — see [Adding a module](#adding-a-module).
`analysis/figures/_metrics.py` assembles those declarations into the vocabulary
every figure reads, and holds a `RESIDUAL` block for the modules that have not
declared theirs yet. A column in neither is still drawable —
`some_new_readout_px` becomes "Some new readout (px)" — and a per-cell summary
of a known column keeps that column's wording, so `turnover_index_median` is
still "Pixels replaced".

Nothing here is loaded until a label is actually asked for, because building the
vocabulary means importing every module, and a figure that labels no column
should not pay for scikit-image.

### Adding a figure

The same arrangement as adding a module: one file, decorated, and nothing else
in the package changes. `python -m analysis figures`, the `--help`, the option
checking and the audited bundle all come from the declaration.

```python
# analysis/figures/37_my_page.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import motility as motility_panels


@figure(
    number=37,
    slug="my-page",                       # bundle folder, config key, catalogue
    summary="one line for `python -m analysis figures`",
    title="What is on the axes",          # may hold {placeholders}
    grammar="histogram",                  # the ReproFig grammar of the drawing
    reads=(Table("cell_frame.csv", module="motility"),),
    panels=(Panel("distribution", motility_panels.step_histogram,
                  title="Step size"),),
    options=(Option("bins", default=45),),   # names from the vocabulary above
)
def build(ctx: FigureContext) -> FigureResult:
    frame = ctx.table("cell_frame.csv")            # copied and hashed into the bundle
    figure_, axes = ctx.layout(ctx.panels())       # only the panels asked for
    ctx.drew("distribution", motility_panels.step_histogram(
        axes["distribution"], frame["step_px_gapless"].dropna(), ctx.theme,
        bins=ctx.option("bins")))                  # its table becomes figure_data
    return FigureResult(figure=figure_, axes=list(axes.values()),
                        subtitle="what the reader is looking at")


if __name__ == "__main__":
    run_figure("my-page")
```

What the declaration buys, none of which is written twice: `--help` lists these
options and refuses any other; `--panels` names these panels and refuses any
other; a panel with `needs=("presence.csv",)` is skipped with a reason on a run
that has no presence table, and one with
`needs=("presence_frame.csv:unclaimed_px",)` on a run whose table has that
column missing; every table read through `ctx` is copied into `data/src/` with
its SHA256; the title, subtitle, footnote and README are placed by `_finish`
exactly as they are on the other thirty-six pages; and the panel passed through
`ctx.drew` becomes `data/der/figure_data.csv` unless the build sets one itself,
which several pages do because they combine two panels' tables.

An option name must be one of the shared vocabulary. If yours is genuinely new,
add it to `_options.OPTIONS` with a one-line meaning and to the table above —
`test_hour_ticks` fails if a declared flag is undocumented.

### Drafts

Deciding what a figure should look like takes many rounds and none of them is a
result. `--draft` writes the figure and nothing else, flat, into
`outputs/figures/<slug>.svg` and `.png`, over the top of the last round:

```powershell
python analysis\figures\build_all.py outputs\a03_my_run --draft
```

No bundle, no copied sources, no new run folder - the tables have not changed,
only the drawing has. Build **without** `--draft` once a figure is settled: what
a draft skips is exactly what makes a figure checkable later, so a draft is not
meant to leave the machine.

`build_all` passes on only what every figure takes — `--stem` and `--draft`. A
figure's own options belong to that figure: `--bins` means something on six of
the thirty-six and nothing on the rest, so giving it to all of them is refused
rather than silently ignored by thirty. Tune one figure by running its builder
alone, and keep the result by putting it in the `figures` block of the
configuration, where it survives into the run folder.

Ticks on an hours axis land on **day boundaries**, not on round decimals.
Matplotlib counts in tens, so a two-day recording gets ticks at 10, 20, 30, 40
and 50 h - none of which is a time of day, and a reader comparing a peak at 21 h
with one at 45 h has nothing to line them up against. The step defaults to 24 h
and must be a factor or a multiple of a day (1, 2, 3, 4, 6, 8, 12, 24, 48, 72),
so every tick sits at the same clock time whichever you pick. Anything else -
10 h, say - is refused before the figure draws.

```powershell
python analysis\figures\05_cell_report_card.py outputs\a02_my_run --hour-ticks 12
```

For every figure, set it in the theme block instead, where it travels with the
run in `theme.json`:

```json
"theme": { "hours_per_tick": 12 }
```

The one exception is the peak-time histogram in figure 1, which is 24 h wide and
so keeps a 6 h step of its own - still a factor of a day - until a step is
actually chosen, on the command line or in that figure's `options` block. A
chosen step applies to every time axis on the figure.

**What the builders do still compute is description, not interpretation.** The
cell count, frame interval, surrogate count, fit threshold and whether the
dataset is calibrated are read from the run manifest rather than typed into a
caption, because a typed one goes stale the first time somebody changes a
setting. If you replace a subtitle, you replace those too - it is your sentence
from then on.

## The aesthetic engine

`analysis/theme.py`. Every figure the package draws goes through one `Theme`;
no builder writes a colour, a type size or a line width of its own. Preview one
before committing to it:

```powershell
python -m analysis theme --list                               # every preset
python -m analysis theme --preset talk --preview swatch.svg   # see it
python -m analysis theme --check                              # has the house style drifted
```

**Colours are named by meaning, not by hue.** A builder asks for `reporter` or
`surveillance`; it never asks for red or teal. So the same readout is the same
colour in every figure, and recolouring a concept is one line of configuration
rather than a search through five scripts. `python -m analysis theme` prints all
20 roles and what each is for.

**Type and geometry come from the theme too.** `theme.size("note")`,
`theme.stroke("guide")`, `theme.canvas(14.5, 12)`, `theme.point_area(1.5)`.
The canvas call is the one that matters: a multi-panel figure is laid out in
fractional rectangles, so its proportions survive any canvas size, but the
ratio between canvas and type does not. Routing the nominal inches through the
theme keeps that ratio fixed, which is why `print` yields a small dense figure
rather than a huge one with unreadable labels.

| Preset | For | Ticks | Line | Canvas |
|---|---|---|---|---|
| `house` | **the default** | 25.0 pt | 2.88 | ×1.00 |
| `talk` | projected slides; opaque background | 31.0 pt | 3.36 | ×1.00 |
| `print` | a journal column, 1200 dpi | 8.4 pt | 1.32 | ×0.49 |
| `colourblind` | Okabe-Ito roles, PuOr diverging | as house | as house | ×1.00 |
| `mono` | greyscale, when colour costs money | as house | as house | ×1.00 |

Behind all of them sits the **lab contract** — the numbers in
`analysis_kit.style` that `PyFLASH.aesthetics` also reproduces. It is
deliberately *not* a preset: its name means nothing outside this lab, and
offering it beside `house` would only ask a user to choose between two things
they cannot tell apart from the name. Every preset is derived from it by scaling
rather than by writing a second set of numbers, and `--check` asserts it still
matches key for key. `house` runs a quarter larger than the contract because the
contract was drawn for one 7x5 panel, while this package's figures are
multi-panel pages read as a whole.

If a figure does have to sit beside published lab panels at contract size, scale
back to it rather than reaching for a preset:

```json
"theme": { "type_scale": 0.8, "stroke_scale": 0.833 }
```

**Two invariants the presets must hold**, both asserted by `test_theme.py`: the
five measurement families never share a colour, and a test outcome
(`rhythmic`, `significant`) never wears a family colour. The second is why
`rhythmic` is violet and not teal — teal already means "turnover", and a reader
who learns that in one figure would misread the next.

`mono` cannot hold either invariant: greyscale has nowhere near enough
distinguishable values for five families plus states, so it collapses
`reporter`/`surveillance` and `motility`/`evidence`. A test pins this so nobody
"fixes" it by inventing greys that are not actually separable in print. The real
fix, when a mono figure needs it, is a linestyle or a hatch in the builder.

Tune from `analysis_config.json`:

```json
"theme": {
  "preset": "talk",
  "roles": {"reporter": "#7a1fa2", "surveillance": "orange"},
  "palette": {"lab_purple": "#7a1fa2"},
  "type_scale": 1.1
}
```

A typo in a role or a setting raises rather than being ignored, because a wrong
theme is otherwise invisible until someone notices the wrong figure.

**Legends have a placement, not a guess.** `theme.legend(ax, ...)` honours
`legend_location`, and **`above` is the package default** — a single row just
above the axes. Also available: `above left`, `above right`, `below`, `right`,
and `inside`/`auto` for matplotlib's `best`.

`best` picks the emptiest corner *of the data*, so it moves when the data moves
and two runs of the same figure can disagree about where the legend is. `above`
depends on the axes instead: it never covers a point and never shifts. If the
axes carries a title, the title is pushed up by the legend's height so the two
cannot collide, and passing an explicit `loc=` opts that panel out entirely —
the theme's anchor is dropped rather than combined with it.

**Where the default numbers come from.** The lab contract, vendored here so the
package needs no extra install. `python -m analysis theme --check` (and
`analysis/test_theme.py`) asserts the vendored copy still agrees with the
installed `analysis_kit`, key for key, wherever that package is importable.

**The look is recorded.** `run` writes the resolved theme into the manifest and
drops the block into the run folder as `theme.json`, so rebuilding a run's
figures a year later reproduces them rather than restyling them.

## Two conventions worth keeping

**Units go through `Scale`, never through a bare multiplication.** An
uncalibrated dataset reports pixels; setting `microns_per_pixel` adds calibrated
columns beside the pixel ones rather than changing what the existing columns
mean. No figure can silently change units.

**A measurement that cannot be trusted is flagged, not dropped.** Rows keep
`is_fragment`, `n_components`, `shape_ratios_valid`, `touches_border`,
`background_is_zero` and `gap_frames`. A downstream filter is then an explicit,
visible choice rather than something buried in a measurement function.

## Known limits

- Nothing checks that a channel is the channel you meant. `description` is free
  text and `channel_index` is a number; point them at the wrong plane and the
  run produces a full set of confident numbers about the wrong dye. Every
  alignment setting is recorded in `manifest.json` under `channel:<name>` for
  exactly that reason, `unsampled_fraction` included.
- A channel is aligned by whole pixels. Sub-pixel drift is not corrected, so a
  channel is at best within half a pixel of the outlines — which matters here,
  where the median cell is about 11 px across.
- `channel_inside_over_outside` compares outline pixels with everything else,
  including foreground that carries no name. It is a contrast at the outlines,
  not a cell-against-empty-field ratio.
- A channel stack is kept as float32 when the source is 16-bit or narrower,
  which is exact, and float64 otherwise. Every statistic is computed in float64
  either way, so `channels` and `intensity` return the same number for the same
  pixels rather than nearly the same one.
- `evidence_*_sum` earns its place only on a tracker that writes genuinely
  graded change maps. On the pinned movie it does not: the total and the count
  correlate at r = 1.0000 for `held` (that channel is binary, so the total *is*
  the count), 0.998 for `trail`, and 0.992 for both `lost` and `gained`, whose
  values are about half saturated. Only `base` differs at all (0.883), and
  `base` is the next frame's raw signal, which `intensity` already measures.
  The columns are right and they are cheap; on this encoder they are close to a
  restatement, and a figure drawn from them here is a figure about pixel counts.
- `evidence_*_sum` is a share of full scale, and full scale for an integer
  channel is what its dtype holds. A tracker that writes into the bottom of a
  16-bit range reports small totals that are correct within a movie and not
  comparable across two movies encoded differently.
- The trail age assumes the encoding every trail channel this package has seen
  uses: brightest is one transition ago, and each step down is one transition
  further back. `trail_levels` is read off the stack when it is not configured,
  and the age columns are left blank rather than guessed when the channel is
  not an evenly spaced ramp.
- `trail_age_frames_*` counts only pixels that carry a trail at all. Ground the
  tracker has no memory of is absent from the mean rather than counted as
  infinitely old, so the mean is "how long ago, among the ground with an
  answer" and not "how long ago, everywhere".

- An object set is checked for shape, never for meaning. Nothing verifies that
  the vessels file holds vessels; `description`, the fingerprint and the
  recorded alignment in `manifest.json` are what make a wrong file findable.
- A side table is checked for shape, never for meaning. Nothing verifies that
  the focus score is a focus score or that the genotype call is right; the
  fingerprint and the three coverage counts in `manifest.json` are what make a
  wrong file findable afterwards.
- A side table joins on a timepoint or on a cell and on nothing else. A fact
  about a subject or a condition has no table at that grain to land in, so it
  stays in the movie block where `condition` and `subject` already are.

- Spatial scale is uncalibrated for the VID95 dataset — no µm/pixel exists in
  the acquisition TIFFs, so everything is pixels until a value is supplied.
- Hour 0 is the first frame of the recording, not a clock or circadian time.
  There is nowhere yet to record the acquisition start, so phases are relative.
- No group comparison yet. Conditions are declared, derived, coloured and
  carried into every table, but nothing tests across them.
- The VID95 dataset is one condition and one movie, so it has no conditions
  block at all — which is a supported state, not a missing one. The only
  two-group run that exists is the synthetic demo.
- Rhythm fits over ~2 cycles cannot resolve a period. Every row carries
  `cycles_covered` and `period_underdetermined` so that caveat travels with the
  number.
- Over half the pinned movie's outline pixels carry a reconstructed owner. The
  package reports that share and never filters on it; whether a frame is too
  reconstructed to fit a rhythm to is a scientific decision, not a default.
  `renamed_px` and `added_px` separate the common case (the microscope showed
  the pixel, the tracker decided whose it was) from the rare one (nothing was
  detected there at all), so the share can be read for what it is.
- A log-spaced histogram used to drop its largest observation about half the
  time. `np.logspace` goes through `log10` and back, so its last edge lands
  within about 1e-14 of the maximum rather than on it, and `np.histogram` drops
  anything above the last edge. Which way it fell was a coin toss that changed
  with the data. `panels.common.histogram` now pins both end points, which is
  what `np.linspace` already guarantees for even bands. Figure 09 counts 5,820
  cell-frames where it used to count 5,819.
- The Sholl ring width used to be set by the single largest reach anywhere in
  the movie. On the pinned dataset one two-piece object - a tracking artefact
  whose halves sat 120 px apart - set a 20.7 px ring against a typical cell
  reach of 8.4 px mean, 7.5 px median, so 96% of cell-frames occupied exactly
  one ring and 58.5% had no skeleton crossing at any radius. The profile was
  computable and meaningless. The width is now a percentile
  (`global_reach_percentile`, default 99) and the profile is measured under two
  widths, so no single cell can set the scale for everyone.
- Every Sholl summary exists twice, `_scale_cell` and `_scale_global`, because
  they are the same quantity on two different x axes. `_scale_cell` divides each
  cell's own reach, so ring 3 is "half way out" for every cell and shapes are
  comparable between a small cell and a large one; `_scale_global` uses one
  width across the movie, so a radius is an absolute distance and profiles can
  be pooled. Averaging the two, or reading one as the other, is meaningless.
  `reach_p95`, `centre_y` and `centre_x` come from raw pixel distances and carry
  no suffix because no ring width can change them.
- The Sholl decay slope is fitted through however many rings hold a crossing,
  which under `_scale_global` is a median of 2. `sholl_regression_points` and
  `sholl_regression_r2` are reported beside it: an r-squared of 1.0 on two
  points is a straight line through two points, not a good fit.
- `cell_summary` used to offer only a median and an inter-quartile range. On
  `extension_bias` the per-cell mean and median rank the cells at a Spearman
  agreement of 0.66 - which cell extends most had a different answer under each,
  and only one of the two was available to ask. Both are now written for every
  metric.
- Only four of the 28 Sholl summaries are rolled up per cell. The rest are cell
  size measured again: per cell, `sholl_auc` correlates with `area_px` at
  r = 0.97 and is 99% predicted by the shape columns already there. The four
  kept are the decay slope, outer-ring occupancy, the inner/outer ratio - the
  last two almost uncorrelated with size, |r| of 0.05 and 0.03 - and peak
  crossings, which is 92% predictable but is the number readers look for.
- Turning angles are only as real as the noise floor under them. A step is a
  difference of two centres and a turn is a difference of two steps, so a 0.3 px
  wobble becomes a 180 degree "reversal". `walk.min_step_px` defaults to 0.5 px,
  at which 2,792 of the pinned movie's 6,073 cell-frames carry a turn and 69 of
  83 cells have more than five. With no floor it is 5,623 turns and 78 cells; at
  1 px, 1,226 turns and 46 cells. 32% of soma steps are under half a pixel and
  60% under one, so the parameter, not the arithmetic, decides what the column
  means.
- The soma's direction does not persist, even above that floor. `mean_turn_cos`
  has a median of **-0.138**: consecutive steps lean towards reversing rather
  than continuing. The directional half-life is 0.44 frames and the median run
  before a large turn is one step. Anti-correlated steps are what a noisy
  position estimate produces *and* what a cell that surveys without
  translocating produces; nothing here separates the two, and the columns should
  be read as a description of the measured track, not as evidence of a walk.
- `turn_angle_dispersion` is a circular standard deviation, not the
  inter-quartile range the plan asked for. A cell that reverses on most steps
  has its turns packed against +pi and -pi, which are the same direction; the
  wrapped IQR calls that the widest spread possible when it is one of the
  narrowest. `sqrt(-2 ln R)` on the resultant length is 0 when the turns agree
  and grows as they scatter, whichever direction they agree on.
- `nearest_neighbour_index` divides by the **measurable field** and carries no
  edge correction. The null it tests against is "the same cells placed at random
  in the study area", so the study area cannot be defined from where the cells
  went: dividing by the ever-occupied mask instead is circular, and not
  harmlessly - it moves the pinned movie's index from 0.91 to 1.90, which is the
  difference between reporting random placement and reporting a regular tiling.
  The measurable field is the right third option because the *instrument*
  defines it. Where no `valid_mask` is declared it is the whole frame, which is
  the old behaviour and is optimistic by however much of the frame the
  registration left blank. Edge cells have unseen neighbours outside the image,
  which biases the index towards "regular"; no cell-frame in this movie touches
  the border, so the bias does not arise here and will in a dataset that does.
- `local_density` divides by the ground the cells ever occupy, intersected with
  the measurable field. That is not the imaged area - the two differ 4.4x on the
  pinned movie - so the number is not comparable with a published density that
  used the whole frame. `local_density_area_px` carries the denominator actually
  used on every row, and `neighbours.density_ground` switches it in one line. It
  is deliberately *not* the denominator the spacing index uses, for the reason
  above.
- A validity mask is the user's assertion, not a measurement. Nothing checks
  that the blanked region really is the registration margin, and nothing derives
  a mask from a registration log; `manifest.json` records the fingerprint and
  the valid-pixel range under `valid_mask`, and `frame_summary.valid_share` puts
  the number beside every frame.
- `domain_px` sums to the whole field, and a `valid_mask` deliberately does not
  change it: it is a partition of the imaged frame, and every pixel is closest
  to somebody whether or not it was worth measuring. That makes it the one
  spatial column on a different denominator from the rest, which is why
  `domain_occupancy` - the cell's own area over its domain - is the one to
  prefer. A frame with fewer visible cells hands everyone a bigger domain, and
  the occupancy survives that too.
- A territory built from a gapped track is smaller than the truth, and its
  *shape* columns carry the gap more quietly than its size does.
  `territory_components` has a median of 2, so a typical eventual footprint is
  in two or more disconnected pieces, and `territory_solidity` (median 0.64) is
  partly reporting those gaps rather than the cell's own raggedness.
  `observed_frames` and `gap_frames` sit on the same row.
- Territories overlap far more than instantaneous outlines do. 177 of the 3,403
  possible pairs share ground, and the median cell shares 47% of its eventual
  territory with somebody. Microglia hold non-overlapping domains at any one
  instant; over 49 hours they sweep across each other's. `contacts` answers who
  *touched*, `territory_overlap.csv` answers who *shares ground*, and a pair can
  do either without the other.
- `territory_centroid_offset_px` is measured against the plain outline centroid,
  not `motility`'s brightness-weighted soma centre. `territory` reads pixels and
  nothing else, and the centroid is the centre of the same outline the territory
  is built from; the two differ by a pixel or two on these cells.
- `branches.csv` is written with **no minimum branch length**, on purpose. A
  median cell here is 11 px across and its branches are a few pixels long, so
  any floor that excludes thinning artefacts also excludes most real branches.
  Every branch is a row - 53,023 of them over 6,054 cell-frames - with its
  length, its order from the cell body, whether it ends in a free tip and how
  far out that end is, so choosing a floor stays the reader's decision. The
  `*_branch_px` summaries in `cell_frame` are computed over all branches, so
  read them with `branch_count` beside them: a mean over 8.8 mostly-tiny
  segments is not the same statement as a mean over 3.7 segments longer than
  3 px.
- `branch_count` and `skeleton_branches` are different numbers and both stay.
  `skeleton_branches` splits the skeleton at every junction *pixel*;
  `branch_count` contracts each junction cluster to one node first. They agree
  to within 0.03 branches per cell-frame on the pinned movie, so the contraction
  turned out not to matter for the count - but it does matter for length, since
  splitting at junction pixels shortens every segment by about two of them.
  `skeleton_branches` is left exactly as it was because `regimes` clusters on
  it, and changing it would move every regime in every run ever made.
- `branch_order` is measured from the brightness-weighted centre of the cell,
  which lands outside the outline in 2.6% of cell-frames - a C-shaped cell can
  put its own centre in the gap. `soma_centre_inside` flags those rather than
  dropping them, so the orders are still there and can still be excluded.
- One column name means two things. `surveillance.held_px` counts pixels a cell
  kept between frames; the `held_px` copied into `history_residency.csv` counts
  pixels a *host* held of a cell it had swallowed. The vocabulary is keyed by
  column name, so plotting the second gets the first's label. The fix is a
  rename in the tracking stage that writes it, not a second entry here.
- `regime_transitions.csv` is one row per pair of consecutive observed frames,
  not one row per change of regime — most of its rows are a cell staying where
  it was. That is correct and load-bearing: figure 24 normalises the transition
  matrix including its diagonal, and figure 34 reads "most likely next state"
  off it, which is usually "the same one". Only the name is wrong.
  `regime_steps` would be honest; renaming it would move five call sites and
  any user script that reads a run, so it is recorded rather than done.
- Five `history_*` tables sit in `tables/` rather than `tracker/`.
  `history_gap_frames`, `history_lifespans`, `history_residency`,
  `history_sources` and `history_join_audit` are computed by this package, so
  the folder is right and the prefix is misleading. Renaming sixteen files to
  fix a prefix that the folder already makes redundant is its own change.
- `motion_evidence_frame.csv` is one row per *gap between* frames, keyed on
  `from_frame_index` — 98 rows for a 99-frame movie. It reads like frame grain
  and is not, so it is not folded into `frame_summary`; folding it would leave
  the last frame blank.
- Nine significant figures is the written precision, not the computed one.
  Every number is carried in full through the arithmetic and rounded only on
  the way to the CSV, holding each value to within 5e-9 of itself. Six was
  tried first and was enough for every *measured* column, but a figure that
  **re-derives** something from the file — a z-score, a circular difference, a
  surrogate band — subtracts nearly-equal numbers and amplifies the rounding;
  at six figures that reached 0.07% of an axis, and at nine it is under one
  part in ten million. Widen `float_format` in `run._write_table` further if an
  analysis needs an exact round-trip of the stored double.
- A column whose values are all whole numbers reads back from CSV as an integer
  rather than a float, because the format drops a trailing `.0`. Seven columns
  in the pinned movie are affected — `area_px`, `convex_area_px`, `signal_max`
  and four medians. The values are unchanged; a script that asserts a dtype
  rather than a value will notice.

- **The package has no clock, and for this preparation it does not need one.**
  `cell_frame` carries `hours`, meaning hours since the recording started, so an
  `onset_hour` of 7.5 is "seven and a half hours in" and the affected columns
  say `h from start` in their unit. There is no wall-clock time and no light
  schedule anywhere in the configuration, because these are **slices in a dish**:
  a slice cannot see light, so lights-on and Zeitgeber time do not exist for it
  and are not the missing piece. The anchor that does exist is the experimental
  one — slice preparation, the last medium change, or whatever synchronising
  stimulus was given — and where every slice is handled to one protocol,
  hours-since-recording-start *is* that anchor and phases are already comparable
  between recordings. What is not recorded is the interval from that event to
  the first frame. If it ever differs between slices, six columns become
  incomparable and nothing will say so: `onset_hour`, `offset_hour`,
  `l5_onset_hour`, `m10_onset_hour`, `cosinor_peak_hour` and
  `free_cosinor_peak_hour`. Every other rhythm column — amplitude, period,
  relative amplitude, interdaily stability, intradaily variability — is
  anchor-free and unaffected either way.

- **Interdaily stability on a two-day recording is one comparison.** It asks how
  much one day resembles the next, and the pinned movie holds 49 hours. The
  number is computed and written rather than withheld, with `days_covered`
  beside it and `stability_underdetermined` set, on the same principle as
  `period_underdetermined`: publishing it silently and withholding it silently
  are equally unhelpful. M10 is weak here for the same reason — ten hours is
  nearly half the record.

- **Three amplitudes live on `rhythms.csv` and they are three different
  numbers.** `relative_amplitude` is `(M10 - L5) / (M10 + L5)`, the busiest ten
  hours of the average day against the quietest five, which is what the field
  means by the term. `cosinor_relative_amplitude` is a fitted amplitude over a
  mean level at the period the fit was told to assume, and
  `free_cosinor_relative_amplitude` is the same ratio at the period the search
  found. None of the three is labelled "relative amplitude" on an axis; each
  label says what it divides.

- **Onset and offset come from ClockLab's template method**, transcribed from
  the lab's own `CircadianWorkbench` v0.5.0 along with L5, M10 and the two
  stability measures rather than derived here. `analysis/test_nonparametric_
  circadian.py` runs one trace through both implementations and requires the
  same answer, so the copy cannot drift silently. Three things were adapted and
  each is marked in `analysis/modules/rhythms.py`: the day is folded on
  `hours % 24` because there is no clock, a bin is a mean rather than a sum
  because these are levels rather than counts, and no bin is excluded as a
  structural zero because a turnover of zero is a real measurement rather than
  an unbroken beam.

- **Pooling is a concatenation, not a check.** Nothing verifies that two movies
  were measured with the same module settings, the same frame interval or the
  same calibration. `manifest.json` records every one of those per movie and
  `pooled/manifest.json` records who contributed what, so a mismatch is findable
  — but it is findable, not caught. Two movies analysed under different settings
  will pool into one table without complaint.

- **`unit: "cell"` is offered and is usually the wrong answer.** It is there
  because it is occasionally right — when cells come from many movies and the
  movie effect is what is being tested — and because refusing to offer it would
  send people to a spreadsheet, where they would do it without the label. Any
  result at `unit: "cell"` from a single movie is pseudoreplicated, and the
  `methods` sentence on the row says so in those words.

- **A nested design is handled by aggregating, not by modelling.** Two movies
  from one animal become one value before testing, which is the standard guard
  and costs the within-subject information a mixed-effects model would use.
  `CircadianWorkbench` carries a REML mixed model for its own profile
  comparisons; posing these scalar contrasts in that shape is real work and is
  not in this package.

- **A resampled interval at the smallest allowed group size is very wide and
  slightly optimistic.** Three units is the floor, and a percentile interval
  from resampling three values has three distinct resamples' worth of
  information in it. It is written because a number with an honest interval
  beats a number with none; it should not be read as though n were larger.

- **`window_change.csv` compares a cell with itself and nothing corrects for
  how long each window was.** A four-hour window against a twenty-hour baseline
  is a legitimate thing to declare and the difference between them is computed
  the same way as any other. `window_frames` and `window_coverage` travel in the
  windowed summary for that reason; nothing enforces that they are comparable.
