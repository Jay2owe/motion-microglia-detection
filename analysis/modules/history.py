"""Why a name is missing, hidden or renamed - in the tracker's own words.

A gap in a label stack is just an absence. Nothing in the pixels distinguishes
a cell that dimmed below threshold with its signal still visible in the raw
frame, a cell sitting inside a merged object under someone else's number, and a
gap the tracker could not explain at all. The tracker knows the difference and
writes it down, and then nothing reads it again.

Think of a ship's log beside a photograph of an empty berth. The photograph
shows the absence; only the log says whether the ship sailed, was moved to
another berth, or was never logged in.

This is the one module allowed to open a tracking CSV, and it exists so that a
presence figure can stop drawing an unexplained hole. It copies the decision
tables into the run as their own tables and joins the few columns that answer a
question the accepted labels cannot:

* ``mechanism`` for each frame of each gap, from the continuity evidence;
* ``silent_nonborder_ending`` per identity, which separates a cell that left
  the field from one the tracker stopped being able to find;
* ``host_identity`` and ``held_px``, which say "this cell was inside that one".

Two things it deliberately does not do. It never ingests a measurement column -
``frame_identities.csv`` and ``tracks.csv`` carry area and intensity that this
package computes better, and two sources of truth for area is worse than one.
And it never second-guesses a ``mechanism`` label; the tracker's classification
is copied verbatim, because disagreeing with it is a separate conversation.

Frame numbering is the trap. The decision tables count **source** ImageJ
frames; the analysis tables count label frames from zero. Every join here goes
through ``frame_table()``. A gap attributed two frames off is worse than no
attribution at all.

The second trap is subtler and is why ``still_missing_in_accepted_labels``
exists. The continuity census and evidence are computed partway through the
accepted history, on the labels as they stood at the seat-identity stage, and
later repairs then fill some of the gaps they recorded and can retire an
identity altogether. On the reference movie 174 of 966 recorded gap frames are
no longer gaps in the accepted labels, and two identities have gone. Every gap
frame therefore carries a flag saying whether it is still a gap in the stack
being measured, and a figure must colour only the ones that are. Saying "this
hole is a merge" over a frame where the cell is plainly on screen would be a
worse failure than saying nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived

#: Decision tables under the accepted-history folder, as name -> relative path.
#: Chains differ between movies, so each one that is absent is skipped and
#: recorded rather than failing the module.
SOURCES: dict[str, str] = {
    "gap_evidence": "21_continuity_evidence/out/gap_evidence.csv",
    "mechanism_summary": "21_continuity_evidence/out/mechanism_summary.csv",
    "termination_evidence": "21_continuity_evidence/out/termination_evidence.csv",
    "termination_audit": "19_continuity_census/out/termination_audit.csv",
    "gap_runs": "19_continuity_census/out/gap_runs.csv",
    "residency_registry": "18_seat_identity/out/residency_registry.csv",
    "seat_table": "18_seat_identity/out/seat_table.csv",
    "partition_actions": "18_seat_identity/out/partition_actions.csv",
    "merge_split_events": "15_merge_split_profile/out/merge_split_events.csv",
    "merge_split_actions": "16_merge_split_reconcile/out/merge_split_actions.csv",
    "quarantine_events": "08_global_quarantine/out/quarantine_events.csv",
    "ordered_actions": "11_ordered_quarantine/out/ordered_actions.csv",
    "hierarchy_actions": "13_hierarchy_reconcile/out/hierarchy_actions.csv",
    "merge_events": "09_general_merge_bridge/out/merge_events.csv",
}

#: Tables written beside the accepted labels rather than into the decision
#: folder. The pixel-evidence table is deliberately left out: it is one row per
#: renamed pixel and belongs in the tracking run, not in every analysis run.
LABEL_SOURCES: dict[str, str] = {
    "stationary_takeover_events": "stationary_takeover_events.csv",
    "stationary_component_runs": "stationary_component_runs.csv",
}


def _read_sources(root: Path | None, sources: dict[str, str], where: str,
                  ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    found: dict[str, pd.DataFrame] = {}
    record: list[dict] = []
    for name, relative in sources.items():
        path = None if root is None else root / relative
        present = path is not None and path.is_file()
        if present:
            found[name] = pd.read_csv(path)
        record.append({
            "table": name,
            "where": where,
            "relative_path": relative,
            "found": bool(present),
            "rows": int(len(found[name])) if present else 0,
        })
    return found, record


#: Only the columns this module computes. Almost everything it writes is a
#: tracking table copied through verbatim, and those columns are the detection
#: stage's vocabulary, not this package's - naming them here would be this file
#: asserting what a column of somebody else's CSV means, and the copies exist
#: precisely so that the tracker's account is not paraphrased.
#:
#: One consequence worth knowing: ``history_residency`` carries a ``held_px``
#: that counts pixels the *host* held of this cell, which is not what
#: ``surveillance.held_px`` counts, and a figure that plots it will get
#: surveillance's wording. Two producers, two meanings, one name - the fix is a
#: rename in the tracking stage rather than a second label here.
PRODUCES = (
    Column("mechanism", "Why the name was missing", "the tracker's own word", "missing"),
    Column("still_missing_in_accepted_labels", "Still a gap in the accepted labels", "0 or 1", "missing"),
    Column("identity_in_accepted_labels", "This identity survived into the accepted labels", "0 or 1", "reference"),
    Column("silent_nonborder_ending", "Ended mid-field rather than by leaving it", "0 or 1", "missing"),
    # The keys of this module's own two bookkeeping tables. They are declared
    # because a table's grain has to name columns something writes, and these
    # two are the only columns of a history_ table that this file invents
    # rather than copies.
    Column("table", "Which decision table this row is about", "name", "reference"),
    Column("check", "Which consistency check this row is", "name", "reference"),
)

#: Every table this module writes. The sixteen copies are built from the
#: dictionaries above so the declaration cannot drift from what is read.
#:
#: ``grain=()`` on a copy is the same reasoning as the ``PRODUCES`` comment:
#: naming its grain would be this package asserting what one row of somebody
#: else's CSV is, and the copies exist precisely so the tracker's account is
#: not paraphrased. ``origin="tracker"`` is what permits the empty grain, and
#: what puts the file in the tracker's own folder.
#:
#: All of them are optional because chains differ between movies: an absent
#: source is skipped and recorded in ``history_sources``, not a failure.
WRITES = tuple(
    Output(f"history_{name}", grain=(), origin="tracker", optional=True)
    for name in (*SOURCES, *LABEL_SOURCES)
) + (
    Output("history_gap_frames", grain=("identity", "frame_index"), optional=True),
    Output("history_lifespans",  grain=("identity",),               optional=True),
    Output("history_residency",  grain=("identity", "frame_index"), optional=True),
    Output("history_sources",    grain=("table",),                  optional=True),
    Output("history_join_audit", grain=("check",),                  optional=True),
)


@register_derived(
    name="history",
    description="The tracker's own record of why a name is missing, hidden or renamed",
    needs_columns=("identity", "frame_index"),
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = context.module_params("history")
    directory = params.get("directory")
    root = Path(directory) if directory and Path(directory).is_dir() else None
    labels_root = params.get("labels_directory")
    labels_root = Path(labels_root) if labels_root and Path(labels_root).is_dir() else None

    found, record = _read_sources(root, SOURCES, "decision folder")
    beside, beside_record = _read_sources(labels_root, LABEL_SOURCES, "beside the labels")
    found.update(beside)
    record.extend(beside_record)
    if not found:
        # run.py records the skip; there is nothing to say and nothing to hide.
        return {}

    frames = context.frame_table()
    to_index = dict(zip(frames["source_imagej_frame"], frames["frame_index"]))
    hours = dict(zip(frames["frame_index"], frames["hours"]))

    tables = {f"history_{name}": table for name, table in found.items()}

    named = set(map(tuple, cell_frame[["identity", "frame_index"]].to_numpy()))
    known = set(cell_frame["identity"].unique())

    gaps = found.get("gap_evidence")
    if gaps is not None and not gaps.empty:
        gap_frames = _expand_gaps(gaps, to_index, hours)
        gap_frames["still_missing_in_accepted_labels"] = [
            tuple(row) not in named
            for row in gap_frames[["identity", "frame_index"]].to_numpy()
        ]
        gap_frames["identity_in_accepted_labels"] = (
            gap_frames["identity"].isin(known))
        tables["history_gap_frames"] = gap_frames

    lifespans = _lifespans(found)
    if lifespans is not None:
        lifespans["identity_in_accepted_labels"] = lifespans["identity"].isin(known)
        tables["history_lifespans"] = lifespans

    residency = found.get("residency_registry")
    if residency is not None and not residency.empty:
        mapped = residency.copy()
        mapped["frame_index"] = mapped["source_imagej_frame"].map(to_index)
        tables["history_residency"] = mapped.dropna(subset=["frame_index"]).astype(
            {"frame_index": int})[
                ["identity", "frame_index", "source_imagej_frame",
                 "host_identity", "held_px"]]

    tables["history_sources"] = pd.DataFrame(record)
    tables["history_join_audit"] = _audit(tables, cell_frame, context)
    return tables


def _expand_gaps(gaps: pd.DataFrame, to_index: dict[int, int],
                 hours: dict[int, float]) -> pd.DataFrame:
    """One row per missing frame of every gap run, carrying its explanation."""
    columns = [c for c in ("identity", "gap", "missing_frames", "mechanism",
                           "prior_owner", "next_owner", "compatibility_scope")
               if c in gaps.columns]
    rows: list[dict] = []
    for record in gaps.to_dict("records"):
        start = int(record["gap_start_source_frame"])
        end = int(record["gap_end_source_frame"])
        for source_frame in range(start, end + 1):
            frame_index = to_index.get(source_frame)
            if frame_index is None:
                # A gap can start before the first label frame; that frame is
                # outside this analysis, not a mapping failure.
                continue
            rows.append({
                **{column: record[column] for column in columns},
                "frame_index": int(frame_index),
                "source_imagej_frame": source_frame,
                "hours": hours.get(int(frame_index)),
            })
    ordered = ["identity", "frame_index", "source_imagej_frame", "hours", "mechanism"]
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=ordered)
    rest = [c for c in table.columns if c not in ordered]
    return table[ordered + rest].sort_values(["identity", "frame_index"]).reset_index(drop=True)


def _lifespans(found: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """One row per identity: how its last frame is to be read."""
    audit = found.get("termination_audit")
    if audit is None or audit.empty:
        return None
    columns = [c for c in ("identity", "last_source_frame", "frames_present",
                           "segments", "last_area_px", "last_mask_touches_border",
                           "silent_nonborder_ending") if c in audit.columns]
    table = audit[columns].copy()
    evidence = found.get("termination_evidence")
    if evidence is not None and "mechanism" in evidence.columns:
        table = table.merge(
            evidence[["identity", "mechanism"]].rename(
                columns={"mechanism": "termination_mechanism"}),
            on="identity", how="left")
    return table


def _audit(tables: dict[str, pd.DataFrame], cell_frame: pd.DataFrame,
           context: MeasurementContext) -> pd.DataFrame:
    """What survived the join, and the one thing that must hold.

    ``expected`` is filled in only where a wrong answer means a wrong join. The
    rest are quantities a reader needs in order to know how much of the
    tracker's record still applies to the stack being measured, and a non-zero
    value there is a fact about the data rather than a fault.
    """
    known = set(cell_frame["identity"].unique())
    checks: list[dict] = []

    def add(check: str, value: int, expected: int | None, note: str) -> None:
        checks.append({"check": check, "value": value, "expected": expected,
                       "passed": expected is None or value == expected,
                       "note": note})

    gap_frames = tables.get("history_gap_frames")
    if gap_frames is not None and not gap_frames.empty:
        out_of_range = int((~gap_frames["frame_index"].between(
            0, context.n_frames - 1)).sum())
        add("gap_frames_outside_the_recording", out_of_range, 0,
            "every mapped frame must fall inside 0..n_frames-1; a failure here "
            "means source_imagej_frame was joined wrongly")
        still = int(gap_frames["still_missing_in_accepted_labels"].sum())
        add("gap_frames_recorded", int(len(gap_frames)), None,
            "missing frames the tracker explained, at the stage it explained them")
        add("gap_frames_still_missing", still, None,
            "of those, the ones that are still gaps in the accepted labels - "
            "the only ones a presence figure may annotate")
        add("gap_frames_since_filled", int(len(gap_frames)) - still, None,
            "gaps a later repair closed, so the explanation no longer applies")

    for name in ("gap_evidence", "termination_audit", "residency_registry"):
        table = tables.get(f"history_{name}")
        if table is None or "identity" not in table.columns:
            continue
        unknown = int(table.loc[~table["identity"].isin(known), "identity"].nunique())
        add(f"{name}_identities_not_in_accepted_labels", unknown, None,
            "identities the record knows that the accepted labels no longer "
            "carry, because a later stage retired them")

    return pd.DataFrame(checks)
