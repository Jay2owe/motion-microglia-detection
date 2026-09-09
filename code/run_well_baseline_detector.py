"""Create a per-well A000 disruption and persistence baseline.

This is a thin adapter around the accepted A3 detector/scorer chain. The legacy
scorer expects its candidate TIFF to be named ``95_A3.tif``; the snapshot stage
creates that internal alias while every manifest and review retains the real
recording stem.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

from well_workflow import find_motion_root


REVIEW_CASE_COLUMNS = (
    "case_id", "case_type", "mechanism", "review_frame_start",
    "review_frame_end", "source_imagej_frame_start",
    "source_imagej_frame_end", "track_ref", "identities",
    "expected_identity", "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_reference(root: Path, supplied: str | Path) -> str:
    """Record a logical project path without resolving cross-volume junctions."""
    value = Path(supplied)
    if not value.is_absolute():
        return str(value)
    try:
        return str(value.relative_to(root))
    except ValueError:
        return str(value)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optional_calibration_has_evidence(
        config: dict, evidence: dict[str, Path], evidence_key: str) -> bool:
    """Return whether an optional score calibration has usable evidence."""
    return bool(config.get("enabled", False)
                and evidence.get(evidence_key) is not None)


def accepted_evidence(run: Path, stem: str) -> dict[str, Path]:
    history = run / "mid/accepted_history"
    paths = {
        "labels": run / f"out/{stem}.tif",
        "unclaimed": run / f"out/{stem}_unclaimed_original_ids.tif",
        "raw": run.parents[1] / f"registered inputs/{stem}.tif",
        "points": history / "26_latent_body_tracks/out/latent_track_points.csv",
        "thresholds": (
            history / "25_raw_physical_hypotheses/out/"
            "frame_evidence_thresholds.csv"),
        "episodes": history / "26_latent_body_tracks/out/encounter_episodes.csv",
        "frames": history / "26_latent_body_tracks/out/encounter_frames.csv",
        "reconnections": (
            history / "26_latent_body_tracks/out/reconnection_hypotheses.csv"),
        "application_audit": (
            history / "46_transient_misownership_recovery/out/application_audit.csv"),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing accepted evidence:\n" + "\n".join(missing))
    projection_stage = history / "72_terminal_projection_diversion/out"
    projection = {
        "terminal_projection_proposals": (
            projection_stage / "terminal_diversion_triples.csv"),
        "terminal_projection_applications": (
            projection_stage / "terminal_diversion_applications.csv"),
    }
    present = {name: path.is_file() for name, path in projection.items()}
    if any(present.values()) and not all(present.values()):
        raise FileNotFoundError(
            "Incomplete accepted terminal-projection evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in projection.items()
                       if not path.is_file()))
    if all(present.values()):
        paths.update(projection)
    reclaim_stage = history / "73_delayed_projection_reclaim/out"
    reclaim = {
        "delayed_reclaim_proposals": (
            reclaim_stage / "delayed_projection_reclaim_audit.csv"),
        "delayed_reclaim_applications": (
            reclaim_stage / "delayed_projection_reclaim_applications.csv"),
    }
    reclaim_present = {name: path.is_file()
                       for name, path in reclaim.items()}
    if any(reclaim_present.values()) and not all(reclaim_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted delayed-reclaim evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in reclaim.items()
                      if not path.is_file()))
    if all(reclaim_present.values()):
        paths.update(reclaim)
    boundary_stage = history / "74_boundary_owner_excursion/out"
    boundary = {
        "boundary_excursion_audit": (
            boundary_stage / "boundary_owner_excursion_audit.csv"),
        "boundary_excursion_applications": (
            boundary_stage / "boundary_owner_excursion_applications.csv"),
    }
    boundary_present = {name: path.is_file()
                        for name, path in boundary.items()}
    if any(boundary_present.values()) and not all(boundary_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted boundary-excursion evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in boundary.items()
                      if not path.is_file()))
    if all(boundary_present.values()):
        paths.update(boundary)
    assimilation_stage = history / "75_terminal_two_seat_assimilation/out"
    assimilation = {
        "terminal_assimilation_audit": (
            assimilation_stage / "terminal_two_seat_assimilation_audit.csv"),
        "terminal_assimilation_applications": (
            assimilation_stage /
            "terminal_two_seat_assimilation_applications.csv"),
    }
    assimilation_present = {name: path.is_file()
                            for name, path in assimilation.items()}
    if any(assimilation_present.values()) and not all(
            assimilation_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted terminal-assimilation evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in assimilation.items()
                       if not path.is_file()))
    if all(assimilation_present.values()):
        paths.update(assimilation)
    flash_stage = history / "76_bracketed_mixed_owner_flash/out"
    flash = {
        "mixed_owner_flash_audit": (
            flash_stage / "bracketed_mixed_owner_flash_audit.csv"),
        "mixed_owner_flash_applications": (
            flash_stage / "bracketed_mixed_owner_flash_applications.csv"),
    }
    flash_present = {name: path.is_file() for name, path in flash.items()}
    if any(flash_present.values()) and not all(flash_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted mixed-owner-flash evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in flash.items()
                       if not path.is_file()))
    if all(flash_present.values()):
        paths.update(flash)
    diversion_stage = history / "77_dominant_seat_projection_diversion/out"
    diversion = {
        "dominant_seat_diversion_audit": (
            diversion_stage / "dominant_seat_projection_diversion_audit.csv"),
        "dominant_seat_diversion_applications": (
            diversion_stage /
            "dominant_seat_projection_diversion_applications.csv"),
        "dominant_seat_diversion_base_audit": (
            diversion_stage / "bracketed_flash_base_audit.csv"),
    }
    diversion_present = {name: path.is_file()
                         for name, path in diversion.items()}
    if any(diversion_present.values()) and not all(diversion_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted dominant-seat-diversion evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in diversion.items()
                       if not path.is_file()))
    if all(diversion_present.values()):
        paths.update(diversion)
    artifact_stage = history / "78_subresolution_point_artifact_calibration/out"
    artifact = {
        "subresolution_events": (
            artifact_stage / "candidate_disruptive_events.csv"),
        "subresolution_members": (
            artifact_stage / "candidate_event_members.csv"),
        "subresolution_track_audit": (
            artifact_stage / "subresolution_track_audit.csv"),
        "subresolution_event_audit": (
            artifact_stage / "subresolution_event_audit.csv"),
    }
    artifact_present = {name: path.is_file() for name, path in artifact.items()}
    if any(artifact_present.values()) and not all(artifact_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted sub-resolution calibration evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in artifact.items()
                       if not path.is_file()))
    if all(artifact_present.values()):
        paths.update(artifact)
    companion_stage = (
        history / "79_short_owned_companion_projection_calibration/out")
    companion = {
        "short_owned_companion_events": (
            companion_stage / "candidate_disruptive_events.csv"),
        "short_owned_companion_members": (
            companion_stage / "candidate_event_members.csv"),
        "short_owned_companion_audit": (
            companion_stage / "short_owned_companion_audit.csv"),
    }
    companion_present = {
        name: path.is_file() for name, path in companion.items()}
    if any(companion_present.values()) and not all(companion_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted short-companion calibration evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in companion.items()
                       if not path.is_file()))
    if all(companion_present.values()):
        paths.update(companion)
    gap_stage = history / "80_ownerless_cohort_latent_gap_completion/out"
    gap = {
        "ownerless_cohort_gap_audit": (
            gap_stage / "ownerless_cohort_latent_gap_audit.csv"),
        "ownerless_cohort_gap_metrics": gap_stage / "producer_metrics.json",
    }
    gap_present = {name: path.is_file() for name, path in gap.items()}
    if any(gap_present.values()) and not all(gap_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted ownerless-cohort gap evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in gap.items()
                       if not path.is_file()))
    if all(gap_present.values()):
        paths.update(gap)
    swarm_stage = (
        history / "81_recurrent_same_owner_branch_swarm_calibration/out")
    swarm = {
        "recurrent_branch_swarm_events": (
            swarm_stage / "candidate_disruptive_events.csv"),
        "recurrent_branch_swarm_members": (
            swarm_stage / "candidate_event_members.csv"),
        "recurrent_branch_swarm_audit": (
            swarm_stage / "recurrent_branch_swarm_audit.csv"),
    }
    swarm_present = {name: path.is_file() for name, path in swarm.items()}
    if any(swarm_present.values()) and not all(swarm_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted branch-swarm calibration evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in swarm.items()
                       if not path.is_file()))
    if all(swarm_present.values()):
        paths.update(swarm)
    aggregate_stage = history / "82_aggregate_shared_core_calibration/out"
    aggregate = {
        "aggregate_shared_core_events": (
            aggregate_stage / "candidate_disruptive_events.csv"),
        "aggregate_shared_core_members": (
            aggregate_stage / "candidate_event_members.csv"),
        "aggregate_shared_core_source_audit": (
            aggregate_stage / "aggregate_shared_core_source_audit.csv"),
        "aggregate_shared_core_event_audit": (
            aggregate_stage / "aggregate_shared_core_event_audit.csv"),
    }
    aggregate_present = {
        name: path.is_file() for name, path in aggregate.items()}
    if any(aggregate_present.values()) and not all(aggregate_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted aggregate shared-core evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in aggregate.items()
                       if not path.is_file()))
    if all(aggregate_present.values()):
        paths.update(aggregate)
    relay_stage = history / "83_weak_terminal_reference_relay_calibration/out"
    relay = {
        "weak_terminal_reference_relay_events": (
            relay_stage / "candidate_disruptive_events.csv"),
        "weak_terminal_reference_relay_members": (
            relay_stage / "candidate_event_members.csv"),
        "weak_terminal_reference_relay_audit": (
            relay_stage / "weak_terminal_reference_relay_audit.csv"),
    }
    relay_present = {name: path.is_file() for name, path in relay.items()}
    if any(relay_present.values()) and not all(relay_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted weak terminal relay evidence:\n" +
            "\n".join(f"{name}: {path}" for name, path in relay.items()
                       if not path.is_file()))
    if all(relay_present.values()):
        paths.update(relay)
    multireference_stage = (
        history / "84_single_owner_multireference_relay_calibration/out")
    multireference = {
        "single_owner_multireference_relay_events": (
            multireference_stage / "candidate_disruptive_events.csv"),
        "single_owner_multireference_relay_members": (
            multireference_stage / "candidate_event_members.csv"),
        "single_owner_multireference_relay_audit": (
            multireference_stage / "single_owner_multireference_relay_audit.csv"),
    }
    multireference_present = {
        name: path.is_file() for name, path in multireference.items()}
    if (any(multireference_present.values())
            and not all(multireference_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted single-owner multireference relay evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in multireference.items()
                if not path.is_file()))
    if all(multireference_present.values()):
        paths.update(multireference)
    event_local_stage = (
        history / "85_event_local_subresolution_artifact_calibration/out")
    event_local = {
        "event_local_subresolution_events": (
            event_local_stage / "candidate_disruptive_events.csv"),
        "event_local_subresolution_members": (
            event_local_stage / "candidate_event_members.csv"),
        "event_local_subresolution_audit": (
            event_local_stage / "event_local_subresolution_audit.csv"),
    }
    event_local_present = {
        name: path.is_file() for name, path in event_local.items()}
    if any(event_local_present.values()) and not all(event_local_present.values()):
        raise FileNotFoundError(
            "Incomplete accepted event-local subresolution evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in event_local.items()
                if not path.is_file()))
    if all(event_local_present.values()):
        paths.update(event_local)
    short_ownerless_stage = (
        history / "86_short_ownerless_subcellular_reference_calibration/out")
    short_ownerless = {
        "short_ownerless_subcellular_events": (
            short_ownerless_stage / "candidate_disruptive_events.csv"),
        "short_ownerless_subcellular_members": (
            short_ownerless_stage / "candidate_event_members.csv"),
        "short_ownerless_subcellular_audit": (
            short_ownerless_stage / "short_ownerless_subcellular_audit.csv"),
    }
    short_ownerless_present = {
        name: path.is_file() for name, path in short_ownerless.items()}
    if (any(short_ownerless_present.values())
            and not all(short_ownerless_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted short ownerless subcellular evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in short_ownerless.items()
                if not path.is_file()))
    if all(short_ownerless_present.values()):
        paths.update(short_ownerless)
    recording_start_stage = (
        history / "87_recording_start_two_seat_inheritance/out")
    recording_start = {
        "recording_start_two_seat_audit": (
            recording_start_stage / "recording_start_two_seat_audit.csv"),
        "recording_start_two_seat_applications": (
            recording_start_stage /
            "recording_start_two_seat_applications.csv"),
        "recording_start_two_seat_metrics": (
            recording_start_stage / "producer_metrics.json"),
    }
    recording_start_present = {
        name: path.is_file() for name, path in recording_start.items()}
    if (any(recording_start_present.values())
            and not all(recording_start_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted recording-start two-seat evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in recording_start.items()
                if not path.is_file()))
    if all(recording_start_present.values()):
        paths.update(recording_start)
    projection_relay_stage = (
        history / "88_detached_projection_owner_relay_calibration/out")
    projection_relay = {
        "detached_projection_owner_relay_events": (
            projection_relay_stage / "candidate_disruptive_events.csv"),
        "detached_projection_owner_relay_members": (
            projection_relay_stage / "candidate_event_members.csv"),
        "detached_projection_owner_relay_audit": (
            projection_relay_stage /
            "detached_projection_owner_relay_audit.csv"),
    }
    projection_relay_present = {
        name: path.is_file() for name, path in projection_relay.items()}
    if (any(projection_relay_present.values())
            and not all(projection_relay_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted detached projection relay evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in projection_relay.items()
                if not path.is_file()))
    if all(projection_relay_present.values()):
        paths.update(projection_relay)
    right_censored_stage = (
        history / "89_right_censored_novel_body_calibration/out")
    right_censored = {
        "right_censored_novel_body_events": (
            right_censored_stage / "candidate_disruptive_events.csv"),
        "right_censored_novel_body_members": (
            right_censored_stage / "candidate_event_members.csv"),
        "right_censored_novel_body_audit": (
            right_censored_stage / "right_censored_novel_body_audit.csv"),
    }
    right_censored_present = {
        name: path.is_file() for name, path in right_censored.items()}
    if (any(right_censored_present.values())
            and not all(right_censored_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted right-censored novel-body evidence:\n"
            + "\n".join(
                f"{name}: {path}" for name, path in right_censored.items()
                if not path.is_file()))
    if all(right_censored_present.values()):
        paths.update(right_censored)
    persistent_flash_stage = (
        history / "95_persistent_single_owner_flash/out")
    persistent_flash = {
        "persistent_single_owner_flash_audit": (
            persistent_flash_stage / "persistent_single_owner_flash_audit.csv"),
        "persistent_single_owner_flash_applications": (
            persistent_flash_stage
            / "persistent_single_owner_flash_applications.csv"),
        "persistent_single_owner_flash_metrics": (
            persistent_flash_stage / "producer_metrics.json"),
    }
    persistent_flash_present = {
        name: path.is_file() for name, path in persistent_flash.items()}
    if (any(persistent_flash_present.values())
            and not all(persistent_flash_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted persistent-flash evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in persistent_flash.items()
                if not path.is_file()))
    if all(persistent_flash_present.values()):
        paths.update(persistent_flash)
    delayed_projection_stage = (
        history / "97_delayed_owner_projection_flash/out")
    delayed_projection = {
        "delayed_owner_projection_flash_audit": (
            delayed_projection_stage /
            "delayed_owner_projection_flash_audit.csv"),
        "delayed_owner_projection_flash_applications": (
            delayed_projection_stage /
            "delayed_owner_projection_flash_applications.csv"),
        "delayed_owner_projection_flash_metrics": (
            delayed_projection_stage / "producer_metrics.json"),
    }
    delayed_projection_present = {
        name: path.is_file() for name, path in delayed_projection.items()}
    if (any(delayed_projection_present.values())
            and not all(delayed_projection_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted delayed-owner projection evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in delayed_projection.items()
                if not path.is_file()))
    if all(delayed_projection_present.values()):
        paths.update(delayed_projection)
    anchored_relay_stage = history / "99_anchored_projection_owner_relay/out"
    alias_retirement_stage = (
        history / "100_ephemeral_speckle_alias_retirement/out")
    raw_gap_stage = (
        history / "101_anchored_projection_raw_gap_completion/out")
    retirement_persistence_stage = (
        history / "102_retirement_aware_persistence/out")
    anchored_relay = {
        "anchored_projection_owner_relay_audit": (
            anchored_relay_stage / "anchored_projection_owner_relay_audit.csv"),
        "anchored_projection_owner_relay_applications": (
            anchored_relay_stage /
            "anchored_projection_owner_relay_applications.csv"),
        "anchored_projection_owner_relay_metrics": (
            anchored_relay_stage / "producer_metrics.json"),
        "ephemeral_speckle_alias_retirement_audit": (
            alias_retirement_stage /
            "ephemeral_speckle_alias_retirement_audit.csv"),
        "ephemeral_speckle_alias_retirement_applications": (
            alias_retirement_stage /
            "ephemeral_speckle_alias_retirement_applications.csv"),
        "ephemeral_speckle_alias_retirement_metrics": (
            alias_retirement_stage / "producer_metrics.json"),
        "anchored_projection_raw_gap_completion_audit": (
            raw_gap_stage /
            "anchored_projection_raw_gap_completion_audit.csv"),
        "anchored_projection_raw_gap_completion_metrics": (
            raw_gap_stage / "producer_metrics.json"),
        "retirement_aware_persistence_metrics": (
            retirement_persistence_stage / "producer_metrics.json"),
        "retirement_aware_persistence_comparison": (
            retirement_persistence_stage /
            "surviving_identity_confidence_comparison.csv"),
    }
    anchored_relay_present = {
        name: path.is_file() for name, path in anchored_relay.items()}
    if (any(anchored_relay_present.values())
            and not all(anchored_relay_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted anchored-projection relay evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in anchored_relay.items()
                if not path.is_file()))
    if all(anchored_relay_present.values()):
        paths.update(anchored_relay)
    terminal_chain_stage = (
        history / "103_terminal_projection_chain_completion/out")
    terminal_chain = {
        "terminal_projection_chain_audit": (
            terminal_chain_stage / "terminal_projection_chain_audit.csv"),
        "terminal_projection_chain_applications": (
            terminal_chain_stage /
            "terminal_projection_chain_applications.csv"),
        "terminal_projection_chain_metrics": (
            terminal_chain_stage / "producer_metrics.json"),
    }
    terminal_chain_present = {
        name: path.is_file() for name, path in terminal_chain.items()}
    if (any(terminal_chain_present.values())
            and not all(terminal_chain_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted terminal projection-chain evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in terminal_chain.items()
                if not path.is_file()))
    if all(terminal_chain_present.values()):
        paths.update(terminal_chain)
    body_scale_stage = history / "105_body_scale_ownerless_allocation/out"
    body_scale = {
        "body_scale_ownerless_audit": (
            body_scale_stage / "body_scale_ownerless_audit.csv"),
        "body_scale_ownerless_applications": (
            body_scale_stage / "body_scale_ownerless_applications.csv"),
        "body_scale_ownerless_metrics": (
            body_scale_stage / "producer_metrics.json"),
    }
    body_scale_present = {
        name: path.is_file() for name, path in body_scale.items()}
    if (any(body_scale_present.values())
            and not all(body_scale_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted body-scale ownerless evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in body_scale.items()
                if not path.is_file()))
    if all(body_scale_present.values()):
        paths.update(body_scale)
    terminal_boundary_stage = (
        history / "106_terminal_boundary_vanished_seat_partition/out")
    terminal_boundary = {
        "terminal_boundary_seat_audit": (
            terminal_boundary_stage / "terminal_boundary_seat_audit.csv"),
        "terminal_boundary_seat_frames": (
            terminal_boundary_stage / "terminal_boundary_seat_frames.csv"),
        "terminal_boundary_seat_metrics": (
            terminal_boundary_stage / "producer_metrics.json"),
    }
    terminal_boundary_present = {
        name: path.is_file() for name, path in terminal_boundary.items()}
    if (any(terminal_boundary_present.values())
            and not all(terminal_boundary_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted terminal-boundary seat evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in terminal_boundary.items()
                if not path.is_file()))
    if all(terminal_boundary_present.values()):
        paths.update(terminal_boundary)
    bracketed_seat_stage = (
        history / "109_bracketed_ownerless_seat_completion/out")
    bracketed_seat = {
        "bracketed_ownerless_seat_audit": (
            bracketed_seat_stage / "bracketed_ownerless_seat_audit.csv"),
        "bracketed_ownerless_seat_frames": (
            bracketed_seat_stage / "bracketed_ownerless_seat_frames.csv"),
        "bracketed_ownerless_seat_metrics": (
            bracketed_seat_stage / "producer_metrics.json"),
    }
    bracketed_seat_present = {
        name: path.is_file() for name, path in bracketed_seat.items()}
    if (any(bracketed_seat_present.values())
            and not all(bracketed_seat_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted bracketed ownerless-seat evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in bracketed_seat.items()
                if not path.is_file()))
    if all(bracketed_seat_present.values()):
        paths.update(bracketed_seat)
    detached_fading_stage = (
        history / "96_detached_fading_projection_calibration/out")
    detached_fading = {
        "detached_fading_projection_events": (
            detached_fading_stage / "candidate_disruptive_events.csv"),
        "detached_fading_projection_members": (
            detached_fading_stage / "candidate_event_members.csv"),
        "detached_fading_projection_audit": (
            detached_fading_stage / "detached_fading_projection_audit.csv"),
    }
    detached_fading_present = {
        name: path.is_file() for name, path in detached_fading.items()}
    if (any(detached_fading_present.values())
            and not all(detached_fading_present.values())):
        raise FileNotFoundError(
            "Incomplete accepted detached fading-projection evidence:\n" +
            "\n".join(
                f"{name}: {path}" for name, path in detached_fading.items()
                if not path.is_file()))
    if all(detached_fading_present.values()):
        paths.update(detached_fading)
    return paths


def validate_review_cases(path: Path) -> Path:
    """Fail before expensive scoring when the issue case table is malformed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cases = pd.read_csv(path)
    missing = [name for name in REVIEW_CASE_COLUMNS if name not in cases.columns]
    if missing:
        raise ValueError(
            "review_cases.csv is missing required columns: "
            + ", ".join(missing))
    if not len(cases):
        raise ValueError(
            "review_cases.csv must contain the neutral baseline control")
    invalid = sorted(set(cases.case_type.astype(str)) - {"failure", "control"})
    if invalid:
        raise ValueError(
            "review_cases.csv has invalid case_type values: "
            + ", ".join(invalid))
    if not cases.case_type.eq("control").any():
        raise ValueError(
            "review_cases.csv must contain at least one control row")
    return path


def semantic_links(tuning: Path, sources: dict[str, Path]) -> dict[str, Path]:
    links = tuning / "inputs/source_links"
    links.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, source in sources.items():
        suffix = "".join(source.suffixes)
        target = links / f"{name}{suffix}"
        if target.exists():
            if not os.path.samefile(target, source) and sha256(target) != sha256(source):
                raise RuntimeError(f"Frozen input drift at {target}")
        else:
            try:
                os.link(source, target)
            except OSError:
                if source.stat().st_size > 8 << 20:
                    os.symlink(source, target)
                else:
                    shutil.copy2(source, target)
        result[name] = target
    return result


def frozen_path(tuning: Path, link: Path) -> Path:
    copied = tuning / "inputs/src" / link.name
    return copied if copied.is_file() else link


def snapshot_stage(_upstream: Path | None, params: dict, out) -> dict:
    labels = out.out / "95_A3.tif"
    unclaimed = out.out / "95_A3_unclaimed_original_ids.tif"
    application_audit = out.out / "application_audit.csv"
    conflicted_audit = out.out / "conflicted_lineage_audit.csv"
    shutil.copy2(params["labels_path"], labels)
    shutil.copy2(params["unclaimed_path"], unclaimed)
    shutil.copy2(params["application_audit_path"], application_audit)
    conflicted_audit.write_text(
        "proposal_id,physical_track,assigned_identity,outcome,reason,changed_pixels,changed_frames\n",
        encoding="utf-8")
    return {
        "outputs": {"detector_labels_alias": labels,
                    "detector_unclaimed_alias": unclaimed,
                    "application_audit": application_audit,
                    "conflicted_lineage_audit": conflicted_audit},
        "summary": {
            "recording_stem": params["recording_stem"],
            "biological_change_pixels": 0,
            "targeting_mode": "field_wide_discovery",
            "identity_targets": 0,
            "track_targets": 0,
            "frame_targets": 0,
            "coordinate_targets": 0,
            "event_targets": 0,
            "region_targets": 0,
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render_reviews(
    root: Path,
    issue: Path,
    stem: str,
    labels: Path,
    raw: Path,
    motion: Path,
    events: Path,
    members: Path,
    issue_number: int,
    review_tag: str | None = None,
) -> dict[str, Path]:
    tag = review_tag or f"I{issue_number:03d}-R01-A000-BASELINE"
    if not re.fullmatch(rf"I{issue_number:03d}-[A-Za-z0-9-]+", tag):
        raise ValueError("review tag must be a canonical issue-local name")
    review = issue / "reviews" / tag / "full_dataset"
    digest = sha256(labels)
    common = [
        "--baseline-labels", str(labels), "--candidate-labels", str(labels),
        "--raw", str(raw), "--motion", str(motion),
        "--output-dir", str(review), "--stem", stem,
        "--baseline-title", "accepted A000",
        "--candidate-title", "accepted A000",
        "--source-frame-offset", "2", "--motion-frame-offset", "3",
        "--expected-baseline-sha256", digest,
        "--expected-candidate-sha256", digest,
    ]
    command = [sys.executable, str(root / "code/build_review_tiff.py"), *common]
    run_checked([*command, "--dry-run"])
    run_checked(command)

    event_dir = review / "event_diagnostics"
    event_common = [
        "--labels", str(labels), "--raw", str(raw),
        "--baseline-events", str(events), "--candidate-events", str(events),
        "--event-members", str(members), "--output-dir", str(event_dir),
        "--stem", stem, "--source-frame-offset", "2",
        "--expected-labels-sha256", digest,
    ]
    event_command = [
        sys.executable, str(root / "code/build_event_review_tiff.py"), *event_common]
    run_checked([*event_command, "--dry-run"])
    run_checked(event_command)
    return {"review": review, "event_review": event_dir}


def copy_accepted_stage(tuning: Path, stage: str, run: str) -> None:
    source = tuning / stage / run
    target = tuning / "accepted" / stage
    if target.exists():
        return
    target.mkdir(parents=True)
    for name in ("out", "qc"):
        item = source / name
        if item.exists():
            shutil.copytree(
                item, target / name, copy_function=_link_or_copy_file)
    _link_or_copy_file(str(source / "run.json"), str(target / "run.json"))


def _link_or_copy_file(source: str, destination: str) -> str:
    """Hardlink immutable score evidence, falling back across volumes."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def append_attempt(path: Path, row: dict) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        existing = list(csv.DictReader(handle))
        fields = list(existing[0]) if existing else list(row)
    if any(item["attempt_id"] == row["attempt_id"] for item in existing):
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    root = find_motion_root(Path(__file__))
    issue = (root / args.issue).resolve()
    tuning = issue / "analysis" / args.tuning_name
    accepted_run = (root / args.accepted_run).resolve()
    motion = (root / args.motion).resolve()
    if not issue.is_relative_to(root / "issues"):
        raise ValueError("issue folder must remain under Motion/issues")
    if not motion.is_file():
        raise FileNotFoundError(motion)

    evidence = accepted_evidence(accepted_run, args.stem)
    cases = validate_review_cases(issue / "review_cases.csv")
    evidence.update({"motion": motion, "cases": cases})
    links = semantic_links(tuning, evidence)

    skill = (root.parents[2] / "Macros and Scripts/Claude/shared-skills"
             / "analysis-tuner")
    sys.path.insert(0, str(skill / "scripts"))
    from tuner import Tuner  # type: ignore  # noqa: E402

    tuner = Tuner(tuning)
    if not (tuning / "inputs/inputs.csv").is_file():
        tuner.freeze_inputs(links.values())
    source = {name: frozen_path(tuning, link) for name, link in links.items()}
    source["window"] = source["labels"]

    scorer_root = (root / "issues/01_95_A3"
                   / "ISSUE-092-duplicate-soma-owner-exclusivity"
                   / "analysis/duplicate_soma_exclusivity_tuning")
    scorer_driver = load_module(
        "accepted_a3_scorer_driver_for_wells", scorer_root / "scripts/run_round01.py")
    scorer = load_module(
        "accepted_a3_disruption_scorer_for_wells", scorer_root / "code/m3_disruption_score.py")
    persistence = load_module(
        "accepted_a3_persistence_scorer_for_wells", scorer_root / "code/m4_persistence_score.py")
    event_calibrator = load_module(
        "accepted_conserved_single_core_encounter_calibrator",
        root / "code/conserved_single_core_encounter.py")
    terminal_projection_calibrator = load_module(
        "accepted_terminal_projection_calibrator",
        root / "code/terminal_projection_calibration.py")
    subresolution_calibrator = load_module(
        "accepted_subresolution_point_artifact_calibrator",
        root / "code/subresolution_point_artifact_calibration.py")
    short_companion_calibrator = load_module(
        "accepted_short_owned_companion_projection_calibrator",
        root / "code/short_owned_companion_projection_calibration.py")
    branch_swarm_calibrator = load_module(
        "accepted_recurrent_same_owner_branch_swarm_calibrator",
        root / "code/recurrent_same_owner_branch_swarm_calibration.py")
    aggregate_shared_core_calibrator = load_module(
        "accepted_aggregate_shared_core_calibrator",
        root / "code/aggregate_shared_core_calibration.py")
    weak_terminal_relay_calibrator = load_module(
        "accepted_weak_terminal_reference_relay_calibrator",
        root / "code/weak_terminal_reference_relay_calibration.py")
    single_owner_multireference_relay_calibrator = load_module(
        "accepted_single_owner_multireference_relay_calibrator",
        root / "code/single_owner_multireference_relay_calibration.py")
    event_local_subresolution_calibrator = load_module(
        "accepted_event_local_subresolution_calibrator",
        root / "code/event_local_subresolution_artifact_calibration.py")
    short_ownerless_subcellular_calibrator = load_module(
        "accepted_short_ownerless_subcellular_calibrator",
        root / "code/short_ownerless_subcellular_reference_calibration.py")
    detached_projection_relay_calibrator = load_module(
        "accepted_detached_projection_relay_calibrator",
        root / "code/detached_projection_owner_relay_calibration.py")
    right_censored_novel_body_calibrator = load_module(
        "accepted_right_censored_novel_body_calibrator",
        root / "code/right_censored_novel_body_calibration.py")
    detached_fading_projection_calibrator = load_module(
        "accepted_detached_fading_projection_calibrator",
        root / "code/detached_fading_projection_calibration.py")
    explained_projection_event_calibrator = load_module(
        "accepted_explained_projection_event_calibrator",
        root / "code/explained_projection_event_calibration.py")
    terminal_projection_chain_calibrator = load_module(
        "accepted_terminal_projection_chain_event_calibrator",
        root / "code/terminal_projection_chain_event_calibration.py")
    conserved_multi_anchor_calibrator = load_module(
        "accepted_conserved_multi_anchor_projection_calibrator",
        root / "code/conserved_multi_anchor_projection_calibration.py")
    terminal_boundary_event_calibrator = load_module(
        "accepted_terminal_boundary_event_calibrator",
        root / "code/terminal_boundary_event_calibration.py")
    fragmented_subcellular_calibrator = load_module(
        "accepted_fragmented_subcellular_episode_calibrator",
        root / "code/fragmented_subcellular_episode_calibration.py")
    fragmented_encounter_calibrator = load_module(
        "accepted_fragmented_encounter_lineage_calibrator",
        root / "code/fragmented_encounter_lineage_calibration.py")
    ownerless_handoff_calibrator = load_module(
        "accepted_ownerless_reference_handoff_calibrator",
        root / "code/ownerless_reference_handoff_calibration.py")
    onset_split_transaction_calibrator = load_module(
        "accepted_shared_donor_onset_split_transaction_calibrator",
        root / "code/shared_donor_onset_split_calibration.py")

    snapshot = tuner.stage_run(
        "m1_accepted_snapshot", "r01c_a000_baseline",
        params={"recording_stem": args.stem,
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "application_audit_path": str(source["application_audit"]),
                "targeting_mode": "field_wide_discovery"},
        fn=snapshot_stage, code=[Path(__file__)],
        notes="Byte-identical accepted parent; legacy 95_A3 filename is an internal detector adapter only.")
    score = tuner.stage_run(
        "m2_disruption_score", "r01c_a000_baseline",
        upstream=f"m1_accepted_snapshot/{snapshot['run']}",
        params=scorer_driver.score_params(source), fn=scorer.run,
        code=[scorer_root / "code/m3_disruption_score.py",
              scorer_root / "code/m4_persistence_score.py",
              root / "code/indexed_event_pairs.py"],
        notes="Current accepted target-free disruption detector, run fresh on this well.")
    score_out = tuning / score["dir"] / "out"
    raw_score_out = score_out
    calibrated_score = tuner.stage_run(
        "m2b_disruption_calibration", "r01c_a000_baseline",
        params={
            "stem": args.stem,
            "events_path": str(score_out / "candidate_disruptive_events.csv"),
            "members_path": str(score_out / "candidate_event_members.csv"),
            "points_path": str(score_out / "candidate_point_scores.csv"),
            "raw_path": str(source["raw"]),
            "labels_path": str(source["labels"]),
            "unclaimed_path": str(source["unclaimed"]),
            "enabled": True,
            "targeting_mode": "field_wide_discovery",
            "single_core_raw_sigma_px": 1.0,
            "single_core_maximum_two_core_valley_ratio": 0.80,
            "single_core_minimum_two_core_endpoint_balance": 0.25,
            "single_core_minimum_two_core_separation_sum_radii": 1.0,
            "single_core_minimum_shared_core_valley_ratio": 0.90,
            "single_core_maximum_shared_core_separation_sum_radii": 1.5,
            "single_core_minimum_shared_core_frames": 2,
        },
        fn=event_calibrator.run,
        code=[root / "code/conserved_single_core_encounter.py"],
        notes=("Accepted target-free source-level raw-core calibration; "
               "no biological labels or ledgers are changed."))
    calibration_stages = [("m2b_disruption_calibration", calibrated_score)]
    final_calibrated_score = calibrated_score
    if "terminal_projection_proposals" in source:
        calibrated_out = tuning / calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2c_terminal_projection_calibration", "r01c_a000_baseline",
            params={
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "projection_proposals_path": str(
                    source["terminal_projection_proposals"]),
                "projection_applications_path": str(
                    source["terminal_projection_applications"]),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "minimum_qualified_two_core_frames": 2,
                "targeting_mode": "field_wide_discovery",
            },
            fn=terminal_projection_calibrator.run,
            code=[root / "code/terminal_projection_calibration.py"],
            notes=("Accepted target-free projection-event calibration from "
                   "the current accepted producer audit; no biological "
                   "labels or ledgers are changed."))
        calibration_stages.append(
            ("m2c_terminal_projection_calibration", final_calibrated_score))
    accepted_config = json.loads(
        (root / "config.json").read_text(encoding="utf-8"))
    transaction_calibration = None
    transaction_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_shared_donor_onset_split_calibration", {})
    if transaction_config.get("enabled", False):
        transaction_calibration = tuner.stage_run(
            "m2a_shared_donor_onset_split_calibration",
            "r01c_a000_baseline",
            params={
                **transaction_config.get("parameters", {}),
                "transactions_path": str(
                    raw_score_out / "candidate_owner_transactions.csv"),
                "labels_path": str(source["labels"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "enabled": True,
                "targeting_mode": "field_wide_discovery",
            },
            fn=onset_split_transaction_calibrator.run,
            code=[root / "code/shared_donor_onset_split_calibration.py"],
            notes=("Accepted field-wide calibration of owner transactions "
                   "created by resolved shared-donor onset splits; biological "
                   "labels, ledgers, and event catalogues are unchanged."))
    artifact_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_subresolution_point_artifact_calibration", {})
    if artifact_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2d_subresolution_point_artifact_calibration",
            "r01c_a000_baseline",
            params={
                **artifact_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "raw_path": str(source["raw"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "thresholds_path": str(source["thresholds"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=subresolution_calibrator.run,
            code=[root / "code/subresolution_point_artifact_calibration.py",
                  root / "code/isolated_unowned_lifetime.py",
                  root / "code/separable_merge_recovery.py"],
            notes=("Accepted field-wide sub-resolution point-artifact "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2d_subresolution_point_artifact_calibration",
             final_calibrated_score))
    companion_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_short_owned_companion_projection_calibration", {})
    if companion_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2e_short_owned_companion_projection_calibration",
            "r01c_a000_baseline",
            params={
                **companion_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(source["points"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=short_companion_calibrator.run,
            code=[
                root / "code/short_owned_companion_projection_calibration.py",
                root / "code/separable_merge_recovery.py",
            ],
            notes=("Accepted field-wide short same-owner companion projection "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2e_short_owned_companion_projection_calibration",
             final_calibrated_score))
    swarm_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_recurrent_same_owner_branch_swarm_calibration", {})
    if swarm_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2f_recurrent_same_owner_branch_swarm_calibration",
            "r01c_a000_baseline",
            params={
                **swarm_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(source["points"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=branch_swarm_calibrator.run,
            code=[
                root / "code/recurrent_same_owner_branch_swarm_calibration.py",
                root / "code/separable_merge_recovery.py",
            ],
            notes=("Accepted field-wide recurrent same-owner branch-swarm "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2f_recurrent_same_owner_branch_swarm_calibration",
             final_calibrated_score))
    aggregate_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_aggregate_shared_core_calibration", {})
    if aggregate_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2g_aggregate_shared_core_calibration",
            "r01c_a000_baseline",
            params={
                **aggregate_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "raw_path": str(source["raw"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=aggregate_shared_core_calibrator.run,
            code=[
                root / "code/aggregate_shared_core_calibration.py",
                root / "code/conserved_single_core_encounter.py",
            ],
            notes=("Accepted field-wide aggregate shared-core calibration; "
                   "biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2g_aggregate_shared_core_calibration",
             final_calibrated_score))
    relay_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_weak_terminal_reference_relay_calibration", {})
    if relay_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2h_weak_terminal_reference_relay_calibration",
            "r01c_a000_baseline",
            params={
                **relay_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "raw_path": str(source["raw"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=weak_terminal_relay_calibrator.run,
            code=[root / "code/weak_terminal_reference_relay_calibration.py"],
            notes=("Accepted field-wide weak terminal reference-relay "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2h_weak_terminal_reference_relay_calibration",
             final_calibrated_score))
    multireference_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_single_owner_multireference_relay_calibration", {})
    if multireference_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2i_single_owner_multireference_relay_calibration",
            "r01c_a000_baseline",
            params={
                **multireference_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "targeting_mode": "field_wide_discovery",
            },
            fn=single_owner_multireference_relay_calibrator.run,
            code=[
                root / "code/single_owner_multireference_relay_calibration.py"],
            notes=("Accepted field-wide single-owner multireference-relay "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2i_single_owner_multireference_relay_calibration",
             final_calibrated_score))
    event_local_config = accepted_config.get("accepted_postprocessing", {}).get(
        "field_wide_event_local_subresolution_artifact_calibration", {})
    if event_local_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2j_event_local_subresolution_artifact_calibration",
            "r01c_a000_baseline",
            params={
                **event_local_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "thresholds_path": str(source["thresholds"]),
                "raw_path": str(source["raw"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=event_local_subresolution_calibrator.run,
            code=[
                root / "code/event_local_subresolution_artifact_calibration.py",
                root / "code/isolated_unowned_lifetime.py",
            ],
            notes=("Accepted field-wide event-local subresolution-artifact "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2j_event_local_subresolution_artifact_calibration",
             final_calibrated_score))
    short_ownerless_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_short_ownerless_subcellular_reference_calibration", {})
    if short_ownerless_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2k_short_ownerless_subcellular_reference_calibration",
            "r01c_a000_baseline",
            params={
                **short_ownerless_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "thresholds_path": str(source["thresholds"]),
                "raw_path": str(source["raw"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=short_ownerless_subcellular_calibrator.run,
            code=[
                root / "code/short_ownerless_subcellular_reference_calibration.py",
                root / "code/isolated_unowned_lifetime.py",
            ],
            notes=("Accepted field-wide short ownerless subcellular-reference "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2k_short_ownerless_subcellular_reference_calibration",
             final_calibrated_score))
    projection_relay_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_detached_projection_owner_relay_calibration", {})
    if projection_relay_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2l_detached_projection_owner_relay_calibration",
            "r01c_a000_baseline",
            params={
                **projection_relay_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "targeting_mode": "field_wide_discovery",
            },
            fn=detached_projection_relay_calibrator.run,
            code=[
                root / "code/detached_projection_owner_relay_calibration.py",
                root / "code/separable_merge_recovery.py",
            ],
            notes=("Accepted field-wide detached projection owner-relay "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2l_detached_projection_owner_relay_calibration",
             final_calibrated_score))
    right_censored_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_right_censored_novel_body_calibration", {})
    if right_censored_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2m_right_censored_novel_body_calibration",
            "r01c_a000_baseline",
            params={
                **right_censored_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "thresholds_path": str(
                    accepted_run / "mid/accepted_history"
                    / "25_raw_physical_hypotheses/out"
                    / "frame_evidence_thresholds.csv"),
                "raw_path": str(source["raw"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=right_censored_novel_body_calibrator.run,
            code=[
                root / "code/right_censored_novel_body_calibration.py",
                root / "code/isolated_unowned_lifetime.py",
                root / "code/separable_merge_recovery.py",
            ],
            notes=("Accepted field-wide right-censored novel-body score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2m_right_censored_novel_body_calibration",
             final_calibrated_score))
    explained_projection_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_explained_projection_event_calibration", {})
    applications_path = evidence.get(
        "delayed_owner_projection_flash_applications")
    if optional_calibration_has_evidence(
            explained_projection_config, evidence,
            "delayed_owner_projection_flash_applications"):
        # Older accepted wells predate this producer.  With no producer ledger
        # there are no explained projection applications to calibrate, so the
        # mathematically correct cross-well behaviour is an unchanged score.
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2n_explained_projection_event_calibration",
            "r01c_a000_baseline",
            params={
                **explained_projection_config.get("parameters", {}),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "applications_path": str(applications_path),
                "targeting_mode": "field_wide_discovery",
            },
            fn=explained_projection_event_calibrator.run,
            code=[root / "code/explained_projection_event_calibration.py"],
            notes=("Accepted producer-ledger projection-event score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2n_explained_projection_event_calibration",
             final_calibrated_score))
    detached_fading_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_detached_fading_projection_calibration", {})
    if detached_fading_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2n_detached_fading_projection_calibration",
            "r01c_a000_baseline",
            params={
                **detached_fading_config.get("parameters", {}),
                "stem": args.stem,
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "targeting_mode": "field_wide_discovery",
            },
            fn=detached_fading_projection_calibrator.run,
            code=[root / "code/detached_fading_projection_calibration.py"],
            notes=("Accepted field-wide detached fading-projection score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2n_detached_fading_projection_calibration",
             final_calibrated_score))
    terminal_chain_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_terminal_projection_chain_event_calibration", {})
    applications_path = source.get(
        "terminal_projection_chain_applications")
    terminal_chain_calibration_applied = optional_calibration_has_evidence(
        terminal_chain_config, evidence,
        "terminal_projection_chain_applications")
    terminal_chain_constituents = None
    if terminal_chain_calibration_applied:
        # Accepted wells that predate this producer have no qualifying
        # applications by construction, so their score remains unchanged.
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2o_terminal_projection_chain_event_calibration",
            "r01c_a000_baseline",
            params={
                **terminal_chain_config.get("parameters", {}),
                "applications_path": str(applications_path),
                "baseline_events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "baseline_members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "candidate_events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "candidate_members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "targeting_mode": "field_wide_discovery",
            },
            fn=terminal_projection_chain_calibrator.run,
            code=[
                root / "code/terminal_projection_chain_event_calibration.py"],
            notes=("Accepted symmetric producer-ledger terminal projection "
                   "source calibration; biological labels and ledgers are "
                   "unchanged."))
        calibration_stages.append(
            ("m2o_terminal_projection_chain_event_calibration",
             final_calibrated_score))
        terminal_chain_constituents = final_calibrated_score["summary"][
            "candidate"]
    conserved_anchor_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_conserved_multi_anchor_projection_calibration", {})
    if conserved_anchor_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2p_conserved_multi_anchor_projection_calibration",
            "r01c_a000_baseline",
            params={
                **conserved_anchor_config.get("parameters", {}),
                "stem": args.stem,
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "physical_track_points_path": str(source["points"]),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "targeting_mode": "field_wide_discovery",
            },
            fn=conserved_multi_anchor_calibrator.run,
            code=[
                root / "code/conserved_multi_anchor_projection_calibration.py"],
            notes=("Accepted field-wide conserved multi-anchor score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2p_conserved_multi_anchor_projection_calibration",
             final_calibrated_score))
    terminal_boundary_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_terminal_boundary_event_calibration", {})
    terminal_boundary_calibration_applied = optional_calibration_has_evidence(
        terminal_boundary_config, evidence, "terminal_boundary_seat_audit")
    if terminal_boundary_calibration_applied:
        # Accepted wells predating the producer have no partition proof and
        # therefore skip this calibration as a mathematical no-op.
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2q_terminal_boundary_event_calibration",
            "r01c_a000_baseline",
            params={
                **terminal_boundary_config.get("parameters", {}),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "applications_path": str(
                    source["terminal_boundary_seat_audit"]),
                "application_frames_path": str(
                    source["terminal_boundary_seat_frames"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=terminal_boundary_event_calibrator.run,
            code=[root / "code/terminal_boundary_event_calibration.py",
                  root / "code/terminal_boundary_vanished_seat.py"],
            notes=("Accepted field-wide terminal boundary producer-proof "
                   "event calibration; biological labels and ledgers are "
                   "unchanged."))
        calibration_stages.append(
            ("m2q_terminal_boundary_event_calibration",
             final_calibrated_score))
    fragmented_subcellular_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_fragmented_subcellular_episode_calibration", {})
    if fragmented_subcellular_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2r_fragmented_subcellular_episode_calibration",
            "r01c_a001_accepted",
            params={
                **fragmented_subcellular_config.get("parameters", {}),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "raw_path": str(source["raw"]),
                "thresholds_path": str(source["thresholds"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=fragmented_subcellular_calibrator.run,
            code=[
                root / "code/fragmented_subcellular_episode_calibration.py",
                root / "code/isolated_unowned_lifetime.py",
            ],
            notes=("Accepted field-wide fragmented subcellular-reference "
                   "score calibration; biological labels and ledgers are "
                   "unchanged."))
        calibration_stages.append(
            ("m2r_fragmented_subcellular_episode_calibration",
             final_calibrated_score))
    fragmented_encounter_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_fragmented_encounter_lineage_calibration", {})
    if fragmented_encounter_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        score_points_path = raw_score_out / "candidate_point_scores.csv"
        movie_frames = int(pd.read_csv(
            score_points_path, usecols=["frame"])["frame"].nunique())
        final_calibrated_score = tuner.stage_run(
            "m2s_fragmented_encounter_lineage_calibration",
            "r01c_a001_accepted",
            params={
                **fragmented_encounter_config.get("parameters", {}),
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "points_path": str(score_points_path),
                "movie_frames": movie_frames,
                "targeting_mode": "field_wide_discovery",
            },
            fn=fragmented_encounter_calibrator.run,
            code=[
                root / "code/fragmented_encounter_lineage_calibration.py"],
            notes=("Accepted field-wide fragmented encounter-lineage score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2s_fragmented_encounter_lineage_calibration",
             final_calibrated_score))
    ownerless_handoff_config = accepted_config.get(
        "accepted_postprocessing", {}).get(
            "field_wide_ownerless_reference_handoff_calibration", {})
    if ownerless_handoff_config.get("enabled", False):
        calibrated_out = tuning / final_calibrated_score["dir"] / "out"
        final_calibrated_score = tuner.stage_run(
            "m2t_ownerless_reference_handoff_calibration",
            "r01c_a002_accepted",
            params={
                **ownerless_handoff_config.get("parameters", {}),
                "stem": args.stem,
                "labels_path": str(source["labels"]),
                "unclaimed_path": str(source["unclaimed"]),
                "events_path": str(
                    calibrated_out / "candidate_disruptive_events.csv"),
                "members_path": str(
                    calibrated_out / "candidate_event_members.csv"),
                "points_path": str(
                    raw_score_out / "candidate_point_scores.csv"),
                "raw_path": str(source["raw"]),
                "thresholds_path": str(source["thresholds"]),
                "targeting_mode": "field_wide_discovery",
            },
            fn=ownerless_handoff_calibrator.run,
            code=[
                root / "code/ownerless_reference_handoff_calibration.py",
                root / "code/isolated_unowned_lifetime.py",
                root / "code/ownerless_body_recovery.py",
            ],
            notes=("Accepted field-wide ownerless reference-handoff score "
                   "calibration; biological labels and ledgers are unchanged."))
        calibration_stages.append(
            ("m2t_ownerless_reference_handoff_calibration",
             final_calibrated_score))
    persistence_run = tuner.stage_run(
        "m3_persistence_score", "r01c_a000_baseline",
        upstream=f"m1_accepted_snapshot/{snapshot['run']}",
        params={"baseline_labels_path": str(source["labels"]),
                "window_reference_labels_path": str(source["window"])},
        fn=persistence.run,
        code=[scorer_root / "code/m4_persistence_score.py",
              root / "code/continuity_confidence.py"],
        notes="Fixed-window persistence baseline for this well.")
    retirement_aware_persistence = None
    retirement_metrics_path = evidence.get(
        "retirement_aware_persistence_metrics")
    if retirement_metrics_path is not None:
        retirement_aware_persistence = json.loads(
            retirement_metrics_path.read_text(encoding="utf-8"))
        if (not retirement_aware_persistence.get("retirement_proof_exact", False)
                or not retirement_aware_persistence.get(
                    "all_persistence_gates_pass", False)
                or any(retirement_aware_persistence.get(
                    "target_counts", {}).values())):
            raise AssertionError(
                "accepted alias-retirement persistence proof is incomplete")

    score_out = tuning / final_calibrated_score["dir"] / "out"
    events_path = score_out / "candidate_disruptive_events.csv"
    members_path = score_out / "candidate_event_members.csv"
    events = pd.read_csv(events_path).sort_values(
        ["disruption_score", "impact", "first_frame"], ascending=[False, False, True])
    top = events.head(20).copy()
    round_dir = tuning / "rounds" / args.round_name
    round_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(round_dir / "top_events.csv", index=False)
    issue_match = re.match(r"ISSUE-(\d+)-", issue.name)
    if issue_match is None:
        raise ValueError(
            "issue folder must begin with ISSUE-NNN- so review files remain canonical")
    issue_number = int(issue_match.group(1))
    reviews = render_reviews(
        root, issue, args.stem, evidence["labels"], evidence["raw"], motion,
        events_path, members_path, issue_number, args.review_tag)

    accepted_stages = [
        ("m1_accepted_snapshot", snapshot),
        ("m2_disruption_score", score),
        *([("m2a_shared_donor_onset_split_calibration",
            transaction_calibration)] if transaction_calibration else []),
        *calibration_stages,
        ("m3_persistence_score", persistence_run),
    ]
    for stage, manifest in accepted_stages:
        copy_accepted_stage(tuning, stage, manifest["run"])

    top_event = top.iloc[0].to_dict() if len(top) else None
    summary = {
        "schema": "motion.well-baseline.v1",
        "status": "baselined",
        "well": args.stem,
        "accepted_parent": project_reference(root, args.accepted_run),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frames_removed": [1, 2],
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events", "regions")},
        "inputs_csv": str((tuning / "inputs/inputs.csv").relative_to(root)),
        "labels_sha256": sha256(evidence["labels"]),
        "unclaimed_sha256": sha256(evidence["unclaimed"]),
        "disruption": {
            **final_calibrated_score["summary"]["after"],
            "owner_transactions": (
                transaction_calibration["summary"]["after"]
                    ["owner_transactions"]
                if transaction_calibration else
                score["summary"]["candidate"].get(
                    "owner_transactions", 0)),
            "owner_transaction_burden": (
                transaction_calibration["summary"]["after"]
                    ["owner_transaction_burden"]
                if transaction_calibration else
                score["summary"]["candidate"].get(
                    "owner_transaction_burden", 0.0)),
        },
        "persistence": persistence_run["summary"],
        **({"retirement_aware_persistence": retirement_aware_persistence}
           if retirement_aware_persistence is not None else {}),
        **({"terminal_projection_chain_constituents":
                terminal_chain_constituents}
           if terminal_chain_constituents is not None else {}),
        "top_event": top_event,
        "top_events_csv": str((round_dir / "top_events.csv").relative_to(root)),
        "stage_runs": [manifest["dir"] for _, manifest in accepted_stages],
        "review": str(reviews["review"].relative_to(root)),
        "event_review": str(reviews["event_review"].relative_to(root)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(tuning / "accepted_base.json", summary)
    write_json(issue.parent / "WELL_BASELINE.json", summary)

    append_attempt(issue / "attempts.csv", {
        "attempt_id": "R01-A000", "date": date.today().isoformat(),
        "approaches": f"accepted {accepted_run.name} unchanged",
        "stage_runs": "|".join(summary["stage_runs"]),
        "outputs_sha256": sha256(events_path),
        "primary": f"{len(events)} freshly detected events",
        "controls": "1/1 neutral baseline control",
        "regressions": "not applicable; A000 accepted parent",
        "guardrails": "zero biological change; zero targets",
        "elapsed_s": summary["elapsed_seconds"], "identical_to": "accepted parent",
        "review_status": "complete baseline review",
        "decision": "baseline",
        "notes": f"Fresh {args.stem} catalogue; no prior event list reused.",
    })
    print(json.dumps(summary, indent=2, default=str))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stem", required=True)
    result.add_argument("--accepted-run", required=True)
    result.add_argument("--issue", required=True)
    result.add_argument("--motion", required=True)
    result.add_argument(
        "--tuning-name", default="highest_disruption_event_tuning",
        help="Directory name beneath the issue's analysis folder.")
    result.add_argument(
        "--round-name", default="R01-rank-current-disruptions",
        help="Directory name beneath the tuning folder's rounds directory.")
    result.add_argument(
        "--review-tag",
        help="Canonical issue-local review folder name; defaults to A000 baseline.")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
