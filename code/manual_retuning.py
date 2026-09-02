from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), bool)
RETUNE_METHOD = "local_evidence_competition"
RETUNE_DOMAIN_POLICIES = ("vacated_only", "vacated_plus_halo")
ASSISTED_RETUNE_PROVENANCE = 5
BACKGROUND_CREATING_OPERATION_TYPES = frozenset({
    "exclude_identity", "delete_identity_interval",
})


@dataclass(frozen=True)
class RetuneRequest:
    source_operation_ids: tuple[int, ...]
    recipient_identities: tuple[int, ...]
    start_imagej_frame: int
    end_imagej_frame: int
    domain_policy: str = "vacated_only"
    halo_px: int = 0
    maximum_reach_px: float = 12.0
    sensitivity: float = 0.6
    competition_margin: float = 0.08
    method: str = RETUNE_METHOD


@dataclass
class RetuneProposal:
    request: RetuneRequest
    assignments: np.ndarray
    eligible_domain: np.ndarray
    released_domain: np.ndarray
    contested: np.ndarray
    best_score: np.ndarray
    per_frame: list[dict[str, Any]]
    warnings: list[str]
    boundary_errors: list[str]
    parent_state_sha256: str

    @property
    def valid(self) -> bool:
        return not self.boundary_errors


def state_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = json.dumps({
        "dtype": contiguous.dtype.name,
        "shape": list(map(int, contiguous.shape)),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _validate_source_operation_ids(
        source_operation_ids: Iterable[int], operations: list[dict[str, Any]]
        ) -> tuple[int, ...]:
    identifiers = tuple(int(value) for value in source_operation_ids)
    if not identifiers or any(value < 1 for value in identifiers) \
            or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "retuning source operation IDs must be unique positive integers")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("retuning source operation IDs must be in ascending order")
    for operation_id in identifiers:
        if operation_id > len(operations):
            raise ValueError(
                f"retuning source operation {operation_id} is not in active history")
        operation = operations[operation_id - 1]
        if operation.get("operation_id") != operation_id:
            raise ValueError("retuning source operation ID differs from edit history")
        if operation.get("type") not in BACKGROUND_CREATING_OPERATION_TYPES:
            raise ValueError(
                f"operation {operation_id} does not create retunable background")
    return identifiers


def released_background_from_history(
        bundle: Any, operations: list[dict[str, Any]],
        source_operation_ids: Iterable[int]) -> np.ndarray:
    """Reconstruct positive-to-background pixels released by selected operations."""
    identifiers = _validate_source_operation_ids(source_operation_ids, operations)
    from manual_editing import replay_edit_log

    released = np.zeros(bundle.canonical_labels.shape, bool)
    for operation_id in identifiers:
        before = replay_edit_log(bundle, operations[:operation_id - 1])[0]
        after = replay_edit_log(bundle, operations[:operation_id])[0]
        released |= (before > 0) & (after == 0)
    if not np.any(released):
        raise ValueError("selected source edits released no identity pixels")
    return released


def source_operation_frame_range(
        operations: list[dict[str, Any]], source_operation_ids: Iterable[int],
        total_frames: int) -> tuple[int, int]:
    identifiers = _validate_source_operation_ids(source_operation_ids, operations)
    starts: list[int] = []
    ends: list[int] = []
    for operation_id in identifiers:
        operation = operations[operation_id - 1]
        if operation["type"] == "exclude_identity":
            starts.append(1)
            ends.append(total_frames)
        else:
            starts.append(int(operation["start_imagej_frame"]))
            ends.append(int(operation["end_imagej_frame"]))
    return min(starts), max(ends)


def eligible_retune_source_operations(
        operations: list[dict[str, Any]], operation_count: int | None = None
        ) -> list[dict[str, Any]]:
    count = len(operations) if operation_count is None else int(operation_count)
    if count < 0 or count > len(operations):
        raise ValueError("retuning history position is outside the edit log")
    return [operation for operation in operations[:count]
            if operation.get("type") in BACKGROUND_CREATING_OPERATION_TYPES]


def build_eligible_domain(
        labels: np.ndarray, released_background: np.ndarray,
        start_imagej_frame: int, end_imagej_frame: int,
        domain_policy: str, halo_px: int) -> tuple[np.ndarray, np.ndarray]:
    if labels.shape != released_background.shape:
        raise ValueError("released background and current labels differ in shape")
    if domain_policy not in RETUNE_DOMAIN_POLICIES:
        raise ValueError(f"unsupported retuning domain policy: {domain_policy!r}")
    if isinstance(halo_px, bool) or not isinstance(halo_px, int) or halo_px < 0:
        raise ValueError("retuning halo must be a non-negative whole pixel count")
    if domain_policy == "vacated_only" and halo_px != 0:
        raise ValueError("vacated-only retuning must use a zero-pixel halo")
    if start_imagej_frame < 1 or end_imagej_frame < start_imagej_frame \
            or end_imagej_frame > len(labels):
        raise ValueError(f"retuning frame range must be within 1-{len(labels)}")

    selected = np.zeros(labels.shape, bool)
    selected[start_imagej_frame - 1:end_imagej_frame] = True
    released = released_background & selected & (labels == 0)
    if not np.any(released):
        raise ValueError(
            "no source-edit pixels are still background in the selected frame range")
    eligible = released.copy()
    if domain_policy == "vacated_plus_halo" and halo_px:
        for frame_index in range(start_imagej_frame - 1, end_imagej_frame):
            eligible[frame_index] = ndi.binary_dilation(
                released[frame_index], structure=STRUCTURE,
                iterations=halo_px) & (labels[frame_index] == 0)
    return released, eligible


def suggest_recipient_identities(
        labels: np.ndarray, eligible_domain: np.ndarray,
        start_imagej_frame: int, end_imagej_frame: int,
        maximum_reach_px: float = 12.0) -> tuple[int, ...]:
    if not np.isfinite(maximum_reach_px) or maximum_reach_px <= 0:
        raise ValueError("retuning maximum reach must be positive")
    candidates: set[int] = set()
    for frame_index in range(start_imagej_frame - 1, end_imagej_frame):
        domain = eligible_domain[frame_index]
        if not np.any(domain):
            continue
        nearby = ndi.binary_dilation(
            domain, structure=STRUCTURE,
            iterations=max(1, int(math.ceil(maximum_reach_px))))
        candidates.update(int(value) for value in np.unique(
            labels[frame_index][nearby]) if value > 0)
    return tuple(sorted(candidates))


def retune_scope_state_sha256(
        labels: np.ndarray, eligible_domain: np.ndarray,
        recipient_identities: tuple[int, ...], start_imagej_frame: int,
        end_imagej_frame: int) -> str:
    """Fingerprint recipient masks, the declared domain, and its current owners."""
    selected = labels[start_imagej_frame - 1:end_imagej_frame]
    domain = eligible_domain[start_imagej_frame - 1:end_imagej_frame]
    owners = np.zeros(selected.shape, np.int64)
    for identity in recipient_identities:
        owners[selected == identity] = identity
    domain_values = np.where(domain, selected.astype(np.int64), 0)
    scoped = np.stack((owners, domain.astype(np.int64), domain_values), axis=0)
    return state_sha256(scoped)


def _robust_identity_appearance(
        raw: np.ndarray, identity_mask: np.ndarray) -> np.ndarray:
    values = raw[identity_mask].astype(np.float32)
    if not values.size:
        return np.zeros(raw.shape, np.float32)
    centre = float(np.median(values))
    deviation = 1.4826 * float(np.median(np.abs(values - centre)))
    noise_floor = max(math.sqrt(max(abs(centre), 1.0)), 1.0)
    scale = max(deviation, noise_floor)
    z = (raw.astype(np.float32) - centre) / (3.0 * scale)
    return np.exp(-0.5 * np.square(np.clip(z, -8, 8))).astype(np.float32)


def _temporal_support(
        labels: np.ndarray, frame_index: int, identity: int) -> np.ndarray:
    support = np.zeros(labels.shape[1:], np.float32)
    neighbours = 0
    for adjacent in (frame_index - 1, frame_index + 1):
        if adjacent < 0 or adjacent >= len(labels):
            continue
        support += ndi.maximum_filter(
            (labels[adjacent] == identity).astype(np.float32), size=5)
        neighbours += 1
    return support / max(neighbours, 1)


def _request_errors(labels: np.ndarray, request: RetuneRequest) -> list[str]:
    errors: list[str] = []
    if not request.source_operation_ids:
        errors.append("at least one background-creating source edit is required")
    if not request.recipient_identities or any(
            identity <= 0 for identity in request.recipient_identities):
        errors.append("recipient identities must be positive")
    if len(set(request.recipient_identities)) != len(
            request.recipient_identities):
        errors.append("recipient identities must not contain duplicates")
    if request.start_imagej_frame < 1 \
            or request.end_imagej_frame < request.start_imagej_frame \
            or request.end_imagej_frame > len(labels):
        errors.append(f"retuning frame range must be within 1-{len(labels)}")
    if request.domain_policy not in RETUNE_DOMAIN_POLICIES:
        errors.append("retuning domain policy is unsupported")
    if request.domain_policy == "vacated_only" and request.halo_px != 0:
        errors.append("vacated-only retuning must use a zero-pixel halo")
    if isinstance(request.halo_px, bool) or request.halo_px < 0:
        errors.append("retuning halo must be non-negative")
    if not np.isfinite(request.maximum_reach_px) \
            or request.maximum_reach_px <= 0:
        errors.append("retuning maximum reach must be positive")
    if not np.isfinite(request.sensitivity) \
            or request.sensitivity < 0 or request.sensitivity > 1:
        errors.append("retuning sensitivity must be between 0 and 1")
    if not np.isfinite(request.competition_margin) \
            or request.competition_margin < 0 or request.competition_margin > 1:
        errors.append("retuning competition margin must be between 0 and 1")
    if request.method != RETUNE_METHOD:
        errors.append(f"unsupported retuning method: {request.method!r}")
    return errors


def propose_mask_retuning(
        labels: np.ndarray, raw: np.ndarray, released_background: np.ndarray,
        request: RetuneRequest) -> RetuneProposal:
    """Propose simultaneous, collision-safe ownership of edit-created background."""
    labels = np.asarray(labels)
    raw = np.asarray(raw)
    released_background = np.asarray(released_background, dtype=bool)
    if labels.ndim != 3 or raw.shape != labels.shape \
            or released_background.shape != labels.shape:
        raise ValueError("retuning labels, raw data, and released pixels must share TYX")
    errors = _request_errors(labels, request)
    if errors:
        raise ValueError("; ".join(errors))
    recipients = tuple(sorted(map(int, request.recipient_identities)))
    request = RetuneRequest(
        tuple(request.source_operation_ids), recipients,
        request.start_imagej_frame, request.end_imagej_frame,
        request.domain_policy, request.halo_px, request.maximum_reach_px,
        request.sensitivity, request.competition_margin, request.method)
    selected = labels[request.start_imagej_frame - 1:request.end_imagej_frame]
    absent = [identity for identity in recipients
              if not np.any(selected == identity)]
    if absent:
        raise ValueError(
            "recipient identities absent from the selected frame range: "
            + ", ".join(map(str, absent)))
    released, eligible = build_eligible_domain(
        labels, released_background, request.start_imagej_frame,
        request.end_imagej_frame, request.domain_policy, request.halo_px)
    interval = request.end_imagej_frame - request.start_imagej_frame + 1
    shape = (interval, *labels.shape[1:])
    assignments = np.zeros(shape, labels.dtype)
    contested = np.zeros(shape, bool)
    best_scores = np.full(shape, -np.inf, np.float32)
    warnings: list[str] = []
    per_frame: list[dict[str, Any]] = []
    minimum_score = 0.55 - 0.25 * float(request.sensitivity)

    for offset, frame_index in enumerate(range(
            request.start_imagej_frame - 1, request.end_imagej_frame)):
        frame = labels[frame_index]
        frame_domain = eligible[frame_index]
        frame_released = released[frame_index]
        present = [identity for identity in recipients
                   if np.any(frame == identity)]
        score_stack = np.full(
            (len(present), *frame.shape), -np.inf, np.float32)
        for row_index, identity in enumerate(present):
            own = frame == identity
            distance = ndi.distance_transform_edt(~own).astype(np.float32)
            distance_support = np.exp(
                -distance / max(float(request.maximum_reach_px) / 2.0, 1.0)
            ).astype(np.float32)
            appearance = _robust_identity_appearance(raw[frame_index], own)
            temporal = _temporal_support(labels, frame_index, identity)
            score = (0.55 * distance_support
                     + 0.25 * appearance
                     + 0.20 * temporal).astype(np.float32)
            candidate = frame_domain & (distance <= request.maximum_reach_px)
            score_stack[row_index][candidate] = score[candidate]

        frame_assignment = assignments[offset]
        frame_contested = contested[offset]
        if present:
            best_index = np.argmax(score_stack, axis=0)
            best = np.max(score_stack, axis=0)
            if len(present) == 1:
                second = np.full(frame.shape, -np.inf, np.float32)
            else:
                second = np.partition(score_stack, -2, axis=0)[-2]
            best_scores[offset] = best
            supported = frame_domain & (best >= minimum_score)
            margins = np.full(frame.shape, np.inf, np.float32)
            finite_runner_up = np.isfinite(second)
            margins[finite_runner_up] = (
                best[finite_runner_up] - second[finite_runner_up])
            frame_contested[:] = supported & (
                margins < request.competition_margin)
            accepted = supported & ~frame_contested
            winner_values = np.asarray(present, dtype=labels.dtype)[best_index]
            tentative = np.where(accepted, winner_values, 0).astype(labels.dtype)

            # Every retained addition must be connected to its current recipient.
            for identity in present:
                own = frame == identity
                candidate = tentative == identity
                seeds = candidate & ndi.binary_dilation(own, structure=STRUCTURE)
                if not np.any(seeds):
                    continue
                connected = ndi.binary_propagation(
                    seeds, structure=STRUCTURE, mask=candidate)
                frame_assignment[connected] = identity

        reclaimed = int(np.count_nonzero(frame_assignment))
        eligible_count = int(np.count_nonzero(frame_domain))
        contested_count = int(np.count_nonzero(frame_contested))
        unresolved = eligible_count - reclaimed
        imagej_frame = frame_index + 1
        identity_rows = [{
            "identity": identity,
            "reclaimed_pixels": int(np.count_nonzero(
                frame_assignment == identity)),
        } for identity in recipients]
        per_frame.append({
            "imagej_frame": imagej_frame,
            "released_pixels": int(np.count_nonzero(frame_released)),
            "eligible_pixels": eligible_count,
            "reclaimed_pixels": reclaimed,
            "unresolved_pixels": unresolved,
            "contested_pixels": contested_count,
            "per_identity": identity_rows,
        })
        if contested_count:
            warnings.append(
                f"ImageJ frame {imagej_frame}: {contested_count} contested pixel(s) "
                "remain background")
        if eligible_count and not reclaimed:
            warnings.append(
                f"ImageJ frame {imagej_frame}: no eligible pixels met the evidence "
                "and connectivity gates")

    boundary_errors: list[str] = []
    if np.any(assignments > 0):
        full_assignment = np.zeros(labels.shape, labels.dtype)
        full_assignment[
            request.start_imagej_frame - 1:request.end_imagej_frame] = assignments
        if np.any((full_assignment > 0) & ~eligible):
            boundary_errors.append("proposal assigns pixels outside its eligible domain")
        if np.any((full_assignment > 0) & (labels > 0)):
            boundary_errors.append("proposal overwrites a current positive owner")
    return RetuneProposal(
        request=request, assignments=assignments,
        eligible_domain=eligible[
            request.start_imagej_frame - 1:request.end_imagej_frame].copy(),
        released_domain=released[
            request.start_imagej_frame - 1:request.end_imagej_frame].copy(),
        contested=contested, best_score=best_scores, per_frame=per_frame,
        warnings=warnings, boundary_errors=boundary_errors,
        parent_state_sha256=retune_scope_state_sha256(
            labels, eligible, recipients, request.start_imagej_frame,
            request.end_imagej_frame))


def _positive_runs(array: np.ndarray) -> list[list[int]]:
    flat = np.ascontiguousarray(array).ravel()
    runs: list[list[int]] = []
    index = 0
    while index < len(flat):
        value = int(flat[index])
        if value == 0:
            index += 1
            continue
        end = index + 1
        while end < len(flat) and int(flat[end]) == value:
            end += 1
        runs.append([int(index), int(end - index), value])
        index = end
    return runs


def _decode_positive_runs(
        shape: tuple[int, ...], runs: list[list[int]], dtype: np.dtype,
        label: str) -> np.ndarray:
    size = int(np.prod(shape))
    flat = np.zeros(size, dtype=dtype)
    previous_end = 0
    for row in runs:
        if not isinstance(row, list) or len(row) != 3 \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in row):
            raise ValueError(f"retuning {label} contains an invalid run")
        start, length, value = row
        end = start + length
        if start < previous_end or length < 1 or end > size or value <= 0:
            raise ValueError(f"retuning {label} run is out of bounds or overlapping")
        flat[start:end] = value
        previous_end = end
    return flat.reshape(shape)


def encode_retune_patch(proposal: RetuneProposal) -> dict[str, Any]:
    spatial = np.any(proposal.eligible_domain, axis=0)
    yy, xx = np.nonzero(spatial)
    if not len(yy):
        raise ValueError("cannot encode an empty retuning domain")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    assignments = np.ascontiguousarray(
        proposal.assignments[:, y0:y1, x0:x1])
    eligible = np.ascontiguousarray(
        proposal.eligible_domain[:, y0:y1, x0:x1].astype(np.uint8))
    released = np.ascontiguousarray(
        proposal.released_domain[:, y0:y1, x0:x1].astype(np.uint8))
    contested = np.ascontiguousarray(
        proposal.contested[:, y0:y1, x0:x1].astype(np.uint8))
    return {
        "schema": "motion.post-edit-mask-retune-patch",
        "schema_version": 1,
        "origin_yx": [y0, x0],
        "shape": list(map(int, assignments.shape)),
        "label_dtype": assignments.dtype.name,
        "assignment_runs": _positive_runs(assignments),
        "eligible_runs": _positive_runs(eligible),
        "released_runs": _positive_runs(released),
        "contested_runs": _positive_runs(contested),
        "assignments_sha256": state_sha256(assignments),
        "eligible_sha256": state_sha256(eligible),
        "released_sha256": state_sha256(released),
        "contested_sha256": state_sha256(contested),
    }


def decode_retune_patch(
        patch: dict[str, Any], dtype: np.dtype
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if not isinstance(patch, dict) \
            or patch.get("schema") != "motion.post-edit-mask-retune-patch" \
            or patch.get("schema_version") != 1:
        raise ValueError("retuning operation has no supported accepted patch")
    shape_value = patch.get("shape")
    origin_value = patch.get("origin_yx")
    if not isinstance(shape_value, list) or len(shape_value) != 3 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in shape_value):
        raise ValueError("retuning patch shape is invalid")
    if not isinstance(origin_value, list) or len(origin_value) != 2 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in origin_value):
        raise ValueError("retuning patch origin is invalid")
    shape = tuple(shape_value)
    patch_dtype = np.dtype(str(patch.get("label_dtype", "")))
    if not np.can_cast(patch_dtype, dtype, casting="safe"):
        raise ValueError("retuning patch cannot be represented by current labels")
    names = ("assignment", "eligible", "released", "contested")
    hash_keys = {
        "assignment": "assignments_sha256",
        "eligible": "eligible_sha256",
        "released": "released_sha256",
        "contested": "contested_sha256",
    }
    arrays: list[np.ndarray] = []
    for name in names:
        runs = patch.get(f"{name}_runs")
        if not isinstance(runs, list):
            raise ValueError(f"retuning patch {name} runs are invalid")
        array_dtype = patch_dtype if name == "assignment" else np.uint8
        array = _decode_positive_runs(shape, runs, array_dtype, f"{name} patch")
        if state_sha256(array) != patch.get(hash_keys[name]):
            raise ValueError(f"retuning {name} patch fingerprint differs")
        arrays.append(array)
    assignments, eligible, released, contested = arrays
    return (assignments.astype(dtype, copy=False), eligible.astype(bool),
            released.astype(bool), contested.astype(bool),
            (int(origin_value[0]), int(origin_value[1])))


def build_retune_operation(
        proposal: RetuneProposal, reviewed_imagej_frames: Iterable[int]
        ) -> dict[str, Any]:
    if not proposal.valid:
        raise ValueError(
            "cannot commit retuning; " + "; ".join(proposal.boundary_errors))
    if not np.any(proposal.assignments):
        raise ValueError("retuning proposal reclaims no pixels")
    request = proposal.request
    reviewed = sorted(set(map(int, reviewed_imagej_frames)))
    expected = list(range(
        request.start_imagej_frame, request.end_imagej_frame + 1))
    if reviewed != expected:
        raise ValueError("every retuning frame must be reviewed before commit")
    return {
        "type": "retune_masks_after_edit",
        "source_operation_ids": list(map(int, request.source_operation_ids)),
        "recipient_identities": list(map(int, request.recipient_identities)),
        "start_imagej_frame": int(request.start_imagej_frame),
        "end_imagej_frame": int(request.end_imagej_frame),
        "domain_policy": request.domain_policy,
        "halo_px": int(request.halo_px),
        "maximum_reach_px": float(request.maximum_reach_px),
        "sensitivity": float(request.sensitivity),
        "competition_margin": float(request.competition_margin),
        "method": request.method,
        "method_version": 1,
        "parent_state_sha256": proposal.parent_state_sha256,
        "accepted_patch": encode_retune_patch(proposal),
        "proposal_warnings": list(proposal.warnings),
        "proposal_metrics": proposal.per_frame,
        "reviewed_imagej_frames": reviewed,
    }


def normalize_retune_request(
        request: dict[str, Any], total_frames: int) -> dict[str, Any]:
    allowed = {
        "type", "user", "source_operation_ids", "recipient_identities",
        "start_imagej_frame", "end_imagej_frame", "domain_policy", "halo_px",
        "maximum_reach_px", "sensitivity", "competition_margin", "method",
        "method_version", "parent_state_sha256", "accepted_patch",
        "proposal_warnings", "proposal_metrics", "reviewed_imagej_frames",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(
            "unsupported fields for retune_masks_after_edit: " + ", ".join(unknown))

    def integer(key: str, minimum: int = 1) -> int:
        value = request.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"batch operation {key} must be an integer >= {minimum}")
        return int(value)

    def number(key: str, minimum: float, maximum: float | None = None) -> float:
        value = request.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not np.isfinite(value) or value < minimum \
                or (maximum is not None and value > maximum):
            suffix = f" and <= {maximum:g}" if maximum is not None else ""
            raise ValueError(
                f"batch operation {key} must be finite and >= {minimum:g}{suffix}")
        return float(value)

    start = integer("start_imagej_frame")
    end = integer("end_imagej_frame")
    if end < start or end > total_frames:
        raise ValueError(f"retuning frame range must be within 1-{total_frames}")
    sources = request.get("source_operation_ids")
    recipients = request.get("recipient_identities")
    for key, values in (("source_operation_ids", sources),
                        ("recipient_identities", recipients)):
        if not isinstance(values, list) or not values \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in values) \
                or len(set(values)) != len(values):
            raise ValueError(
                f"retuning {key} must contain unique positive integers")
    if sources != sorted(sources):
        raise ValueError("retuning source operation IDs must be in ascending order")
    domain_policy = request.get("domain_policy")
    if domain_policy not in RETUNE_DOMAIN_POLICIES:
        raise ValueError("retuning domain policy is unsupported")
    halo = integer("halo_px", 0)
    if domain_policy == "vacated_only" and halo:
        raise ValueError("vacated-only retuning must use a zero-pixel halo")
    if request.get("method") != RETUNE_METHOD or integer("method_version") != 1:
        raise ValueError("retuning method or method version is unsupported")
    parent_hash = request.get("parent_state_sha256")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError("retuning parent-state fingerprint is invalid")
    warnings = request.get("proposal_warnings", [])
    metrics = request.get("proposal_metrics", [])
    if not isinstance(warnings, list) or any(
            not isinstance(value, str) for value in warnings):
        raise ValueError("retuning proposal warnings must be text")
    if not isinstance(metrics, list) or any(
            not isinstance(value, dict) for value in metrics):
        raise ValueError("retuning proposal metrics must be objects")
    reviewed = request.get("reviewed_imagej_frames")
    if reviewed != list(range(start, end + 1)):
        raise ValueError("every retuning frame must be recorded as reviewed")
    patch = request.get("accepted_patch")
    if not isinstance(patch, dict):
        raise ValueError("retuning accepted patch must be an object")
    decode_retune_patch(patch, np.dtype(str(patch.get("label_dtype", "uint16"))))
    normalized = {
        "type": "retune_masks_after_edit",
        "source_operation_ids": list(map(int, sources)),
        "recipient_identities": list(map(int, recipients)),
        "start_imagej_frame": start,
        "end_imagej_frame": end,
        "domain_policy": domain_policy,
        "halo_px": halo,
        "maximum_reach_px": number("maximum_reach_px", 1e-12),
        "sensitivity": number("sensitivity", 0, 1),
        "competition_margin": number("competition_margin", 0, 1),
        "method": RETUNE_METHOD,
        "method_version": 1,
        "parent_state_sha256": parent_hash,
        "accepted_patch": patch,
        "proposal_warnings": warnings,
        "proposal_metrics": metrics,
        "reviewed_imagej_frames": reviewed,
    }
    if "user" in request:
        normalized["user"] = request["user"]
    return normalized


def apply_retune_operation(
        labels: np.ndarray, provenance: np.ndarray,
        operation: dict[str, Any], released_background: np.ndarray,
        excluded_identities: set[int] | None = None) -> dict[str, Any]:
    """Apply a frozen retuning patch after validating its exact released domain."""
    start = int(operation["start_imagej_frame"])
    end = int(operation["end_imagej_frame"])
    recipients = tuple(sorted(map(int, operation["recipient_identities"])))
    if excluded_identities and any(
            identity in excluded_identities for identity in recipients):
        raise ValueError("an excluded identity cannot receive retuned pixels")
    assignments, eligible_crop, released_crop, contested_crop, (y0, x0) = \
        decode_retune_patch(operation["accepted_patch"], labels.dtype)
    if assignments.shape[0] != end - start + 1:
        raise ValueError("retuning patch frame count differs from its interval")
    y1, x1 = y0 + assignments.shape[1], x0 + assignments.shape[2]
    if y1 > labels.shape[1] or x1 > labels.shape[2]:
        raise ValueError("retuning patch lies outside the label field")
    if set(map(int, np.unique(assignments))) - {0, *recipients}:
        raise ValueError("retuning patch contains an undeclared recipient identity")
    if np.any((assignments > 0) & ~eligible_crop):
        raise ValueError("retuning patch assigns pixels outside its eligible domain")
    if np.any(released_crop & ~eligible_crop):
        raise ValueError("retuning released domain is outside its eligible domain")
    if np.any(contested_crop & ~eligible_crop):
        raise ValueError("retuning contested pixels are outside its eligible domain")

    exact_released, exact_eligible = build_eligible_domain(
        labels, released_background, start, end, operation["domain_policy"],
        int(operation["halo_px"]))
    full_eligible = np.zeros(labels.shape, bool)
    full_released = np.zeros(labels.shape, bool)
    full_eligible[start - 1:end, y0:y1, x0:x1] = eligible_crop
    full_released[start - 1:end, y0:y1, x0:x1] = released_crop
    if not np.array_equal(full_released, exact_released):
        raise ValueError("retuning patch released domain differs from source edits")
    if not np.array_equal(full_eligible, exact_eligible):
        raise ValueError("retuning patch eligible domain differs from declared policy")
    if retune_scope_state_sha256(
            labels, full_eligible, recipients, start, end) \
            != operation["parent_state_sha256"]:
        raise ValueError(
            "retuning proposal is stale relative to the replayed parent state")

    per_identity = {identity: 0 for identity in recipients}
    per_frame: list[dict[str, Any]] = []
    affected: list[int] = []
    total_reclaimed = 0
    total_unresolved = 0
    total_contested = 0
    for offset, imagej_frame in enumerate(range(start, end + 1)):
        frame = labels[imagej_frame - 1]
        domain = full_eligible[imagej_frame - 1]
        if np.any(frame[domain] != 0):
            raise ValueError(
                f"retuning eligible pixels are no longer background on ImageJ frame "
                f"{imagej_frame}")
        candidate = np.zeros(frame.shape, labels.dtype)
        candidate[y0:y1, x0:x1] = assignments[offset]
        reclaimed = int(np.count_nonzero(candidate))
        contested_count = int(np.count_nonzero(contested_crop[offset]))
        eligible_count = int(np.count_nonzero(domain))
        unresolved = eligible_count - reclaimed
        identity_rows: list[dict[str, int]] = []
        for identity in recipients:
            mask = candidate == identity
            count = int(np.count_nonzero(mask))
            if count:
                frame[mask] = identity
                provenance[imagej_frame - 1][mask] = ASSISTED_RETUNE_PROVENANCE
                per_identity[identity] += count
            identity_rows.append({
                "identity": identity, "reclaimed_pixels": count})
        if reclaimed:
            affected.append(imagej_frame)
        total_reclaimed += reclaimed
        total_unresolved += unresolved
        total_contested += contested_count
        per_frame.append({
            "imagej_frame": imagej_frame,
            "released_pixels": int(np.count_nonzero(
                full_released[imagej_frame - 1])),
            "eligible_pixels": eligible_count,
            "reclaimed_pixels": reclaimed,
            "unresolved_pixels": unresolved,
            "contested_pixels": contested_count,
            "per_identity": identity_rows,
        })
    if not total_reclaimed:
        raise ValueError("retuning operation reclaims no pixels")
    return {
        "affected_imagej_frames": affected,
        "reclaimed_pixels": total_reclaimed,
        "changed_pixels": total_reclaimed,
        "unresolved_pixels": total_unresolved,
        "contested_pixels": total_contested,
        "per_identity": [{
            "identity": identity,
            "reclaimed_pixels": per_identity[identity],
        } for identity in recipients],
        "per_frame": per_frame,
    }
