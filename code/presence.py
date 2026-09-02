"""Cells do not disappear: locate every identity in every frame of its lifetime.

An identity that stops moving stops producing lag evidence, so a motion-first detector
drops it even though the cell is still sitting in the raw movie. This layer takes the
identity movie produced upstream and, for every frame in which an identity is missing,
searches the likely area implied by its movement and the red/green motion evidence, then
sculpts a mask out of that area to match the identity's own descriptor.

Only the image border may end a life. Every other recovery failure is reported.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from tracking import frame_observations


DEFAULTS = {
    # Where the cell is allowed to be.
    "search_px": 14.0,              # radius at one step; grows with sqrt(steps)
    "maximum_search_px": 44.0,
    "velocity_decay": 0.6,          # a stale velocity is trusted less each step
    "border_px": 8,                 # a mask this close to the edge may exit the field
    # What the cell looks like.
    "intensity_floor_fraction": 0.30,
    "minimum_intensity": 120.0,
    "maximum_mean_log2_change": 1.0,
    "minimum_recovered_area_ratio": 0.40,
    "maximum_recovered_area_ratio": 1.60,
    # Both calibrate themselves from the movie. An outline drawn on empty film must
    # stand out from what surrounds it, and an identity already too big to be one cell
    # is reported rather than grown further.
    "background_percentile": 95.0,
    "background_multiple": 1.0,
    "local_contrast": 1.4,
    "contrast_ring_px": 3,
    "maximum_identity_area_ratio": 4.0,
    "descriptor_memory": 9,         # frames of history behind the median descriptor
    # Carving a lost cell back out of the neighbour that swallowed it. Only the pixels a
    # host cannot account for by its own size are ever available.
    "carve_hosts": True,
    "carve_penalty": 0.35,
    "host_surplus_fraction": 0.15,
    "minimum_carve_px": 10,
    "minimum_host_area_px": 12,
    "carve_only_new_host_gain": False,
    # Candidate workflows can isolate the two recovery domains without naming a cell
    # or location. Defaults preserve historical behaviour.
    "allow_new_foreground": True,
    # Established cell bodies remain open claims. Tiny or fleeting fragments are not
    # forced through the movie because they are below reliable soma evidence.
    "minimum_persistent_area_px": 30,
    # Optional movie-relative form used by later tuning rounds. When supplied, the
    # threshold is this fraction of the median identity's median observed area. The
    # fixed-pixel setting above remains the default and therefore preserves every
    # accepted run that predates this option.
    "minimum_persistent_area_ratio_to_movie_median": None,
    "minimum_persistent_area_floor_px": 4,
    # A supplementary pass can be restricted to the identities excluded by an
    # earlier fixed floor. The upper bound is exclusive and disabled by default.
    "maximum_persistent_area_px_exclusive": None,
    "minimum_persistent_observations": 3,
    # A first-ever tiny fragment must not permanently poison the descriptor. When
    # enabled by an audited recovery arm, several consecutive mutually consistent
    # substantial observations can establish a fresh descriptor and robust endpoint
    # velocity. The default is off so existing pipeline behaviour remains unchanged.
    "allow_descriptor_reestablishment": False,
    "descriptor_reestablishment_observations": 3,
    "descriptor_reestablishment_area_ratio": 1.6,
    "descriptor_reestablishment_maximum_step_px": 14.0,
    "allow_field_exit": True,
    # Review attempts can first hard-fill only gaps bracketed by real observations.
    # End-of-track cells remain open claims in the audit without painting uncertain
    # outlines beyond their last evidence.
    "limit_to_observed_lifetime": False,
    # Restrict recovery using the complete gap in the unmodified input labels. Using
    # descriptor staleness here would reset after each inferred frame and could walk
    # through an arbitrarily long interruption.
    "maximum_original_gap_frames": None,
    # A supplementary duration pass may start strictly beyond an earlier accepted
    # ceiling. This prevents reapplying the earlier short-gap population.
    "minimum_original_gap_frames": None,
    "forced_identity_ids": [],
    "forced_intervals": [],
    # How the likely area is scored.
    "intensity_weight": 1.00,
    "proximity_weight": 0.80,
    "template_weight": 0.90,        # scaled down by how much the cell has moved
    "red_weight": 0.45,
    "green_weight": 0.35,
    "template_decay_px": 4.0,
    "motion_log2": 1.5,             # |lag| at or above this is real movement
    "stable_log2": 1.0,             # |lag| below this is stable structure
    "minimum_seed_score": 0.25,
    "minimum_pixel_score": 0.12,
}


@dataclass
class Descriptor:
    """What makes this the same cell: size, shape, movement, intensity, location."""

    identity: int
    t: int
    mask: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    areas: list[int] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stale: int = 0
    retired: str = ""

    @property
    def area(self) -> float:
        """The cell's own size, not the size of a blob it is sharing.

        An outline that suddenly holds two cells must not teach the cell that it is
        twice as big, or it would stop looking like a blob with room in it.
        """
        return float(np.median(self.areas)) if self.areas else 0.0

    @property
    def mean(self) -> float:
        return float(np.median(self.means)) if self.means else 0.0

    def observe(self, t: int, mask: np.ndarray, raw_frame: np.ndarray,
                memory: int, merged_area_ratio: float = 1.6,
                position: np.ndarray | None = None,
                area: int | None = None,
                mean: float | None = None) -> None:
        if position is None:
            yy, xx = np.nonzero(mask)
            position = np.array([yy.mean(), xx.mean()], float)
        else:
            position = np.asarray(position, float)
        steps = max(t - self.t, 1)
        if self.areas:
            step_velocity = (position - self.position) / steps
            self.velocity = 0.65 * step_velocity + 0.35 * self.velocity
        self.t, self.mask, self.position, self.stale = t, mask.copy(), position, 0
        area = int(mask.sum()) if area is None else int(area)
        if not self.areas or area <= merged_area_ratio * self.area:
            self.areas.append(area)
            del self.areas[:-memory]
        self.means.append(
            float(raw_frame[mask].mean()) if mean is None else float(mean))
        del self.means[:-memory]


def _identity_frames(labels: np.ndarray) -> dict[int, list[int]]:
    present: dict[int, list[int]] = {}
    for t in range(len(labels)):
        for value in np.unique(labels[t]):
            value = int(value)
            if value > 0:
                present.setdefault(value, []).append(t)
    return present


def _touches_border(mask: np.ndarray, border_px: int) -> bool:
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return False
    height, width = mask.shape
    return bool(yy.min() < border_px or xx.min() < border_px
                or yy.max() >= height - border_px or xx.max() >= width - border_px)


def motion_coverage(mask: np.ndarray, lag_frame: np.ndarray | None,
                    threshold: float) -> float:
    """Fraction of the cell's own area carrying real movement across this transition.

    This is the measure of "a lot of movement": it decides how far the outline is
    allowed to depart from the shape the cell had in the previous frame.
    """
    if lag_frame is None or not mask.any():
        return 0.0
    values = np.abs(np.nan_to_num(lag_frame[mask], nan=0.0))
    return float(np.count_nonzero(values >= threshold) / mask.sum())


def _window(shape: tuple[int, int], centre: np.ndarray, radius: float
            ) -> tuple[slice, slice]:
    y0 = max(int(np.floor(centre[0] - radius)), 0)
    y1 = min(int(np.ceil(centre[0] + radius)) + 1, shape[0])
    x0 = max(int(np.floor(centre[1] - radius)), 0)
    x1 = min(int(np.ceil(centre[1] + radius)) + 1, shape[1])
    return slice(y0, y1), slice(x0, x1)


def carvable_pixels(frame_labels: np.ndarray, rows: slice, cols: slice,
                    inside: np.ndarray, distance: np.ndarray,
                    expected_area: dict[int, float], params: dict,
                    previous_frame_labels: np.ndarray | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Pixels a neighbouring identity is holding but cannot account for by its own size.

    A cell that has been swallowed by the outline of the cell beside it is still there.
    The host may give up only its surplus - the pixels beyond its own expected area -
    and only the ones closest to where the lost cell is predicted to be.
    """
    local_labels = frame_labels[rows, cols]
    contested = np.zeros(inside.shape, bool)
    newly_gained = np.zeros(inside.shape, bool)
    if not bool(params.get("carve_hosts", True)):
        return contested, newly_gained
    minimum_carve = int(params["minimum_carve_px"])
    minimum_host = int(params["minimum_host_area_px"])
    fraction = float(params["host_surplus_fraction"])
    for host in np.unique(local_labels[inside]):
        host = int(host)
        if host <= 0:
            continue
        held = int(np.count_nonzero(frame_labels == host))
        expected = float(expected_area.get(host, 0.0))
        surplus = held - expected
        if expected <= 0 or surplus < max(minimum_carve, fraction * expected):
            continue
        available = int(min(surplus, held - minimum_host))
        if available < minimum_carve:
            continue
        candidates = inside & (local_labels == host)
        ys, xs = np.nonzero(candidates)
        if not len(ys):
            continue
        if previous_frame_labels is None:
            gained = np.zeros_like(candidates)
        else:
            gained = candidates & (
                previous_frame_labels[rows, cols] != host)
        newly_gained |= gained
        gained_y, gained_x = np.nonzero(gained)
        old_y, old_x = np.nonzero(candidates & ~gained)
        gained_order = np.argsort(distance[gained_y, gained_x])
        old_order = np.argsort(distance[old_y, old_x])
        ordered = [(int(gained_y[index]), int(gained_x[index]))
                   for index in gained_order]
        if not bool(params.get("carve_only_new_host_gain", False)):
            ordered += [(int(old_y[index]), int(old_x[index]))
                        for index in old_order]
        for y, x in ordered[:available]:
            contested[y, x] = True
    return contested, newly_gained


def likely_area(state: Descriptor, t: int, raw: np.ndarray, lag: np.ndarray,
                frame_labels: np.ndarray, expected_area: dict[int, float], params: dict
                , previous_frame_labels: np.ndarray | None = None
                ) -> tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray,
                           dict] | None:
    """Score every pixel the cell could plausibly occupy in this frame.

    Movement gives the search area; the red and green motion evidence and the cell's own
    previous outline say which parts of it are likely.
    """
    steps = max(t - state.t, 1)
    radius = min(float(params["search_px"]) * np.sqrt(steps),
                 float(params["maximum_search_px"]))
    decay = float(params["velocity_decay"])
    drift = state.velocity * sum(decay ** k for k in range(steps))
    predicted = state.position + drift
    height, width = raw.shape[1:]
    if not (-radius <= predicted[0] < height + radius
            and -radius <= predicted[1] < width + radius):
        return None
    rows, cols = _window((height, width), predicted, radius)
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return None

    yy, xx = np.mgrid[rows, cols]
    distance = np.hypot(yy - predicted[0], xx - predicted[1])
    inside = distance <= radius
    local_labels = frame_labels[rows, cols]
    contested, newly_gained = carvable_pixels(
        frame_labels, rows, cols, inside, distance, expected_area, params,
        previous_frame_labels)

    intensity = raw[t][rows, cols].astype(np.float32)
    floor = max(float(params["intensity_floor_fraction"]) * state.mean,
                float(params["minimum_intensity"]))
    # The growth domain is deliberately wider than the bright core: a hard brightness
    # mask breaks a dim cell into islands the sculpt cannot cross.
    blank = ((local_labels == 0)
             & bool(params.get("allow_new_foreground", True)))
    free = (inside & (intensity >= float(params["minimum_intensity"]))
            & (blank | contested))
    if not free.any():
        return None

    intensity_term = np.clip(intensity / max(state.mean, 1.0), 0.0, 1.25)
    sigma = max(radius / 2.0, 3.0)
    proximity_term = np.exp(-0.5 * (distance / sigma) ** 2)

    # The previous outline, carried to where the movement says the cell now is. A cell
    # that has barely moved should keep its shape; one that has moved a lot need not.
    template = np.zeros((height, width), bool)
    shift = np.rint(drift).astype(int)
    source_y, source_x = np.nonzero(state.mask)
    target_y = np.clip(source_y + shift[0], 0, height - 1)
    target_x = np.clip(source_x + shift[1], 0, width - 1)
    template[target_y, target_x] = True
    local_template = template[rows, cols]
    if local_template.any():
        outside = ndi.distance_transform_edt(~local_template)
        template_term = np.exp(-outside / max(float(params["template_decay_px"]), 1e-6))
    else:
        template_term = np.zeros_like(intensity_term)

    lag_index = min(max(t - 1, 0), len(lag) - 1)
    lag_frame = np.nan_to_num(lag[lag_index][rows, cols], nan=0.0)
    red = lag_frame >= float(params["motion_log2"])
    green = (np.abs(lag_frame) < float(params["stable_log2"])) & (intensity >= floor)
    previous_lag = lag[min(max(state.t - 1, 0), len(lag) - 1)] if len(lag) else None
    coverage = motion_coverage(state.mask, previous_lag, float(params["motion_log2"]))

    score = (float(params["intensity_weight"]) * intensity_term
             + float(params["proximity_weight"]) * proximity_term
             + float(params["template_weight"]) * (1.0 - coverage) * template_term
             + float(params["red_weight"]) * red
             + float(params["green_weight"]) * green
             - float(params["carve_penalty"]) * contested)
    score = np.where(free, score, -np.inf)
    context = {
        "search_radius_px": float(radius), "gap_steps": int(steps),
        "predicted_y": float(predicted[0]), "predicted_x": float(predicted[1]),
        "motion_coverage": float(coverage), "free_px": int(free.sum()),
        "carvable_px": int((contested & free).sum()),
        "new_host_gain_px": int(newly_gained.sum()),
        "carvable_new_host_gain_px": int((contested & newly_gained & free).sum()),
        "red_px": int((red & free).sum()), "green_px": int((green & free).sum()),
        "intensity_floor": float(floor),
    }
    return (rows, cols), score, free, contested, context


def sculpt(score: np.ndarray, free: np.ndarray, target_area: int,
           params: dict) -> np.ndarray | None:
    """Carve a connected mask of about the expected size out of the likely area.

    Growth follows the score downhill from the best seed, so the outline is shaped by
    the evidence rather than inherited from whatever the detector happened to draw.
    """
    if not free.any() or target_area <= 0:
        return None
    flat = int(np.argmax(np.where(free, score, -np.inf)))
    seed = np.unravel_index(flat, score.shape)
    if score[seed] < float(params["minimum_seed_score"]):
        return None
    floor_score = float(params["minimum_pixel_score"])
    chosen = np.zeros(score.shape, bool)
    visited = np.zeros(score.shape, bool)
    heap: list[tuple[float, int, int]] = [(-float(score[seed]), int(seed[0]), int(seed[1]))]
    visited[seed] = True
    height, width = score.shape
    while heap and int(chosen.sum()) < target_area:
        negative, y, x = heapq.heappop(heap)
        if -negative < floor_score:
            break
        chosen[y, x] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] \
                        and free[ny, nx]:
                    visited[ny, nx] = True
                    heapq.heappush(heap, (-float(score[ny, nx]), ny, nx))
    return chosen if chosen.any() else None


def local_contrast(mask: np.ndarray, raw_frame: np.ndarray, params: dict
                   ) -> tuple[float, float]:
    """How far the proposed outline stands above the film immediately around it.

    A real cell is a local peak. This is what stops a recovery being sculpted out of
    noise in an empty part of the field.
    """
    ring = ndi.binary_dilation(mask, iterations=int(params["contrast_ring_px"])) & ~mask
    inside = float(np.median(raw_frame[mask])) if mask.any() else 0.0
    around = float(np.median(raw_frame[ring])) if ring.any() else 0.0
    return inside, around


def _recovery_reason(mask: np.ndarray, state: Descriptor, raw_frame: np.ndarray,
                     new_mask: np.ndarray, context: dict, params: dict
                     ) -> tuple[bool, str, dict]:
    area = int(mask.sum())
    mean = float(raw_frame[mask].mean()) if area else 0.0
    yy, xx = np.nonzero(mask)
    centre = np.array([yy.mean(), xx.mean()], float)
    distance = float(np.hypot(centre[0] - context["predicted_y"],
                              centre[1] - context["predicted_x"]))
    area_ratio = area / max(state.area, 1.0)
    mean_change = abs(float(np.log2((mean + 1.0) / (state.mean + 1.0))))
    median_intensity, surround = local_contrast(mask, raw_frame, params)
    new_fraction = float(new_mask.sum() / max(area, 1))
    measures = {
        "recovered_area_px": area, "recovered_mean_intensity": mean,
        "recovered_median_intensity": median_intensity,
        "surrounding_median_intensity": surround,
        "new_foreground_fraction": new_fraction,
        "recovered_y": float(centre[0]), "recovered_x": float(centre[1]),
        "area_ratio": float(area_ratio), "mean_log2_change": mean_change,
        "centre_offset_px": distance,
    }
    # A carve sits inside signal another identity was already holding, so only an
    # outline made mostly of new foreground has to prove it is not empty film.
    if new_fraction > 0.5:
        if median_intensity < float(params["_background_level"]):
            return False, "no_brighter_than_background", measures
        if median_intensity < float(params["local_contrast"]) * surround:
            return False, "no_local_contrast", measures
    if area_ratio < float(params["minimum_recovered_area_ratio"]):
        return False, "too_small_for_this_identity", measures
    if area_ratio > float(params["maximum_recovered_area_ratio"]):
        return False, "too_large_for_this_identity", measures
    if mean_change > float(params["maximum_mean_log2_change"]):
        return False, "brightness_does_not_match", measures
    if distance > context["search_radius_px"]:
        return False, "outside_the_movement_search_area", measures
    return True, "recovered", measures


def _background_level(labels: np.ndarray, raw: np.ndarray, params: dict) -> float:
    """How bright this movie's empty space is, so a recovery cannot be made of noise."""
    empty = raw[labels == 0]
    if not empty.size:
        return float(params["minimum_intensity"])
    level = float(np.percentile(empty, float(params["background_percentile"])))
    return max(level * float(params["background_multiple"]),
               float(params["minimum_intensity"]))


def _maximum_identity_area(labels: np.ndarray, params: dict) -> float:
    """An outline several times the size of a typical cell is already holding more
    than one. This layer reports it instead of growing it further."""
    areas: list[int] = []
    for identity, ts in _identity_frames(labels).items():
        areas.append(int(np.median([int(np.count_nonzero(labels[t] == identity))
                                    for t in ts])))
    if not areas:
        return float("inf")
    return float(np.median(areas)) * float(params["maximum_identity_area_ratio"])


def resolve_minimum_persistent_area(labels: np.ndarray, params: dict) -> float:
    """Resolve the small-cell persistence floor without naming a cell or movie."""
    ratio = params.get("minimum_persistent_area_ratio_to_movie_median")
    if ratio is None:
        return float(params["minimum_persistent_area_px"])
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError(
            "minimum_persistent_area_ratio_to_movie_median must be positive")
    identity_medians = []
    for identity, ts in _identity_frames(labels).items():
        identity_medians.append(float(np.median([
            int(np.count_nonzero(labels[t] == identity)) for t in ts])))
    if not identity_medians:
        return float(params["minimum_persistent_area_floor_px"])
    typical = float(np.median(identity_medians))
    return max(float(params["minimum_persistent_area_floor_px"]),
               ratio * typical)


def _settle_carves(mask: np.ndarray, frame_labels: np.ndarray, params: dict
                   ) -> tuple[np.ndarray, str]:
    """Keep only the carves a host can survive: it stays whole and above its floor.

    The host keeps its largest remaining piece. Anything the cut strands is handed to the
    cell doing the carving, so no pixel is ever orphaned and no host is ever split.
    """
    minimum_host = int(params["minimum_host_area_px"])
    structure = np.ones((3, 3), int)
    refused: list[str] = []
    result = mask.copy()
    for host in np.unique(frame_labels[result]):
        host = int(host)
        if host <= 0:
            continue
        held = frame_labels == host
        remainder = held & ~result
        components, count = ndi.label(remainder, structure=structure)
        if count == 0:
            result &= ~held
            refused.append(str(host))
            continue
        sizes = np.bincount(components.ravel())[1:]
        keep = components == (int(np.argmax(sizes)) + 1)
        if int(keep.sum()) < minimum_host:
            result &= ~held
            refused.append(str(host))
            continue
        result |= remainder & ~keep
    return result, ";".join(refused)


def complete_presence(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                      params: dict | None = None
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Give every identity a located mask in every frame from its birth to the end.

    Returns the completed identity movie, a boolean stack marking recovered pixels, and
    one row per attempted recovery, whether it succeeded or not.
    """
    settings = dict(DEFAULTS)
    settings.update(params or {})
    if labels.shape != raw.shape:
        raise ValueError("identity movie and raw movie differ in shape")
    settings.setdefault("_background_level", _background_level(labels, raw, settings))
    settings.setdefault("_maximum_identity_area",
                        _maximum_identity_area(labels, settings))
    settings.setdefault("_minimum_persistent_area",
                        resolve_minimum_persistent_area(labels, settings))
    frames = len(labels)
    output = labels.copy()
    recovered = np.zeros(labels.shape, bool)
    present = _identity_frames(labels)
    births = {identity: ts[0] for identity, ts in present.items()}
    deaths = {identity: ts[-1] for identity, ts in present.items()}
    memory = int(settings["descriptor_memory"])
    border_px = int(settings["border_px"])
    states: dict[int, Descriptor] = {}
    reestablishment_buffers: dict[int, list[dict]] = {}
    rows: list[dict] = []
    forced_ids = {int(value) for value in settings.get(
        "forced_identity_ids", [])}
    forced_intervals: dict[int, list[tuple[int, int]]] = {}
    for interval in settings.get("forced_intervals", []):
        forced_intervals.setdefault(int(interval["identity"]), []).append(
            (int(interval["start_t"]), int(interval["end_t"])))

    minimum_original_gap = settings.get("minimum_original_gap_frames")
    maximum_original_gap = settings.get("maximum_original_gap_frames")
    allowed_original_gap: dict[tuple[int, int], bool] = {}
    if minimum_original_gap is not None or maximum_original_gap is not None:
        minimum_original_gap = (1 if minimum_original_gap is None
                                else int(minimum_original_gap))
        maximum_original_gap = (frames if maximum_original_gap is None
                                else int(maximum_original_gap))
        if minimum_original_gap < 1 or maximum_original_gap < 1:
            raise ValueError("original gap frame limits must be positive or null")
        if minimum_original_gap > maximum_original_gap:
            raise ValueError(
                "minimum_original_gap_frames cannot exceed the maximum")
        for identity, observed_frames in present.items():
            observed_set = set(observed_frames)
            first, last = observed_frames[0], observed_frames[-1]
            missing = np.asarray(
                [t not in observed_set for t in range(first, last + 1)], bool)
            padded = np.concatenate(([False], missing, [False])).astype(np.int8)
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            for start, end in zip(starts, ends):
                length = int(end - start + 1)
                accepted = minimum_original_gap <= length <= maximum_original_gap
                for relative_t in range(int(start), int(end) + 1):
                    allowed_original_gap[(identity, first + relative_t)] = accepted

    for t in range(frames):
        # Presence uses mask, centroid, area, and intensity only. Tracking's
        # covariance/perimeter descriptor is intentionally skipped here.
        observations = frame_observations(
            labels[t], raw[t], include_shape=False)
        for identity, observation in observations.items():
            state = states.get(identity)
            if state is None:
                state = Descriptor(identity, t, observation["mask"].copy(),
                                   observation["position"].copy(), np.zeros(2, float))
                states[identity] = state
            reestablished = False
            observed_area = int(np.count_nonzero(observation["mask"]))
            minimum_area = float(settings["_minimum_persistent_area"])
            if (bool(settings.get("allow_descriptor_reestablishment", False))
                    and (not forced_ids or identity in forced_ids)
                    and (not forced_intervals
                         or identity in forced_intervals)
                    and state.areas and state.area < minimum_area
                    and observed_area >= minimum_area):
                buffer = reestablishment_buffers.setdefault(identity, [])
                if buffer and int(buffer[-1]["t"]) != t - 1:
                    buffer.clear()
                buffer.append({
                    "t": int(t), "mask": observation["mask"].copy(),
                    "position": observation["position"].copy(),
                    "area": observed_area,
                    "mean": float(raw[t][observation["mask"]].mean()),
                })
                required = int(settings[
                    "descriptor_reestablishment_observations"])
                del buffer[:-required]
                if len(buffer) == required:
                    areas = np.asarray([item["area"] for item in buffer], float)
                    positions = np.asarray(
                        [item["position"] for item in buffer], float)
                    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    consistent_area = bool(
                        float(areas.max() / max(areas.min(), 1.0))
                        <= float(settings[
                            "descriptor_reestablishment_area_ratio"]))
                    consistent_motion = bool(
                        not len(steps) or float(steps.max()) <= float(settings[
                            "descriptor_reestablishment_maximum_step_px"]))
                    if consistent_area and consistent_motion:
                        first, last = buffer[0], buffer[-1]
                        elapsed = max(int(last["t"]) - int(first["t"]), 1)
                        velocity = ((last["position"] - first["position"])
                                    / elapsed)
                        state = Descriptor(
                            identity, t, last["mask"].copy(),
                            last["position"].copy(), velocity)
                        state.areas = [int(item["area"]) for item in buffer]
                        state.means = [float(item["mean"]) for item in buffer]
                        states[identity] = state
                        rows.append({
                            "t": t, "imagej_frame": t + 1,
                            "identity": identity,
                            "outcome": "descriptor_reestablished",
                            "reestablishment_observations": required,
                            "expected_area_px": state.area,
                            "expected_mean_intensity": state.mean,
                            "endpoint_velocity_y": float(velocity[0]),
                            "endpoint_velocity_x": float(velocity[1]),
                        })
                        reestablished = True
                        buffer.clear()
            elif observed_area < minimum_area:
                reestablishment_buffers.pop(identity, None)
            if not reestablished:
                state.observe(
                    t, observation["mask"], raw[t], memory,
                    position=observation["position"],
                    area=observation["area"], mean=observation["mean"])

        due = [state for identity, state in states.items()
               if births[identity] <= t and identity not in observations
               and (minimum_original_gap is None and maximum_original_gap is None
                    or allowed_original_gap.get((identity, t), False))
               and (not forced_ids or identity in forced_ids)
               and (not forced_intervals or any(
                    start <= t <= end
                    for start, end in forced_intervals.get(identity, [])))
               and (not bool(settings["limit_to_observed_lifetime"])
                    or t <= deaths[identity])
               and not state.retired]
        if not due:
            continue

        expected_area = {identity: state.area for identity, state in states.items()}
        proposals: list[tuple[float, int, np.ndarray, dict, dict]] = []
        for state in sorted(due, key=lambda row: row.identity):
            state.stale += 1
            if (state.area < float(settings["_minimum_persistent_area"])
                    or len(state.areas) < int(
                        settings["minimum_persistent_observations"])):
                rows.append({
                    "t": t, "imagej_frame": t + 1,
                    "identity": state.identity,
                    "outcome": "small_or_unestablished_identity_not_forced",
                    "gap_steps": t - state.t,
                    "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean,
                })
                continue
            persistence_ceiling = settings.get(
                "maximum_persistent_area_px_exclusive")
            if (persistence_ceiling is not None
                    and state.area >= float(persistence_ceiling)):
                rows.append({
                    "t": t, "imagej_frame": t + 1,
                    "identity": state.identity,
                    "outcome": "outside_supplementary_area_band",
                    "gap_steps": t - state.t,
                    "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean,
                })
                continue
            if state.area > float(settings["_maximum_identity_area"]):
                rows.append({
                    "t": t, "imagej_frame": t + 1, "identity": state.identity,
                    "outcome": "identity_too_large_to_be_one_cell",
                    "gap_steps": t - state.t, "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean,
                })
                continue
            found = likely_area(
                state, t, raw, lag, output[t], expected_area, settings,
                output[t - 1] if t > 0 else None)
            if found is None:
                left_field = (_touches_border(state.mask, border_px)
                              and bool(settings["allow_field_exit"]))
                rows.append({
                    "t": t, "imagej_frame": t + 1, "identity": state.identity,
                    "outcome": "field_exit" if left_field
                    else "unresolved_persistent_claim_no_candidate_area",
                    "gap_steps": t - state.t, "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean,
                })
                if left_field:
                    state.retired = "field_exit"
                continue
            (window_rows, window_cols), score, free, _, context = found
            local = sculpt(score, free, int(round(state.area)), settings)
            if local is None:
                left_field = (_touches_border(state.mask, border_px)
                              and bool(settings["allow_field_exit"]))
                rows.append({
                    "t": t, "imagej_frame": t + 1, "identity": state.identity,
                    "outcome": "field_exit" if left_field
                    else "unresolved_persistent_claim_no_candidate_pixels",
                    "gap_steps": t - state.t, "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean, **context,
                })
                if left_field:
                    state.retired = "field_exit"
                continue
            mask = np.zeros(labels.shape[1:], bool)
            mask[window_rows, window_cols] = local
            quality = float(np.mean(score[local]))
            proposals.append((quality, state.identity, mask, context, {"state": state}))

        # Highest-confidence recoveries claim their pixels first, so one pixel can never
        # belong to two identities.
        for quality, identity, mask, context, holder in sorted(
                proposals, key=lambda row: (-row[0], row[1])):
            state = holder["state"]
            mask = mask & ~recovered[t]
            mask, refused = _settle_carves(mask, output[t], settings)
            context = {**context, "carved_px": int(np.count_nonzero(
                mask & (labels[t] > 0))), "hosts_refused": refused}
            if not mask.any():
                rows.append({
                    "t": t, "imagej_frame": t + 1, "identity": identity,
                    "outcome": "area_taken_by_a_closer_identity",
                    "gap_steps": t - state.t, "last_imagej_frame": state.t + 1,
                    "expected_area_px": state.area,
                    "expected_mean_intensity": state.mean, **context,
                })
                continue
            accepted, reason, measures = _recovery_reason(
                mask, state, raw[t], mask & (labels[t] == 0), context, settings)
            row = {
                "t": t, "imagej_frame": t + 1, "identity": identity,
                "outcome": reason, "gap_steps": t - state.t,
                "last_imagej_frame": state.t + 1,
                "expected_area_px": state.area,
                "expected_mean_intensity": state.mean,
                "candidate_score": quality, **context, **measures,
            }
            if accepted:
                output[t][mask] = identity
                recovered[t][mask] = True
                state.observe(t, mask, raw[t], memory)
            elif (_touches_border(state.mask, border_px)
                  and bool(settings["allow_field_exit"])):
                row["outcome"] = "field_exit"
                state.retired = "field_exit"
            rows.append(row)

    table = pd.DataFrame(rows)
    _validate(output, labels, recovered)
    return output, recovered, table


def complete_presence_bidirectional(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        params: dict | None = None,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Keep established identities open in both directions from their evidence.

    The forward pass prevents death after an established cell is observed. The
    time-reversed pass prevents an unmatched early observation from being called a
    genuine birth merely because tracking began near the movie midpoint.
    """
    forward, forward_recovered, forward_rows = complete_presence(
        labels, raw, lag, params)
    reverse_params = dict(params or {})
    if reverse_params.get("forced_intervals"):
        reverse_params["forced_intervals"] = [{
            "identity": int(interval["identity"]),
            "start_t": len(labels) - 1 - int(interval["end_t"]),
            "end_t": len(labels) - 1 - int(interval["start_t"]),
        } for interval in reverse_params["forced_intervals"]]
    reversed_labels = forward[::-1]
    reversed_raw = raw[::-1]
    reversed_lag = -lag[::-1]
    backward_reversed, backward_recovered_reversed, backward_rows = \
        complete_presence(reversed_labels, reversed_raw, reversed_lag,
                          reverse_params)
    output = backward_reversed[::-1]
    recovered = forward_recovered | backward_recovered_reversed[::-1]
    forward_rows = forward_rows.copy()
    if len(forward_rows):
        forward_rows.insert(0, "direction", "forward")
    backward_rows = backward_rows.copy()
    if len(backward_rows):
        backward_rows["t"] = len(labels) - 1 - backward_rows["t"].astype(int)
        backward_rows["imagej_frame"] = backward_rows["t"] + 1
        backward_rows.insert(0, "direction", "backward")
    attempts = pd.concat(
        [forward_rows, backward_rows], ignore_index=True, sort=False)
    return output, recovered, attempts


def _validate(output: np.ndarray, labels: np.ndarray, recovered: np.ndarray) -> None:
    if not np.array_equal(output[~recovered], labels[~recovered]):
        raise AssertionError("presence completion altered a pixel it did not claim")
    if np.any(recovered & (output == 0)):
        raise AssertionError("a claimed pixel carries no identity")
    for t in range(len(labels)):
        before = set(map(int, np.unique(labels[t]))) - {0}
        after = set(map(int, np.unique(output[t]))) - {0}
        if not before <= after:
            raise AssertionError(
                f"frame {t}: carving removed identities {sorted(before - after)}")


def presence_report(labels: np.ndarray, border_px: int = 8) -> pd.DataFrame:
    """One row per identity: does it survive to the end of the movie, and if not, why."""
    frames = len(labels)
    present = _identity_frames(labels)
    rows: list[dict] = []
    for identity, ts in sorted(present.items()):
        first, last = ts[0], ts[-1]
        gaps = sorted(set(range(first, last + 1)) - set(ts))
        exits = _touches_border(labels[last] == identity, border_px)
        rows.append({
            "identity": identity,
            "first_imagej_frame": first + 1, "last_imagej_frame": last + 1,
            "observed_frames": len(ts),
            "required_frames": frames - first,
            "interior_gap_frames": len(gaps),
            "frames_missing_after_last": frames - 1 - last,
            "ends_at_border": bool(exits),
            "complete": bool(not gaps and (last == frames - 1 or exits)),
        })
    return pd.DataFrame(rows)
