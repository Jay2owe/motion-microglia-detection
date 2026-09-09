"""Conservative whole-episode ownership repair; never suppress detections.

Physical reference numbers and labels are evidence values only. Discovery is
field-wide; this producer accepts no selectors. It reuses the frozen prior
discovery, but not its component-erasure reconciliation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile

import component_accounting as accounting
import terminal_physical_core_lineage_discovery as discovery
from raw_physical_hypotheses import evidence_image

STRUCTURE = np.ones((3, 3), np.uint8)
EXTRA = {
    'minimum_reliable_anchor_movie_fraction',
    'minimum_displaced_body_movie_fraction',
    'minimum_displaced_history_movie_fraction',
    'minimum_displaced_core_coverage',
}


def check_params(params):
    discovery.assert_target_free({k: v for k, v in params.items() if k not in EXTRA})
    for key in EXTRA:
        if not 0 <= float(params[key]) <= 1:
            raise ValueError(f'{key} must be in [0,1]')


def _inside(mask, row):
    return discovery._inside(mask, row)


def supported_raw_core(raw, row):
    """Accepted detector evidence at an observed core, never a predicted seat.

    Use this core's peak, not a possibly brighter neighbouring host's peak.
    Retain the accepted 35%-peak/half-weak evidence threshold and radius bounds.
    """
    if row.state != 'observed':
        return None
    evidence = evidence_image(raw, {})
    positive = evidence[evidence > 0]
    if not len(positive):
        return None
    weak = float(np.median(positive))
    radius = max(float(row.radius_px), 1.)
    half = max(7, int(math.ceil(2 * radius)))
    cx, cy = int(round(row.x)), int(round(row.y))
    region = (slice(max(0,cy-half), min(raw.shape[0],cy+half+1)),
              slice(max(0,cx-half), min(raw.shape[1],cx+half+1)))
    local = evidence[region]
    yy, xx = np.indices(local.shape)
    distance2 = (xx + region[1].start - row.x)**2 + (yy + region[0].start - row.y)**2
    near = (distance2 <= (.75 * radius)**2) & (raw[region] > 0)
    if not np.any(near):
        return None
    py, px = np.unravel_index(np.argmax(np.where(near, local, -np.inf)), local.shape)
    peak = float(local[py,px])
    if peak < weak:
        return None
    components, _ = ndi.label((local >= max(.35*peak, .5*weak)) & (raw[region] > 0), STRUCTURE)
    number = int(components[py,px])
    if not number:
        return None
    local_mask = components == number
    area = int(local_mask.sum())
    # Tracking radii are clipped to >=2 px, even for smaller raw cores. That
    # tracking floor must not become an invented segmentation-area minimum.
    if not 2 <= area <= 4*math.pi*radius**2:
        return None
    core = np.zeros(raw.shape, bool)
    core[region] = local_mask
    return core


def keep_host_fragments_detected(before, trial, recovered, anchor, frame_points, canonical):
    """Reconcile a cut-off host tip without deleting any foreground.

    Keep every remainder containing a host witness, or the largest remainder
    when the detector has no witness. Only an unseeded, smaller remainder
    actually touching the recovered body can transfer with it.
    """
    for owner in np.unique(before[recovered]):
        owner = int(owner)
        if owner <= 0 or owner == anchor:
            continue
        parts, count = ndi.label(trial == owner, STRUCTURE)
        old_count = ndi.label(before == owner, STRUCTURE)[1]
        if count <= old_count:
            continue
        retained = set()
        for point in frame_points.itertuples(index=False):
            stable = canonical.get(int(point.track_id))
            if stable is not None and int(stable['owner']) == owner:
                number = discovery.physical._component_at(parts, point.x, point.y, search_radius=0)
                if number > 0:
                    retained.add(number)
        if not retained:
            retained.add(int(np.argmax(np.bincount(parts.ravel())[1:]))+1)
        for number in range(1, count+1):
            fragment = parts == number
            if number not in retained and np.any(ndi.binary_dilation(recovered, structure=STRUCTURE) & fragment):
                trial[fragment] = anchor
    return trial


def displacement_options(labels, raw, scored, proposal, params):
    """Find independently witnessed long-lived bodies occupying the anchor.

    A body's first supported history must have a different positive owner.
    This rejects new core observations born inside the anchor's own projection.
    Allocation is explicit, stable and to named foreground only.
    """
    anchor = int(proposal['anchor_owner'])
    last = int(proposal['anchor_last'])
    group = proposal['group'].set_index('frame')
    minimum_history = max(2, math.ceil(len(labels) * params['minimum_displaced_history_movie_fraction']))
    minimum_span = max(2, math.ceil(len(labels) * params['minimum_displaced_body_movie_fraction']))
    options = []
    for track, history in scored[scored.physically_visible].groupby('track_id'):
        if int(track) in set(map(int, str(proposal['lineage_tracks']).split('|'))):
            continue
        history = history.sort_values('frame')
        early = history.iloc[:minimum_history]
        if len(early) < minimum_history:
            continue
        owner, support, purity = discovery.physical._mode(early.accepted_owner.tolist())
        if owner <= 0 or owner == anchor or support < math.ceil(.8 * minimum_history) or purity < .8:
            continue
        invaded = history[(history.frame > last) & (history.accepted_owner == anchor)]
        if len(invaded) < minimum_span:
            continue
        evidence = []
        parts = {}
        for row in invaded.itertuples(index=False):
            f = int(row.frame)
            if f not in group.index:
                continue
            reference = group.loc[f]
            component = discovery.relay._component_at_owner(labels[f], anchor, row)
            if component is None or _inside(component, reference):
                continue
            separation = float(np.hypot(row.x - reference.x, row.y - reference.y) /
                               max(row.radius_px + reference.radius_px, 1.))
            valley = discovery.physical._valley_ratio(raw[f], (row.x, row.y),
                                                     (reference.x, reference.y))
            evidence.append(separation >= params['minimum_core_separation_sum_radii'] and
                            valley <= params['maximum_separable_valley_ratio'])
            parts[f] = component
        coverage = len(parts) / max(len(invaded), 1)
        if len(parts) < minimum_span or coverage < params['minimum_displaced_core_coverage']:
            continue
        if np.mean(evidence) < params['minimum_displaced_core_coverage']:
            continue
        options.append(dict(track=int(track), previous_owner=int(owner), parts=parts,
                            frames=len(parts), separation_support=float(np.mean(evidence))))
    return options


def partition_body(plane, raw, row, reference_tracks, by_frame, canonical, params):
    """Partition only already named pixels; independently supported hosts stay."""
    owner = discovery.physical._disk_owner(plane, row.x, row.y, row.radius_px)
    if owner <= 0:
        return None, dict(reason='no_named_detection_at_reference')
    shared = discovery.relay._component_at_owner(plane, owner, row)
    if shared is None:
        return None, dict(reason='no_named_component_at_reference')
    companions = []
    for other in by_frame.get(int(row.frame), pd.DataFrame()).itertuples(index=False):
        stable = canonical.get(int(other.track_id))
        if int(other.track_id) in reference_tracks or stable is None or int(stable['owner']) != owner:
            continue
        if not _inside(shared, other):
            continue
        separation = np.hypot(row.x - other.x, row.y - other.y) / max(row.radius_px + other.radius_px, 1.)
        if separation < params['minimum_core_separation_sum_radii']:
            continue
        # A reference born after the takeover can simply describe the same
        # reassigned projection. It is not independent prior host evidence.
        if int(stable['first_frame']) > int(params['_episode_start']):
            continue
        companions.append(other)
    detail = dict(source_owner=int(owner), source_area=int(shared.sum()),
                  host_cores='|'.join(str(int(r.track_id)) for r in companions))
    if not companions:
        ratio = shared.sum() / (math.pi * max(float(row.radius_px), 1.) ** 2)
        expected_area = float(params['_reference_body_area'])
        reference_support = float(np.count_nonzero(shared & params['_reference_footprint'])) / max(int(shared.sum()), 1)
        shape_supported = (shared.sum() <= 2 * expected_area and reference_support >= .5)
        if ratio > params['maximum_component_area_radius_ratio'] and not shape_supported:
            return None, {**detail, 'reason': 'large_shared_body_without_independent_host'}
        return shared, {**detail, 'reason': 'separate_body_relabel'}
    markers = np.zeros(plane.shape, np.int32)
    marker = discovery.physical._marker_pixel(shared, row.x, row.y)
    markers[marker] = 1
    valleys = []
    for other in companions:
        host_marker = discovery.physical._marker_pixel(shared, other.x, other.y)
        if host_marker is not None and host_marker != marker:
            markers[host_marker] = 2
            valleys.append(discovery.physical._valley_ratio(raw, (row.x, row.y), (other.x, other.y)))
    if not valleys or min(valleys) > params['maximum_separable_valley_ratio']:
        return None, {**detail, 'reason': 'insufficient_raw_separation'}
    elevation = -ndi.gaussian_filter(raw.astype(np.float32), params['watershed_sigma_px'])
    recovered = watershed(elevation, markers=markers, mask=shared) == 1
    # Watershed must account for every host fragment as well as the target.
    # A separated remainder with no host seed is attached to its adjacent
    # target partition, never dropped or left as a duplicate host label.
    host_parts, host_count = ndi.label(shared & ~recovered, STRUCTURE)
    for part in range(1, host_count + 1):
        fragment = host_parts == part
        if not np.any(fragment & (markers == 2)):
            if np.any(ndi.binary_dilation(recovered, structure=STRUCTURE) & fragment):
                recovered |= fragment
    if recovered.sum() < params['minimum_changed_pixels']:
        return None, {**detail, 'reason': 'partition_below_minimum_body_area'}
    return recovered, {**detail, 'reason': 'two_body_foreground_partition',
                       'valley_ratio': float(min(valleys))}


def produce(labels, unclaimed, raw, points, params):
    check_params(params)
    discovery_params = {k: v for k, v in params.items() if k not in EXTRA}
    audit, proposals, scored, links = discovery.discover(labels, points, discovery_params)
    visible = scored[scored.physically_visible]
    by_frame = {int(f): g for f, g in visible.groupby('frame')}
    canonical = discovery.physical.canonical_owners(visible, max(2, math.ceil(len(labels)*.05)), .8, False)
    for track, stable in canonical.items():
        stable['first_frame'] = int(visible[visible.track_id == track].frame.min())
    # Physical-core detector restarts do not erase established host history.
    # Reuse only the same mutually unique scale-normalised endpoint links used
    # by target discovery, and only when both ends agree on the host owner.
    for _ in range(len(links)):
        advanced = False
        for edge in links:
            left, right = canonical.get(int(edge['left'])), canonical.get(int(edge['right']))
            if left is not None and right is not None and left['owner'] == right['owner']:
                first = min(left['first_frame'], right['first_frame'])
                if first < right['first_frame']:
                    right['first_frame'] = first
                    advanced = True
        if not advanced:
            break
    candidate = labels.copy()
    candidate_ledger = unclaimed.copy()
    next_id = int(labels.max()) + 1
    frame_audit = []
    allocation_audit = []
    min_anchor = math.ceil(len(labels) * params['minimum_reliable_anchor_movie_fraction'])
    for prop in proposals:
        if not prop['eligible']:
            continue
        if int(prop['anchor_run_frames']) < min_anchor:
            frame_audit.append(dict(proposal=prop['proposal_id'], anchor=prop['anchor_owner'],
                                    frame=-1, applied=False, reason='insufficient_reliable_initial_history'))
            continue
        anchor = int(prop['anchor_owner'])
        last = int(prop['anchor_last'])
        trial = candidate.copy()
        trial_ledger = candidate_ledger.copy()
        displacement = displacement_options(labels, raw, scored, prop, params)
        allocated = []
        for option in displacement:
            previous = int(option['previous_owner'])
            frames = sorted(option['parts'])
            previous_free = all(not np.any(trial[f] == previous) for f in frames)
            destination = previous if previous_free else next_id
            if not previous_free:
                next_id += 1
            for f, mask in option['parts'].items():
                trial[f][mask] = destination
            allocated.append(dict(proposal=prop['proposal_id'], anchor=anchor,
                physical_track=option['track'], previous_owner=previous,
                destination=destination, new_identity=not previous_free,
                first_frame=min(frames), last_frame=max(frames), frames=len(frames),
                separation_support=option['separation_support']))
        reference_rows = prop['group'][(prop['group'].frame <= last) &
                                      (prop['group'].accepted_owner == anchor)]
        reference_masks = []
        for ref in reference_rows.itertuples(index=False):
            mask = discovery.relay._component_at_owner(labels[int(ref.frame)], anchor, ref)
            if mask is not None:
                reference_masks.append(mask)
        reference_footprint = np.mean(reference_masks, axis=0) >= .25
        local_params = {**params, '_episode_start': last,
                        '_reference_body_area': float(np.median([m.sum() for m in reference_masks])),
                        '_reference_footprint': reference_footprint}
        reference_tracks = set(map(int, str(prop['lineage_tracks']).split('|')))
        local_audit = []
        visible_frames = prop['group'].loc[prop['group'].physically_visible, 'frame'].to_numpy(int)
        for row in prop['group'].itertuples(index=False):
            f = int(row.frame)
            if f <= last:
                continue
            bracketed = False
            if not row.physically_visible:
                left, right = visible_frames[visible_frames < f], visible_frames[visible_frames > f]
                bracketed = (len(left) > 0 and len(right) > 0 and
                             int(right[0]-left[-1]-1) <= math.ceil(.04*len(labels)))
                if not bracketed:
                    continue
                # A bounded missing reference may relabel a small *existing*
                # detection; it cannot introduce new foreground or split a
                # large neighbour on the strength of a prediction.
                owner = discovery.physical._disk_owner(labels[f], row.x, row.y, row.radius_px)
                mask = discovery.relay._component_at_owner(labels[f], owner, row) if owner > 0 else None
                if mask is None or mask.sum() > 2*local_params['_reference_body_area']:
                    local_audit.append(dict(proposal=prop['proposal_id'], anchor=anchor, frame=f,
                        applied=False, recovered_pixels=0, reason='bracketed_prediction_has_no_small_existing_body'))
                    continue
            recovered, detail = partition_body(labels[f], raw[f], row, reference_tracks,
                                               by_frame, canonical, local_params)
            detail.update(proposal=prop['proposal_id'], anchor=anchor, frame=f,
                          applied=False, recovered_pixels=0)
            if recovered is None:
                raw_core = supported_raw_core(raw[f], row)
                if raw_core is not None:
                    if np.any(raw_core & (trial[f] != labels[f]) & (trial[f] != anchor)):
                        detail['raw_recovery'] = 'conflicts_with_displaced_body'
                    else:
                        recovered = raw_core
                        detail['reason'] = 'observed_raw_core_recovery'
            if recovered is not None:
                frame_trial = trial[f].copy()
                frame_trial[recovered] = anchor
                frame_trial = keep_host_fragments_detected(labels[f], frame_trial, recovered, anchor,
                                                           by_frame.get(f, pd.DataFrame()), canonical)
                if accounting.new_duplicate_components(labels[f:f+1], frame_trial[None]) > 0:
                    detail['reason'] = 'would_create_duplicate_without_dropping_foreground'
                else:
                    trial[f] = frame_trial
                    trial_ledger[f][recovered] = 0
                    detail.update(applied=True, recovered_pixels=int(recovered.sum()))
            local_audit.append(detail)
        if any(row['applied'] for row in local_audit):
            candidate = trial
            candidate_ledger = trial_ledger
            allocation_audit.extend(allocated)
        frame_audit.extend(local_audit)
    assert not np.any((labels > 0) & (candidate == 0)), 'Named foreground suppressed'
    reclaimed = (unclaimed > 0) & (candidate_ledger == 0)
    assert not np.any(reclaimed & (candidate == 0)), 'Ledger removed without detected replacement'
    assert np.array_equal(candidate_ledger[~reclaimed], unclaimed[~reclaimed])
    duplicates = accounting.new_duplicate_components(labels, candidate)
    assert duplicates == 0, f'{duplicates} new duplicate components'
    summary = dict(targeting_mode='field_wide_discovery',
        target_counts={k: 0 for k in ['identities','tracks','frames','coordinates','regions','wells']},
        changed_label_pixels=int((candidate != labels).sum()),
        changed_frames=int(np.any(candidate != labels, axis=(1,2)).sum()),
        suppressed_foreground_pixels=int(((labels > 0) & (candidate == 0)).sum()),
        added_foreground_pixels=int(((labels == 0) & (candidate > 0)).sum()),
        named_foreground_exact=bool(np.array_equal(candidate > 0, labels > 0)),
        named_foreground_preserved=True, unclaimed_exact=bool(np.array_equal(candidate_ledger, unclaimed)),
        reclaimed_unclaimed_pixels=int(reclaimed.sum()), new_duplicate_components=duplicates,
        new_identities=sorted(map(int, set(np.unique(candidate)) - set(np.unique(labels)))),
        allocations=allocation_audit)
    return candidate, candidate_ledger, audit, pd.DataFrame(frame_audit), pd.DataFrame(allocation_audit), summary


def run(_upstream, params, out):
    labels = tifffile.imread(params['labels_path'])
    unclaimed = tifffile.imread(params['unclaimed_path'])
    raw = tifffile.imread(params['raw_path'])
    points = pd.read_csv(params['physical_track_points_path'])
    candidate, ledger, audit, frames, allocations, summary = produce(labels, unclaimed, raw, points, params)
    folder = Path(out.out)
    folder.mkdir(parents=True, exist_ok=True)
    stem = params['output_stem']
    outputs = dict(labels=folder/f'{stem}.tif', unclaimed=folder/f'{stem}_unclaimed_original_ids.tif',
                   audit=folder/'discovery_audit.csv', frames=folder/'episode_frames.csv',
                   allocations=folder/'displaced_body_allocations.csv', metrics=folder/'producer_metrics.json')
    for key, arr in [('labels', candidate), ('unclaimed', ledger)]:
        tifffile.imwrite(outputs[key], arr, imagej=True, compression='zlib', photometric='minisblack',
                         metadata=dict(axes='TYX', finterval=1800., tunit='sec', unit='pixel'))
    audit.to_csv(outputs['audit'], index=False)
    frames.to_csv(outputs['frames'], index=False)
    allocations.to_csv(outputs['allocations'], index=False)
    outputs['metrics'].write_text(json.dumps(summary, indent=2, default=int)+'\n')
    return dict(outputs=outputs, summary=summary)
