"""Field-wide body continuity with conservative competing-component reassignment.

No identity, track, frame or spatial selectors. Raw boundary partitioning is used
where several detected owners touch the conflict. The early stable body is found
by accepted discovery. Independent observed cores may replace a drifting raw
reference; no prediction or new raw foreground is introduced by this attempt.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile
from skimage.segmentation import watershed
import conservative_episode_ownership as base

STRUCTURE=base.STRUCTURE

def choose_point(points, previous, radius, peak):
    """Choose a strong local core from body continuity, never a track ID."""
    group=points[(points.state=='observed') & (points.evidence>=.25*peak) &
                 (points.radius_px>=.6*radius)].copy()
    if group.empty:
        return None,'no_observed_scale_supported_core'
    group['step']=np.hypot(group.x-previous[0],group.y-previous[1])/(group.radius_px+radius)
    group=group[group.step<=1.25].sort_values(['step','evidence'],ascending=[True,False])
    if group.empty:
        return None,'no_continuous_core'
    best=group.iloc[0]
    if len(group)>1 and float(group.iloc[1].step-best.step)<.20:
        # Close references may be the same raw peak; different peaks are ambiguous.
        other=group.iloc[1]
        if np.hypot(best.x-other.x,best.y-other.y)>.5*(best.radius_px+other.radius_px):
            return None,'ambiguous_continuation'
    return best,'observed_body_continuation'

def historic_owner_options(history, anchor, frame):
    visible=history[history.physically_visible & (history.accepted_owner>0)]
    prior=visible[visible.frame<frame]
    before=prior[prior.accepted_owner!=anchor]
    choices=set()
    if len(before)>=3:
        recent=before.tail(5)
        owner,count,purity=base.discovery.physical._mode(recent.accepted_owner.tolist())
        if count>=3 and purity>=.6:
            choices.add(owner)
    nonanchor=visible[visible.accepted_owner!=anchor]
    if len(nonanchor)>=5:
        owner,count,purity=base.discovery.physical._mode(nonanchor.accepted_owner.tolist())
        if count>=5 and purity>=.8:
            choices.add(owner)
    return choices

def temporal_owner_evidence(stack, frame, mask, anchor):
    """Unchanged parent evidence, not votes from the proposed correction."""
    evidence={}
    span=max(2,math.ceil(len(stack)*.05))
    for f in range(max(0,frame-span),min(len(stack),frame+span+1)):
        if f==frame:
            continue
        owners,counts=np.unique(stack[f][mask],return_counts=True)
        for owner,count in zip(owners,counts):
            if int(owner)<=0 or int(owner)==anchor:
                continue
            if count/max(int(mask.sum()),1)>=.25:
                item=evidence.setdefault(int(owner),dict(frames=0,support=0.))
                item['frames']+=1
                item['support']+=float(count)/max(int(mask.sum()),1)
    if not evidence:
        return set(),evidence
    maximum=max(item['support'] for item in evidence.values())
    total=sum(item['support'] for item in evidence.values())
    choices={owner for owner,item in evidence.items() if item['frames']>=3 and
             item['support']==maximum and item['support']/total>=.60}
    return choices,evidence


def partition_detected_contact(trial, mask, raw):
    """Partition a former owner boundary among the bodies it actually touches.

    Every transferred pixel is connected through this mask to a pre-existing
    neighbouring owner seed. This does not fill a gap or add/delete foreground.
    """
    ring=ndi.binary_dilation(mask,STRUCTURE)&~mask&(trial>0)
    if not np.any(ring):
        return None
    markers=np.where(ring,trial,0).astype(np.int32)
    assigned=watershed(-ndi.gaussian_filter(raw.astype(np.float32),1.),
                       markers,mask=mask|ring,connectivity=STRUCTURE)
    if not np.all(assigned[mask]>0):
        return None
    result=trial.copy();result[mask]=assigned[mask]
    return result


def supported_partition(plane, raw, row, reference_tracks, by_frame, canonical, params):
    recovered,detail=base.partition_body(plane,raw,row,reference_tracks,by_frame,canonical,params)
    if recovered is not None or detail['reason']!='large_shared_body_without_independent_host':
        return recovered,detail
    owner=base.discovery.physical._disk_owner(plane,row.x,row.y,row.radius_px)
    shared=base.discovery.relay._component_at_owner(plane,owner,row)
    markers=np.zeros(plane.shape,np.int32)
    seed=base.discovery.physical._marker_pixel(shared,row.x,row.y)
    markers[seed]=1
    hosts=[]
    for other in by_frame.get(int(row.frame),pd.DataFrame()).itertuples(index=False):
        if int(other.track_id)==int(row.track_id) or other.state!='observed':
            continue
        if other.evidence<.25*row.evidence or other.radius_px<.5*row.radius_px:
            continue
        if not base._inside(shared,other):
            continue
        group=ALL_HISTORIES[int(other.track_id)]
        if int(group.physically_visible.sum())<math.ceil(.10*len(PARENT_LABELS)):
            continue
        separation=np.hypot(other.x-row.x,other.y-row.y)/(other.radius_px+row.radius_px)
        valley=base.discovery.physical._valley_ratio(raw,(row.x,row.y),(other.x,other.y))
        if separation<params['minimum_core_separation_sum_radii'] or valley>params['maximum_separable_valley_ratio']:
            continue
        host=base.discovery.physical._marker_pixel(shared,other.x,other.y)
        if host!=seed:
            markers[host]=2;hosts.append(int(other.track_id))
    if not hosts:
        return None,{**detail,'reason':'no_independent_observed_host_in_shared_body'}
    recovered=watershed(-ndi.gaussian_filter(raw.astype(np.float32),params['watershed_sigma_px']),
                         markers,mask=shared,connectivity=STRUCTURE)==1
    if recovered.sum()<params['minimum_changed_pixels']:
        return None,{**detail,'reason':'observed_host_partition_too_small'}
    return recovered,{**detail,'reason':'sustained_independent_raw_core_partition','host_cores':'|'.join(map(str,hosts))}


def reconcile_competitors(before, trial, recovered, anchor, points, histories):
    """Resolve substantial competing bodies, not merely a component-count budget.

    Parent temporal overlap supplies a second independent ownership witness.
    No numeric tie-breaking. Remaining disconnected projections are counted and
    force manual review; the strict component gate is not relabelled as passing.
    """
    changes=[];failure=[]
    parts,count=ndi.label(trial==anchor,STRUCTURE)
    frame=int(points.iloc[0].frame)
    for number in range(1,count+1):
        mask=parts==number
        if np.any(mask&recovered):
            continue
        history_options=set();witnesses=[]
        distance=ndi.distance_transform_edt(~mask)
        for row in points.itertuples(index=False):
            cy,cx=round(row.y),round(row.x)
            if 0<=cy<trial.shape[0] and 0<=cx<trial.shape[1] and distance[cy,cx]<=.75*row.radius_px:
                options=historic_owner_options(histories[int(row.track_id)],anchor,frame)
                history_options.update(options)
                if options:
                    witnesses.append(int(row.track_id))
        temporal,temporal_detail=temporal_owner_evidence(PARENT_LABELS,frame,mask,anchor)
        contact=trial[ndi.binary_dilation(mask,STRUCTURE)&~mask]
        neighbors=set(map(int,np.unique(contact[(contact>0)&(contact!=anchor)])))
        supported=history_options&temporal
        if not supported:
            supported=history_options&neighbors
        if not supported and len(temporal)==1:
            supported=temporal
        if not supported and len(neighbors)==1:
            supported=neighbors
        if not supported and len(history_options)==1:
            supported=history_options
        if len(supported)!=1 and len(neighbors)>1:
            partition=partition_detected_contact(trial,mask,PARENT_RAW[frame])
            if partition is not None:
                for destination in np.unique(partition[mask]):
                    changes.append(dict(from_owner=anchor,to_owner=int(destination),
                        pixels=int(np.count_nonzero(mask&(partition==destination))),
                        witnesses='contact_partition',temporal=json.dumps(temporal_detail)))
                trial=partition
                continue
        if len(supported)!=1:
            failure.append(dict(area=int(mask.sum()),history=sorted(history_options),
                                temporal=temporal_detail,neighbors=sorted(neighbors)))
            if mask.sum()>=5 or witnesses:
                return None,changes,failure
            # Retain unresolved sub-display speckles as-is; never erase them.
            continue
        destination=next(iter(supported))
        trial[mask]=destination
        changes.append(dict(from_owner=anchor,to_owner=destination,pixels=int(mask.sum()),
                            witnesses='|'.join(map(str,witnesses)),temporal=json.dumps(temporal_detail)))
    return trial,changes,failure


def produce(labels,ledger,raw,points,params):
    global PARENT_LABELS,PARENT_RAW,ALL_HISTORIES
    PARENT_LABELS=labels
    PARENT_RAW=raw
    base.check_params(params)
    audit,proposals,scored,links=base.discovery.discover(labels,points,{k:v for k,v in params.items() if k not in base.EXTRA})
    visible=scored[scored.physically_visible]
    by_frame={int(f):g for f,g in visible.groupby('frame')}
    histories={int(k):v for k,v in scored.groupby('track_id')}
    ALL_HISTORIES=histories
    canonical=base.discovery.physical.canonical_owners(visible,max(2,math.ceil(len(labels)*.05)),.8,False)
    for track,item in canonical.items():
        item['first_frame']=int(histories[track].frame.min())
    for _ in range(len(links)):
        changed=False
        for link in links:
            left,right=canonical.get(int(link["left"])),canonical.get(int(link["right"]))
            if left and right and left["owner"]==right["owner"] and left["first_frame"]<right["first_frame"]:
                right["first_frame"]=left["first_frame"];changed=True
        if not changed:
            break
    candidate=labels.copy();rows=[];transfers=[]
    for prop in proposals:
        if not prop['eligible'] or prop['anchor_run_frames']<math.ceil(len(labels)*params['minimum_reliable_anchor_movie_fraction']):
            continue
        anchor=int(prop['anchor_owner'])
        # Start after the durable initial run, not the last brief recurrence.
        runs=base.discovery._positive_runs(prop['group'])
        early=next(r for r in runs if r['owner']==anchor and r['first']==prop['anchor_first'])
        last=int(early['last'])
        reference=prop['group'][(prop['group'].frame<=last)&(prop['group'].accepted_owner==anchor)]
        masks=[base.discovery.relay._component_at_owner(labels[int(r.frame)],anchor,r) for r in reference.itertuples(index=False)]
        masks=[m for m in masks if m is not None]
        prior_mask=masks[-1]
        radius=float(reference.radius_px.median());peak=float(reference.evidence.median())
        yy,xx=np.nonzero(prior_mask);previous=(float(xx.mean()),float(yy.mean()))
        local={**params,'_episode_start':last,'_reference_body_area':float(np.median([m.sum() for m in masks])),
               '_reference_footprint':np.mean(masks,axis=0)>=.25}
        reference_tracks=set(map(int,str(prop['lineage_tracks']).split('|')))
        for f in range(last+1,len(labels)):
            frame_points=by_frame.get(f,visible.iloc[0:0])
            row,reason=choose_point(frame_points,previous,radius,peak)
            detail=dict(proposal=prop['proposal_id'],anchor=anchor,frame=f,applied=False,reason=reason)
            if row is None:
                rows.append(detail);continue
            detail.update(reference_track=int(row.track_id),x=float(row.x),y=float(row.y))
            recovered,partition=supported_partition(labels[f],raw[f],row,reference_tracks|{int(row.track_id)},by_frame,canonical,local)
            detail.update(partition)
            if recovered is None:
                rows.append(detail);continue
            # Follow the detected body even when ownership reconciliation fails.
            yy,xx=np.nonzero(recovered)
            previous=(float(xx.mean()),float(yy.mean())) if recovered.sum()<=2*local['_reference_body_area'] else (float(row.x),float(row.y))
            trial=candidate[f].copy();trial[recovered]=anchor
            trial=base.keep_host_fragments_detected(labels[f],trial,recovered,anchor,frame_points,canonical)
            trial,changed,failures=reconcile_competitors(labels[f],trial,recovered,anchor,frame_points,histories)
            detail['competing_components']=json.dumps(failures)
            if trial is None:
                detail['reason']='unresolved_competing_identity_without_suppression'
            else:
                candidate[f]=trial
                detail.update(applied=True,recovered_pixels=int(recovered.sum()))
                transfers.extend(dict(frame=f,proposal=prop['proposal_id'],**item) for item in changed)
            rows.append(detail)
    assert np.array_equal(candidate>0,labels>0)
    duplicates=base.accounting.new_duplicate_components(labels,candidate)
    metrics=dict(targeting_mode='field_wide_discovery',target_counts={k:0 for k in ['identities','tracks','frames','coordinates','regions','wells']},
        changed_label_pixels=int((candidate!=labels).sum()),changed_frames=int(np.any(candidate!=labels,axis=(1,2)).sum()),
        suppressed_foreground_pixels=0,added_foreground_pixels=0,new_duplicate_components=duplicates,new_identities=[],
        unclaimed_exact=True,applied_frames=sum(r['applied'] for r in rows))
    return candidate,ledger.copy(),audit,pd.DataFrame(rows),pd.DataFrame(transfers),metrics

def run(_upstream,params,out):
    result=produce(tifffile.imread(params['labels_path']),tifffile.imread(params['unclaimed_path']),
                   tifffile.imread(params['raw_path']),pd.read_csv(params['physical_track_points_path']),params)
    labels,ledger,audit,frames,transfers,metrics=result
    paths={}
    for name,arr in [('labels',labels),('unclaimed',ledger)]:
        name_out=params['output_stem']+('_unclaimed_original_ids' if name=='unclaimed' else '')+'.tif'
        paths[name]=out.out/name_out
        tifffile.imwrite(paths[name],arr,imagej=True,compression='zlib',photometric='minisblack',metadata=dict(axes='TYX',finterval=1800.,tunit='sec',unit='pixel'))
    for name,table in [('discovery_audit',audit),('episode_frames',frames),('component_transfers',transfers)]:
        paths[name]=out.out/(name+'.csv');table.to_csv(paths[name],index=False)
    paths['metrics']=out.out/'producer_metrics.json';paths['metrics'].write_text(json.dumps(metrics,indent=2)+'\n')
    return dict(outputs=paths,summary=metrics)
