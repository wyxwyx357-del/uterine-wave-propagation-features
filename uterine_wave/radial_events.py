"""Automatic, reviewable biphasic propagation candidates from radial A(s,t).

One candidate needs two adjacent opposite-sign phases at multiple positions.
Engineering support/fit settings are not physiological or pregnancy thresholds.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .amplitude import band_limited_signal


@dataclass(frozen=True)
class RadialWaveConfig:
    min_space_points: int = 4
    min_path_span: float = .15
    min_fit_r2: float = .5
    synchronous_tolerance_s: float = .5


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _fit(s, t):
    slope, intercept = np.polyfit(s, t, 1)
    residual = t-(slope*s+intercept)
    total = float(np.sum((t-t.mean())**2))
    return dict(slope=float(slope), intercept=float(intercept),
                r2=float(1-np.sum(residual**2)/total) if total > 1e-20 else None,
                rmse_s=float(np.sqrt(np.mean(residual**2))))


def _lobes(y, valid, t, threshold):
    filtered = np.full_like(y, np.nan)
    columns = [[] for _ in range(y.shape[1])]
    minimum = max(3, int(np.ceil(12/np.median(np.diff(t)))))
    for col in range(y.shape[1]):
        for run_id, (begin, end) in enumerate(_runs(valid[:, col])):
            if end-begin < minimum:
                continue
            signal = band_limited_signal(y[begin:end, col], t[begin:end])
            filtered[begin:end, col] = signal
            edges = np.r_[0, np.flatnonzero(np.diff(signal >= 0))+1, len(signal)]
            for sequence, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                # A censored phase cannot establish a complete bipolar event.
                if a == 0 or b == len(signal):
                    continue
                peak = int(a+np.argmax(abs(signal[a:b])))
                amplitude = float(abs(signal[peak]))
                if amplitude <= max(float(threshold[col]), 1e-12):
                    continue
                columns[col].append(dict(column=col, run_id=run_id, sequence=sequence,
                    sign=1 if signal[peak] > 0 else -1, peak_index=int(begin+peak),
                    peak_time_s=float(t[begin+peak]), amplitude=amplitude,
                    first=int(begin+a), last=int(begin+b-1),
                    start_s=float(t[begin+a]), end_s=float(t[begin+b-1])))
    return filtered, columns


def _ridges(columns, t, s):
    nodes=[]
    for column in columns:
        for node in column:
            node['id']=len(nodes);nodes.append(node)
    parent=list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i

    for left, right in zip(columns[:-1], columns[1:]):
        choices=[]
        for a in left:
            for b in right:
                delay=abs(a['peak_time_s']-b['peak_time_s'])
                scale=.5*((a['end_s']-a['start_s'])+(b['end_s']-b['start_s']))
                overlap=min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s'])
                if a['sign']==b['sign'] and delay<=scale and overlap>=0:
                    choices.append((delay,a['id'],b['id']))
        used_left=set();used_right=set()
        for _, a, b in sorted(choices):
            if a in used_left or b in used_right:continue
            parent[find(a)]=find(b);used_left.add(a);used_right.add(b)
    groups={}
    for node in nodes:groups.setdefault(find(node['id']),[]).append(node)
    ridges=[]
    for group in groups.values():
        if len(group)<3:continue
        group=sorted(group,key=lambda node:node['column'])
        ids=np.array([node['column'] for node in group])
        arrival=np.array([node['peak_time_s'] for node in group])
        ridges.append(dict(id=len(ridges), sign=group[0]['sign'], nodes=group,
                           fit=_fit(s[ids],arrival), center_s=float(np.median(arrival))))
    return ridges


def _phase_edges(ridges):
    edges={}
    for i,a in enumerate(ridges):
        for j in range(i+1,len(ridges)):
            b=ridges[j]
            if a['sign']==b['sign']:continue
            left={node['column']:node for node in a['nodes']}
            right={node['column']:node for node in b['nodes']}
            common=sorted(set(left)&set(right))
            # Adjacent phases must belong to the same continuous track run in
            # each shared column. No phase pairing over a missing interval.
            common=[k for k in common if left[k]['run_id']==right[k]['run_id'] and
                    abs(left[k]['sequence']-right[k]['sequence'])==1]
            if len(common)<3 or np.any(np.diff(common)!=1):continue
            delta=np.array([right[k]['peak_time_s']-left[k]['peak_time_s'] for k in common])
            if not (np.all(delta>0) or np.all(delta<0)):continue
            early,late=(a,b) if delta.mean()>0 else (b,a)
            edges[i,j]=dict(first=early['id'], second=late['id'], columns=common,
                            score=(1,len(common),int(early['sign']<0)))
    return edges


def _pair_phases(ridges, edges):
    """Non-overlapping adjacent phase pairs, without counting both polarities.

    Maximum-cardinality bipartite matching, then shared support and leading
    polarity as tie breakers. Branches remain flagged for review, but do not
    discard every supported pair in the connected component.
    """
    neighbors={i:[] for i in range(len(ridges))}
    for i,j in edges:neighbors[i].append(j);neighbors[j].append(i)
    ambiguous=[i for i in neighbors if len(neighbors[i])>1]
    negative=[i for i in neighbors if ridges[i]['sign']<0 and neighbors[i]]
    positive=[i for i in neighbors if ridges[i]['sign']>0 and neighbors[i]]
    if not negative or not positive:return [],ambiguous
    n=len(negative);m=len(positive)
    # Dummy columns permit unmatched phases. Integer weights encode the
    # lexicographic objective without a tunable physiological coefficient.
    support_bound=sum(len(e['columns']) for e in edges.values())+1
    tie_bound=min(n,m)+1
    cost=np.zeros((n,m+n),float)
    cost[:,:m]=1
    for row,i in enumerate(negative):
        for col,j in enumerate(positive):
            edge=edges.get(tuple(sorted((i,j))))
            if edge is not None:
                cost[row,col]=-((support_bound+len(edge['columns']))*tie_bound+edge['score'][2])
    # Rectangular shortest-augmenting-path assignment (rows <= columns).
    u=np.zeros(n+1);v=np.zeros(m+n+1)
    owner=np.zeros(m+n+1,int);previous=np.zeros(m+n+1,int)
    for row in range(1,n+1):
        owner[0]=row;col=0;minimum=np.full(m+n+1,np.inf);used=np.zeros(m+n+1,bool)
        while True:
            used[col]=True;current_row=owner[col];delta=np.inf;next_col=0
            for j in range(1,m+n+1):
                if used[j]:continue
                reduced=cost[current_row-1,j-1]-u[current_row]-v[j]
                if reduced<minimum[j]:minimum[j]=reduced;previous[j]=col
                if minimum[j]<delta:delta=minimum[j];next_col=j
            u[owner[used]]+=delta;v[used]-=delta;minimum[~used]-=delta
            col=next_col
            if owner[col]==0:break
        while col:
            before=previous[col];owner[col]=owner[before];col=before
    chosen=[]
    for col in range(1,m+1):
        if owner[col] and cost[owner[col]-1,col-1]<0:
            chosen.append(tuple(sorted((negative[owner[col]-1],positive[col-1]))))
    return chosen,ambiguous


def detect_radial_events(signal, valid, time_s, space, *, noise_floor, config=None):
    """Find paired contraction/relaxation candidates, with an explicit noise scale.

    noise_floor is an absolute A threshold (s^-1), scalar or one per section.
    It must come from a stated noise reference, not per-video robust-z scaling.
    Space stays on the original full cervix-to-fundus axis, without rescaling.
    """
    config=config or RadialWaveConfig()
    y=np.asarray(signal,float);m=np.asarray(valid,bool);t=np.asarray(time_s,float);s=np.asarray(space,float)
    if (y.shape!=m.shape or y.shape!=(len(t),len(s)) or len(t)<3 or len(s)<3 or
            not np.isfinite(t).all() or not np.isfinite(s).all() or np.any(np.diff(s)<=0) or
            np.any((s<0)|(s>1)) or np.any(np.diff(t)<=0) or
            not np.allclose(np.diff(t),np.median(np.diff(t)),rtol=1e-5,atol=1e-9)):
        raise ValueError('Expected matching native-uniform time x ordered normalized-space maps')
    threshold=np.broadcast_to(np.asarray(noise_floor,float),(len(s),))
    if not np.isfinite(threshold).all() or np.any(threshold<0):raise ValueError('Invalid absolute noise floor')
    if (config.min_space_points<3 or not 0<=config.min_path_span<=1 or
            not 0<=config.min_fit_r2<=1 or config.synchronous_tolerance_s<0):
        raise ValueError('Invalid radial event support settings')
    filtered,columns=_lobes(y,m&np.isfinite(y),t,threshold)
    ridges=_ridges(columns,t,s);edges=_phase_edges(ridges)
    pairs,ambiguous=_pair_phases(ridges,edges);events=[]
    for key in pairs:
        edge=edges[key];first=ridges[edge['first']];second=ridges[edge['second']]
        ids=np.asarray(edge['columns']);left={x['column']:x for x in first['nodes']};right={x['column']:x for x in second['nodes']}
        # The same phase transition at each section is an arrival marker in
        # band-limited finite-interval A, not an instantaneous tissue extremum.
        arrival=[]
        for j in ids:
            a,b=left[j],right[j];k=a['last'];k2=b['first']
            before,after=filtered[k,j],filtered[k2,j]
            arrival.append(float(t[k]+(-before)/(after-before)*(t[k2]-t[k])))
        arrival=np.asarray(arrival);fit=_fit(s[ids],arrival)
        span=float(s[ids[-1]]-s[ids[0]])
        tolerance=max(float(np.median(np.diff(t))),config.synchronous_tolerance_s)
        delta=np.diff(arrival)
        native_tolerance=float(np.median(np.diff(t)))
        monotone=np.all(delta>=-native_tolerance) or np.all(delta<=native_tolerance)
        phase_fits=[_fit(s[ids],np.array([phase[j]['peak_time_s'] for j in ids]))
                    for phase in (left,right)]
        if len(ids)<config.min_space_points or span<config.min_path_span:
            kind='local'
        elif np.ptp(arrival)<=tolerance:
            kind='synchronous'
        elif (not monotone or fit['r2'] is None or fit['r2']<config.min_fit_r2 or
              any(p['r2'] is None or p['r2']<config.min_fit_r2 or p['slope']*fit['slope']<=0 for p in phase_fits)):
            kind='complex'
        else:kind='C2F' if fit['slope']>0 else 'F2C'
        events.append(dict(event_id='',classification=kind,reviewed=False,
            start_s=min(left[j]['start_s'] for j in ids),end_s=max(right[j]['end_s'] for j in ids),
            sections=ids.tolist(),s_start=float(s[ids[0]]),s_end=float(s[ids[-1]]),spatial_span=span,
            arrival_time_s=arrival.tolist(),arrival_fit_R2=fit['r2'],arrival_fit_RMSE_s=fit['rmse_s'],
            arrival_time_slope=fit['slope'],
            velocity_paths_per_s=float(1/fit['slope']) if kind in ('C2F','F2C') else None,
            leading_polarity=first['sign'],phase_ridge_ids=[first['id'],second['id']],
            phase_pairing_ambiguous=any(i in ambiguous for i in key),
            phase_arrival_fits=phase_fits,
            first_phase_arrival_s=[left[j]['peak_time_s'] for j in ids],
            second_phase_arrival_s=[right[j]['peak_time_s'] for j in ids],
            peak_absolute_rate_per_phase=[max(left[j]['amplitude'] for j in ids),max(right[j]['amplitude'] for j in ids)]))
    events.sort(key=lambda event:(np.median(event['arrival_time_s']),event['s_start']))
    for i,event in enumerate(events,1):event['event_id']=f'candidate_{i:03d}'
    return dict(status='automatic_radial_wave_candidates_require_review',config=asdict(config),
        noise_floor_rate=threshold.copy(),filter_band_cpm=[.5,5.],filtered_A=filtered,
        events=events,candidate_count=len(events),
        traveling_candidate_count=sum(e['classification'] in ('C2F','F2C') for e in events),
        ambiguous_phase_ridge_ids=ambiguous,unpaired_ridge_count=len(ridges)-2*len(pairs),
        event_count=None,event_frequency_cpm=None,
        phase_ridges=[dict(ridge_id=r['id'],sign=r['sign'],sections=[x['column'] for x in r['nodes']],
                           peak_time_s=[x['peak_time_s'] for x in r['nodes']],fit=r['fit']) for r in ridges])


def radial_candidates_from_maps(spacetime, *, noise_floor, s_range=(0.,1.), config=None):
    """Use current 2 s radial-map interface and retain original section IDs."""
    a=np.asarray(spacetime['A'],float);valid=np.asarray(spacetime['valid_mask'],bool)
    t=np.asarray(spacetime['interval_center_s'],float);s=np.asarray(spacetime['s'],float)
    if a.shape!=(len(t),2,len(s)) or valid.shape!=a.shape:
        raise ValueError('Expected time x two-wall x section radial maps')
    if len(s_range)!=2 or not 0<=s_range[0]<s_range[1]<=1:
        raise ValueError('Invalid anatomical s_range')
    ids=np.flatnonzero((s>=s_range[0])&(s<=s_range[1]))
    if len(ids)<3:raise ValueError('Selected region needs at least three native sections')
    filtered=np.full_like(a,np.nan);walls=[]
    for wall in (0,1):
        result=detect_radial_events(a[:,wall][:,ids],valid[:,wall][:,ids],t,s[ids],
                                   noise_floor=noise_floor,config=config)
        filtered[:,wall,ids]=result.pop('filtered_A')
        for event in result['events']:
            event['event_id']=f"wall{wall}_{event['event_id']}"
            event['wall']=wall;event['sections']=ids[event['sections']].tolist()
        for ridge in result['phase_ridges']:ridge['sections']=ids[ridge['sections']].tolist()
        walls.append(dict(wall=wall,**result))
    report=dict(status='automatic_radial_wave_candidates_require_review',
        measurement_source='A_rate',measurement_units='s^-1',source_section_indices=ids.tolist(),
        requested_s_range=list(s_range),sampled_s_range=[float(s[ids[0]]),float(s[ids[-1]])],
        center_time_range_s=[float(t[0]),float(t[-1])],
        nominal_interval_s=float(spacetime.get('nominal_interval_s',np.median(spacetime['interval_dt_s']))),
        walls=walls,event_count=None,event_frequency_cpm=None,
        note='Two-wall candidate counts are not summed; review event identity and arrivals before patient rates.')
    arrays=dict(filtered_A=filtered,candidate_valid_mask=valid&np.isfinite(filtered),
                interval_center_s=t.copy(),s=s.copy())
    return report,arrays
