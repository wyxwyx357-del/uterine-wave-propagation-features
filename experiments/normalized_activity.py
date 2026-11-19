"""Wang-inspired pointwise relative salience on unchanged radial maps.

The paper's magnitude is |curvature change|; here it is |filtered radial A|.
Min/max are fixed over the complete video and BOTH walls, not each candidate
or column. The paper does not specify this pooling scope. Cutoff=.6 and three
adjacent native supports reuse the published cutoff and current fit support.
"""
import numpy as np


def normalize_magnitude(filtered, valid):
    y = np.asarray(filtered, float); m = np.asarray(valid, bool)
    if y.shape != m.shape or y.ndim != 3 or y.shape[1] != 2:
        raise ValueError('Expected time x two walls x native sections with valid mask')
    supported = m & np.isfinite(y)
    magnitude = abs(y[supported]); normalized = np.full_like(y, np.nan)
    if not len(magnitude):
        return normalized, dict(status='no_filtered_support',minimum=None,maximum=None,
            physical_cutoff=None,cutoff=.6,normalization_scope='whole_video_both_walls')
    low = float(magnitude.min()); high = float(magnitude.max())
    normalized[supported] = (magnitude-low)/(high-low) if high > low else 0.
    return normalized, dict(status='normalized' if high>low else 'constant_magnitude',
        minimum=low,maximum=high,physical_cutoff=low+.6*(high-low),cutoff=.6,
        normalization_scope='whole_video_both_walls',valid_filtered_samples=int(supported.sum()),
        salient_samples=int((normalized[supported]>.6).sum()))


def score_salience(event, source_evidence, normalized, valid, time_s):
    entity = source_evidence[event['canonical_source_entity_id']]
    t = np.asarray(time_s, float); y = np.asarray(normalized, float); m = np.asarray(valid, bool)
    if y.shape != m.shape or y.ndim != 2 or y.shape[0] != len(t):
        raise ValueError('Expected one native wall with matching time and validity')
    ids = sorted(int(c) for c in entity['profile']); supported=[]; per_section=[]
    for col in ids:
        phases = entity['profile'].get(str(col),entity['profile'].get(col))
        lo = min(p['start_s'] for p in phases); hi = max(p['end_s'] for p in phases)
        a = int(np.searchsorted(t,lo-1e-8)); b = int(np.searchsorted(t,hi+1e-8,side='right'))
        good = b>a and m[a:b,col].all() and np.isfinite(y[a:b,col]).all()
        peak = float(y[a:b,col].max()) if good else None
        samples = int((y[a:b,col]>.6).sum()) if good else 0
        if good and peak>.6:supported.append(col)
        per_section.append(dict(section=col,normalized_peak=peak,salient_samples=samples,
                                phase_support_complete=bool(good)))
    blocks = [b.tolist() for b in np.split(np.asarray(supported,int),
              np.flatnonzero(np.diff(supported)!=1)+1) if len(b)>=3]
    retained = sorted(c for block in blocks for c in block)
    return dict(event_proposal_id=event['event_proposal_id'],
        canonical_source=event['canonical_source_entity_id'],cutoff=.6,
        decision='salient_support' if retained else 'insufficient_salient_support',
        original_sections=ids,salient_sections=supported,retained_sections=retained,
        native_contiguous_blocks=blocks,per_section=per_section,
        direction=event['direction'],direction_source='unchanged_original_proposal_not_refit_on_subset',
        ownership_status=event['ownership_status'],event_count=None,event_frequency=None)
