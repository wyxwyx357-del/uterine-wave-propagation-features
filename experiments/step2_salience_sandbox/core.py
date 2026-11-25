"""Continuous section scores, common retention, and unchanged downstream code."""
import numpy as np
from analysis.mse_pregnancy_association.code.normalized_activity import normalize_magnitude, score_salience
from analysis.mse_pregnancy_association.code.patient_activity_aggregation import candidate_nodes
from analysis.mse_pregnancy_association.code.activity_counting import count_activities
from analysis.mse_pregnancy_association.code.batch_wave_counts import count_map, COUNT, RATE
from .upstream import stable_hash, pool_hash

MIN_ADJACENT = 3  # Existing production support rule, not a new exclusion threshold.


def contiguous_blocks(ids):
    ids=np.asarray(sorted(set(ids)),int)
    return [block.tolist() for block in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)
            if len(block)>=MIN_ADJACENT]


def s_crit(section_scores):
    """None for no eligible triple; otherwise max of adjacent triple minima.

    Longer blocks cannot improve their own minimum over all contained triples.
    Missing/nonfinite scores break support. Compare None as no passing block.
    """
    available={int(c):float(v) for c,v in section_scores.items() if v is not None and np.isfinite(v)}
    minima=[min(available[c],available[c+1],available[c+2]) for c in available
            if c+1 in available and c+2 in available]
    return max(minima) if minima else None


def event_windows(shared, wall, event):
    entity=shared['walls'][wall]['reconciliation']['source_evidence'][event['canonical_source_entity_id']]
    rows=[]
    for c in sorted(int(c) for c in entity['profile']):
        phases=entity['profile'].get(str(c),entity['profile'].get(c))
        lo=min(p['start_s'] for p in phases);hi=max(p['end_s'] for p in phases)
        a=int(np.searchsorted(shared['t'],lo-1e-8))
        b=int(np.searchsorted(shared['t'],hi+1e-8,side='right'))
        y=shared['filtered'][a:b,wall,c]
        complete=bool(b>a and shared['valid'][a:b,wall,c].all() and np.isfinite(y).all())
        rows.append(dict(section=c,first=a,last=b,complete=complete,
                         values=y, start_s=float(shared['t'][a]) if b>a else None,
                         end_s=float(shared['t'][b-1]) if b>a else None))
    return rows


class BaselineMinMaxScorer:
    name='S1_baseline_minmax'
    numerator='production event-section peak of normalized |filtered A|'

    def __init__(self,shared):
        self.normalized,self.metadata=normalize_magnitude(shared['filtered'],shared['valid'])

    def score(self,shared,wall,event):
        rows=[]
        for window in event_windows(shared,wall,event):
            c=window['section']
            samples=self.normalized[window['first']:window['last'],wall,c]
            complete=window['complete'] and bool(np.isfinite(samples).all())
            rows.append(dict(section=c,score=float(samples.max()) if complete else None,
                             complete=complete,sample_scores=samples if complete else None))
        return rows


def retain(event,rows,threshold):
    """One selection rule for all scorers; strict >, never >=."""
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    selected=[r['section'] for r in rows if r['complete'] and r['score'] is not None
              and np.isfinite(r['score']) and r['score']>threshold]
    blocks=contiguous_blocks(selected)
    return dict(event_proposal_id=event['event_proposal_id'],canonical_source=event['canonical_source_entity_id'],
                cutoff=float(threshold),decision='salient_support' if blocks else 'insufficient_salient_support',
                original_sections=[r['section'] for r in rows],salient_sections=selected,
                retained_sections=sorted(c for block in blocks for c in block),native_contiguous_blocks=blocks,
                direction=event['direction'],direction_source='unchanged_original_proposal_not_refit_on_subset',
                ownership_status=event['ownership_status'],event_count=None,event_frequency=None)


def production_score_record(event,rows,threshold):
    """Baseline gate adapter only; no scoring or retention mathematics copied."""
    result=retain(event,rows,threshold)
    result['per_section']=[dict(section=r['section'],normalized_peak=r['score'],
                              salient_samples=int((r['sample_scores']>threshold).sum()) if r['complete'] else 0,
                              phase_support_complete=r['complete']) for r in rows]
    return result


def continuous_branch(shared,scorer):
    before=pool_hash(shared)
    if before != shared['pool_hash']:
        raise RuntimeError('Shared upstream was mutated')
    walls=[]
    for wall,record in enumerate(shared['walls']):
        proposals=[]
        for event in record['reconciliation']['event_proposals']:
            rows=scorer.score(shared,wall,event)
            proposals.append(dict(event=event,sections=rows,
                                  S_crit=s_crit({r['section']:r['score'] for r in rows})))
        walls.append(dict(wall=wall,proposals=proposals))
    if pool_hash(shared)!=before:
        raise RuntimeError('Scorer changed shared upstream')
    result=dict(scorer=scorer.name,pool_hash=before,walls=walls)
    if isinstance(scorer,BaselineMinMaxScorer):
        result['normalized_values']=scorer.normalized
        result['normalization_metadata']=scorer.metadata
    return result


def evaluate_threshold(shared,branch,threshold):
    """Always rerun full retained-section-dependent identity/propagation/count."""
    if branch['pool_hash']!=pool_hash(shared):
        raise RuntimeError('Different candidate pool')
    selected=dict(walls=[dict(wall=w['wall'],scores=[retain(p['event'],p['sections'],threshold)
                             for p in w['proposals']]) for w in branch['walls']])
    ownership=dict(walls=[dict(reconciliation=w['reconciliation']) for w in shared['walls']])
    nodes=candidate_nodes(selected,ownership)
    if shared['observable']:
        result=count_activities(nodes,shared['s'],shared['span'],shared['dt'],shared['exposure']['seconds'])
    else:
        result={COUNT:None,RATE:None,**{prefix+suffix:None for prefix in (COUNT,RATE)
                                      for suffix in ('_min','_max','_exact')},'activities':[]}
    return dict(selection=selected,nodes=nodes,result=result,pool_hash=branch['pool_hash'],threshold=float(threshold))


class BaselineEquivalenceError(RuntimeError):
    pass


def baseline_equivalence_gate(maps,noise_floor,shared,branch,evaluation):
    """Independent production reference. Any differing layer is fatal."""
    def equal(layer,a,b):
        if stable_hash(a)!=stable_hash(b):
            raise BaselineEquivalenceError('Baseline mismatch at '+layer)
    normalized,_=normalize_magnitude(shared['filtered'],shared['valid'])
    equal('threshold',evaluation['threshold'],.6)
    equal('normalized_values',branch['normalized_values'],normalized)
    equal('normalization_metadata',branch['normalization_metadata'],normalize_magnitude(shared['filtered'],shared['valid'])[1])
    selected=[]
    for wall in branch['walls']:
        scores=[]
        ownership=shared['walls'][wall['wall']]['reconciliation']
        for proposal in wall['proposals']:
            event=proposal['event']
            production=score_salience(event,ownership['source_evidence'],normalized[:,wall['wall']],
                                     shared['valid'][:,wall['wall']],shared['t'])
            sandbox=production_score_record(event,proposal['sections'],.6)
            equal(f"wall{wall['wall']}/{event['event_proposal_id']}/proposal_sections_retention",sandbox,production)
            scores.append(production)
        selected.append(dict(wall=wall['wall'],scores=scores))
    equal('actual_evaluated_proposal_retention',evaluation['selection'],dict(walls=[
        dict(wall=w['wall'],scores=[{k:v for k,v in score.items() if k!='per_section'} for score in w['scores']])
        for w in selected]))
    production_nodes=candidate_nodes(dict(walls=selected),dict(walls=[dict(reconciliation=w['reconciliation']) for w in shared['walls']]))
    equal('candidate_nodes',evaluation['nodes'],production_nodes)
    production=count_map(maps,noise_floor)
    if shared['observable']:
        equal('propagation_and_activity_ledger',evaluation['result']['activities'],production['activities'])
        equal('identity_comparisons',evaluation['result']['identity_comparisons'],production['identity_comparisons'])
    for prefix in (COUNT,RATE):
        equal(prefix,evaluation['result'][prefix],production[prefix])
        for suffix in ('_min','_max','_exact'):
            equal(prefix+suffix,evaluation['result'][prefix+suffix],production[prefix+suffix])
    return dict(passed=True,pool_hash=shared['pool_hash'],
                proposals=sum(len(w['proposals']) for w in branch['walls']),
                count_range=[production.get(COUNT+'_min'),production.get(COUNT+'_max')],
                layers=['normalized_values','every_proposal_and_section','salient_sections','retained_sections',
                        'native_contiguous_blocks','candidate_nodes','propagation_and_activity_ledger',
                        'identity_comparisons','v1_count_rate','v2_count_frequency_min_max_exact'])
