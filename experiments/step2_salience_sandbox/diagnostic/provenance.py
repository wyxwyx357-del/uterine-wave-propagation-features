"""Read actual frozen ownership chains; reconstruct only diagnostic graphs."""
from collections import Counter
from .attribution_rules import stable_identity, maximum_matching, edge_in_some_maximum, overlap

def phase_kind(entity):
    kinds={m['evidence_kind'] for m in entity['original_measurements']}
    if not kinds:
        kinds={'paired_phase' if len(ps)==2 and {p['sign'] for p in ps}=={-1,1} else 'single_phase'
               for ps in entity['profile'].values()}
    if kinds=={'paired_phase'}:return 'paired_phase'
    if kinds=={'single_phase'}:return 'single_phase'
    return 'mixed' if kinds else 'unknown'

def enrich(record,name):
    point=record['points'][name];ev=point['evaluation'];result=ev['result'];shared=record['shared']
    nodes={n['node_id']:n for n in ev['nodes']}
    edges=result.get('identity_comparisons',[])
    predictions=[]
    for basic,activity in zip(point['predictions'],result['activities']):
        assert basic['prediction_id']==activity['activity_id']
        group=[nodes[k] for k in activity['source_nodes']];chains=[]
        for node in group:
            wall=node['wall'];rec=shared['walls'][wall]['reconciliation']
            event=next(e for e in rec['event_proposals'] if e['event_proposal_id']==node['source_proposal_id'])
            entity=rec['source_evidence'][node['canonical_source']]
            parents={(wall,p) for p in event['source_parent_ids']}
            branch=record['branches']['baseline_minmax' if name=='baseline' else 'mad_oob']
            score=next(p for p in branch['walls'][wall]['proposals'] if p['event']['event_proposal_id']==event['event_proposal_id'])
            selection=next(s for s in ev['selection']['walls'][wall]['scores'] if s['event_proposal_id']==event['event_proposal_id'])
            chains.append(dict(wall=wall,node_id=node['node_id'],canonical_source_entity_id=node['canonical_source'],
                source_entity_ids=event['source_entity_ids'],source_parent_keys=sorted(parents),
                phase_origin=phase_kind(entity),source_kind=entity['source_kind'],
                source_sections=sorted(int(c) for c in entity['profile']),retained_sections=node['sections'],
                source_phase_profile=entity['profile'],retained_phase_profile=node['profile'],
                canonical_measurement=entity['canonical_measurement'],original_measurements=entity['original_measurements'],
                contributor_phase_origins={k:phase_kind(rec['source_evidence'][k]) for k in event['source_entity_ids']},
                phase_ridge_provenance=[dict(wall=wall,**p) for p in event['phase_ridge_provenance']],
                split_decisions=[{k:p[k] for k in ('parent_candidate_id','split_status','reason','conflict_reasons') if k in p}
                                 for p in shared['walls'][wall]['split_proposals'] if (wall,p['parent_candidate_id']) in parents],
                split_actions=[p for p in rec['split_actions'] if (wall,p['parent_id']) in parents],
                ownership_status=event['ownership_status'],ownership_conflicts=event['ownership_conflicts'],
                canonical_child=entity['source_kind']=='child',has_reconciled_child_evidence=event['from_split'],
                salience_selection=selection,section_scores=[{k:v for k,v in s.items() if k!='sample_scores'} for s in score['sections']],
                S_crit=score['S_crit']))
        kinds={c['phase_origin'] for c in chains}
        phase=next(iter(kinds)) if len(kinds)==1 else 'mixed'
        related=[e for e in edges if (e['node_a'] in activity['source_nodes']) != (e['node_b'] in activity['source_nodes'])]
        rels={e['relation'] for e in related}
        wall_status='cross_wall_merged' if len(group)==2 else 'single_wall_unresolved_cross_wall' if 'unresolved' in rels else (
            'single_wall_shared_identity_not_merged' if 'shared_identity_supported' in rels else 'single_wall_clean')
        predictions.append(dict(**basic,stable_source_identity=stable_identity(group),source_nodes=activity['source_nodes'],
            source_node_keys=sorted((n['wall'],n['canonical_source']) for n in group),walls=activity['walls'],
            midpoint_s=(basic['evidence_start_s']+basic['evidence_end_s'])/2,
            source_sections=sorted({c for q in chains for c in q['source_sections']}),
            phase_origin=phase,split_origin='split_child' if any(c['canonical_child'] for c in chains) else 'unsplit_base',
            wall_status=wall_status,ownership_statuses=sorted({c['ownership_status'] for c in chains}),
            ownership_remainder=False,has_reconciled_child_evidence=any(c['has_reconciled_child_evidence'] for c in chains),
            parent_keys=sorted({tuple(p) for c in chains for p in c['source_parent_keys']}),
            source_chain=chains,propagation_fit=activity['block_evidence'],cross_wall_relations=related,
            unresolved_related_nodes=activity['unresolved_related_nodes']))
    matched_ids={m['prediction_id'] for m in point['match']['matches']}
    for p in predictions:
        parents=set(p['parent_keys']);tp=[q for q in predictions if q['prediction_id'] in matched_ids]
        p['matched_TP']=p['prediction_id'] in matched_ids
        p['shared_parent_with_TP']=any(parents&set(q['parent_keys']) for q in tp if q is not p)
        p['sibling_prediction_count']=sum(bool(parents&set(q['parent_keys'])) for q in predictions if q is not p)
    return predictions

def graph(point):
    ev=point['evaluation'];result=ev['result'];nodes={n['node_id']:n for n in ev['nodes']}
    used={n for a in result['activities'] if len(a['source_nodes'])==2 for n in a['source_nodes']}
    prop={e['node_id'] for a in result['activities'] for e in a['block_evidence'] if e['propagation_supported']}-used
    left={n for n in prop if nodes[n]['wall']==0};right=prop-left
    edges=set()
    for e in result['identity_comparisons']:
        a,b=e['node_a'],e['node_b']
        if a in prop and b in prop and e['relation'] in ('shared_identity_supported','unresolved'):
            edges.add((a,b) if a in left else (b,a))
    m=maximum_matching(left,right,edges)['cardinality']
    assert m==result['ambiguous_cross_wall_propagation_pairs']
    return left,right,edges,m

def duplicate_links(pred,relations,truth_to_tp,point,g):
    left,right,edges,m=g;links=[]
    lookup={frozenset((e['node_a'],e['node_b'])):e for e in point['evaluation']['result']['identity_comparisons']}
    for rel in relations:
        if not rel['matcher_eligible']:continue
        assert rel['truth_id'] in truth_to_tp,'Maximum-cardinality matcher left eligible truth unoccupied'
        tp=truth_to_tp[rel['truth_id']]
        sharedwall=bool(set(pred['walls'])&set(tp['walls']))
        comparisons=[]
        for a in pred['source_nodes']:
            for b in tp['source_nodes']:
                edge=lookup.get(frozenset((a,b)))
                if edge is None:continue
                pair=(a,b) if a in left else (b,a)
                in_graph=pair in edges
                if edge['relation']=='unresolved':why='unresolved local identity evidence'
                elif edge['relation']=='no_shared_identity_evidence':why='no shared identity evidence; broad truth eligibility is not phase identity'
                else:
                    neighbors=[e for e in point['evaluation']['result']['identity_comparisons'] if
                               e['relation'] in ('shared_identity_supported','unresolved') and
                               ({e['node_a'],e['node_b']}&{a,b}) and frozenset((e['node_a'],e['node_b']))!=frozenset((a,b))]
                    why='additional unresolved or nonunique shared-identity neighbors block strict pair'
                    assert neighbors,'Unexpected unmerged unique supported pair'
                comparisons.append(dict(**edge,display_relation='unrelated' if edge['relation']=='no_shared_identity_evidence' else edge['relation'],
                    enters_Step1_ambiguity_graph=in_graph,edge_in_some_maximum_matching=edge_in_some_maximum(left,right,edges,pair,m),
                    why_not_merged=why,unmatched_node=a,TP_node=b,
                    ambiguous_neighbor_count=sum(e['relation'] in ('unresolved','shared_identity_supported') and
                        bool({a,b}&{e['node_a'],e['node_b']}) for e in point['evaluation']['result']['identity_comparisons'])))
        seconds,iou=overlap(pred['evidence_start_s'],pred['evidence_end_s'],tp['evidence_start_s'],tp['evidence_end_s'])
        same_wall_pairs=[]
        for a in pred['source_chain']:
            for b in tp['source_chain']:
                if a['wall']!=b['wall']:continue
                sections=sorted(set(a['retained_phase_profile'])&set(b['retained_phase_profile']))
                sequences=[dict(section=int(c),unmatched_signs=[p['sign'] for p in a['retained_phase_profile'][c]],
                                TP_signs=[p['sign'] for p in b['retained_phase_profile'][c]]) for c in sections]
                common_parent=bool(set(map(tuple,a['source_parent_keys']))&set(map(tuple,b['source_parent_keys'])))
                same_wall_pairs.append(dict(wall=a['wall'],unmatched_canonical=a['canonical_source_entity_id'],TP_canonical=b['canonical_source_entity_id'],
                    same_parent=common_parent,different_split_children=common_parent and a['canonical_child'] and b['canonical_child'] and
                    a['canonical_source_entity_id']!=b['canonical_source_entity_id'],
                    different_phase_sources=a['phase_ridge_provenance']!=b['phase_ridge_provenance'],
                    ownership_remainder=False,phase_sequences=sequences,
                    all_shared_sections_reverse_biphasic_order=bool(sequences) and all(len(x['unmatched_signs'])==len(x['TP_signs'])==2 and
                        x['unmatched_signs']==x['TP_signs'][::-1] for x in sequences)))
        links.append(dict(truth_id=rel['truth_id'],TP_stable_identity=tp['stable_source_identity'],
            TP_source_nodes=tp['source_nodes'],TP_walls=tp['walls'],unmatched_walls=pred['walls'],
            subtype='A2' if sharedwall else 'A1',shared_parent=bool(set(pred['parent_keys'])&set(tp['parent_keys'])),
            different_canonical_source=pred['stable_source_identity']!=tp['stable_source_identity'],
            TP_phase_origin=tp['phase_origin'],TP_split_origin=tp['split_origin'],
            evidence_overlap_seconds=seconds,evidence_IoU=iou,shared_sections=sorted(set(pred['sections'])&set(tp['sections'])),
            comparisons=comparisons,same_wall_phase_comparisons=same_wall_pairs,TP_unresolved_related_nodes=tp['unresolved_related_nodes']))
    return links

def graph_accounting(record,name,unmatched,g):
    left,right,edges,m=g;marked=set()
    a1=[p for p in unmatched if p['primary_class']=='A' and p['subtype']=='A1']
    for p in a1:
        for link in p['duplicate_links']:
            if link['subtype']!='A1':continue
            for c in link['comparisons']:
                a,b=c['node_a'],c['node_b'];pair=(a,b) if a in left else (b,a)
                if pair in edges:marked.add(pair)
    minimum=maximum_matching(left,right,edges,marked,False);maximum=maximum_matching(left,right,edges,marked,True)
    assert minimum['cardinality']==maximum['cardinality']==m
    result=record['points'][name]['evaluation']['result'];truth=len(record['truth'])
    return dict(case_id=record['case_id'],point=name,truth_count=truth,ledger_unmatched=len(unmatched),
        count_min=result['significant_propagation_activity_count_min'],count_max=result['significant_propagation_activity_count_max'],
        min_count_equivalent_excess=max(result['significant_propagation_activity_count_min']-truth,0),
        max_count_equivalent_excess=max(result['significant_propagation_activity_count_max']-truth,0),
        identity_reduction=m,A1_count=len(a1),A1_with_unresolved_relation=sum(any(c['relation']=='unresolved' for l in p['duplicate_links']
            if l['subtype']=='A1' for c in l['comparisons']) for p in a1),
        A1_individually_foldable=sum(any(c['edge_in_some_maximum_matching'] for l in p['duplicate_links']
            if l['subtype']=='A1' for c in l['comparisons']) for p in a1),
        A1_jointly_absorbable_min=minimum['marked_pairs'],A1_jointly_absorbable_max=maximum['marked_pairs'],
        marked_A1_edges=sorted(marked),min_marked_matching=minimum['pairs'],max_marked_matching=maximum['pairs'])
