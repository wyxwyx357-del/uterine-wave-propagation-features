"""Phase3 one-time reporting of the prelocked outcomes and exact gates."""
from collections import Counter
import numpy as np
from ..benchmark_data import save
from ..mechanism_2x2.metrics import aggregate
from ..diagnostic.attribute import write_csv
from .evaluate import coexist_summary,flow_summary,lost_gained,MERGE_DEFINITION
from .gates import decide

def interval_report(results):
    a={v['row']['case_id']:v['row'] for v in results['M0']};b={v['row']['case_id']:v['row'] for v in results['M1']}
    assert a.keys()==b.keys()
    transitions={k:0 for k in ('improved','worsened','unchanged','both_missing','one_missing')}
    distances=dict(transitions)
    for key,x in a.items():
        y=b[key];c0=x['count_interval_coverage'];c1=y['count_interval_coverage'];d0=x['distance_to_interval'];d1=y['distance_to_interval']
        if c0 is None or c1 is None:
            category='both_missing' if c0 is None and c1 is None else 'one_missing'
            transitions[category]+=1;distances[category]+=1;continue
        transitions['improved' if c1>c0 else 'worsened' if c1<c0 else 'unchanged']+=1
        distances['improved' if d1<d0 else 'worsened' if d1>d0 else 'unchanged']+=1
    output={}
    for model,rows in [('M0',a.values()),('M1',b.values())]:
        rs=list(rows);valid=[r for r in rs if r['count_interval_coverage'] is not None];ds=[r['distance_to_interval'] for r in valid]
        output[model]=dict(evaluable=len(valid),missing=len(rs)-len(valid),coverage=sum(r['count_interval_coverage'] for r in valid)/len(valid) if valid else None,
            mean_distance=float(np.mean(ds)) if ds else None,median_distance=float(np.median(ds)) if ds else None)
    return dict(models=output,coverage_difference=output['M1']['coverage']-output['M0']['coverage'] if output['M1']['coverage'] is not None and output['M0']['coverage'] is not None else None,
        coverage_transitions=transitions,distance_transitions=distances,significance_test_performed=False)

def build(results,formal_eligible):
    models=('M0','M1');summary={m:aggregate([v['row'] for v in results[m]]) for m in models}
    textures={}
    for t in sorted({v['row']['texture_id'] for v in results['M0']}):
        textures[t]={m:aggregate([v['row'] for v in results[m] if v['row']['texture_id']==t]) for m in models}
    for row in list(summary.values())+[r for t in textures.values() for r in t.values()]:
        for label in row['strata']:row['strata'][label]['FN']=row['strata'][label]['truth']-row['strata'][label]['TP']
    coexist=coexist_summary(results)
    negative={m:dict(pooled=summary[m]['negative'],
        zero=aggregate([v['row'] for v in results[m] if v['row']['scenario']['family']=='zero'])['negative'],
        artifact_negative=aggregate([v['row'] for v in results[m] if v['row']['scenario']['family']=='artifact' and not v['row']['truth_count']])['negative'],
        per_texture={t:dict(pooled=textures[t][m]['negative'],
            zero=aggregate([v['row'] for v in results[m] if v['row']['texture_id']==t and v['row']['scenario']['family']=='zero'])['negative'],
            artifact_negative=aggregate([v['row'] for v in results[m] if v['row']['texture_id']==t and v['row']['scenario']['family']=='artifact' and not v['row']['truth_count']])['negative'])
            for t in textures}) for m in models}
    gate=decide(summary,textures,{m:coexist[m]['pooled'] for m in models},{m:negative[m]['pooled'] for m in models},formal_eligible)
    recall=lost_gained(results);structures={}
    for label in recall:
        recall[label]['lost_count']=len(recall[label]['lost']);recall[label]['gained_count']=len(recall[label]['gained'])
        recall[label]['one_event_percentage_points']=100/recall[label]['denominator'] if recall[label]['denominator'] else None
    for m in models:
        ps=[r for v in results[m] for r in v['phase']['pattern_rows']]
        structures[m]=dict(truth_wall_denominator=len(ps),main_edge_available=sum(r['main_pair_available'] for r in ps),
            central_main_pair_selected=sum(r['central_main_pair_selected'] for r in ps),
            central_main_pair_selected_rate=sum(r['central_main_pair_selected'] for r in ps)/len(ps) if ps else None,
            peripheral_two_pair=sum(r['four_lobe_flanking_pairs'] for r in ps),
            peripheral_two_pair_rate=sum(r['four_lobe_flanking_pairs'] for r in ps)/len(ps) if ps else None,
            truth_wall_rows=ps)
    return dict(summary=summary,textures=textures,coexist=coexist,negative=negative,gate=gate,recall=recall,
        structures=structures,flow={m:flow_summary(results[m]) for m in models},interval=interval_report(results))

def write(output,results,formal_eligible):
    report=build(results,formal_eligible);models=('M0','M1')
    save(output/'M1_UNSEEN_TEXTURE_PRIMARY_RESULTS.json',dict(A2={m:report['summary'][m]['A2'] for m in models},
        paired_phase_A2={m:report['summary'][m]['paired_phase_A2'] for m in models},
        effect_evidence=report['gate']['evidence'],recall_safety=report['recall'],
        matcher_eligible_count={m:report['summary'][m]['matcher_eligible_count'] for m in models}))
    truth_rows=[dict(t,model=m) for m in models for v in results[m] for t in v['row']['truth_rows']]
    write_csv(output/'M1_UNSEEN_TEXTURE_TRUTH_LEVEL.csv',truth_rows)
    texture_rows=[]
    for t in report['textures']:
        for m in models:
            selected=[v for v in results[m] if v['row']['texture_id']==t]
            s=report['textures'][t][m];co=aggregate([v['row'] for v in selected if v['row']['scenario']['family']=='coexist'])
            texture_rows.append(dict(texture_id=t,model=m,**{k:s[k] for k in ('TP','FP','FN','precision','recall','F1','A2','paired_phase_A2','A1','positive_unmatched','strata','matcher_eligible_count')},
                coexist_recall=co['recall'],negative=s['negative'],single_phase_flow=flow_summary(selected)))
    write_csv(output/'M1_UNSEEN_TEXTURE_TEXTURE_LEVEL.csv',texture_rows)
    save(output/'M1_UNSEEN_TEXTURE_COEXIST_SAFETY.json',dict(models=report['coexist'],merge_two_truth_error_definition=MERGE_DEFINITION,
        paired_ratio_differences={ratio:dict(TP_difference=report['coexist']['M1'][ratio]['TP']-report['coexist']['M0'][ratio]['TP'],
            FN_difference=report['coexist']['M1'][ratio]['FN']-report['coexist']['M0'][ratio]['FN'],
            prediction_count_difference=report['coexist']['M1'][ratio]['propagation_predictions']-report['coexist']['M0'][ratio]['propagation_predictions'],
            recall_difference=report['coexist']['M1'][ratio]['recall']-report['coexist']['M0'][ratio]['recall']) for ratio in ('2','3','5')}))
    save(output/'M1_UNSEEN_TEXTURE_NEGATIVE_CONTROLS.json',report['negative'])
    save(output/'M1_UNSEEN_TEXTURE_PAIRING_STRUCTURE.json',report['structures'])
    save(output/'M1_UNSEEN_TEXTURE_SINGLE_PHASE_FLOW.json',dict(models=report['flow'],
        units='formed/passed/propagation are canonical wall-proposal/node counts; final ledger counts are activities after cross-wall grouping. Detector raw events additionally saved. All raw scenarios include duplicate brightness controls; truth-bearing flow separately saved.',
        rows={m:[v['flow'] for v in results[m]] for m in models}))
    save(output/'M1_UNSEEN_TEXTURE_FINAL_LEDGER.json',dict(summary=report['summary'],recall_lost_gained=report['recall'],
        Step1_interval=report['interval'],rows={m:[v['row'] for v in results[m]] for m in models}))
    save(output/'M1_UNSEEN_TEXTURE_GATE_DECISION.json',report['gate'])
    return report
