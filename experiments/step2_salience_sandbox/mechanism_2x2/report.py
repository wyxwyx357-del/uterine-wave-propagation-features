"""Numerical 2x2 reports; final qualitative interpretation requires evidence review."""
import numpy as np
from ..benchmark_data import save
from ..diagnostic.attribute import write_csv
from .metrics import aggregate
from .mechanisms import MODELS

def effects(values):
    d0,d1,d2,d3=[values[m] for m in MODELS]
    return dict(D=values,R_P=(d0-d1)/d0 if d0 else None,R_F=(d0-d2)/d0 if d0 else None,
                R_C=(d0-d3)/d0 if d0 else None,R_F_given_P=(d1-d3)/d0 if d0 else None)

def produce(output,rows,formation,phases,pairings,runs,fixtures,unsupported,config_hash):
    summary={m:{s:aggregate(rows[m][s]) for s in ('S1','S2')} for m in MODELS}
    formation_summary={m:aggregate(formation[m]) for m in MODELS}
    pattern={}
    filters={}
    for m in MODELS:
        ps=phases[m]['pattern_rows'];fr=phases[m]['filter_rows'];ratios=[x for r in fr for x in r['peripheral_main_amplitude_ratios']]
        pattern[m]=dict(truth_wall_rows=len(ps),main_edge_available=sum(r['main_pair_available'] for r in ps),
            central_main_pair_selected=sum(r['central_main_pair_selected'] for r in ps),
            central_main_pair_selected_fraction=sum(r['central_main_pair_selected'] for r in ps)/len(ps) if ps else None,
            four_lobe_flanking_pairs=sum(r['four_lobe_flanking_pairs'] for r in ps),
            four_lobe_flanking_fraction=sum(r['four_lobe_flanking_pairs'] for r in ps)/len(ps) if ps else None)
        filters[m]=dict(truth_component_native_lobes=sum(r['truth_component_lobe_count'] for r in fr),
            peripheral_native_lobes=sum(r['peripheral_lobe_count'] for r in fr),
            peripheral_main_amplitude_ratio_median=float(np.median(ratios)) if ratios else None,
            peripheral_main_amplitude_ratio_P90=float(np.percentile(ratios,90)) if ratios else None,
            ringing_like_selected_native_wall_pairs=len(phases[m]['ringing_like_selected_pairs']))
    primary=dict(fixed_S1_A2=effects({m:summary[m]['S1']['A2'] for m in MODELS}),
        formation_no_salience_A2=effects({m:formation_summary[m]['A2'] for m in MODELS}),
        fixed_S2_A2=effects({m:summary[m]['S2']['A2'] for m in MODELS}),
        fixed_downstream=summary,formation_no_salience=formation_summary,four_lobe_mechanism=pattern,
        unsupported_truth_inputs=[r for r in unsupported if r['truth_bearing']],common_configuration_hash=config_hash,
        primary_scope='S1 A2 is directly comparable to prior 73; formation_no_salience is a separate native full-support mechanism endpoint, not numerically interchangeable.')
    save(output/'PAIRING_FILTER_2X2_PRIMARY_RESULTS.json',primary)
    truths=[];textures=[]
    for m in MODELS:
        for layer,rs in [('formation_no_salience',formation[m])]+list(rows[m].items()):
            truths.extend(dict(t,model=m) for row in rs for t in row['truth_rows'])
            for texture in sorted({r['texture_id'] for r in rs}):
                sub=[r for r in rs if r['texture_id']==texture];value=aggregate(sub)
                coexist=aggregate([r for r in sub if r['scenario']['family']=='coexist'])
                textures.append(dict(model=m,layer=layer,texture_id=texture,A2=value['A2'],paired_phase_A2=value['paired_phase_A2'],
                    clear_recall=value['strata']['clear']['recall'],medium_recall=value['strata']['medium']['recall'],
                    coexist_recall=coexist['recall'],negative_FP_min=value['negative']['FP_min'],negative_FP_max=value['negative']['FP_max'],
                    TP=value['TP'],FP=value['FP'],FN=value['FN'],matcher_eligible_count=value['matcher_eligible_count']))
    write_csv(output/'PAIRING_FILTER_2X2_TRUTH_LEVEL.csv',truths)
    write_csv(output/'PAIRING_FILTER_2X2_TEXTURE_LEVEL.csv',textures)
    coexist={}
    for m in MODELS:
        coexist[m]={}
        for ratio in (2,3,5):
            cases={r['case_id'] for r in rows[m]['S1'] if r['scenario']['family']=='coexist' and r['scenario']['ratio']==ratio}
            layers={s:aggregate([r for r in rs if r['case_id'] in cases]) for s,rs in [('formation_no_salience',formation[m])]+list(rows[m].items())}
            selected=[r for r in phases[m]['cross_truth'] if r['case_id'] in cases]
            coexist[m][str(ratio)]=dict(layers=layers,cross_truth_pairing_error=sum(r['cross_truth_pairing_error'] for r in selected),
                possible_cross_truth_pairing=sum(r['possible_cross_truth'] for r in selected),
                selected_phase_pairs=len(selected),phase_pair_unassigned=sum(r['unassigned_phase'] for r in selected),
                scope='Original native-wall selected phase pairs; unknown phase assignment is not counted as proven safe',phase_assignments=selected)
    save(output/'PAIRING_FILTER_2X2_COEXIST_SAFETY.json',coexist)
    negative={m:{s:dict(all_unique=summary[m][s]['negative'],
        zero=aggregate([r for r in rows[m][s] if r['scenario']['family']=='zero'])['negative'],
        artifact=aggregate([r for r in rows[m][s] if r['scenario']['family']=='artifact' and not r['truth_count']])['negative'],
        by_scenario={name:aggregate([r for r in rows[m][s] if r['scenario']['scenario']==name])['negative']
                     for name in sorted({r['scenario']['scenario'] for r in rows[m][s] if not r['truth_count']})}) for s in ('S1','S2')} for m in MODELS}
    save(output/'PAIRING_FILTER_2X2_NEGATIVE_CONTROLS.json',negative)
    eligible=len(runs);unsupported_runs=sum(r['M2_filter_unsupported'] for r in runs)
    save(output/'PAIRING_FILTER_2X2_FILTER_DIAGNOSTIC.json',dict(summary=filters,
        eligible_M2_native_runs=eligible,unsupported_M2_native_runs=unsupported_runs,unsupported_fraction=unsupported_runs/eligible if eligible else None,
        runs=runs,by_truth_wall={m:phases[m]['filter_rows'] for m in MODELS},
        interpretation='Pure known-truth component diagnostics are separate from actual mixed inputs. Native lobe counts are not patient/event counts.'))
    save(output/'PAIRING_FILTER_2X2_PAIRING_DIAGNOSTIC.json',dict(pattern_summary=pattern,edge_audit=pairings,
        truth_wall_selection={m:phases[m]['pattern_rows'] for m in MODELS},
        asymmetric_mechanism_fixture=fixtures,ringing_like_selected_pairs={m:phases[m]['ringing_like_selected_pairs'] for m in MODELS}))
    save(output/'PAIRING_FILTER_2X2_DOWNSTREAM_LEDGER.json',dict(summary=summary,rows=rows,formation_rows=formation))
    save(output/'PAIRING_FILTER_2X2_FINAL_INTERPRETATION.json',dict(status='pending evidence review',
        allowable_mechanism_conclusions=['pairing-dominant','filter-dominant','coupled filter x pairing','M1 reduces duplication but causes unacceptable event loss/merging','mechanism still unresolved'],
        production_adoption=False,threshold_retuned=False,asymmetric_fixtures_excluded_from_winner=True))
    return primary

def plot_cases(output,cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=output/'plots';folder.mkdir(exist_ok=True);manifest=[]
    for case in sorted({case for case,model in cache}):
        if any((case,m) not in cache for m in MODELS):continue
        fig,axes=plt.subplots(4,2,figsize=(13,12))
        for row,m in enumerate(MODELS):
            r,logs=cache[case,m];s=r['shared'];wall=0;col=len(s['s'])//2;t=s['t'];log=logs[0]
            axes[row,0].plot(t,s['filtered'][:,wall,col],color='black',lw=1)
            for truth in r['truth']:axes[row,0].axvspan(truth['truth_start_s'],truth['truth_end_s'],color='green',alpha=.1)
            axes[row,0].set_title(m+' | filtered A, wall 0 / section '+str(col));axes[row,0].set_ylabel('A (1/s)')
            ridges={r['id']:r for r in log['ridges']}
            for edge in log['edges']:
                if not edge['selected']:continue
                a,b=edge['phase_ridge_ids'];x,y=ridges[a]['center_s'],ridges[b]['center_s']
                axes[row,1].plot([x,y],[0,0],lw=2)
                axes[row,1].scatter([x,y],[0,0],c=['red' if ridges[a]['sign']>0 else 'blue','red' if ridges[b]['sign']>0 else 'blue'],s=30)
                axes[row,1].text((x+y)/2,.07,f"{a}-{b}\nW={edge['W_used']:.2f}",ha='center',fontsize=7)
            axes[row,1].set_ylim(-.2,.35);axes[row,1].set_xlim(t[0],t[-1]);axes[row,1].set_yticks([]);axes[row,1].set_title(m+' | actual selected phase pairs')
        for ax in axes[-1]:ax.set_xlabel('time (s)')
        fig.suptitle(case);fig.tight_layout(rect=(0,0,1,.97));path=folder/(case.replace('/','__')+'.png');fig.savefig(path,dpi=130);plt.close(fig)
        manifest.append(dict(case_id=case,path=str(path.relative_to(output))))
    return manifest
