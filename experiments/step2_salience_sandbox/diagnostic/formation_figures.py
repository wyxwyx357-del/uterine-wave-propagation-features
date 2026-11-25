"""Balanced existing-input figures and continuous analytical-component readouts."""
import numpy as np

def component_readout(records,traces,pairs):
    rows=[];seen=set()
    for pair in pairs:
        trace=traces[pair['case_id'],pair['wall']];record=records[pair['case_id']]
        raw,filtered,arrival=trace['components'][pair['truth_id']];wall=pair['wall'];t=record['shared']['t']
        for parents in pair['parent_pair_relations']:
            for bridge in parents['existing_bridge_pairs']:
                if not bridge['connects_main_positive_and_negative'] or bridge['selected']:continue
                core_ids=set(bridge['phase_ridge_ids'])
                a=trace['original'][parents['parent_a']];b=trace['original'][parents['parent_b']]
                if not parents['both_distinct_selected_pairs']:continue
                outer=(set(a['phase_ridge_ids'])|set(b['phase_ridge_ids']))-core_ids
                for role,ids in [('main',core_ids),('peripheral',outer)]:
                    for rid in sorted(ids):
                        ridge=trace['ridges'][rid]
                        for node in ridge['nodes']:
                            c=node['column'];idx=node['peak_index'];key=(pair['case_id'],wall,pair['truth_id'],rid,c,role)
                            if key in seen:continue
                            seen.add(key)
                            actual=record['shared']['filtered'][idx,wall,c];value=filtered[idx,wall,c]
                            rows.append(dict(case_id=pair['case_id'],texture_id=record['source']['texture_id'],wall=wall,truth_id=pair['truth_id'],
                                ridge_id=rid,section=c,role_relative_to_unselected_main_pair=role,sign=node['sign'],peak_s=float(t[idx]),
                                truth_arrival_s=float(arrival[c]),offset_from_truth_arrival_s=float(t[idx]-arrival[c]),
                                actual_raw_A=float(record['shared']['A'][idx,wall,c]),actual_filtered_A=float(actual),
                                injected_truth_raw_A=float(raw[idx,wall,c]),injected_truth_filtered_A=float(value),
                                other_components_and_residual_filtered_A=float(actual-value),
                                signed_component_fraction=float(value/actual) if actual else None,
                                truth_raw_component_zero=bool(raw[idx,wall,c]==0),
                                filtered_truth_component_sign_agrees=bool(value*node['sign']>0)))
    return rows

def produce_figures(records,traces,sources,pairs,output):
    selected=[]
    for texture in sorted({r['texture_id'] for r in pairs}):
        choices=[r for r in pairs if r['point']=='baseline' and r['texture_id']==texture and r['both_final_representations_are_original_bases']]
        choices.sort(key=lambda r:(r['scenario_id']!='medium_C2F',r['scenario_id'],r['wall'],r['unmatched_canonical_source']))
        if choices:selected.append(('base_'+texture,choices[0]))
    for label,select in [('TP_child_scope',lambda r:r['TP_source_kind']=='child'),
                         ('MAD_single_phase_scope',lambda r:any(not p['both_original_parents_paired'] for p in r['parent_pair_relations']))]:
        choices=[r for r in pairs if select(r)]
        choices.sort(key=lambda r:(r['case_id'],r['point'],r['wall']))
        if choices:selected.append((label,choices[0]))
    result=[];folder=output/'plots';folder.mkdir(exist_ok=True)
    for label,pair in selected:
        path=folder/(label+'.png');draw_pair(records[pair['case_id']],traces[pair['case_id'],pair['wall']],sources,pair,path)
        result.append(dict(label=label,case_id=pair['case_id'],point=pair['point'],wall=pair['wall'],truth_id=pair['truth_id'],path=str(path.relative_to(output))))
    return result

def draw_pair(record,trace,sources,pair,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    fig,axes=plt.subplots(3,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1,1.1]})
    a=sources[pair['case_id'],pair['wall'],pair['unmatched_canonical_source']]
    b=sources[pair['case_id'],pair['wall'],pair['TP_canonical_source']]
    relation=next((r for r in pair['parent_pair_relations'] if r['main_pair_partitioned_into_flanking_pairs']),pair['parent_pair_relations'][0])
    pa=next(p for p in a['original_parents'] if p['event_id']==relation['parent_a'])
    pb=next(p for p in b['original_parents'] if p['event_id']==relation['parent_b'])
    ids=set(pa['sections'])&set(pb['sections']);col=sorted(ids)[len(ids)//2]
    wall=pair['wall'];shared=record['shared'];t=shared['t'];raw,filtered,arrival=trace['components'][pair['truth_id']]
    axes[0].plot(t,shared['A'][:,wall,col],color='gray',label='actual raw A',lw=.8)
    axes[0].plot(t,raw[:,wall,col],color='green',label='known injected truth A',lw=1.2)
    axes[1].plot(t,shared['filtered'][:,wall,col],color='black',label='actual filtered A',lw=1.2)
    axes[1].plot(t,filtered[:,wall,col],color='green',label='filtered truth component',lw=1.2)
    axes[1].plot(t,shared['filtered'][:,wall,col]-filtered[:,wall,col],color='gray',label='other truth + residual',lw=.7)
    for ax in axes[:2]:
        ax.axvline(arrival[col],color='green',ls=':',label='truth deformation center')
        ax.axvspan(pa['start_s'],pa['end_s'],color='red',alpha=.07,label='unmatched ORIGINAL parent bounds')
        ax.axvspan(pb['start_s'],pb['end_s'],color='blue',alpha=.07,label='TP ORIGINAL parent bounds')
        ax.set_xlim(max(t[0],min(pa['start_s'],pb['start_s'])-2),min(t[-1],max(pa['end_s'],pb['end_s'])+2))
        ax.legend(ncol=2,fontsize=7);ax.set_xlabel('time (s)');ax.set_ylabel('A (1/s)')
    graph=axes[2];ridge_ids=sorted(set(pa['phase_ridge_ids'])|set(pb['phase_ridge_ids']),key=lambda rid:trace['ridges'][rid]['center_s'])
    xs={rid:i for i,rid in enumerate(ridge_ids)}
    for rid in ridge_ids:
        ridge=trace['ridges'][rid];x=xs[rid];sign=ridge['sign']
        graph.scatter([x],[0],s=150,color='red' if sign>0 else 'blue',zorder=5)
        graph.text(x,-.13,f"r{rid}: {'+' if sign>0 else '-'}\nmedian peak {ridge['center_s']:.2f}s",ha='center',va='top',fontsize=9)
    for parent,color,curve in [(pa,'red',.4),(pb,'blue',-.4)]:
        if len(parent['phase_ridge_ids'])!=2:continue
        u,v=parent['phase_ridge_ids'];weight=parent['association_weight']
        graph.add_patch(FancyArrowPatch((xs[u],0),(xs[v],0),connectionstyle=f'arc3,rad={curve}',arrowstyle='-',color=color,lw=2))
        graph.text((xs[u]+xs[v])/2,.45 if curve>0 else -.53,
            f"SELECTED {parent['event_id']}\nw={weight:.4f}",ha='center',fontsize=8,color=color)
    bridges=[e for e in relation['existing_bridge_pairs'] if e['connects_main_positive_and_negative']]
    for bridge in bridges:
        u,v=bridge['phase_ridge_ids']
        graph.plot([xs[u],xs[v]],[.12,.12],color='black',ls='--',lw=2)
        graph.text((xs[u]+xs[v])/2,.2,f"main + to -: {'SELECTED' if bridge['selected'] else 'NOT SELECTED'}\nw={bridge['association_weight']:.4f}",ha='center',fontsize=8)
    if b['source_kind']=='child':
        graph.text(.02,.97,'TP is an auxiliary split child: graph shows its ORIGINAL parent, not child phase IDs.',transform=graph.transAxes,va='top',fontsize=9,color='purple')
    if not relation['both_original_parents_paired']:
        graph.text(.02,.97,'One source is single-phase: not the paired-base duplication pattern.',transform=graph.transAxes,va='top',fontsize=9,color='purple')
    graph.set_ylim(-.9,.95);graph.set_xlim(-.45,len(ridge_ids)-.55);graph.set_xticks([]);graph.set_yticks([])
    graph.set_xlabel('Original phase ridges in temporal order; red/blue arcs are actual selected pairs')
    fig.suptitle(f"{record['case_id']} | {pair['point']} | wall {wall}, section {col}\n{pair['unmatched_canonical_source']} vs TP {pair['TP_canonical_source']}",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(path,dpi=140);plt.close(fig)
