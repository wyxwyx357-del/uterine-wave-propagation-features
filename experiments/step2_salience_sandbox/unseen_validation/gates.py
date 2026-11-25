"""User-prespecified gates; exact rational comparisons, no fitted tolerance."""
from fractions import Fraction

DEFINITIONS=dict(
 G1_1='M0 A2 > 0 and (A2_M0-A2_M1)/A2_M0 >= 4/5',
 G1_2='nonworsening texture fraction >= 3/4',
 G1_3='strict improvement fraction among M0-positive A2 textures >= 3/4',
 G1_4='max signed texture contribution / sum signed reductions <= 1/2; denominator must be positive',
 G1_5='every leave-one-texture-out M0 denominator >0 and reduction >= 3/4',
 G2='F1_M1-F1_M0 >= 1/10',
 G3='each clear/medium/structured_weak recall difference >= -1/50',
 G4_1='pooled coexist cross_truth_pairing_error_M1 <= M0',
 G4_2='pooled coexist merge_two_truth_error_M1 <= M0',
 G4_3='pooled coexist recall difference >= -1/50; no per-ratio veto',
 G5='unique pooled negative Step1 upper FP/min_M1 <= M0')

def ratio(a,b):return Fraction(a,b) if b else None
def f1(row):return ratio(2*row['TP'],2*row['TP']+row['FP']+row['FN'])
def recall(row):return ratio(row['TP'],row['truth'])
def display(value):return float(value) if value is not None else None

def decide(aggregate,texture,coexist,negative,formal_eligible):
    ids=sorted(texture);N=len(ids);d0=aggregate['M0']['A2'];d1=aggregate['M1']['A2'];total=d0-d1
    reduction=ratio(total,d0);delta={t:texture[t]['M0']['A2']-texture[t]['M1']['A2'] for t in ids}
    positive=[t for t in ids if texture[t]['M0']['A2']>0]
    nonworsening=ratio(sum(delta[t]>=0 for t in ids),N)
    strict=ratio(sum(delta[t]>0 for t in positive),len(positive))
    contributions={t:ratio(delta[t],total) if total>0 else None for t in ids}
    loo={t:ratio(total-delta[t],d0-texture[t]['M0']['A2']) for t in ids}
    sub={
        'G1.1':d0>0 and reduction is not None and reduction>=Fraction(4,5),
        'G1.2':nonworsening is not None and nonworsening>=Fraction(3,4),
        'G1.3':strict is not None and strict>=Fraction(3,4),
        'G1.4':total>0 and bool(contributions) and max(contributions.values())<=Fraction(1,2),
        'G1.5':bool(loo) and all(v is not None and v>=Fraction(3,4) for v in loo.values())}
    F0=f1(aggregate['M0']);F1=f1(aggregate['M1']);difference=F1-F0 if F0 is not None and F1 is not None else None
    g3={};recall_differences={}
    for label in ('clear','medium','structured_weak'):
        a=aggregate['M0']['strata'][label];b=aggregate['M1']['strata'][label]
        r0=ratio(a['TP'],a['truth']);r1=ratio(b['TP'],b['truth']);dr=r1-r0 if r0 is not None and r1 is not None else None
        g3[label]=dr is not None and dr>=Fraction(-1,50);recall_differences[label]=display(dr)
    c0=coexist['M0'];c1=coexist['M1'];r0=ratio(c0['TP'],c0['truth_count']);r1=ratio(c1['TP'],c1['truth_count'])
    dc=r1-r0 if r0 is not None and r1 is not None else None
    g4={'G4.1':c1['cross_truth_pairing_error']<=c0['cross_truth_pairing_error'],
        'G4.2':c1['merge_two_truth_error']<=c0['merge_two_truth_error'],
        'G4.3':dc is not None and dc>=Fraction(-1,50)}
    def negative_rate(row):
        return Fraction(row['FP_max'])/Fraction(str(row['minutes'])) if row['FP_max'] is not None and row['minutes']>0 else None
    n0=negative_rate(negative['M0']);n1=negative_rate(negative['M1'])
    gates=dict(Gate1=all(sub.values()),Gate2=difference is not None and difference>=Fraction(1,10),
        Gate3=all(g3.values()),Gate4=all(g4.values()),Gate5=n0 is not None and n1 is not None and n1<=n0)
    status=('PASS' if all(gates.values()) else 'FAIL') if formal_eligible else 'PILOT_ONLY_NO_FORMAL_DECISION'
    return dict(M1_UNSEEN_VALIDATION=status,formal_eligible=formal_eligible,gates=gates,Gate1_subgates=sub,Gate3_subgates=g3,Gate4_subgates=g4,
        evidence=dict(A2_M0=d0,A2_M1=d1,A2Reduction=display(reduction),signed_delta_A2=delta,
            nonworsening_fraction=display(nonworsening),strict_improvement_positive_fraction=display(strict),
            positive_M0_A2_texture_count=len(positive),contribution_denominator=total,
            texture_contributions={t:display(v) for t,v in contributions.items()},
            max_single_texture_contribution=display(max(contributions.values())) if total>0 else None,
            leave_one_texture_out={t:display(v) for t,v in loo.items()},
            leave_one_texture_out_min=display(min(loo.values())) if loo and all(v is not None for v in loo.values()) else None,
            F1_difference=display(difference),recall_differences=recall_differences,pooled_coexist_recall_difference=display(dc)),
        comparison_arithmetic='Exact fractions from counts; no epsilon, no rounding before gates',definitions=DEFINITIONS)
