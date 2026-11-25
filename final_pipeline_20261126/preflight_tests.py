from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from analysis.mse_pregnancy_association.code.measurement.deformation import interval_support
from analysis.mse_pregnancy_association.code.record_mse_core import summarize_paired_windows
from analysis.mse_pregnancy_association.code.normalized_activity import normalize_magnitude

def test_no_cross_segment():
    tracks=dict(target_valid=np.ones((6,1),bool),step_valid=np.array([[0],[1],[1],[0],[1],[1]],bool),
        segment_id=np.array([[0],[0],[0],[1],[1],[1]]),pose_epoch=np.zeros(6,int),time_s=np.arange(6,dtype=float))
    assert interval_support(tracks,2)[:,0].tolist()==[True,False,False,True]

def test_mse_core():
    p=[dict(first_section=1,last_section=3,window_start_s=float(i),window_end_s=float(i+20),window_center_s=float(i+10),
        anterior_ER=.2*(i+1),posterior_ER=.1*(i+1)) for i in range(3)]
    r=summarize_paired_windows(p)
    assert abs(r['record_MSE_time_equal']-np.mean(np.square([.1,.2,.3])))<1e-12 and r['unique_window_count']==3

def test_minmax():
    y=np.array([[[0.,1.,2.],[3.,4.,5.]],[[6.,7.,8.],[9.,10.,11.]]]); v=np.ones_like(y,dtype=bool)
    _,m=normalize_magnitude(y,v)
    assert m['cutoff']==.6 and m['normalization_scope']=='whole_video_both_walls' and abs(m['physical_cutoff']-6.6)<1e-12

def test_r2_synthetic():
    path=Path(__file__).resolve().parent/'r2_frozen'/'extract_longitudinal_frequency.py'
    spec=importlib.util.spec_from_file_location('r2_extract',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    r=mod.synthetic_checks(); k=r['known_frequency_recovery_cpm']
    assert r.get('status')=='pass' and k['1.0']==1.0 and k['2.0']==2.0 and k['3.0']==3.0

def main():
    out=[]
    for fn in [test_no_cross_segment,test_mse_core,test_minmax,test_r2_synthetic]:
        try: fn(); out.append({'test':fn.__name__,'status':'PASS'})
        except Exception as e: out.append({'test':fn.__name__,'status':'FAIL','error':type(e).__name__+': '+str(e)})
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if all(x['status']=='PASS' for x in out) else 2
if __name__=='__main__': raise SystemExit(main())
