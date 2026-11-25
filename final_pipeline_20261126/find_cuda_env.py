from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

CHECK = r'''import json
try:
 import cv2
 d=int(cv2.cuda.getCudaEnabledDeviceCount())
 ok=bool(getattr(cv2,'cuda_SparsePyrLKOpticalFlow',None) or getattr(cv2.cuda,'SparsePyrLKOpticalFlow_create',None))
 print(json.dumps({'ok': d>0 and ok, 'opencv':cv2.__version__, 'cuda_devices':d, 'sparse_lk':ok}))
except Exception as e:
 print(json.dumps({'ok':False,'error':type(e).__name__+': '+str(e)}))
'''

def probe(python):
    try:
        p=subprocess.run([str(python),'-c',CHECK],capture_output=True,text=True,timeout=20)
        line=(p.stdout.strip().splitlines() or ['{}'])[-1]
        data=json.loads(line)
        data['python']=str(python); data['returncode']=p.returncode
        return data
    except Exception as e:
        return {'ok':False,'python':str(python),'error':type(e).__name__+': '+str(e)}

def main():
    candidates=[Path(sys.executable)]
    try:
        p=subprocess.run(['conda','env','list','--json'],capture_output=True,text=True,timeout=20,check=True)
        for root in json.loads(p.stdout).get('envs',[]):
            py=Path(root)/('python.exe' if sys.platform.startswith('win') else 'bin/python')
            if py.exists() and py not in candidates: candidates.append(py)
    except Exception: pass
    results=[probe(p) for p in candidates]
    print(json.dumps(results,ensure_ascii=False,indent=2))
    good=[r for r in results if r.get('ok')]
    if good:
        print('\nUSE_PYTHON='+good[0]['python']); return 0
    print('\nNO_CUDA_OPENCV_ENV_FOUND'); return 2

if __name__=='__main__': raise SystemExit(main())
