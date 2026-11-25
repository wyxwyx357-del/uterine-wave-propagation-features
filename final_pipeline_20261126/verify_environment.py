from __future__ import annotations
import json, sys, hashlib
from pathlib import Path

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    result = {"python": sys.version.split()[0]}
    import numpy, pandas, scipy
    result.update(numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__)
    try:
        import cv2
        result["opencv"] = cv2.__version__
        result["cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount())
        result["cuda_sparse_pyr_lk_available"] = bool(
            getattr(cv2, "cuda_SparsePyrLKOpticalFlow", None) or
            getattr(cv2.cuda, "SparsePyrLKOpticalFlow_create", None)
        )
    except Exception as e:
        result["opencv_error"] = f"{type(e).__name__}: {e}"
    try:
        from analysis.mse_pregnancy_association.code.measurement.tracking import track_segments
        from analysis.mse_pregnancy_association.code.record_mse_core import summarize_paired_windows
        from analysis.mse_pregnancy_association.code.batch_wave_counts import count_map
        from experiments.step2_salience_sandbox.mechanism_2x2.mechanisms import intervention
        result["core_imports"] = "ok"
    except Exception as e:
        result["core_imports"] = f"FAILED: {type(e).__name__}: {e}"
    root = Path(__file__).resolve().parent
    repo = root.parent
    lock = json.loads((root / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    bad_git=[]
    for rel, expected in lock["git_blob_sha1"].items():
        path=repo/rel
        if not path.is_file() or git_blob_sha1(path) != expected:
            bad_git.append(rel)
    bad_r2=[]
    for rel, expected in lock["r2_sha256"].items():
        path=root/"r2_frozen"/rel
        if not path.is_file() or sha256(path) != expected:
            bad_r2.append(rel)
    result["source_lock_git_mismatch"] = bad_git
    result["source_lock_r2_mismatch"] = bad_r2
    r2 = root / "r2_frozen" / "code" / "verify_cuda_backend.py"
    result["r2_source_present"] = r2.is_file()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    bad = result.get("core_imports") != "ok" or result.get("cuda_devices", 0) < 1 or not result.get("cuda_sparse_pyr_lk_available", False) or not r2.is_file() or bool(result["source_lock_git_mismatch"]) or bool(result["source_lock_r2_mismatch"])
    raise SystemExit(2 if bad else 0)

if __name__ == "__main__":
    main()
