"""Existing CUDA LK and grayscale QC, without machine-specific imports."""
import cv2
import numpy as np

_FLOWS = {}


def local_contrast(gray, sigma=5.):
    source = gray.astype(np.float32)
    mean = cv2.GaussianBlur(source, (0, 0), sigma)
    square = cv2.GaussianBlur(source*source, (0, 0), sigma)
    std = np.sqrt(np.maximum(square-mean*mean, 0))
    mapped = 128+35*(source-mean)/(std+5)
    return np.uint8(np.where(gray > 12, np.clip(np.rint(mapped), 0, 255), 0))


def patch(image, point, radius):
    x, y = point
    h, w = image.shape
    if not np.isfinite(point).all() or min(x,y) < radius+1 or x >= w-radius-1 or y >= h-radius-1:
        return None
    return cv2.getRectSubPix(image, (2*radius+1, 2*radius+1), (float(x), float(y)))


def flow(a, b, p, window, initial=None):
    p = np.asarray(p, np.float32).reshape(1, -1, 2)
    if initial is not None:
        initial = np.asarray(initial, np.float32)
        if initial.shape != p.reshape(-1, 2).shape or not np.isfinite(initial).all():
            raise ValueError("Initial target must be finite Nx2")
    key = (window, initial is not None)
    if key not in _FLOWS:
        if cv2.cuda.getCudaEnabledDeviceCount() < 1:
            raise RuntimeError("CUDA OpenCV with cudaoptflow is required; no CPU fallback is used")
        cv2.cuda.setDevice(0)
        factory = getattr(cv2, 'cuda_SparsePyrLKOpticalFlow', None)
        _FLOWS[key] = (factory.create((window, window), 3, 30, initial is not None)
                      if factory is not None else
                      cv2.cuda.SparsePyrLKOpticalFlow_create((window, window), 3, 30, initial is not None))
    def upload(x):
        gpu = cv2.cuda_GpuMat()
        gpu.upload(np.ascontiguousarray(x))
        return gpu
    target = upload(initial.reshape(1, -1, 2)) if initial is not None else None
    q, status, err = _FLOWS[key].calc(upload(a), upload(b), upload(p), target)
    return q.download().reshape(-1, 1, 2), status.download().reshape(-1, 1), None


def match(a,b,qa,qb,p,window,initial=None,*,max_fb_px=.75,qc_scale=1.,qc_window=None,return_diagnostics=False):
    found,ok,_=flow(a,b,p,window,initial)
    points=found[:,0];back,bok,_=flow(b,a,points,window,p if initial is not None else None)
    fb=np.linalg.norm(back[:,0]-p,axis=1)
    good=ok.ravel().astype(bool)&bok.ravel().astype(bool)&np.isfinite(points).all(1)&(fb<=max_fb_px)
    # Pixel-center convention of spatial downsampling; localization stays native.
    qp,qq=(p,points) if qc_scale==1 else ((p+.5)/qc_scale-.5,(points+.5)/qc_scale-.5)
    radius=int(round((window//2)/qc_scale)) if qc_window is None else qc_window//2
    score=np.full(len(p),-1.);reason=np.full(len(p),'fb_or_flow_rejected',dtype='U24')
    for j in np.flatnonzero(good):
        x,y=patch(qa,qp[j],radius),patch(qb,qq[j],radius)
        if x is None or y is None:good[j]=False;reason[j]='patch_outside';continue
        if min(x.std(),y.std())<2:good[j]=False;reason[j]='low_texture';continue
        x=x.astype(float).ravel();y=y.astype(float).ravel();x-=x.mean();y-=y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
        score[j]=float(x@y/den) if den>1e-10 else -1.
        good[j]=score[j]>=.75;reason[j]='accepted' if good[j] else 'ncc_rejected'
    result = (points,good,score,fb,reason)
    if return_diagnostics:
        return (*result, dict(forward_status=ok.ravel().astype(bool),
                              backward_status=bok.ravel().astype(bool)))
    return result

