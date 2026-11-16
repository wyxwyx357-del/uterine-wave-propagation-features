"""Local video decoding. Retain all decoded frames and source presentation times."""
from pathlib import Path
import numpy as np


def read_video(path, *, time_factor, coordinate_scale=.5):
    """time_factor=3 for a verified 3x speed-up; factor is never inferred from a name."""
    import cv2
    if time_factor <= 0 or not np.isfinite(time_factor) or not 0 < coordinate_scale <= 1:
        raise ValueError("Invalid time factor or coordinate scale")
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        raise ValueError("Cannot open video")
    frames, times = [], []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000*time_factor)
            if coordinate_scale != 1:
                image = cv2.resize(image, None, fx=coordinate_scale, fy=coordinate_scale,
                                   interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    t = np.asarray(times, float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Decoder did not provide increasing timestamps; supply explicit acquisition times via the Python API")
    return frames, t-t[0]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
