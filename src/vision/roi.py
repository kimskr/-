"""
roi.py — Region-of-interest helpers shared by all detectors.
Owner: DEV1

The ROI is the part of the frame that contains the inspected part.
Detectors analyse only the ROI so that the conveyor, background and
part outline do not produce false positives.
"""
import numpy as np


def get_roi(frame: np.ndarray, roi) -> tuple[np.ndarray, int, int]:
    """
    Crop `frame` to `roi` ([x, y, w, h] or None).
    Returns (crop, offset_x, offset_y). The ROI is clipped to the frame,
    and an ROI that falls completely outside the frame raises ValueError.
    """
    if roi is None:
        return frame, 0, 0
    x, y, w, h = (int(v) for v in roi)
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside the frame ({fw}x{fh})")
    return frame[y0:y1, x0:x1], x0, y0


def finding_score(ratio: float) -> tuple[float, float]:
    """
    Convert "how far past the threshold" into (score, confidence).
    ratio = measured / threshold, so ratio 1.0 means exactly at threshold.

        ratio < 1.0  -> score 1.0, confidence 0.0   (no finding)
        ratio = 1.0  -> score 0.5, confidence 0.5   (WARN band)
        ratio >= 2.0 -> score 0.0, confidence 1.0   (FAIL)
    """
    if ratio < 1.0:
        return 1.0, 0.0
    confidence = min(1.0, ratio / 2.0)
    return 1.0 - confidence, confidence
