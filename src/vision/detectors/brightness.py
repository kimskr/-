"""
brightness.py — Detects over/under exposure and uneven lighting using global stats.
Owner: DEV1

Measures the mean and standard deviation of gray levels inside the ROI.
  mean < mean_min  -> 'underexposed'
  mean > mean_max  -> 'overexposed'
  std  > std_max   -> 'uneven_lighting' (glare, shadow, reflection)
How far a value is past its limit, relative to `margin`, sets the score.
"""
import time

import cv2
import numpy as np

from src.utils.config import get_config
from src.vision.detectors.base import Detector, DetectionResult, DefectRegion
from src.vision.roi import get_roi, finding_score


class BrightnessDetector(Detector):
    @property
    def name(self) -> str:
        return "brightness"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        cfg = get_config()["detectors"]
        bcfg = cfg["brightness"]
        roi, ox, oy = get_roi(frame, cfg.get("roi"))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        mean, std = cv2.meanStdDev(gray)
        mean, std = float(mean[0][0]), float(std[0][0])
        margin = float(bcfg.get("margin", 20))

        # (defect_type, distance past the limit in gray levels)
        checks = [
            ("underexposed", bcfg["mean_min"] - mean),
            ("overexposed", mean - bcfg["mean_max"]),
            ("uneven_lighting", std - bcfg["std_max"]),
        ]

        score = 1.0
        defects = []
        h, w = gray.shape[:2]
        for defect_type, excess in checks:
            if excess <= 0:
                continue
            # excess just above 0 -> ratio 1.0 (WARN), excess >= margin -> ratio 2.0 (FAIL)
            s, conf = finding_score(1.0 + excess / margin)
            score = min(score, s)
            defects.append(DefectRegion(x=ox, y=oy, w=w, h=h,
                                        confidence=round(conf, 3),
                                        defect_type=defect_type))

        duration = (time.perf_counter() - start) * 1000
        return DetectionResult(score=score, defects=defects,
                               duration_ms=duration, detector_name=self.name)
