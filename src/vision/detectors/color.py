"""
color.py — Detects color anomalies (e.g. missing coating) using HSV range analysis.
Owner: DEV1

Pixels inside [hsv_lower, hsv_upper] are treated as anomalous. The default
range (low saturation, dark) catches bare/uncoated spots on a coated part.
If the anomalous fraction of the ROI reaches `anomaly_threshold`, the
connected anomalous areas are reported as 'color' defects.
"""
import time

import cv2
import numpy as np

from src.utils.config import get_config
from src.vision.detectors.base import Detector, DetectionResult, DefectRegion
from src.vision.roi import get_roi, finding_score


class ColorDetector(Detector):
    @property
    def name(self) -> str:
        return "color"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        cfg = get_config()["detectors"]
        ccfg = cfg["color"]
        roi, ox, oy = get_roi(frame, cfg.get("roi"))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(ccfg["hsv_lower"], dtype=np.uint8)
        upper = np.array(ccfg["hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        ratio = cv2.countNonZero(mask) / float(mask.size)
        score, conf = finding_score(ratio / ccfg["anomaly_threshold"])

        defects = []
        if conf > 0:
            min_area = ccfg.get("min_region_area_px", 50)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                defects.append(DefectRegion(x=x + ox, y=y + oy, w=w, h=h,
                                            confidence=round(conf, 3),
                                            defect_type="color"))

        duration = (time.perf_counter() - start) * 1000
        return DetectionResult(score=score, defects=defects,
                               duration_ms=duration, detector_name=self.name)
