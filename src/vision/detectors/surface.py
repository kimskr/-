"""
surface.py — Detects scratches and dust using morphology + Canny edge support.
Owner: DEV1

Pipeline (inside the ROI):
  1. Gray + Gaussian blur.
  2. Black-hat and top-hat with a `morph_kernel_px` kernel: keeps only dark and
     bright features SMALLER than the kernel, so the part's own shape and slow
     lighting gradients are ignored.
  3. Threshold at `contrast_threshold` gray levels -> defect candidates.
  4. Morphological close to join broken scratch segments.
  5. Keep candidates that touch Canny edges (rejects soft blotches) and are at
     least `min_defect_area_px` in area.
  6. Elongated candidates -> 'scratch', compact ones -> 'dust'.
Each defect's confidence comes from its contrast relative to the threshold.
"""
import time

import cv2
import numpy as np

from src.utils.config import get_config
from src.vision.detectors.base import Detector, DetectionResult, DefectRegion
from src.vision.roi import get_roi, finding_score


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


class SurfaceDetector(Detector):
    @property
    def name(self) -> str:
        return "surface"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        cfg = get_config()["detectors"]
        scfg = cfg["surface"]
        roi, ox, oy = get_roi(frame, cfg.get("roi"))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        k = _odd(scfg.get("blur_ksize", 5))
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        # 2. Small dark (black-hat) and bright (top-hat) features
        mk = _odd(scfg.get("morph_kernel_px", 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        contrast = cv2.max(blackhat, tophat)

        # 3. Threshold + 4. close small gaps
        thr = float(scfg["contrast_threshold"])
        _, mask = cv2.threshold(contrast, thr - 1, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # 5. Canny edge support (dilated so thin edges overlap the candidate)
        edges = cv2.Canny(gray, scfg["canny_low"], scfg["canny_high"])
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        min_area = scfg["min_defect_area_px"]
        min_edge_ratio = scfg.get("min_edge_ratio", 0.05)
        aspect_limit = scfg.get("scratch_aspect_ratio", 3.0)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        defects = []
        score = 1.0
        for i in range(1, n):  # 0 is the background
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            component = labels[y:y + h, x:x + w] == i
            edge_ratio = np.count_nonzero(edges[y:y + h, x:x + w][component]) / area
            if edge_ratio < min_edge_ratio:
                continue

            peak = float(np.percentile(contrast[y:y + h, x:x + w][component], 90))
            s, conf = finding_score(peak / thr)
            score = min(score, s)

            # 6. Classify by elongation of the rotated bounding box
            pts = np.column_stack(np.nonzero(component))[:, ::-1].astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            elongation = max(rw, rh) / max(1.0, min(rw, rh))
            defect_type = "scratch" if elongation >= aspect_limit else "dust"

            defects.append(DefectRegion(x=int(x) + ox, y=int(y) + oy, w=int(w), h=int(h),
                                        confidence=round(conf, 3),
                                        defect_type=defect_type))

        duration = (time.perf_counter() - start) * 1000
        return DetectionResult(score=score, defects=defects,
                               duration_ms=duration, detector_name=self.name)
