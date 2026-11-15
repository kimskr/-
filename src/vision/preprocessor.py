"""
preprocessor.py — Frame preprocessing before detection.
Owner: DEV1
Input: raw BGR frame from camera
Output: normalized frame ready for detectors
Target: < 10 ms per 1280x800 frame.

Denoise options (config: preprocess.denoise):
  gaussian — fast (~1 ms). Default.
  nlmeans  — strong but slow (often 100 ms+ at 1280x800); it can also erase
             fine scratches, so use it only for experiments.
  none     — no denoising.
"""
import cv2
import numpy as np

from src.utils.config import get_config


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize (only if needed) -> denoise -> return BGR frame ready for detectors."""
    if frame is None or frame.size == 0:
        raise ValueError("preprocess() received an empty frame")

    cfg = get_config()
    cam = cfg["camera"]
    pcfg = cfg.get("preprocess", {})

    if frame.shape[1] != cam["width"] or frame.shape[0] != cam["height"]:
        frame = cv2.resize(frame, (cam["width"], cam["height"]), interpolation=cv2.INTER_AREA)

    method = pcfg.get("denoise", "gaussian")
    if method == "gaussian":
        k = int(pcfg.get("gaussian_ksize", 3))
        k = k if k % 2 == 1 else k + 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    elif method == "nlmeans":
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
    elif method != "none":
        raise ValueError(f"Unknown preprocess.denoise: {method!r}")
    return frame
