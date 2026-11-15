"""
capture.py — Camera initialization and frame capture.
Owner: DEV1
Camera: Arducam UC852 (OV9782) — USB 2.0, no driver needed.
"""
import sys

import cv2
import numpy as np
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}


def resolve_backend(name: str = "auto") -> int:
    """Pick the OpenCV capture backend. 'auto' chooses by operating system."""
    if name in (None, "auto"):
        if sys.platform.startswith("win"):
            return cv2.CAP_DSHOW
        if sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY
    if name not in _BACKENDS:
        raise ValueError(f"Unknown camera.backend: {name!r}")
    return _BACKENDS[name]


class Camera:
    def __init__(self):
        cfg = get_config()["camera"]
        self.device_id = cfg["device_id"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.fps = cfg["fps"]
        self.backend = resolve_backend(cfg.get("backend", "auto"))
        self.fourcc = cfg.get("fourcc")
        self._cap = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_id, self.backend)
        if self.fourcc:
            # Set before size/fps: USB 2.0 needs MJPG for 1280x800 @ 30fps
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok = self._cap.isOpened()
        if ok:
            # Log what the driver actually gave us — it may differ from what we asked for
            aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = self._cap.get(cv2.CAP_PROP_FPS)
            log.info(f"Camera opened | requested {self.width}x{self.height}@{self.fps} "
                     f"| actual {aw}x{ah}@{afps:.0f}")
            if (aw, ah) != (self.width, self.height):
                log.warning("Camera resolution differs from config — check fourcc/USB port")
        else:
            log.error(f"Camera {self.device_id} failed to open")
        return ok

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def grab_frame(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            log.error("Camera not open")
            return None
        ret, frame = self._cap.read()
        if not ret:
            log.warning("Failed to grab frame")
            return None
        return frame

    def release(self):
        if self._cap:
            self._cap.release()
            log.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
