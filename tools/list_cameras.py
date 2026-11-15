"""
list_cameras.py — Find which camera index / backend is the Arducam.
Owner: DEV1

Usage:
    python tools/list_cameras.py

Tries camera indexes 0-5 with every backend available on this OS, asks each
for 1280x800 MJPG, and prints the resolution it actually delivers.
The UC852 should report 1280x800; a laptop webcam usually reports 640x480 or 1280x720.
Put the working index and backend into config.yaml (camera.device_id / camera.backend).
"""
import sys

import cv2

if sys.platform.startswith("win"):
    BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
elif sys.platform == "darwin":
    BACKENDS = {"avfoundation": cv2.CAP_AVFOUNDATION}
else:
    BACKENDS = {"v4l2": cv2.CAP_V4L2}


def probe(index: int, backend: int):
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        ok, frame = cap.read()
        if not ok or frame is None:
            return "opens, but no frame"
        return f"{frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def main():
    cv2.setLogLevel(0) if hasattr(cv2, "setLogLevel") else None
    found = False
    print("index | backend      | result")
    print("------+--------------+---------------------")
    for index in range(6):
        for name, backend in BACKENDS.items():
            result = probe(index, backend)
            if result:
                found = True
                tag = "  <- likely UC852" if result == "1280x800" else ""
                print(f"{index:5d} | {name:12s} | {result}{tag}")
    if not found:
        print("No camera found.")
    print("\nIf only one index shows up, Windows sees only one camera — "
          "check the Arducam USB cable / Device Manager.")


if __name__ == "__main__":
    main()
