"""
test_camera.py — Week 1 camera check + fixture photo capture tool.
Owner: DEV1

Run this FIRST to confirm the camera works:
    python test_camera.py

Keys in the preview window:
    P  save frame to tests/fixtures/pass/  (good part)
    F  save frame to tests/fixtures/fail/  (defective part — rename afterwards,
       e.g. scratch_01.png, dust_01.png, color_anomaly_01.png)
    D  toggle live detection overlay (runs the real ensemble on each frame)
    Focus: the number in the top-left is image sharpness. Turn the lens slowly
           and stop where the number is highest.
    Q  quit

Uses camera settings from config.yaml (device, resolution, backend, fourcc).
Frames are saved as PNG (lossless) so JPEG artifacts don't look like dust.
"""
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.vision.capture import Camera

FIXTURES = Path(__file__).parent / "tests" / "fixtures"
COLORS = {"PASS": (0, 200, 0), "WARN": (0, 200, 255), "FAIL": (0, 0, 255)}


def save(frame, label: str) -> Path:
    folder = FIXTURES / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(path), frame)
    print(f"Saved {path}")
    return path


def sharpness(frame) -> float:
    """Variance of the Laplacian on the center of the frame — higher = sharper."""
    h, w = frame.shape[:2]
    center = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_focus(view, frame, best: float) -> float:
    value = sharpness(frame)
    best = max(best, value)
    color = (0, 200, 0) if value >= 0.9 * best else (0, 200, 255)
    cv2.putText(view, f"Focus {value:.0f} (best {best:.0f})", (10, view.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return best


def draw_detection(frame, ensemble):
    from src.vision.preprocessor import preprocess
    decision = ensemble.run(preprocess(frame))
    view = frame.copy()
    for d in decision.all_defects:
        cv2.rectangle(view, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 0, 255), 2)
        cv2.putText(view, f"{d.defect_type} {d.confidence:.2f}", (d.x, max(15, d.y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    text = f"{decision.result} {decision.confidence:.2f} | {decision.total_duration_ms:.0f}ms"
    cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                COLORS[decision.result], 2)
    return view


def main():
    with Camera() as cam:
        if not cam.is_open():
            print("Camera failed to open. Check USB connection and camera.device_id / backend.")
            return
        print("P=save PASS  F=save FAIL  D=detection overlay  R=reset focus best  Q=quit")
        ensemble = None
        best_focus = 0.0
        frames, t0 = 0, time.perf_counter()
        while True:
            frame = cam.grab_frame()
            if frame is None:
                print("Frame grab failed")
                break
            frames += 1
            if frames == 60:
                fps = frames / (time.perf_counter() - t0)
                print(f"Measured FPS: {fps:.1f} | frame {frame.shape[1]}x{frame.shape[0]}")

            view = draw_detection(frame, ensemble) if ensemble else frame.copy()
            best_focus = draw_focus(view, frame, best_focus)
            cv2.imshow("Arducam UC852 — P:pass F:fail D:detect Q:quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                save(frame, "pass")
            elif key == ord("f"):
                save(frame, "fail")
            elif key == ord("r"):
                best_focus = 0.0
            elif key == ord("d"):
                if ensemble is None:
                    from src.vision.detectors.ensemble import Ensemble
                    ensemble = Ensemble()
                else:
                    ensemble = None
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
