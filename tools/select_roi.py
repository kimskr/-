"""
select_roi.py — Pick the inspection area (ROI) with the mouse and save it to config.yaml.
Owner: DEV1

Usage:
    python tools/select_roi.py                     # uses the newest image in tests/fixtures/pass
    python tools/select_roi.py path/to/image.png

Drag a rectangle over the FLAT area of the part (leave out edges, logos, holes,
camera bumps and background), then press ENTER or SPACE. Press C to cancel.
The ROI is written to config.yaml as detectors.roi: [x, y, w, h].
"""
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"


def newest_fixture() -> Path:
    images = sorted((ROOT / "tests" / "fixtures" / "pass").glob("*.png"),
                    key=lambda p: p.stat().st_mtime)
    if not images:
        sys.exit("No images in tests/fixtures/pass. Capture some with: python test_camera.py")
    return images[-1]


def write_roi(roi) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    new_line = f"  roi: [{', '.join(str(int(v)) for v in roi)}]"
    text, n = re.subn(r"^  roi: .*?(?=\s+#|$)", new_line, text, count=1, flags=re.M)
    if n != 1:
        sys.exit("Could not find 'roi:' under detectors in config.yaml")
    CONFIG.write_text(text, encoding="utf-8")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_fixture()
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Cannot read {path}")
    print(f"Image: {path}")
    print("Drag over the flat area of the part, then ENTER. C = cancel.")
    x, y, w, h = cv2.selectROI("Select inspection area (ENTER = ok, C = cancel)", img,
                               showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("Cancelled — config.yaml not changed.")
        return
    write_roi([x, y, w, h])
    print(f"Saved to config.yaml: detectors.roi: [{x}, {y}, {w}, {h}]")


if __name__ == "__main__":
    main()
