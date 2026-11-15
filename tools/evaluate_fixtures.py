"""
evaluate_fixtures.py — Detection accuracy report on the fixture images.
Owner: DEV1

Runs preprocess + ensemble on every image in
    tests/fixtures/pass/   (expected PASS)
    tests/fixtures/fail/   (expected FAIL)
and prints accuracy, false-positive rate, false-negative rate and timing as a
Markdown report you can paste into the weekly update to the CTO.

Usage:
    python tools/evaluate_fixtures.py
    python tools/evaluate_fixtures.py --warn-as fail     # count WARN as a rejection
    python tools/evaluate_fixtures.py --fixtures other/dir --details
    python tools/evaluate_fixtures.py --overlays          # save images with red boxes
                                                          # to tests/fixtures/overlays/

Plan targets: accuracy >= 80% (W3), >= 85% (W8); false positives < 5% (W7);
full pipeline < 100 ms (W6).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.detectors.ensemble import Ensemble  # noqa: E402
from src.vision.preprocessor import preprocess  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def save_overlay(frame, decision, out: Path, roi=None):
    """Draw ROI (blue) and every defect box (red) and write the image."""
    view = frame.copy()
    if roi:
        x, y, w, h = (int(v) for v in roi)
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for d in decision.all_defects:
        cv2.rectangle(view, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 0, 255), 2)
        cv2.putText(view, d.defect_type, (d.x, max(15, d.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(view, f"{decision.result} {decision.confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), view)


def evaluate(fixtures: Path, warn_as: str = "pass", overlays: Path = None) -> dict:
    from src.utils.config import get_config
    roi = get_config()["detectors"].get("roi")
    ensemble = Ensemble()
    rows = []
    for expected in ("pass", "fail"):
        for path in load_images(fixtures / expected):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"WARNING: cannot read {path}", file=sys.stderr)
                continue
            t0 = time.perf_counter()
            decision = ensemble.run(preprocess(frame))
            total_ms = (time.perf_counter() - t0) * 1000
            if overlays is not None:
                save_overlay(frame, decision, overlays / f"{expected}_{path.stem}.png", roi)
            verdict = decision.result
            rejected = verdict == "FAIL" or (verdict == "WARN" and warn_as == "fail")
            rows.append({
                "file": path.name,
                "expected": expected.upper(),
                "result": verdict,
                "score": decision.confidence,
                "correct": rejected == (expected == "fail"),
                "defects": sorted({d.defect_type for d in decision.all_defects}),
                "ms": total_ms,
            })

    good = [r for r in rows if r["expected"] == "PASS"]
    bad = [r for r in rows if r["expected"] == "FAIL"]
    fp = sum(not r["correct"] for r in good)
    fn = sum(not r["correct"] for r in bad)
    times = [r["ms"] for r in rows]
    return {
        "rows": rows,
        "total": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "fp_rate": fp / len(good) if good else 0.0,
        "fn_rate": fn / len(bad) if bad else 0.0,
        "fp": fp, "fn": fn, "n_pass": len(good), "n_fail": len(bad),
        "ms_mean": statistics.mean(times) if times else 0.0,
        "ms_max": max(times) if times else 0.0,
    }


def print_report(rep: dict, details: bool):
    print("## Detection accuracy report\n")
    if rep["total"] == 0:
        print("No fixture images found. Capture some with `python test_camera.py` (P / F keys).")
        return
    print("| Metric | Value | Target |")
    print("|---|---|---|")
    print(f"| Images | {rep['total']} ({rep['n_pass']} PASS / {rep['n_fail']} FAIL) | 30+ |")
    print(f"| Accuracy | {rep['accuracy']:.1%} | >= 85% |")
    print(f"| False positive rate (good part rejected) | {rep['fp_rate']:.1%} ({rep['fp']}) | < 5% |")
    print(f"| False negative rate (defect missed) | {rep['fn_rate']:.1%} ({rep['fn']}) | < 10% |")
    print(f"| Pipeline time mean / max | {rep['ms_mean']:.1f} / {rep['ms_max']:.1f} ms | < 100 ms |")
    wrong = [r for r in rep["rows"] if not r["correct"]]
    shown = rep["rows"] if details else wrong
    if shown:
        print("\n| File | Expected | Result | Score | Defects | ms |")
        print("|---|---|---|---|---|---|")
        for r in shown:
            mark = "" if r["correct"] else " ❌"
            print(f"| {r['file']} | {r['expected']} | {r['result']}{mark} | {r['score']:.2f} "
                  f"| {', '.join(r['defects']) or '-'} | {r['ms']:.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fixtures", type=Path,
                    default=Path(__file__).resolve().parent.parent / "tests" / "fixtures")
    ap.add_argument("--warn-as", choices=("pass", "fail"), default="pass",
                    help="how to count WARN results (default: pass)")
    ap.add_argument("--details", action="store_true", help="list every image, not only errors")
    ap.add_argument("--overlays", action="store_true",
                    help="save images with defect boxes to <fixtures>/overlays/")
    args = ap.parse_args()
    overlays = args.fixtures / "overlays" if args.overlays else None
    print_report(evaluate(args.fixtures, args.warn_as, overlays), args.details)
    if overlays:
        print(f"\nOverlay images saved to {overlays}")


if __name__ == "__main__":
    main()
