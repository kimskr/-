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


def evaluate(fixtures: Path, warn_as: str = "pass") -> dict:
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
    args = ap.parse_args()
    print_report(evaluate(args.fixtures, args.warn_as), args.details)


if __name__ == "__main__":
    main()
