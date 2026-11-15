# AI Manufacturing Quality Platform

**Automotive parts quality inspection — Detection → Root Cause → Pre-ERP Alert**

## Stack
- **Vision**: Python 3.10 + OpenCV (→ C++17 in Phase 2)
- **API**: FastAPI + WebSocket
- **DB**: SQLite (→ PostgreSQL in production)
- **AI**: Claude API (Anthropic)
- **Camera**: Arducam UC852 (OV9782 color global shutter, USB 2.0)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in .env
cp .env.example .env            # Windows PowerShell: copy .env.example .env

# 3. Find the camera, then test it first
python tools/list_cameras.py    # put the index into config.yaml camera.device_id
python test_camera.py

# 4. Run inspection loop
python main.py

# 5. Run API server (separate terminal)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Run tests
pytest tests/ -v
```

## Project Structure
```
quality_platform/
├── cpp/                    # C++ vision layer (Phase 2)
├── src/
│   ├── vision/             # Camera + detectors (DEV1)
│   │   ├── capture.py      #   camera open / grab
│   │   ├── preprocessor.py #   resize + denoise
│   │   ├── roi.py          #   inspection area helpers
│   │   └── detectors/      #   surface, color, brightness, ensemble
│   ├── decision/           # PASS/FAIL engine (CTO) — empty; ensemble.py decides today
│   ├── analytics/          # Root cause engine (CTO)
│   ├── db/                 # Database layer (DEV2)
│   ├── api/                # FastAPI server (DEV2)
│   └── utils/              # Config, logger, Claude
├── tests/                  # All tests
│   └── fixtures/pass|fail  # Real camera images (PASS / FAIL ground truth)
├── tools/                  # DEV1 tools: camera finder, ROI picker, accuracy report
├── claude_prompts/         # Saved Claude prompts
├── config.yaml             # All thresholds — tune here
└── main.py                 # Entry point
```

## Status

| Week | Milestone | Status |
|------|-----------|--------|
| W1 | Camera live, dev environment | ✅ Done — UC852 at 1280x800, fixture capture tool |
| W2 | Vision pipeline core | ✅ Done — capture + preprocess (~1.6 ms) |
| W3 | Surface defect detector | ✅ Done — scratch / dust |
| W4 | Color + ensemble + full loop | 🟡 Vision done (color, brightness, ensemble); API/DB loop pending |
| W5–W8 | Analytics, dashboard, pre-ERP, hardening | ⬜ Not started |

### Detection accuracy (DEV1)
Stand-in part: phone back. Fixed camera, lighting and part position.
Images: `tests/fixtures/pass` and `tests/fixtures/fail`.

| Metric | Result | Target |
|--------|--------|--------|
| Fixtures | 30 (15 PASS / 15 FAIL) | 30+ |
| Accuracy | 100% | ≥ 85% |
| False positive rate | 0% | < 5% |
| False negative rate | 0% | < 10% |
| Pipeline time per frame | 3–13 ms | < 100 ms |
| 10-min live run | 0 false FAILs, defect → FAIL immediately | 2-hour run (W7) |

> Real factory parts are still needed. These numbers come from a stand-in object.
> `tests/unit/test_real_fixtures.py` fails if accuracy drops below 85% or false positives reach 5%.

## Vision Pipeline (DEV1)

```
camera → preprocess → [surface, color, brightness] → ensemble → PASS / WARN / FAIL
```

| Module | What it does |
|--------|--------------|
| `src/vision/capture.py` | Opens the camera; picks the backend by OS (DSHOW / V4L2 / AVFoundation); requests MJPG for 1280x800 @ 30 fps over USB 2.0 |
| `src/vision/preprocessor.py` | Resizes if needed and applies a Gaussian denoise (~1.6 ms) |
| `src/vision/roi.py` | Restricts detection to the inspection area (`detectors.roi`) |
| `detectors/surface.py` | Finds scratches and dust: black-hat / top-hat morphology + Canny edge support; elongated = scratch, compact = dust |
| `detectors/color.py` | Finds coating anomalies: fraction of ROI pixels inside an HSV range |
| `detectors/brightness.py` | Flags underexposed, overexposed or unevenly lit frames (mean / std) |
| `detectors/ensemble.py` | Weighted score, capped by the worst detector, so one clear defect fails the part |

**Score convention (all detectors):** 1.0 = no finding, 0.5 = exactly at threshold (WARN), 0.0 = 2× threshold (FAIL).

## DEV1 Tools

```bash
python tools/list_cameras.py                   # find the camera index / backend
python test_camera.py                          # live view: P = save PASS, F = save FAIL,
                                               # D = detection overlay, focus meter, Q = quit
python tools/select_roi.py                     # drag the inspection area -> config.yaml
python tools/evaluate_fixtures.py --details    # accuracy / FP / FN / timing report
python tools/evaluate_fixtures.py --overlays   # save images with defect boxes
```

### Fixture capture rules
1. Fix the camera, lighting and lens focus. Don't change them afterwards.
2. Place the part in the same position every time. Mark it with tape or use a jig.
3. Set the ROI to the flat inspection area only. Leave out edges, logos, holes and background.
4. If the camera or lighting changes, re-capture all fixtures and re-select the ROI.

## Known Issues
- CI triggers on `main` / `develop`, but the default branch is `master`, so CI never runs.
- About 40 flake8 errors in `src/api`, `src/db` and `src/utils` will fail the lint job.
- `python-dotenv` is missing from `requirements.txt` (`main.py` imports it).
- On Python 3.12+, `pip install -r requirements.txt` may fail on the pinned numpy. Use Python 3.10, or `pip install opencv-python numpy pyyaml` for the vision tools only.
- Defect type labels are sometimes wrong even when PASS/FAIL is right. For example, a small black tape patch is labeled `scratch`, not `color`.

## Roles
- 🔵 **CTO**: Architecture, CI/CD, root cause engine, Claude integration
- 🟢 **DEV1**: Vision algorithms, camera pipeline, test fixtures
- 🟡 **DEV2**: Database, FastAPI, dashboard, ERP webhook

## 8-Week MVP
| Week | Milestone |
|------|-----------|
| W1 | Camera live, dev environment |
| W2 | Vision pipeline core |
| W3 | Surface defect detector |
| W4 | Color + ensemble + full loop |
| W5 | Root cause analytics engine |
| W6 | Live dashboard + WebSocket |
| W7 | Pre-ERP alert + Claude reports |
| W8 | Production hardening + CI/CD |
