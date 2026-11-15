"""Unit tests for ROI helpers, preprocessor, ensemble decision and camera wrapper."""
import time
from unittest import mock

import numpy as np
import pytest

from src.vision import capture
from src.vision.detectors.ensemble import Ensemble
from src.vision.preprocessor import preprocess
from src.vision.roi import finding_score, get_roi
from tests import synthetic_images as s


# ── ROI ──────────────────────────────────────────────

def test_get_roi_none_returns_frame():
    frame = np.zeros((10, 20, 3), np.uint8)
    crop, ox, oy = get_roi(frame, None)
    assert crop is frame and (ox, oy) == (0, 0)


def test_get_roi_clips_to_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop, ox, oy = get_roi(frame, [150, 80, 100, 100])
    assert crop.shape[:2] == (20, 50) and (ox, oy) == (150, 80)


def test_get_roi_outside_frame_raises():
    with pytest.raises(ValueError):
        get_roi(np.zeros((100, 200, 3), np.uint8), [300, 300, 10, 10])


@pytest.mark.parametrize("ratio,score,conf", [
    (0.5, 1.0, 0.0), (1.0, 0.5, 0.5), (1.5, 0.25, 0.75), (2.0, 0.0, 1.0), (9.0, 0.0, 1.0),
])
def test_finding_score(ratio, score, conf):
    assert finding_score(ratio) == pytest.approx((score, conf))


# ── Preprocessor ─────────────────────────────────────

def test_preprocess_keeps_size_and_type():
    out = preprocess(s.clean_metal())
    assert out.shape == (800, 1280, 3) and out.dtype == np.uint8


def test_preprocess_resizes_other_resolutions():
    out = preprocess(np.zeros((480, 640, 3), np.uint8))
    assert out.shape == (800, 1280, 3)


def test_preprocess_under_10ms():
    img = s.clean_metal()
    preprocess(img)  # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        preprocess(img)
    assert (time.perf_counter() - t0) / 10 * 1000 < 10


def test_preprocess_rejects_empty_frame():
    with pytest.raises(ValueError):
        preprocess(np.zeros((0, 0, 3), np.uint8))


def test_preprocess_unknown_method(cfg):
    cfg["preprocess"]["denoise"] = "magic"
    with pytest.raises(ValueError):
        preprocess(s.clean_metal())


def test_preprocess_none_is_identity(cfg):
    cfg["preprocess"]["denoise"] = "none"
    img = s.clean_metal()
    assert np.array_equal(preprocess(img), img)


# ── Ensemble decision ────────────────────────────────

@pytest.mark.parametrize("name,img,expected", [
    ("clean metal", s.clean_metal(), "PASS"),
    ("clean coating", s.clean_coated(), "PASS"),
    ("scratch", s.with_scratch(s.clean_metal()), "FAIL"),
    ("dust", s.with_dust(s.clean_metal()), "FAIL"),
    ("missing coating", s.with_color_patch(s.clean_coated()), "FAIL"),
    ("underexposed", s.exposure(s.clean_metal(), 0.2), "FAIL"),
    ("overexposed", s.exposure(s.clean_metal(), 1.9), "FAIL"),
])
def test_ensemble_end_to_end(name, img, expected):
    assert Ensemble().run(preprocess(img)).result == expected, name


def test_single_failing_detector_fails_the_part():
    # Weighted average alone would be 0.5*0 + 0.3*1 + 0.2*1 = 0.5 -> WARN.
    result = Ensemble().run(preprocess(s.with_scratch(s.clean_metal())))
    assert result.scores["color"] == 1.0 and result.scores["brightness"] == 1.0
    assert result.result == "FAIL"


def test_full_pipeline_under_100ms():
    img = s.with_dust(s.clean_metal())
    ens = Ensemble()
    ens.run(preprocess(img))  # warm-up
    t0 = time.perf_counter()
    ens.run(preprocess(img))
    assert (time.perf_counter() - t0) * 1000 < 100


# ── Camera wrapper (no hardware: VideoCapture is mocked) ──

@pytest.mark.parametrize("platform,expected", [
    ("win32", "CAP_DSHOW"), ("linux", "CAP_V4L2"), ("darwin", "CAP_AVFOUNDATION"),
])
def test_resolve_backend_auto(platform, expected):
    with mock.patch.object(capture.sys, "platform", platform):
        assert capture.resolve_backend("auto") == getattr(capture.cv2, expected)


def test_resolve_backend_unknown():
    with pytest.raises(ValueError):
        capture.resolve_backend("firewire")


def _fake_cap(opened=True, frame=True):
    cap = mock.MagicMock()
    cap.isOpened.return_value = opened
    cap.get.side_effect = lambda prop: {3: 1280, 4: 800, 5: 30}.get(prop, 0)
    cap.read.return_value = (True, np.zeros((800, 1280, 3), np.uint8)) if frame else (False, None)
    return cap


def test_camera_open_grab_release():
    fake = _fake_cap()
    with mock.patch.object(capture.cv2, "VideoCapture", return_value=fake) as vc:
        with capture.Camera() as cam:
            assert cam.is_open()
            assert cam.grab_frame().shape == (800, 1280, 3)
        vc.assert_called_once()
    fourcc = capture.cv2.VideoWriter_fourcc(*"MJPG")
    fake.set.assert_any_call(capture.cv2.CAP_PROP_FOURCC, fourcc)
    fake.release.assert_called_once()


def test_camera_grab_failure_returns_none():
    with mock.patch.object(capture.cv2, "VideoCapture", return_value=_fake_cap(frame=False)):
        with capture.Camera() as cam:
            assert cam.grab_frame() is None


def test_camera_not_open():
    with mock.patch.object(capture.cv2, "VideoCapture", return_value=_fake_cap(opened=False)):
        cam = capture.Camera()
        assert cam.open() is False
        assert cam.grab_frame() is None
