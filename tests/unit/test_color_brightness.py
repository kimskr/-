"""Unit tests for ColorDetector and BrightnessDetector."""
import pytest

from src.vision.detectors.brightness import BrightnessDetector
from src.vision.detectors.color import ColorDetector
from tests import synthetic_images as s


# ── Color ────────────────────────────────────────────

def test_color_clean_coating_passes():
    result = ColorDetector().detect(s.clean_coated())
    assert result.score == 1.0
    assert result.defects == []


def test_color_missing_coating_fails():
    result = ColorDetector().detect(s.with_color_patch(s.clean_coated(), frac=0.15))
    assert result.score == 0.0          # 15% >= 2 x 5% threshold
    assert len(result.defects) == 1
    assert result.defects[0].defect_type == "color"


def test_color_small_patch_below_threshold_passes():
    result = ColorDetector().detect(s.with_color_patch(s.clean_coated(), frac=0.02))
    assert result.score == 1.0


def test_color_patch_at_threshold_is_warn_band():
    # 6% anomalous vs 5% threshold -> ratio 1.2 -> score 0.4 .. 0.5
    result = ColorDetector().detect(s.with_color_patch(s.clean_coated(), frac=0.06))
    assert 0.3 < result.score <= 0.5


def test_color_patch_box_location():
    d = ColorDetector().detect(s.with_color_patch(s.clean_coated(), frac=0.15)).defects[0]
    cx, cy = d.x + d.w / 2, d.y + d.h / 2
    assert abs(cx - s.W / 2) < 5 and abs(cy - s.H / 2) < 5


def test_color_roi_ignores_dark_background(cfg):
    img, roi = s.part_on_background()
    assert ColorDetector().detect(img).score == 0.0   # dark conveyor looks like bare metal
    cfg["detectors"]["roi"] = roi
    assert ColorDetector().detect(img).score == 1.0


# ── Brightness ───────────────────────────────────────

def test_brightness_normal_passes():
    result = BrightnessDetector().detect(s.clean_metal())
    assert result.score == 1.0
    assert result.defects == []


@pytest.mark.parametrize("factor,expected", [(0.2, "underexposed"), (1.9, "overexposed")])
def test_brightness_exposure_problems(factor, expected):
    result = BrightnessDetector().detect(s.exposure(s.clean_metal(), factor))
    assert result.score < 0.5
    assert expected in [d.defect_type for d in result.defects]


def test_brightness_slightly_dark_is_warn_band():
    # mean ~35 vs mean_min 40 -> 5 levels past limit, margin 20 -> score 0.375
    img = s.exposure(s.clean_metal(level=140), 35 / 140)
    result = BrightnessDetector().detect(img)
    assert 0.3 < result.score < 0.5


def test_brightness_glare_is_uneven_lighting():
    result = BrightnessDetector().detect(s.with_glare(s.clean_metal(level=100)))
    assert result.score < 1.0
    assert [d.defect_type for d in result.defects] == ["uneven_lighting"]


def test_brightness_accepts_grayscale():
    gray = s.clean_metal()[:, :, 1]
    assert BrightnessDetector().detect(gray).score == 1.0
