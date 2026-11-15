"""Unit tests for SurfaceDetector (scratch / dust)."""
import pytest

from src.vision.detectors.surface import SurfaceDetector
from src.vision.preprocessor import preprocess
from tests import synthetic_images as s


def run(img):
    return SurfaceDetector().detect(preprocess(img))


@pytest.mark.parametrize("seed,level", [(0, 90), (1, 140), (2, 200), (3, 140)])
def test_clean_metal_has_no_defects(seed, level):
    result = run(s.clean_metal(seed, level))
    assert result.defects == []
    assert result.score == 1.0


def test_clean_coated_has_no_defects():
    assert run(s.clean_coated()).defects == []


def test_scratch_detected_and_classified():
    result = run(s.with_scratch(s.clean_metal()))
    assert result.score < 0.5
    assert [d.defect_type for d in result.defects] == ["scratch"]


def test_scratch_bounding_box_covers_the_line():
    result = run(s.with_scratch(s.clean_metal(), p1=(300, 250), p2=(800, 420)))
    d = result.defects[0]
    assert d.x <= 305 and d.y <= 255
    assert d.x + d.w >= 795 and d.y + d.h >= 415


def test_dust_detected_and_classified():
    result = run(s.with_dust(s.clean_metal(), n=4))
    assert result.score < 0.5
    assert len(result.defects) == 4
    assert {d.defect_type for d in result.defects} == {"dust"}


def test_stronger_defect_gives_lower_score():
    faint = run(s.with_scratch(s.clean_metal(), depth=25)).score
    strong = run(s.with_scratch(s.clean_metal(), depth=60)).score
    assert strong < faint < 1.0


def test_defect_confidence_in_range():
    for d in run(s.with_dust(s.clean_metal())).defects:
        assert 0.5 <= d.confidence <= 1.0


def test_tiny_specks_below_min_area_ignored(cfg):
    cfg["detectors"]["surface"]["min_defect_area_px"] = 500
    assert run(s.with_dust(s.clean_metal())).defects == []


def test_roi_excludes_background_and_offsets_boxes(cfg):
    img, roi = s.part_on_background()
    cfg["detectors"]["roi"] = roi
    assert run(img).defects == []

    result = run(s.with_scratch(img, p1=(560, 350), p2=(720, 450)))
    assert len(result.defects) == 1
    d = result.defects[0]
    # Box must be in full-frame coordinates, not ROI coordinates
    assert 550 <= d.x <= 565 and 340 <= d.y <= 355


def test_duration_reported():
    result = run(s.clean_metal())
    assert result.duration_ms > 0
    assert result.detector_name == "surface"
