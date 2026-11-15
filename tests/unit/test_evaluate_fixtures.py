"""Unit tests for tools/evaluate_fixtures.py (accuracy report)."""
import importlib.util
from pathlib import Path

import cv2

from tests import synthetic_images as s

_spec = importlib.util.spec_from_file_location(
    "evaluate_fixtures", Path(__file__).resolve().parents[2] / "tools" / "evaluate_fixtures.py")
evaluate_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_fixtures)


def _write(folder: Path, name: str, img):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / name), img)


def test_evaluate_counts_errors(tmp_path):
    _write(tmp_path / "pass", "clean_01.png", s.clean_metal(0))
    _write(tmp_path / "pass", "clean_02.png", s.clean_coated(1))
    _write(tmp_path / "pass", "mislabeled.png", s.with_scratch(s.clean_metal()))  # -> false positive
    _write(tmp_path / "fail", "scratch_01.png", s.with_scratch(s.clean_metal()))
    _write(tmp_path / "fail", "dust_01.png", s.with_dust(s.clean_metal()))

    rep = evaluate_fixtures.evaluate(tmp_path)
    assert rep["total"] == 5
    assert rep["n_pass"] == 3 and rep["n_fail"] == 2
    assert rep["fp"] == 1 and rep["fn"] == 0
    assert rep["accuracy"] == 0.8


def test_evaluate_empty_folder(tmp_path, capsys):
    rep = evaluate_fixtures.evaluate(tmp_path)
    assert rep["total"] == 0
    evaluate_fixtures.print_report(rep, details=False)
    assert "No fixture images found" in capsys.readouterr().out


def test_report_lists_wrong_files(tmp_path, capsys):
    _write(tmp_path / "fail", "missed.png", s.clean_metal())  # labelled FAIL but clean -> FN
    rep = evaluate_fixtures.evaluate(tmp_path)
    evaluate_fixtures.print_report(rep, details=False)
    out = capsys.readouterr().out
    assert "missed.png" in out and "❌" in out


def test_overlays_written(tmp_path):
    _write(tmp_path / "fail", "scratch_01.png", s.with_scratch(s.clean_metal()))
    evaluate_fixtures.evaluate(tmp_path, overlays=tmp_path / "overlays")
    assert (tmp_path / "overlays" / "fail_scratch_01.png").exists()


def test_select_roi_writes_config(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "select_roi", Path(__file__).resolve().parents[2] / "tools" / "select_roi.py")
    select_roi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(select_roi)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("detectors:\n  roi: null            # comment\n  surface:\n", encoding="utf-8")
    monkeypatch.setattr(select_roi, "CONFIG", cfg)
    select_roi.write_roi([10, 20, 300, 400])
    assert "  roi: [10, 20, 300, 400]            # comment" in cfg.read_text(encoding="utf-8")
