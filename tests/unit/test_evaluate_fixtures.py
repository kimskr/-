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
