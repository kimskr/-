"""
Regression test on the real camera fixtures in tests/fixtures/pass|fail.

Uses the ROI from config.yaml (the real capture setup), unlike the synthetic
unit tests. Fails if a code or config change drops accuracy below the plan's
targets, so tuning cannot silently break detection on real images.
"""
from pathlib import Path

import pytest
import yaml

from tests.unit.test_evaluate_fixtures import evaluate_fixtures

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

has_fixtures = any((FIXTURES / "pass").glob("*.png")) and any((FIXTURES / "fail").glob("*.png"))


@pytest.mark.skipif(not has_fixtures, reason="no real fixture images captured yet")
def test_real_fixtures_meet_targets(cfg):
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg["detectors"]["roi"] = yaml.safe_load(f)["detectors"]["roi"]

    rep = evaluate_fixtures.evaluate(FIXTURES)
    wrong = [r["file"] for r in rep["rows"] if not r["correct"]]
    assert rep["accuracy"] >= 0.85, f"accuracy {rep['accuracy']:.1%}, wrong: {wrong}"
    assert rep["fp_rate"] < 0.05, f"false positive rate {rep['fp_rate']:.1%}, wrong: {wrong}"
