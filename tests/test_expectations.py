import json
from pathlib import Path
import pytest
from src.data import validation

EXPECT_PATH = Path("data/expectations/expectations.json")


def test_expectations_file_exists_and_loadable():
    assert EXPECT_PATH.exists(), f"{EXPECT_PATH} not found"
    suite = json.load(open(EXPECT_PATH))
    assert isinstance(suite, dict)
    assert "expectations" in suite
    assert isinstance(suite["expectations"], list)
    assert len(suite["expectations"]) > 0


def test_each_expectation_has_type_and_kwargs():
    suite = json.load(open(EXPECT_PATH))
    for exp in suite["expectations"]:
        assert "expectation_type" in exp
        assert "kwargs" in exp
        assert isinstance(exp["kwargs"], dict)


def test_validate_features_runs_and_returns_result():
    res = validation.validate_features(raise_on_fail=False)
    assert isinstance(res, dict)
    assert "success" in res
