"""
Test 2: JSONL Output Validation
Validates every line of predictions.jsonl matches the autograder format.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="module")
def predictions_and_test_ids():
    """Run the full pipeline once and return (predictions, test_ids)."""
    from solution import DocFusionSolution

    train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
    test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")

    with tempfile.TemporaryDirectory() as work_dir:
        sol = DocFusionSolution()
        model_dir = sol.train(train_dir, work_dir)
        out_path = os.path.join(work_dir, "predictions.jsonl")
        sol.predict(model_dir, test_dir, out_path)

        with open(out_path) as f:
            raw_lines = f.readlines()

        predictions = []
        for line in raw_lines:
            stripped = line.strip()
            if stripped:
                predictions.append(json.loads(stripped))

        test_jsonl = os.path.join(test_dir, "test.jsonl")
        with open(test_jsonl) as f:
            test_ids = {json.loads(l)["id"] for l in f if l.strip()}

        yield predictions, test_ids


class TestJSONLFormat:
    def test_all_lines_valid_json(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        assert len(preds) > 0

    def test_every_record_has_id(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            assert "id" in pred
            assert isinstance(pred["id"], str)
            assert pred["id"].strip() != ""

    def test_every_record_has_is_forged(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            assert "is_forged" in pred

    def test_is_forged_is_int_0_or_1(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            assert type(pred["is_forged"]) is int
            assert pred["is_forged"] in (0, 1)

    def test_vendor_is_string_or_null(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            if "vendor" in pred:
                assert pred["vendor"] is None or isinstance(pred["vendor"], str)

    def test_date_is_string_or_null(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            if "date" in pred:
                assert pred["date"] is None or isinstance(pred["date"], str)

    def test_total_is_string_or_null(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            if "total" in pred:
                assert pred["total"] is None or isinstance(pred["total"], str)

    def test_dates_are_iso_format_when_present(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            date_val = pred.get("date")
            if date_val is not None:
                assert re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", date_val
                ), f"Date not ISO: {date_val}"

    def test_totals_are_numeric_when_present(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        for pred in preds:
            total_val = pred.get("total")
            if total_val is not None:
                float(total_val)  # should not raise

    def test_no_duplicate_ids(self, predictions_and_test_ids):
        preds, _ = predictions_and_test_ids
        ids = [p["id"] for p in preds]
        assert len(ids) == len(set(ids))

    def test_ids_match_test_data(self, predictions_and_test_ids):
        preds, test_ids = predictions_and_test_ids
        pred_ids = {p["id"] for p in preds}
        assert pred_ids == test_ids, (
            f"Missing: {test_ids - pred_ids}, Extra: {pred_ids - test_ids}"
        )

    def test_prediction_count_matches(self, predictions_and_test_ids):
        preds, test_ids = predictions_and_test_ids
        assert len(preds) == len(test_ids)
