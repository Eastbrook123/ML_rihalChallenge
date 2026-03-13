"""
Test 9: End-to-End Smoke Test
Replicates the full check_submission.py flow programmatically.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestEndToEndSmoke:
    def test_full_pipeline(self):
        """Mirrors the official check_submission.py validation flow."""
        from solution import DocFusionSolution

        train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
        test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")
        test_jsonl_path = os.path.join(test_dir, "test.jsonl")

        with tempfile.TemporaryDirectory() as work_dir:
            # 1. Instantiate
            sol = DocFusionSolution()

            # 2. Train
            model_dir = sol.train(train_dir, work_dir)
            assert isinstance(model_dir, str)
            assert model_dir.strip() != ""
            assert os.path.isdir(model_dir)

            # 3. Predict
            out_path = os.path.join(work_dir, "predictions.jsonl")
            sol.predict(model_dir, test_dir, out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

            # 4. Load predictions
            predictions = []
            with open(out_path) as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    pred = json.loads(line)
                    assert isinstance(pred, dict), f"Line {idx} is not a dict"
                    predictions.append(pred)

            # 5. Load test IDs
            with open(test_jsonl_path) as f:
                test_records = [json.loads(l) for l in f if l.strip()]
            expected_ids = {r["id"] for r in test_records}

            # 6. Validate each prediction
            pred_ids = set()
            for idx, pred in enumerate(predictions, start=1):
                assert "id" in pred, f"Line {idx}: missing 'id'"
                assert "is_forged" in pred, f"Line {idx}: missing 'is_forged'"

                record_id = pred["id"]
                assert isinstance(record_id, str) and record_id.strip()
                assert record_id not in pred_ids, f"Duplicate ID: {record_id}"
                pred_ids.add(record_id)

                is_forged = pred["is_forged"]
                assert type(is_forged) is int
                assert is_forged in (0, 1)

                for field in ("vendor", "date", "total"):
                    if field in pred:
                        val = pred[field]
                        assert val is None or isinstance(val, str), (
                            f"Line {idx}: '{field}' must be string or null"
                        )

            # 7. Check ID coverage
            missing = expected_ids - pred_ids
            extra = pred_ids - expected_ids
            assert not missing, f"Missing IDs: {missing}"
            assert not extra, f"Extra IDs: {extra}"
            assert len(predictions) == len(test_records)

    def test_idempotent_predictions(self):
        """Running predict twice should produce identical output."""
        from solution import DocFusionSolution

        train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
        test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")

        with tempfile.TemporaryDirectory() as work_dir:
            sol = DocFusionSolution()
            model_dir = sol.train(train_dir, work_dir)

            out1 = os.path.join(work_dir, "pred1.jsonl")
            out2 = os.path.join(work_dir, "pred2.jsonl")
            sol.predict(model_dir, test_dir, out1)
            sol.predict(model_dir, test_dir, out2)

            with open(out1) as f:
                content1 = f.read()
            with open(out2) as f:
                content2 = f.read()
            assert content1 == content2
