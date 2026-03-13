"""
Test 8: Performance & Resource Checks
Benchmarks inference time, memory, and model size.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestInferencePerformance:
    @pytest.fixture(scope="class")
    def trained_pipeline(self):
        from solution import DocFusionSolution

        train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
        work_dir = tempfile.mkdtemp()
        sol = DocFusionSolution()
        model_dir = sol.train(train_dir, work_dir)
        return sol, model_dir, work_dir

    def test_predict_latency(self, trained_pipeline):
        sol, model_dir, work_dir = trained_pipeline
        test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")
        out_path = os.path.join(work_dir, "perf_predictions.jsonl")

        start = time.perf_counter()
        sol.predict(model_dir, test_dir, out_path)
        elapsed = time.perf_counter() - start

        per_doc = elapsed / 10  # 10 test docs
        assert per_doc < 5.0, f"Per-doc latency {per_doc:.2f}s exceeds 5s budget"

    def test_peak_memory(self, trained_pipeline):
        sol, model_dir, work_dir = trained_pipeline
        test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")
        out_path = os.path.join(work_dir, "mem_predictions.jsonl")

        tracemalloc.start()
        sol.predict(model_dir, test_dir, out_path)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 1024, f"Peak memory {peak_mb:.1f}MB exceeds 1GB budget"


class TestModelSize:
    def test_model_artifacts_size(self):
        train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
        with tempfile.TemporaryDirectory() as work_dir:
            from solution import DocFusionSolution

            sol = DocFusionSolution()
            model_dir = sol.train(train_dir, work_dir)

            total_size = 0
            for root, dirs, files in os.walk(model_dir):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))

            size_mb = total_size / (1024 * 1024)
            assert size_mb < 50, f"Model size {size_mb:.1f}MB exceeds 50MB budget"


class TestTrainingPerformance:
    def test_training_time(self):
        from solution import DocFusionSolution

        train_dir = os.path.join(PROJECT_ROOT, "dummy_data", "train")
        with tempfile.TemporaryDirectory() as work_dir:
            sol = DocFusionSolution()
            start = time.perf_counter()
            sol.train(train_dir, work_dir)
            elapsed = time.perf_counter() - start

            assert elapsed < 120, f"Training time {elapsed:.1f}s exceeds 120s budget"
