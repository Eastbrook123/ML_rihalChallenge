"""
Test 1: Interface Validation
Validates that DocFusionSolution matches the autograder contract exactly.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestDocFusionSolutionInterface:
    """Validate the class exists and has the correct method signatures."""

    def test_solution_file_exists(self, project_root):
        assert os.path.exists(os.path.join(project_root, "solution.py"))

    def test_class_exists(self):
        from solution import DocFusionSolution

        assert DocFusionSolution is not None

    def test_class_is_instantiable(self):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        assert sol is not None

    def test_train_method_exists(self):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        assert hasattr(sol, "train")
        assert callable(sol.train)

    def test_predict_method_exists(self):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        assert hasattr(sol, "predict")
        assert callable(sol.predict)

    def test_train_signature(self):
        from solution import DocFusionSolution

        sig = inspect.signature(DocFusionSolution.train)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "train_dir" in params
        assert "work_dir" in params

    def test_predict_signature(self):
        from solution import DocFusionSolution

        sig = inspect.signature(DocFusionSolution.predict)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "model_dir" in params
        assert "data_dir" in params
        assert "out_path" in params

    def test_train_returns_string(self, dummy_train_dir, tmp_work_dir):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        result = sol.train(dummy_train_dir, tmp_work_dir)
        assert isinstance(result, str)
        assert result.strip() != ""

    def test_train_returns_valid_directory(self, dummy_train_dir, tmp_work_dir):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        result = sol.train(dummy_train_dir, tmp_work_dir)
        assert os.path.isdir(result)

    def test_predict_creates_output_file(
        self, dummy_train_dir, dummy_test_dir, tmp_work_dir
    ):
        from solution import DocFusionSolution

        sol = DocFusionSolution()
        model_dir = sol.train(dummy_train_dir, tmp_work_dir)
        out_path = os.path.join(tmp_work_dir, "predictions.jsonl")
        sol.predict(model_dir, dummy_test_dir, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
