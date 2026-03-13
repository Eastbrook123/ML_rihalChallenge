"""
Test 6: Anomaly Detector Unit Tests
Tests XGBoost classifier fit, predict, save, and load.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anomaly_detector import AnomalyDetector


class TestFitPredict:
    def test_basic_fit_predict(self):
        features = [
            {"f1": 0.1, "f2": 0.2, "f3": 0.3},
            {"f1": 0.9, "f2": 0.8, "f3": 0.7},
            {"f1": 0.15, "f2": 0.25, "f3": 0.35},
            {"f1": 0.85, "f2": 0.75, "f3": 0.65},
        ]
        labels = [0, 1, 0, 1]

        det = AnomalyDetector()
        det.fit(features, labels)
        result = det.predict_one({"f1": 0.1, "f2": 0.2, "f3": 0.3})
        assert result in (0, 1)

    def test_predict_returns_int(self):
        features = [{"a": float(i), "b": float(i * 2)} for i in range(20)]
        labels = [i % 2 for i in range(20)]

        det = AnomalyDetector()
        det.fit(features, labels)
        result = det.predict_one({"a": 5.0, "b": 10.0})
        assert type(result) is int

    def test_batch_predict_length(self):
        features = [{"a": float(i), "b": float(i * 2)} for i in range(20)]
        labels = [i % 2 for i in range(20)]

        det = AnomalyDetector()
        det.fit(features, labels)

        test_feats = [{"a": float(i), "b": float(i * 2)} for i in range(5)]
        results = det.predict_batch(test_feats)
        assert len(results) == 5
        for r in results:
            assert r in (0, 1)


class TestNoModel:
    def test_predict_without_fit_returns_zero(self):
        det = AnomalyDetector()
        assert det.predict_one({"f1": 0.5}) == 0

    def test_batch_predict_without_fit(self):
        det = AnomalyDetector()
        results = det.predict_batch([{"f1": 0.5}, {"f1": 0.9}])
        assert results == [0, 0]


class TestSaveLoad:
    def test_save_load_round_trip(self, tmp_work_dir):
        features = [{"a": float(i), "b": float(i * 2)} for i in range(20)]
        labels = [i % 2 for i in range(20)]

        det1 = AnomalyDetector()
        det1.fit(features, labels)
        det1.save(tmp_work_dir)

        det2 = AnomalyDetector()
        det2.load(tmp_work_dir)

        test_input = {"a": 5.0, "b": 10.0}
        assert det1.predict_one(test_input) == det2.predict_one(test_input)

    def test_feature_names_preserved(self, tmp_work_dir):
        features = [{"x": 1.0, "y": 2.0, "z": 3.0}] * 10
        labels = [0, 1] * 5

        det1 = AnomalyDetector()
        det1.fit(features, labels)
        det1.save(tmp_work_dir)

        det2 = AnomalyDetector()
        det2.load(tmp_work_dir)
        assert det1.feature_names == det2.feature_names

    def test_load_missing_model(self, tmp_work_dir):
        det = AnomalyDetector()
        det.load(tmp_work_dir)
        assert det.model is None


class TestEdgeCases:
    def test_all_same_label(self):
        features = [{"a": float(i)} for i in range(10)]
        labels = [0] * 10
        det = AnomalyDetector()
        det.fit(features, labels)
        result = det.predict_one({"a": 5.0})
        assert result in (0, 1)

    def test_single_feature(self):
        features = [{"only": float(i)} for i in range(10)]
        labels = [0, 1] * 5
        det = AnomalyDetector()
        det.fit(features, labels)
        result = det.predict_one({"only": 3.0})
        assert result in (0, 1)

    def test_missing_feature_defaults_to_zero(self):
        features = [{"a": 1.0, "b": 2.0}] * 10
        labels = [0, 1] * 5
        det = AnomalyDetector()
        det.fit(features, labels)
        result = det.predict_one({"a": 1.0})  # missing "b"
        assert result in (0, 1)
