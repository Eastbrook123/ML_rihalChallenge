"""
Test 5: Feature Extractor Unit Tests
Tests visual and statistical feature extraction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_extractor import FeatureExtractor
from tests.conftest import make_mock_ocr_result


class TestVisualFeatures:
    def test_white_image_ela_near_zero(self, white_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(white_image)
        assert feats["ela_mean"] < 5.0
        assert feats["edge_density"] < 0.01

    def test_noisy_image_high_noise(self, noisy_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(noisy_image)
        assert feats["noise_level"] > 100

    def test_returns_all_visual_keys(self, white_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(white_image)
        for key in FeatureExtractor.VISUAL_KEYS:
            assert key in feats, f"Missing key: {key}"

    def test_all_values_are_float(self, white_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(white_image)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_brightness_white(self, white_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(white_image)
        assert feats["brightness_mean"] > 250

    def test_brightness_noisy(self, noisy_image):
        fe = FeatureExtractor()
        feats = fe.visual_features(noisy_image)
        assert 100 < feats["brightness_mean"] < 160


class TestStatisticalFeatures:
    def test_complete_fields(self):
        fe = FeatureExtractor()
        fields = {"vendor": "Store", "date": "2024-01-01", "total": "10.50"}
        ocr = make_mock_ocr_result([("Store", 10), ("2024-01-01", 50), ("TOTAL 10.50", 200)])
        feats = fe.statistical_features(fields, ocr)
        assert feats["field_completeness"] == 1.0

    def test_no_fields(self):
        fe = FeatureExtractor()
        fields = {"vendor": None, "date": None, "total": None}
        ocr = make_mock_ocr_result([])
        feats = fe.statistical_features(fields, ocr)
        assert feats["field_completeness"] == 0.0

    def test_returns_all_stat_keys(self):
        fe = FeatureExtractor()
        fields = {"vendor": "X", "date": "2024-01-01", "total": "5.00"}
        ocr = make_mock_ocr_result([("X", 10)])
        feats = fe.statistical_features(fields, ocr)
        for key in FeatureExtractor.STAT_KEYS:
            assert key in feats, f"Missing key: {key}"

    def test_total_zscore_with_stats(self):
        fe = FeatureExtractor()
        fe._train_stats = {"total_mean": 100.0, "total_std": 50.0}
        fields = {"vendor": "X", "date": "2024-01-01", "total": "200.00"}
        ocr = make_mock_ocr_result([("X", 10)])
        feats = fe.statistical_features(fields, ocr)
        assert abs(feats["total_zscore"] - 2.0) < 0.01


class TestDefaultFeatures:
    def test_default_has_all_keys(self):
        fe = FeatureExtractor()
        default = fe.default_features()
        for key in FeatureExtractor.ALL_KEYS:
            assert key in default

    def test_default_all_zero(self):
        fe = FeatureExtractor()
        default = fe.default_features()
        for v in default.values():
            assert v == 0.0


class TestSaveLoadStats:
    def test_round_trip(self, tmp_work_dir):
        fe = FeatureExtractor()
        features = [
            {"total_value": 100.0, "other": 1.0},
            {"total_value": 200.0, "other": 2.0},
            {"total_value": 300.0, "other": 3.0},
        ]
        fe.save_stats(features, tmp_work_dir)

        fe2 = FeatureExtractor()
        fe2.load_stats(tmp_work_dir)
        assert abs(fe2._train_stats["total_mean"] - 200.0) < 0.01
        assert fe2._train_stats["total_std"] > 0

    def test_load_missing_file(self, tmp_work_dir):
        fe = FeatureExtractor()
        fe.load_stats(tmp_work_dir)
        assert fe._train_stats["total_mean"] == 0.0
        assert fe._train_stats["total_std"] == 1.0
