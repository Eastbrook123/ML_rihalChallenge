"""
Extracts visual and statistical features for anomaly detection.
Visual: ELA, edge density, noise, texture.
Statistical: field completeness, amount z-scores, OCR confidence.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


class FeatureExtractor:
    VISUAL_KEYS = [
        "ela_mean", "ela_std", "ela_max", "ela_high_ratio",
        "edge_density", "noise_level",
        "brightness_mean", "brightness_std",
        "lbp_uniformity", "lbp_entropy",
    ]
    STAT_KEYS = [
        "field_completeness", "total_value", "total_zscore",
        "total_is_round", "ocr_conf_mean", "ocr_conf_min",
        "text_length", "num_lines",
    ]
    ALL_KEYS = VISUAL_KEYS + STAT_KEYS

    def __init__(self):
        self._train_stats: dict | None = None

    def visual_features(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        ela = self._error_level_analysis(image)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        noise = self._estimate_noise(gray)
        lbp_hist = self._lbp_features(gray)

        return {
            "ela_mean": float(np.mean(ela)),
            "ela_std": float(np.std(ela)),
            "ela_max": float(np.max(ela)),
            "ela_high_ratio": float(np.mean(ela > 30)),
            "edge_density": edge_density,
            "noise_level": float(noise),
            "brightness_mean": float(np.mean(gray)),
            "brightness_std": float(np.std(gray)),
            "lbp_uniformity": float(lbp_hist[0]) if len(lbp_hist) > 0 else 0.0,
            "lbp_entropy": float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))),
        }

    def _error_level_analysis(self, image: np.ndarray) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if recompressed is None:
            return np.zeros(image.shape[:2], dtype=np.float32)

        if len(image.shape) == 3:
            diff = np.abs(
                image.astype(np.float32) - recompressed.astype(np.float32)
            )
            ela = np.mean(diff, axis=2)
        else:
            ela = np.abs(
                image.astype(np.float32) - recompressed.astype(np.float32)
            )
        return ela

    def _estimate_noise(self, gray: np.ndarray) -> float:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(lap))

    def _lbp_features(
        self, gray: np.ndarray, num_points: int = 8, radius: int = 1
    ) -> np.ndarray:
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            dx = int(round(radius * np.cos(angle)))
            dy = int(round(radius * np.sin(angle)))

            y_s = max(0, -dy)
            y_e = min(h, h - dy)
            x_s = max(0, -dx)
            x_e = min(w, w - dx)

            region = gray[y_s:y_e, x_s:x_e]
            neighbor = gray[y_s + dy : y_e + dy, x_s + dx : x_e + dx]

            min_h = min(region.shape[0], neighbor.shape[0])
            min_w = min(region.shape[1], neighbor.shape[1])
            if min_h > 0 and min_w > 0:
                mask = (neighbor[:min_h, :min_w] >= region[:min_h, :min_w]).astype(
                    np.uint8
                )
                lbp[y_s : y_s + min_h, x_s : x_s + min_w] |= mask << i

        hist, _ = np.histogram(lbp.ravel(), bins=2**num_points, range=(0, 2**num_points))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-10
        return hist

    def statistical_features(self, fields: dict, ocr_result) -> dict:
        completeness = sum(1 for v in fields.values() if v is not None) / 3.0

        total_val = None
        if fields.get("total"):
            try:
                total_val = float(fields["total"])
            except (ValueError, TypeError):
                pass

        conf_mean = (
            float(np.mean(ocr_result.confidences))
            if ocr_result.confidences
            else 0.0
        )
        conf_min = (
            float(np.min(ocr_result.confidences))
            if ocr_result.confidences
            else 0.0
        )
        text_length = sum(len(t) for t in ocr_result.texts)
        num_lines = len(ocr_result.texts)

        total_zscore = 0.0
        if total_val is not None and self._train_stats:
            mean = self._train_stats.get("total_mean", 0)
            std = self._train_stats.get("total_std", 1)
            total_zscore = (total_val - mean) / (std + 1e-10)

        return {
            "field_completeness": completeness,
            "total_value": total_val if total_val is not None else 0.0,
            "total_zscore": total_zscore,
            "total_is_round": float(
                total_val is not None and total_val == int(total_val)
            ),
            "ocr_conf_mean": conf_mean,
            "ocr_conf_min": conf_min,
            "text_length": float(text_length),
            "num_lines": float(num_lines),
        }

    def default_features(self) -> dict:
        return {k: 0.0 for k in self.ALL_KEYS}

    def save_stats(self, features_list: list[dict], model_dir: str):
        totals = [
            f["total_value"]
            for f in features_list
            if f.get("total_value", 0) > 0
        ]
        stats = {
            "total_mean": float(np.mean(totals)) if totals else 0.0,
            "total_std": float(np.std(totals)) if totals else 1.0,
        }
        with open(os.path.join(model_dir, "feature_stats.json"), "w") as f:
            json.dump(stats, f)
        self._train_stats = stats

    def load_stats(self, model_dir: str):
        path = os.path.join(model_dir, "feature_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                self._train_stats = json.load(f)
        else:
            self._train_stats = {"total_mean": 0.0, "total_std": 1.0}
