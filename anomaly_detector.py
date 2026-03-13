"""
XGBoost-based binary classifier for receipt forgery detection.
Lightweight, fast, CPU-friendly.
"""

from __future__ import annotations

import os
import pickle

import numpy as np


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.feature_names: list[str] | None = None
        self.threshold: float = 0.5
        self._default_label: int = 0

    def fit(self, features_list: list[dict], labels: list[int]):
        try:
            from xgboost import XGBClassifier

            ModelClass = XGBClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            ModelClass = GradientBoostingClassifier

        self.feature_names = sorted(features_list[0].keys())
        X = self._to_matrix(features_list)
        y = np.array(labels, dtype=np.int32)

        unique_labels = set(y.tolist())
        if len(unique_labels) < 2:
            self.model = None
            self.threshold = 0.5
            self._default_label = int(y[0]) if len(y) > 0 else 0
            return

        if ModelClass.__name__ == "XGBClassifier":
            self.model = ModelClass(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            self.model = ModelClass(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        self.model.fit(X, y)

    def predict_one(self, features: dict) -> int:
        if self.model is None:
            return self._default_label
        X = self._to_matrix([features])
        proba = self.model.predict_proba(X)[0, 1]
        return 1 if proba >= self.threshold else 0

    def predict_batch(self, features_list: list[dict]) -> list[int]:
        if self.model is None:
            return [self._default_label] * len(features_list)
        X = self._to_matrix(features_list)
        probas = self.model.predict_proba(X)[:, 1]
        return [1 if p >= self.threshold else 0 for p in probas]

    def _to_matrix(self, features_list: list[dict]) -> np.ndarray:
        return np.array(
            [[f.get(name, 0.0) for name in self.feature_names] for f in features_list],
            dtype=np.float32,
        )

    def save(self, model_dir: str):
        with open(os.path.join(model_dir, "anomaly_model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "default_label": self._default_label,
                },
                f,
            )

    def load(self, model_dir: str):
        path = os.path.join(model_dir, "anomaly_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.threshold = data.get("threshold", 0.5)
            self._default_label = data.get("default_label", 0)
