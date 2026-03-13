"""
DocFusion Solution -- End-to-end document processing pipeline.
Combines OCR-based field extraction with visual+statistical anomaly detection.
Conforms to the DocFusion Autograder Harness interface contract.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling modules are importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ocr_engine import OCREngine
from field_extractor import FieldExtractor
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from utils import load_jsonl, load_image, normalize_date, normalize_total


class DocFusionSolution:
    def __init__(self):
        self._ocr: OCREngine | None = None

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = OCREngine()

    def train(self, train_dir: str, work_dir: str) -> str:
        model_dir = os.path.join(work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        self._ensure_ocr()
        extractor = FieldExtractor()
        feat_extractor = FeatureExtractor()
        detector = AnomalyDetector()

        train_path = os.path.join(train_dir, "train.jsonl")
        records = load_jsonl(train_path)
        images_dir = os.path.join(train_dir, "images")

        features_list = []
        labels = []

        for rec in records:
            img_name = Path(rec["image_path"]).name
            img_path = os.path.join(images_dir, img_name)
            image = load_image(img_path)

            if image is None:
                features_list.append(feat_extractor.default_features())
                labels.append(rec["label"]["is_forged"])
                continue

            ocr_result = self._ocr.run(image)
            fields = extractor.extract(ocr_result)
            visual_feats = feat_extractor.visual_features(image)
            stat_feats = feat_extractor.statistical_features(fields, ocr_result)
            combined = {**visual_feats, **stat_feats}
            features_list.append(combined)
            labels.append(rec["label"]["is_forged"])

        detector.fit(features_list, labels)
        detector.save(model_dir)
        feat_extractor.save_stats(features_list, model_dir)

        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        self._ensure_ocr()
        extractor = FieldExtractor()
        feat_extractor = FeatureExtractor()
        feat_extractor.load_stats(model_dir)
        detector = AnomalyDetector()
        detector.load(model_dir)

        test_path = os.path.join(data_dir, "test.jsonl")
        records = load_jsonl(test_path)
        images_dir = os.path.join(data_dir, "images")

        predictions = []

        for rec in records:
            img_name = Path(rec["image_path"]).name
            img_path = os.path.join(images_dir, img_name)
            image = load_image(img_path)

            if image is None:
                predictions.append(
                    {
                        "id": rec["id"],
                        "vendor": None,
                        "date": None,
                        "total": None,
                        "is_forged": 0,
                    }
                )
                continue

            ocr_result = self._ocr.run(image)
            fields = extractor.extract(ocr_result)
            visual_feats = feat_extractor.visual_features(image)
            stat_feats = feat_extractor.statistical_features(fields, ocr_result)
            combined = {**visual_feats, **stat_feats}

            is_forged = detector.predict_one(combined)

            predictions.append(
                {
                    "id": rec["id"],
                    "vendor": fields.get("vendor"),
                    "date": normalize_date(fields.get("date")),
                    "total": normalize_total(fields.get("total")),
                    "is_forged": int(is_forged),
                }
            )

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
