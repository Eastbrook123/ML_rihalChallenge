"""
Test 10: Visual & Manual Checks
Generates an HTML report for manual inspection of OCR and extraction results.
Always passes -- the output is for human review.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _img_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


class TestVisualReport:
    def test_generate_visual_report(self):
        """Generate visual_report.html for manual inspection."""
        from ocr_engine import OCREngine
        from field_extractor import FieldExtractor
        from feature_extractor import FeatureExtractor
        from anomaly_detector import AnomalyDetector
        from utils import load_jsonl, load_image

        test_dir = os.path.join(PROJECT_ROOT, "dummy_data", "test")
        test_path = os.path.join(test_dir, "test.jsonl")
        images_dir = os.path.join(test_dir, "images")

        records = load_jsonl(test_path)

        ocr = OCREngine()
        extractor = FieldExtractor()
        feat_ext = FeatureExtractor()

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>DocFusion Visual Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            ".card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".card h3 { margin-top: 0; }",
            ".fields td { padding: 4px 12px; }",
            ".genuine { color: #2e7d32; font-weight: bold; }",
            ".forged { color: #c62828; font-weight: bold; }",
            "img { max-width: 400px; border: 1px solid #ddd; }",
            ".row { display: flex; gap: 20px; align-items: flex-start; }",
            "</style></head><body>",
            "<h1>DocFusion Visual Inspection Report</h1>",
            f"<p>Generated from {len(records)} test images</p>",
        ]

        for rec in records:
            img_name = Path(rec["image_path"]).name
            img_path = os.path.join(images_dir, img_name)
            image = load_image(img_path)

            card_html = f'<div class="card"><h3>ID: {rec["id"]}</h3>'
            card_html += '<div class="row">'

            if image is not None:
                ocr_result = ocr.run(image)
                fields = extractor.extract(ocr_result)
                vis_feats = feat_ext.visual_features(image)

                annotated = image.copy()
                for text, box, conf in ocr_result.lines_top_to_bottom():
                    pts = np.array(box, dtype=np.int32)
                    color = (0, 0, 255) if conf < 0.5 else (0, 180, 0)
                    cv2.polylines(annotated, [pts], True, color, 2)

                b64 = _img_to_base64(annotated)
                card_html += f'<div><img src="data:image/png;base64,{b64}"/></div>'

                gt = rec.get("fields", {})
                card_html += '<div>'
                card_html += '<table class="fields">'
                card_html += "<tr><th>Field</th><th>Extracted</th><th>Ground Truth</th></tr>"
                for field in ("vendor", "date", "total"):
                    ext_val = fields.get(field, "N/A")
                    gt_val = gt.get(field, "N/A")
                    match = "OK" if ext_val == gt_val else "MISS"
                    card_html += f"<tr><td>{field}</td><td>{ext_val}</td><td>{gt_val} ({match})</td></tr>"
                card_html += "</table>"

                card_html += f'<p>OCR lines: {len(ocr_result.texts)}</p>'
                card_html += f'<p>ELA mean: {vis_feats["ela_mean"]:.2f}</p>'
                card_html += "</div>"
            else:
                card_html += "<p>Image not found or corrupt</p>"

            card_html += "</div></div>"
            html_parts.append(card_html)

        html_parts.append("</body></html>")

        report_path = os.path.join(PROJECT_ROOT, "visual_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        assert os.path.exists(report_path)
        print(f"\nVisual report generated: {report_path}")
