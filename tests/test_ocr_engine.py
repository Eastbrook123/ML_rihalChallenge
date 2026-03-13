"""
Test 3: OCR Engine Unit Tests
Tests the OCR wrapper with synthetic images.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr_engine import OCREngine, OCRResult


class TestOCRResult:
    def test_empty_result(self):
        r = OCRResult([], [], [], (600, 400, 3))
        assert r.lines_top_to_bottom() == []
        assert r.full_text == ""

    def test_sorting_by_y(self):
        boxes = [
            [[0, 100], [200, 100], [200, 120], [0, 120]],
            [[0, 10], [200, 10], [200, 30], [0, 30]],
            [[0, 50], [200, 50], [200, 70], [0, 70]],
        ]
        texts = ["middle", "top", "center"]
        confs = [0.9, 0.95, 0.88]
        r = OCRResult(boxes, texts, confs, (200, 200, 3))
        sorted_items = r.lines_top_to_bottom()
        assert sorted_items[0][0] == "top"
        assert sorted_items[1][0] == "center"
        assert sorted_items[2][0] == "middle"

    def test_full_text(self):
        boxes = [
            [[0, 10], [200, 10], [200, 30], [0, 30]],
            [[0, 50], [200, 50], [200, 70], [0, 70]],
        ]
        r = OCRResult(boxes, ["Hello", "World"], [0.9, 0.9], (100, 200, 3))
        assert r.full_text == "Hello\nWorld"

    def test_parallel_lists(self):
        boxes = [[[0, i * 20], [100, i * 20], [100, i * 20 + 15], [0, i * 20 + 15]] for i in range(5)]
        texts = [f"text_{i}" for i in range(5)]
        confs = [0.5 + i * 0.1 for i in range(5)]
        r = OCRResult(boxes, texts, confs, (200, 200, 3))
        assert len(r.boxes) == len(r.texts) == len(r.confidences) == 5


class TestOCREngine:
    @pytest.fixture(scope="class")
    def engine(self):
        return OCREngine()

    def test_white_image(self, engine, white_image):
        result = engine.run(white_image)
        assert isinstance(result, OCRResult)
        assert len(result.texts) == len(result.boxes) == len(result.confidences)

    def test_text_image(self, engine, text_image):
        result = engine.run(text_image)
        assert isinstance(result, OCRResult)
        assert len(result.texts) == len(result.boxes)

    def test_large_image_resized(self, engine):
        large = np.ones((4000, 3000, 3), dtype=np.uint8) * 255
        result = engine.run(large)
        assert isinstance(result, OCRResult)

    def test_small_image(self, engine):
        small = np.ones((50, 50, 3), dtype=np.uint8) * 255
        result = engine.run(small)
        assert isinstance(result, OCRResult)

    def test_grayscale_image(self, engine):
        gray = np.ones((200, 200), dtype=np.uint8) * 255
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = engine.run(bgr)
        assert isinstance(result, OCRResult)
