"""Shared pytest fixtures for DocFusion test suite."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DUMMY_DATA = os.path.join(PROJECT_ROOT, "dummy_data")
TRAIN_DIR = os.path.join(DUMMY_DATA, "train")
TEST_DIR = os.path.join(DUMMY_DATA, "test")


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def dummy_train_dir():
    return TRAIN_DIR


@pytest.fixture
def dummy_test_dir():
    return TEST_DIR


@pytest.fixture
def tmp_work_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def white_image():
    return np.ones((600, 400, 3), dtype=np.uint8) * 255


@pytest.fixture
def noisy_image():
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (600, 400, 3), dtype=np.uint8)


@pytest.fixture
def text_image():
    """Image with programmatically rendered text for OCR testing."""
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "ACME Corp", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Date: 2024-01-15", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, "Item A    5.00", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Item B    3.50", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "TOTAL    10.50", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def make_mock_ocr_result(lines):
    """Create a mock OCRResult from a list of (text, y_position) tuples."""
    from ocr_engine import OCRResult

    sorted_lines = sorted(lines, key=lambda x: x[1])
    boxes = []
    texts = []
    confs = []
    for text, y in sorted_lines:
        box = [[0, y], [200, y], [200, y + 20], [0, y + 20]]
        boxes.append(box)
        texts.append(text)
        confs.append(0.95)
    return OCRResult(boxes, texts, confs, (600, 400, 3))


@pytest.fixture
def mock_ocr_receipt():
    """Mock OCR result simulating a typical receipt."""
    return make_mock_ocr_result([
        ("ACME Corp", 10),
        ("123 Main Street", 30),
        ("Date: 24/01/2024", 60),
        ("Item A       5.00", 100),
        ("Item B       3.50", 120),
        ("Item C       2.00", 140),
        ("TOTAL       10.50", 200),
    ])


@pytest.fixture
def mock_ocr_empty():
    """Mock OCR result with no text detected."""
    from ocr_engine import OCRResult

    return OCRResult([], [], [], (600, 400, 3))


@pytest.fixture
def train_records():
    train_path = os.path.join(TRAIN_DIR, "train.jsonl")
    with open(train_path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def test_records():
    test_path = os.path.join(TEST_DIR, "test.jsonl")
    with open(test_path) as f:
        return [json.loads(line) for line in f if line.strip()]
