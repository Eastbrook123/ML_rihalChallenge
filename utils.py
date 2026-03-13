"""
Utility functions for image loading, JSONL I/O, and field normalization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image(path: str) -> np.ndarray | None:
    try:
        img = cv2.imread(str(Path(path).resolve()))
        return img
    except Exception:
        return None


def normalize_date(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    try:
        from dateutil import parser as dp

        dt = dp.parse(raw, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return raw


def normalize_total(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.replace(",", "").replace(" ", "").strip()
    cleaned = re.sub(r"[^\d.]", "", cleaned)
    if not cleaned:
        return raw
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return raw
