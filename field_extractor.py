"""
Rule-based field extraction from OCR output.
Uses spatial heuristics + regex patterns for vendor, date, total.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np

DATE_PATTERNS = [
    (r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", "DMY"),
    (r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", "YMD"),
    (
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+(\d{4})",
        "DMonY",
    ),
    (
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+(\d{1,2}),?\s+(\d{4})",
        "MonDY",
    ),
]

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

TOTAL_KEYWORDS = [
    r"grand\s*total",
    r"total\s*amount",
    r"total\s*due",
    r"amount\s*due",
    r"balance\s*due",
    r"net\s*total",
    r"total",
    r"sum",
]

MONEY_PATTERN = re.compile(
    r"(?:[\$\£\€\¥]|CHF|RM|Rs\.?|kr\.?|R\b)"
    r"\s*(\d{1,3}(?:[,\s]\d{3})*\.\d{1,2})\b"
    r"|"
    r"(\d{1,3}(?:[,\s]\d{3})*\.\d{2})\b"
    r"|"
    r"(\d{1,5})\s*(?:[\$\£\€\¥]|CHF|RM|Rs|kr)\b"
)

_MONEY_SIMPLE = re.compile(r"(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)")

NON_MONETARY_LINE = re.compile(
    r"\b(?:tel|fax|phone|hp|mob|rech|nr|no\.|reg|gstin|abn|uen)\b",
    re.IGNORECASE,
)

MAX_PLAUSIBLE_TOTAL = 99_999.0

SKIP_VENDOR_PATTERNS = re.compile(
    r"\b(?:tax|gst|receipt|invoice|tel|fax|phone|date|time|cash|change|subtotal|total|rech|reg|bon)\b"
    r"|(?:^nr\b|^no\.)",
    re.IGNORECASE,
)

OCR_ARTIFACTS = re.compile(r"[\[\]|{}]")


class FieldExtractor:
    def extract(self, ocr_result) -> dict:
        lines = ocr_result.lines_top_to_bottom()
        if not lines:
            return {"vendor": None, "date": None, "total": None}

        vendor = self._extract_vendor(lines)
        date = self._extract_date(lines)
        total = self._extract_total(lines)

        return {"vendor": vendor, "date": date, "total": total}

    # ── Vendor ────────────────────────────────────────────────

    def _extract_vendor(self, lines) -> str | None:
        candidates = []
        for text, box, conf in lines[:6]:
            if conf < 0.25:
                continue
            cleaned = self._clean_vendor_text(text)
            if not cleaned or len(cleaned) < 2:
                continue
            if re.fullmatch(r"[\d\s\-/.:,]+", cleaned):
                continue
            if SKIP_VENDOR_PATTERNS.search(cleaned):
                continue
            candidates.append((cleaned, conf))
            if len(candidates) >= 3:
                break

        if not candidates:
            return None

        merged = self._maybe_merge_vendor(candidates)
        return merged

    def _clean_vendor_text(self, text: str) -> str:
        cleaned = OCR_ARTIFACTS.sub("", text)
        cleaned = cleaned.strip(" \t\n\r.,;:!?-_=+*/\\")
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _maybe_merge_vendor(self, candidates: list[tuple[str, float]]) -> str:
        if len(candidates) == 1:
            return candidates[0][0]

        first = candidates[0][0]
        if len(first) >= 15:
            return first

        parts = [first]
        for text, conf in candidates[1:]:
            if conf < 0.3:
                continue
            if len(text) < 3:
                continue
            if SKIP_VENDOR_PATTERNS.search(text):
                break
            if re.fullmatch(r"[\d\s\-/.:,]+", text):
                break
            if re.search(r"\d", text) and re.search(r"[a-zA-Z]", text):
                break
            parts.append(text)
            if sum(len(p) for p in parts) >= 30:
                break

        return " ".join(parts)

    # ── Date ──────────────────────────────────────────────────

    def _extract_date(self, lines) -> str | None:
        all_text = " ".join(t for t, _, _ in lines)
        for pattern, fmt in DATE_PATTERNS:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                try:
                    return self._parse_date_match(match, fmt)
                except (ValueError, KeyError):
                    continue

        try:
            from dateutil import parser as dateutil_parser

            dt = dateutil_parser.parse(all_text, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _parse_date_match(self, match, fmt: str) -> str:
        if fmt == "DMY":
            d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        elif fmt == "YMD":
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        elif fmt == "DMonY":
            d = int(match.group(1))
            m = MONTH_MAP[match.group(2).lower()[:3]]
            y = int(match.group(3))
        elif fmt == "MonDY":
            m = MONTH_MAP[match.group(1).lower()[:3]]
            d = int(match.group(2))
            y = int(match.group(3))
        else:
            raise ValueError(f"Unknown format: {fmt}")

        dt = datetime(y, m, d)
        return dt.strftime("%Y-%m-%d")

    # ── Total ─────────────────────────────────────────────────

    def _extract_total(self, lines) -> str | None:
        result = self._total_by_keyword(lines)
        if result is not None:
            return result

        result = self._total_from_bottom(lines)
        if result is not None:
            return result

        return self._total_largest_anywhere(lines)

    def _total_by_keyword(self, lines) -> str | None:
        indexed = [(i, t, box, conf) for i, (t, box, conf) in enumerate(lines)]

        for keyword_pat in TOTAL_KEYWORDS:
            for i, text, box, conf in indexed:
                if not re.search(keyword_pat, text, re.IGNORECASE):
                    continue

                amount = self._find_money_in_text(text)
                if amount is not None:
                    return amount

                for j in range(i + 1, min(i + 3, len(lines))):
                    next_text = lines[j][0]
                    amount = self._find_money_in_text(next_text)
                    if amount is not None:
                        return amount

        return None

    def _total_from_bottom(self, lines) -> str | None:
        bottom_start = max(0, int(len(lines) * 0.5))
        bottom_lines = lines[bottom_start:]

        amounts = []
        for text, box, conf in bottom_lines:
            if self._is_non_monetary_line(text):
                continue
            for val in self._extract_plausible_amounts(text):
                amounts.append((val, conf))

        if not amounts:
            return None

        amounts.sort(key=lambda x: x[0], reverse=True)
        return f"{amounts[0][0]:.2f}"

    def _total_largest_anywhere(self, lines) -> str | None:
        amounts = []
        for text, box, conf in lines:
            if self._is_non_monetary_line(text):
                continue
            for val in self._extract_plausible_amounts(text):
                amounts.append((val, conf))

        if not amounts:
            return None

        amounts.sort(key=lambda x: x[0], reverse=True)
        return f"{amounts[0][0]:.2f}"

    def _find_money_in_text(self, text: str) -> str | None:
        for m in MONEY_PATTERN.finditer(text):
            raw = m.group(1) or m.group(2) or m.group(3)
            if raw:
                val = self._parse_amount(raw)
                if val is not None and val <= MAX_PLAUSIBLE_TOTAL:
                    return f"{val:.2f}"

        for raw in _MONEY_SIMPLE.findall(text):
            val = self._parse_amount(raw)
            if val is not None and 0.01 <= val <= MAX_PLAUSIBLE_TOTAL:
                if "." in raw:
                    return f"{val:.2f}"

        return None

    def _extract_plausible_amounts(self, text: str) -> list[float]:
        results = []

        for m in MONEY_PATTERN.finditer(text):
            raw = m.group(1) or m.group(2) or m.group(3)
            if raw:
                val = self._parse_amount(raw)
                if val is not None and 0.01 <= val <= MAX_PLAUSIBLE_TOTAL:
                    results.append(val)

        for raw in _MONEY_SIMPLE.findall(text):
            val = self._parse_amount(raw)
            if val is None:
                continue
            if "." not in raw and val >= 1000:
                continue
            if 0.01 <= val <= MAX_PLAUSIBLE_TOTAL:
                results.append(val)

        return results

    def _is_non_monetary_line(self, text: str) -> bool:
        return bool(NON_MONETARY_LINE.search(text))

    def _parse_amount(self, raw: str) -> float | None:
        cleaned = raw.replace(",", "").replace(" ", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _clean_money(self, raw: str) -> str | None:
        cleaned = raw.replace(",", "").replace(" ", "").strip()
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            return None
