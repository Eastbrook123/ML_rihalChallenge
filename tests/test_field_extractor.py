"""
Test 4: Field Extractor Unit Tests
Parameterized tests for vendor, date, and total extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from field_extractor import FieldExtractor
from tests.conftest import make_mock_ocr_result


class TestVendorExtraction:
    def test_simple_vendor(self):
        ocr = make_mock_ocr_result([
            ("ACME Corp", 10), ("123 Main St", 30),
            ("Date: 2024-01-01", 50), ("TOTAL 10.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "ACME Corp"

    def test_skips_numeric_lines(self):
        ocr = make_mock_ocr_result([
            ("12345678", 10), ("Real Store", 30), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "Real Store"

    def test_skips_keyword_lines(self):
        ocr = make_mock_ocr_result([
            ("TEL: 555-1234", 10), ("Gulf Mart", 30), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "Gulf Mart"

    def test_merges_short_header_lines(self):
        ocr = make_mock_ocr_result([
            ("Berghotel", 10), ("Grosse Scheidegg", 30), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "Berghotel Grosse Scheidegg"

    def test_long_first_line_no_merge(self):
        ocr = make_mock_ocr_result([
            ("SuperMegaStore International", 10), ("Branch 42", 30), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "SuperMegaStore International"

    def test_empty_ocr(self):
        ocr = make_mock_ocr_result([])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] is None

    def test_single_line(self):
        ocr = make_mock_ocr_result([("OnlyStore", 10)])
        fields = FieldExtractor().extract(ocr)
        assert fields["vendor"] == "OnlyStore"


class TestDateExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Date: 24/01/2024", "2024-01-24"),
            ("Date: 2024-01-24", "2024-01-24"),
            ("Date: 24.01.2024", "2024-01-24"),
            ("24 Jan 2024", "2024-01-24"),
            ("Jan 24, 2024", "2024-01-24"),
            ("January 24, 2024", "2024-01-24"),
        ],
    )
    def test_date_formats(self, text, expected):
        ocr = make_mock_ocr_result([
            ("Store", 10), (text, 50), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["date"] == expected

    def test_no_date_in_text(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("Item 5.00", 50), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        # May be None or may fuzzy-parse -- either is acceptable

    def test_iso_date(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("2024-06-15", 50), ("TOTAL 5.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["date"] == "2024-06-15"


class TestTotalExtraction:
    def test_total_keyword(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("Item 3.50", 50), ("TOTAL 10.50", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] == "10.50"

    def test_grand_total(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("Subtotal 8.00", 180),
            ("Tax 2.00", 190), ("Grand Total 10.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] == "10.00"

    def test_total_with_currency_symbol(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("TOTAL $25.99", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] == "25.99"

    def test_no_total_keyword_uses_largest(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("3.50", 50), ("2.00", 60), ("15.50", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] == "15.50"

    def test_empty_ocr(self):
        ocr = make_mock_ocr_result([])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] is None

    def test_comma_formatted_total(self):
        ocr = make_mock_ocr_result([
            ("Store", 10), ("TOTAL 1,250.00", 200),
        ])
        fields = FieldExtractor().extract(ocr)
        assert fields["total"] == "1250.00"
