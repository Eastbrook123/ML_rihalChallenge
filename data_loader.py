"""
Unified data loader for SROIE, Find-It-Again, and CORD-v2 datasets.
Converts all three into a common schema compatible with DocFusion pipeline.

Common record schema:
{
    "id": str,
    "image_path": str,           # relative path within dataset
    "source": str,               # "sroie" | "finditagain" | "cord"
    "split": str,                # "train" | "test" | "val"
    "fields": {
        "vendor": str | None,
        "date": str | None,
        "total": str | None,
    },
    "label": {
        "is_forged": int,        # 0 or 1
        "fraud_type": str,       # "none" | type description
    },
}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator


def _safe_str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


class SROIELoader:
    """
    Loads SROIE dataset (Kaggle urbikn/sroie-datasetv2).
    Structure: {root}/train/ and {root}/test/ each containing .jpg + .txt pairs.
    The .txt files contain 4 lines: vendor, date, address, total.
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def load(self, split: str = "train") -> list[dict]:
        split_dir = self.root / split
        if not split_dir.exists():
            img_dir = split_dir / "img"
            if img_dir.exists():
                split_dir = img_dir

        records = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        img_dir = split_dir / "img" if (split_dir / "img").exists() else split_dir
        box_dir = split_dir / "box" if (split_dir / "box").exists() else split_dir
        entities_dir = split_dir / "entities" if (split_dir / "entities").exists() else split_dir

        image_files = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in image_exts
        ) if img_dir.exists() else []

        for img_file in image_files:
            stem = img_file.stem
            record_id = f"sroie_{split}_{stem}"

            fields = {"vendor": None, "date": None, "total": None}
            entities_file = entities_dir / f"{stem}.txt"
            if entities_file.exists():
                fields = self._parse_entities(entities_file)

            records.append({
                "id": record_id,
                "image_path": str(img_file),
                "source": "sroie",
                "split": split,
                "fields": fields,
                "label": {"is_forged": 0, "fraud_type": "none"},
            })

        return records

    def _parse_entities(self, path: Path) -> dict:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        except Exception:
            return {"vendor": None, "date": None, "total": None}

        vendor = _safe_str(lines[0]) if len(lines) > 0 else None
        date = _safe_str(lines[1]) if len(lines) > 1 else None
        total = _safe_str(lines[3]) if len(lines) > 3 else None
        if total is None and len(lines) > 2:
            total = _safe_str(lines[2])

        return {"vendor": vendor, "date": date, "total": total}


class FindItAgainLoader:
    """
    Loads Find-It-Again dataset for forgery detection.
    Structure: {root}/{split}/ containing .png + .txt pairs,
    plus {root}/{split}.txt with ground truth labels.
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def load(self, split: str = "train") -> list[dict]:
        split_dir = self.root / split
        label_file = self.root / f"{split}.txt"

        labels = {}
        if label_file.exists():
            labels = self._parse_labels(label_file)

        records = []
        if not split_dir.exists():
            return records

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(
            f for f in split_dir.iterdir()
            if f.suffix.lower() in image_exts
        )

        for img_file in image_files:
            stem = img_file.stem
            record_id = f"fia_{split}_{stem}"

            is_forged = 0
            fraud_type = "none"
            if stem in labels:
                lbl = labels[stem]
                is_forged = lbl.get("is_forged", 0)
                fraud_type = lbl.get("fraud_type", "none")

            txt_file = split_dir / f"{stem}.txt"
            ocr_text = ""
            if txt_file.exists():
                try:
                    ocr_text = txt_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            fields = self._extract_fields_from_text(ocr_text)

            records.append({
                "id": record_id,
                "image_path": str(img_file),
                "source": "finditagain",
                "split": split,
                "fields": fields,
                "label": {"is_forged": is_forged, "fraud_type": fraud_type},
            })

        return records

    def _parse_labels(self, path: Path) -> dict:
        labels = {}
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return labels

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",", 1)
            if len(parts) < 2:
                parts = line.split("\t", 1)

            filename = parts[0].strip().replace(".png", "").replace(".jpg", "")

            try:
                annotation = json.loads(parts[1]) if len(parts) > 1 else {}
            except (json.JSONDecodeError, IndexError):
                rest = parts[1].strip() if len(parts) > 1 else ""
                is_forged = 1 if "forg" in rest.lower() else 0
                labels[filename] = {"is_forged": is_forged, "fraud_type": rest or "none"}
                continue

            regions = annotation.get("regions", [])
            is_forged = 1 if regions else 0
            fraud_types = set()
            for r in regions:
                attrs = r.get("region_attributes", {})
                modified = attrs.get("Modified area", {})
                if isinstance(modified, dict):
                    fraud_types.update(k for k, v in modified.items() if v)

            labels[filename] = {
                "is_forged": is_forged,
                "fraud_type": ",".join(fraud_types) if fraud_types else "none",
            }

        return labels

    def _extract_fields_from_text(self, text: str) -> dict:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        vendor = lines[0] if lines else None

        date = None
        total = None
        date_pat = re.compile(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}")
        money_pat = re.compile(r"\d+[.,]\d{2}")

        for line in lines:
            if date is None and date_pat.search(line):
                date = date_pat.search(line).group()
            if "total" in line.lower():
                m = money_pat.findall(line)
                if m:
                    total = m[-1]

        if total is None:
            amounts = []
            for line in lines:
                for m in money_pat.findall(line):
                    try:
                        amounts.append(float(m.replace(",", "")))
                    except ValueError:
                        pass
            if amounts:
                total = f"{max(amounts):.2f}"

        return {"vendor": vendor, "date": date, "total": total}


class CORDLoader:
    """
    Loads CORD-v2 dataset from HuggingFace (naver-clova-ix/cord-v2).
    Uses the `datasets` library if available, otherwise expects local parquet/json.
    """

    def __init__(self, root_dir: str | None = None):
        self.root = Path(root_dir) if root_dir else None

    def load(self, split: str = "train") -> list[dict]:
        try:
            return self._load_from_huggingface(split)
        except Exception:
            if self.root:
                return self._load_from_local(split)
            return []

    def _load_from_huggingface(self, split: str) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset("naver-clova-ix/cord-v2", split=split)
        records = []

        for idx, row in enumerate(ds):
            record_id = f"cord_{split}_{idx}"
            gt = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
            gt_parse = gt.get("gt_parse", {})

            fields = self._extract_cord_fields(gt_parse)

            img = row.get("image")
            img_path = ""
            if img is not None and self.root:
                save_dir = self.root / split / "images"
                save_dir.mkdir(parents=True, exist_ok=True)
                img_path = str(save_dir / f"cord_{idx}.png")
                if not os.path.exists(img_path):
                    img.save(img_path)

            records.append({
                "id": record_id,
                "image_path": img_path,
                "source": "cord",
                "split": split,
                "fields": fields,
                "label": {"is_forged": 0, "fraud_type": "none"},
            })

        return records

    def _load_from_local(self, split: str) -> list[dict]:
        split_dir = self.root / split
        if not split_dir.exists():
            return []

        gt_file = split_dir / "ground_truth.json"
        if not gt_file.exists():
            gt_file = split_dir / "metadata.jsonl"

        if not gt_file.exists():
            return []

        records = []
        if gt_file.suffix == ".jsonl":
            with open(gt_file) as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    gt_parse = data.get("gt_parse", data.get("ground_truth", {}))
                    if isinstance(gt_parse, str):
                        gt_parse = json.loads(gt_parse)
                        gt_parse = gt_parse.get("gt_parse", gt_parse)

                    fields = self._extract_cord_fields(gt_parse)
                    img_path = data.get("file_name", data.get("image_path", ""))
                    if img_path and not os.path.isabs(img_path):
                        img_path = str(split_dir / img_path)

                    records.append({
                        "id": f"cord_{split}_{idx}",
                        "image_path": img_path,
                        "source": "cord",
                        "split": split,
                        "fields": fields,
                        "label": {"is_forged": 0, "fraud_type": "none"},
                    })

        return records

    def _extract_cord_fields(self, gt_parse: dict) -> dict:
        vendor = None
        date = None
        total = None

        total_section = gt_parse.get("total", {})
        if isinstance(total_section, dict):
            total = total_section.get("total_price")
        elif isinstance(total_section, list):
            for item in total_section:
                if isinstance(item, dict) and "total_price" in item:
                    total = item["total_price"]
                    break

        menu = gt_parse.get("menu", [])
        if isinstance(menu, dict):
            vendor = menu.get("nm")
        elif isinstance(menu, list) and menu:
            vendor = menu[0].get("nm") if isinstance(menu[0], dict) else None

        return {"vendor": _safe_str(vendor), "date": _safe_str(date), "total": _safe_str(total)}


class UnifiedDataLoader:
    """
    Loads and merges records from all three dataset sources
    into a single unified format for training/evaluation.
    """

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)

    def load_all(self, split: str = "train") -> list[dict]:
        all_records = []

        sroie_dir = self.data_root / "sroie"
        if sroie_dir.exists():
            loader = SROIELoader(str(sroie_dir))
            records = loader.load(split)
            all_records.extend(records)

        fia_dir = self.data_root / "finditagain"
        if fia_dir.exists():
            loader = FindItAgainLoader(str(fia_dir))
            records = loader.load(split)
            all_records.extend(records)

        cord_dir = self.data_root / "cord"
        loader = CORDLoader(str(cord_dir) if cord_dir.exists() else None)
        try:
            records = loader.load(split)
            all_records.extend(records)
        except Exception:
            pass

        return all_records

    def export_jsonl(self, records: list[dict], out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def get_stats(self, records: list[dict]) -> dict:
        stats = {
            "total_records": len(records),
            "by_source": {},
            "by_split": {},
            "forged_count": 0,
            "genuine_count": 0,
            "fields_present": {"vendor": 0, "date": 0, "total": 0},
        }
        for rec in records:
            src = rec.get("source", "unknown")
            split = rec.get("split", "unknown")
            stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
            stats["by_split"][split] = stats["by_split"].get(split, 0) + 1

            if rec.get("label", {}).get("is_forged", 0):
                stats["forged_count"] += 1
            else:
                stats["genuine_count"] += 1

            for field in ("vendor", "date", "total"):
                if rec.get("fields", {}).get(field):
                    stats["fields_present"][field] += 1

        return stats
