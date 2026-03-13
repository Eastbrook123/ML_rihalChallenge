"""
Download and prepare all three datasets for DocFusion.

Usage:
    python download_datasets.py              # download all
    python download_datasets.py --sroie      # download SROIE only
    python download_datasets.py --cord       # download CORD-v2 only
    python download_datasets.py --finditagain  # download Find-It-Again only
    python download_datasets.py --output data  # custom output dir

Requirements:
    - kaggle (pip install kaggle) + Kaggle API token (~/.kaggle/kaggle.json)
    - datasets (pip install datasets) for CORD-v2
    - requests for Find-It-Again
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def download_sroie(output_dir: Path):
    """
    Download SROIE dataset v2 from Kaggle.
    Requires: kaggle CLI + API token.
    """
    sroie_dir = output_dir / "sroie"
    if sroie_dir.exists() and any(sroie_dir.rglob("*.jpg")):
        print(f"[SROIE] Already exists at {sroie_dir}, skipping.")
        return

    print("[SROIE] Downloading from Kaggle...")
    sroie_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "urbikn/sroie-datasetv2",
                "-p", str(sroie_dir),
                "--unzip",
            ],
            check=True,
        )
        print(f"[SROIE] Downloaded and extracted to {sroie_dir}")
    except FileNotFoundError:
        print("[SROIE] ERROR: kaggle CLI not found. Install with: pip install kaggle")
        print("[SROIE] Also ensure ~/.kaggle/kaggle.json contains your API token.")
        print("[SROIE] Get token from: https://www.kaggle.com/settings -> API -> Create New Token")
        _print_manual_sroie(sroie_dir)
    except subprocess.CalledProcessError as e:
        print(f"[SROIE] ERROR: kaggle download failed: {e}")
        _print_manual_sroie(sroie_dir)


def _print_manual_sroie(sroie_dir: Path):
    print(f"\n[SROIE] Manual download:")
    print(f"  1. Visit https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
    print(f"  2. Download and extract to: {sroie_dir}")
    print(f"  3. Expected structure:")
    print(f"     {sroie_dir}/train/img/  (images)")
    print(f"     {sroie_dir}/train/entities/  (annotations)")
    print(f"     {sroie_dir}/test/img/  (images)")


def download_cord(output_dir: Path):
    """
    Download CORD-v2 from HuggingFace using the datasets library.
    Images are saved locally for offline use.
    """
    cord_dir = output_dir / "cord"
    if cord_dir.exists() and any(cord_dir.rglob("*.png")):
        print(f"[CORD] Already exists at {cord_dir}, skipping.")
        return

    print("[CORD] Downloading from HuggingFace...")
    cord_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        import json

        for split in ("train", "validation", "test"):
            split_dir = cord_dir / split
            img_dir = split_dir / "images"
            meta_path = split_dir / "metadata.jsonl"

            if meta_path.exists():
                with open(meta_path) as f:
                    existing = sum(1 for _ in f)
                if existing > 0:
                    print(f"[CORD]   {split}: {existing} records already exist, skipping")
                    continue

            img_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CORD]   Downloading {split} (streaming)...")

            try:
                ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
            except Exception as e:
                print(f"[CORD]   Warning: Could not load {split}: {e}")
                continue

            metadata = []
            count = 0
            for row in ds:
                img = row.get("image")
                fname = f"cord_{count}.png"
                img_path = img_dir / fname
                if img is not None and not img_path.exists():
                    img.save(str(img_path))

                gt = row.get("ground_truth", "{}")
                metadata.append({"file_name": f"images/{fname}", "ground_truth": gt})
                count += 1
                if count % 200 == 0:
                    print(f"[CORD]     {count} records...", flush=True)

            with open(meta_path, "w") as f:
                for m in metadata:
                    f.write(json.dumps(m) + "\n")

            print(f"[CORD]   Saved {count} records for {split}")

        print(f"[CORD] Done. Data at {cord_dir}")

    except ImportError:
        print("[CORD] ERROR: 'datasets' library not found. Install with: pip install datasets")
        print("[CORD] Then rerun this script.")
    except Exception as e:
        print(f"[CORD] ERROR: {e}")
        print(f"\n[CORD] Manual download:")
        print(f"  Visit https://huggingface.co/datasets/naver-clova-ix/cord-v2")
        print(f"  Download and extract to: {cord_dir}")


def download_finditagain(output_dir: Path):
    """
    Download Find-It-Again dataset.
    Provides instructions since this dataset requires manual agreement/download.
    """
    fia_dir = output_dir / "finditagain"
    if fia_dir.exists() and any(fia_dir.rglob("*.png")):
        print(f"[FindItAgain] Already exists at {fia_dir}, skipping.")
        return

    fia_dir.mkdir(parents=True, exist_ok=True)

    print("[FindItAgain] This dataset requires manual download due to license restrictions.")
    print(f"\n  Steps:")
    print(f"  1. Visit: https://l3i-share.univ-lr.fr/2023Finditagain/index.html")
    print(f"  2. Download the dataset archive")
    print(f"  3. Extract contents to: {fia_dir}")
    print(f"  4. Expected structure:")
    print(f"     {fia_dir}/train/  (images + annotations)")
    print(f"     {fia_dir}/test/   (images)")
    print(f"     {fia_dir}/train.txt  (labels)")
    print(f"     {fia_dir}/test.txt   (labels)")

    readme = fia_dir / "DOWNLOAD_README.txt"
    readme.write_text(
        "Find-It-Again Dataset\n"
        "=====================\n\n"
        "This dataset requires manual download:\n"
        "1. Visit: https://l3i-share.univ-lr.fr/2023Finditagain/index.html\n"
        "2. Download the dataset archive\n"
        "3. Extract images and labels into this directory\n\n"
        "Expected structure:\n"
        "  finditagain/train/ - training images\n"
        "  finditagain/test/  - test images\n"
        "  finditagain/train.txt - training labels (VIA JSON format)\n"
        "  finditagain/test.txt  - test labels\n"
    )
    print(f"\n[FindItAgain] Created readme at {readme}")


def print_summary(output_dir: Path):
    print("\n" + "=" * 60)
    print("  Dataset Download Summary")
    print("=" * 60)

    for name in ("sroie", "cord", "finditagain"):
        d = output_dir / name
        if d.exists():
            n_images = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
            status = f"{n_images} images" if n_images > 0 else "directory exists (no images yet)"
        else:
            status = "not downloaded"
        print(f"  {name:15s} : {status}")

    print("=" * 60)
    print(f"\nAll data is stored in: {output_dir.resolve()}")
    print("Run 'python data_loader.py' or import UnifiedDataLoader to use.\n")


def main():
    parser = argparse.ArgumentParser(description="Download DocFusion datasets")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")
    parser.add_argument("--sroie", action="store_true", help="Download SROIE only")
    parser.add_argument("--cord", action="store_true", help="Download CORD-v2 only")
    parser.add_argument("--finditagain", action="store_true", help="Download Find-It-Again only")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_all = not (args.sroie or args.cord or args.finditagain)

    if download_all or args.sroie:
        download_sroie(output_dir)

    if download_all or args.cord:
        download_cord(output_dir)

    if download_all or args.finditagain:
        download_finditagain(output_dir)

    print_summary(output_dir)


if __name__ == "__main__":
    main()
