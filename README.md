# DocFusion: Operation Intelligent Documents

End-to-end document intelligence pipeline for the **2026 ML Rihal CodeStacker Challenge**.
Extracts structured fields (vendor, date, total) from scanned receipts and detects
forged documents using visual + statistical anomaly detection.

## Architecture

```
Image ──> Preprocess (CLAHE + Denoise) ──> EasyOCR (1536px) ──> Regex Extraction ──> Fields
  │                                                                                    │
  └──> ELA + Edge + LBP Visual Features ──┬──> XGBoost Classifier ──> is_forged        │
                                          │                                            │
              Statistical Features ───────┘                                            v
                                                                               predictions.jsonl
```

**Design rationale:** Heavy transformer models (Donut, LayoutLM) require 800MB+ and
seconds per document on CPU. This pipeline uses CLAHE contrast enhancement + denoising
as preprocessing, EasyOCR (~30MB) at 1536px resolution for OCR, regex + spatial heuristics
for field extraction, and a lightweight XGBoost classifier (<1MB) for anomaly detection.
Total inference: ~300ms/doc on CPU.

## Datasets

| Dataset | Source | Records | Usage |
|---------|--------|---------|-------|
| **SROIE v2** | [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) | ~1,000 | Field extraction training/eval |
| **CORD-v2** | [HuggingFace](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | ~11,000 | Field extraction training/eval |
| **Find-It-Again** | [L3i](https://l3i-share.univ-lr.fr/2023Finditagain/index.html) | ~1,500 | Forgery detection training |

### Download Datasets

```bash
# Download all datasets
python download_datasets.py

# Download individual datasets
python download_datasets.py --sroie
python download_datasets.py --cord
python download_datasets.py --finditagain

# Custom output directory
python download_datasets.py --output ./my_data
```

**Note:** SROIE requires a Kaggle API token (`~/.kaggle/kaggle.json`).
Find-It-Again requires manual download due to license restrictions.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ML-rihal

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download datasets
python download_datasets.py
```

## Training

```bash
python -c "
from solution import DocFusionSolution
sol = DocFusionSolution()
model_dir = sol.train('./dummy_data/train', './work')
print(f'Model saved to: {model_dir}')
"
```

## Prediction

```bash
python -c "
from solution import DocFusionSolution
sol = DocFusionSolution()
sol.predict('./work/model', './dummy_data/test', './predictions.jsonl')
"
```

## Local Validation

```bash
python check_submission.py --submission . --data ./dummy_data --verbose
```

Expected output:
```
[check] Loaded submission: /path/to/ML-rihal
[check] PASSED
```

## Running the Web UI

```bash
streamlit run app.py
```

Open http://localhost:8501. The dashboard has five pages:

| Page | Description |
|------|-------------|
| **Analyze** | Upload a single receipt, view extracted fields + anomaly verdict |
| **Batch** | Upload multiple receipts for bulk analysis with JSONL export |
| **EDA** | Visualize dataset statistics: distributions, field coverage, vendor frequency |
| **Datasets** | Browse individual records from SROIE, CORD-v2, Find-It-Again |
| **About** | Architecture overview, challenge level coverage, performance metrics |

## Running Tests

```bash
# Unit tests (fast, no OCR needed)
pytest tests/test_field_extractor.py tests/test_anomaly_detector.py tests/test_feature_extractor.py -v

# Interface validation
pytest tests/test_interface.py -v

# JSONL output validation
pytest tests/test_jsonl_output.py -v

# OCR engine tests (requires EasyOCR model download on first run)
pytest tests/test_ocr_engine.py -v

# Cross-check against ground truth
pytest tests/test_cross_check.py -v

# Performance benchmarks
pytest tests/test_performance.py -v

# End-to-end smoke test
pytest tests/test_smoke.py -v

# Generate visual report (creates visual_report.html)
pytest tests/test_visual_checks.py -v

# Run everything
pytest tests/ -v
```

## Generating the Word Report

```bash
python generate_report.py
```

Produces `DocFusion_Report.docx` with architecture diagrams, charts, test results,
and complete documentation.

## Docker Deployment

```bash
# Build the image
docker build -t docfusion .

# Run locally
docker run -p 8501:8501 docfusion

# Production deployment
docker run -d -p 80:8501 --restart unless-stopped --name docfusion docfusion
```

## Project Structure

```
ML-rihal/
├── solution.py              # DocFusionSolution class (autograder entry point)
├── ocr_engine.py            # OCR wrapper (EasyOCR primary, PaddleOCR fallback)
├── field_extractor.py       # Regex + spatial heuristic field extraction
├── feature_extractor.py     # Visual (ELA, edge, LBP) + statistical features
├── anomaly_detector.py      # XGBoost forgery classifier
├── utils.py                 # JSONL I/O, image loading, normalization
├── data_loader.py           # Unified loader for SROIE, CORD-v2, Find-It-Again
├── download_datasets.py     # Dataset download script
├── app.py                   # Multi-page Streamlit dashboard
├── requirements.txt         # Pinned dependencies
├── check_submission.py      # Official autograder checker
├── generate_report.py       # Word document report generator
├── generate_dummy_images.py # Dummy PNG generator for smoke tests
├── Dockerfile               # Container configuration
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project metadata (Python 3.13+)
├── notebooks/
│   ├── eda.ipynb            # Level 1 EDA notebook
│   ├── training.ipynb       # Model training & feature importance
│   └── extraction_experiments.ipynb  # Extraction pipeline experiments
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_interface.py    # Autograder interface validation
│   ├── test_jsonl_output.py # JSONL format validation
│   ├── test_ocr_engine.py   # OCR engine unit tests
│   ├── test_field_extractor.py  # Field extraction unit tests
│   ├── test_feature_extractor.py # Feature extraction tests
│   ├── test_anomaly_detector.py  # Anomaly detector tests
│   ├── test_cross_check.py  # Ground truth cross-check
│   ├── test_performance.py  # Performance benchmarks
│   ├── test_smoke.py        # End-to-end smoke test
│   └── test_visual_checks.py # Visual report generator
├── dummy_data/              # Smoke test data
│   ├── train/
│   │   ├── train.jsonl
│   │   └── images/
│   └── test/
│       ├── test.jsonl
│       └── images/
└── data/                    # External datasets (gitignored)
    ├── sroie/               # SROIE v2 (Kaggle)
    ├── cord/                # CORD-v2 (HuggingFace)
    └── finditagain/         # Find-It-Again
```

## Performance

| Metric               | Value           | Budget    |
|----------------------|-----------------|-----------|
| OCR model size       | ~30 MB          | < 100 MB  |
| Anomaly model size   | < 1 MB          | < 50 MB   |
| Peak memory          | < 500 MB        | < 1 GB    |
| Inference per doc    | ~300 ms (CPU)   | < 5 s     |
| Training (20 docs)   | < 60 s          | < 120 s   |

## Challenge Levels Covered

| Level | Task | Implementation |
|-------|------|----------------|
| **1** | Document Understanding & EDA | `notebooks/eda.ipynb` + EDA dashboard page |
| **2** | Structured Information Extraction | `field_extractor.py` – regex + spatial heuristics |
| **3A** | Anomaly Detection | `anomaly_detector.py` + `feature_extractor.py` (XGBoost) |
| **3B** | Basic Web UI | Multi-page Streamlit dashboard (`app.py`) |
| **4A** | Harness Integration | `solution.py` – `DocFusionSolution.train()` / `.predict()` |
| **4B** | Pipeline Efficiency | CPU-only, ~300ms/doc, <500MB RAM |
| **4C** | Reproducibility | `requirements.txt`, deterministic seeds, Docker |
| **Bonus** | Deployment | Dockerfile, Docker Compose ready |

## License

This project was built for the 2026 ML Rihal CodeStacker Challenge.
