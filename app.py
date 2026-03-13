"""
DocFusion: Multi-page Streamlit dashboard.
Premium UI with dark-theme-first design, confidence indicators, and rich interactions.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ocr_engine import OCREngine
from field_extractor import FieldExtractor
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector

st.set_page_config(page_title="DocFusion", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════
#  CSS DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #6C9FFF;
    --accent-dim: rgba(108,159,255,.12);
    --accent-glow: rgba(108,159,255,.25);
    --green: #4ADE80;
    --green-dim: rgba(74,222,128,.12);
    --green-border: rgba(74,222,128,.35);
    --red: #F87171;
    --red-dim: rgba(248,113,113,.12);
    --red-border: rgba(248,113,113,.35);
    --yellow: #FBBF24;
    --yellow-dim: rgba(251,191,36,.12);
    --yellow-border: rgba(251,191,36,.35);
    --surface: rgba(255,255,255,.03);
    --surface-hover: rgba(255,255,255,.06);
    --border: rgba(255,255,255,.08);
    --border-strong: rgba(255,255,255,.14);
    --text-primary: rgba(255,255,255,.92);
    --text-secondary: rgba(255,255,255,.55);
    --text-dim: rgba(255,255,255,.35);
    --radius: .75rem;
    --radius-sm: .5rem;
    --radius-lg: 1rem;
}

html, body, p, input, select, textarea, td, th, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div,
.stTextInput input, .stSelectbox, .stRadio label p,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
.stTabs button, .stExpander summary span {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container { padding: .75rem 1.5rem 2rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
div[data-testid="stMetric"] { background: transparent !important; border: none !important; padding: 0 !important; }

/* ── SIDEBAR ── */
.sb-header {
    padding: 1.25rem 1rem .75rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: .5rem;
}
.sb-logo {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -.03em;
    color: var(--accent);
    line-height: 1.1;
}
.sb-tagline {
    font-size: .72rem;
    color: var(--text-dim);
    margin-top: .2rem;
    letter-spacing: .04em;
    text-transform: uppercase;
}
.sb-footer {
    position: fixed;
    bottom: 0;
    width: inherit;
    padding: .6rem 1rem;
    border-top: 1px solid var(--border);
    font-size: .68rem;
    color: var(--text-dim);
    text-align: center;
    background: inherit;
}

/* ── SIDEBAR RADIO BUTTONS ── */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: .15rem !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: .55rem .85rem !important;
    margin: 0 !important;
    transition: all .12s !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--accent-dim) !important;
    border-color: rgba(108,159,255,.25) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: var(--accent-dim) !important;
    border-color: rgba(108,159,255,.4) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: .88rem !important;
    font-weight: 500 !important;
}

/* ── PAGE HEADER ── */
.page-header {
    display: flex;
    align-items: center;
    gap: .65rem;
    padding-bottom: .65rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-header .ph-icon {
    font-size: 1.5rem;
    line-height: 1;
}
.page-header .ph-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
}
.page-header .ph-desc {
    font-size: .82rem;
    color: var(--text-dim);
    margin-left: .35rem;
}

/* ── HERO EMPTY STATE ── */
.hero-empty {
    text-align: center;
    padding: 4rem 2rem;
    border: 2px dashed var(--border-strong);
    border-radius: var(--radius-lg);
    margin: 1rem 0;
}
.hero-empty .icon { font-size: 3rem; margin-bottom: .75rem; opacity: .4; }
.hero-empty h3 { margin: 0 0 .3rem; font-weight: 600; color: var(--text-secondary); font-size: 1.1rem; }
.hero-empty p { margin: 0; font-size: .85rem; color: var(--text-dim); }

/* ── STATUS BANNERS ── */
.verdict {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .6rem;
    padding: .9rem 1.5rem;
    border-radius: var(--radius);
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: .01em;
}
.verdict-genuine {
    background: linear-gradient(135deg, rgba(74,222,128,.08), rgba(74,222,128,.04));
    border: 1px solid var(--green-border);
    color: var(--green);
}
.verdict-forged {
    background: linear-gradient(135deg, rgba(248,113,113,.1), rgba(248,113,113,.05));
    border: 1px solid var(--red-border);
    color: var(--red);
}
.verdict .verdict-icon { font-size: 1.3rem; }
.verdict .verdict-sub { font-weight: 400; font-size: .82rem; opacity: .7; margin-left: .25rem; }

/* ── CONFIDENCE WARNING ── */
.conf-alert {
    display: flex;
    align-items: center;
    gap: .5rem;
    padding: .6rem 1rem;
    border-radius: var(--radius-sm);
    background: var(--yellow-dim);
    border: 1px solid var(--yellow-border);
    color: var(--yellow);
    font-size: .82rem;
    font-weight: 500;
    margin-top: .5rem;
}

/* ── FIELD CARD ── */
.fc {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
    position: relative;
    transition: border-color .15s, background .15s;
    min-height: 5.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.fc:hover { background: var(--surface-hover); }
.fc.c-green  { border-color: var(--green-border); }
.fc.c-yellow { border-color: var(--yellow-border); }
.fc.c-red    { border-color: var(--red-border); }
.fc-label {
    font-size: .65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-dim);
    margin-bottom: .35rem;
}
.fc-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    word-break: break-word;
    line-height: 1.25;
}
.fc-badge {
    position: absolute;
    top: .45rem;
    right: .55rem;
    font-size: .58rem;
    font-weight: 700;
    padding: .12rem .45rem;
    border-radius: 1rem;
    letter-spacing: .02em;
}
.badge-green  { background: var(--green-dim); color: var(--green); }
.badge-yellow { background: var(--yellow-dim); color: var(--yellow); }
.badge-red    { background: var(--red-dim); color: var(--red); }

/* ── STAT PILLS ── */
.pills { display: flex; flex-wrap: wrap; gap: .35rem; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    padding: .3rem .7rem;
    border-radius: 2rem;
    font-size: .78rem;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
}
.pill b { color: var(--text-primary); font-weight: 700; }

/* ── CHIPS ── */
.chip {
    display: inline-block;
    padding: .18rem .6rem;
    border-radius: 1rem;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .02em;
}
.chip-ok { background: var(--green-dim); color: var(--green); border: 1px solid var(--green-border); }
.chip-bad { background: var(--red-dim); color: var(--red); border: 1px solid var(--red-border); }

/* ── FEATURE GRID ── */
.fg {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: .4rem;
}
.fg-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: .5rem .7rem;
}
.fg-label {
    font-size: .62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--text-dim);
}
.fg-val {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: .1rem;
}

/* ── OCR LINES ── */
.ocr-block {
    max-height: 350px;
    overflow-y: auto;
    padding-right: .5rem;
}
.ol {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: .78rem;
    padding: .3rem .6rem;
    margin-bottom: .15rem;
    background: var(--surface);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.ol-green  { border-left: 3px solid var(--green); }
.ol-yellow { border-left: 3px solid var(--yellow); }
.ol-red    { border-left: 3px solid var(--red); }
.ol-conf {
    flex-shrink: 0;
    font-size: .65rem;
    color: var(--text-dim);
    width: 2.2rem;
    text-align: right;
}
.ol-text { color: var(--text-secondary); }

/* ── DATA TABLE ── */
.dtable { width: 100%; border-collapse: collapse; }
.dtable thead th {
    padding: .55rem .75rem;
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--text-dim);
    text-align: left;
    border-bottom: 2px solid var(--border-strong);
}
.dtable tbody td {
    padding: .5rem .75rem;
    font-size: .84rem;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
}
.dtable tbody tr:hover { background: var(--surface-hover); }
.dtable .fname { font-weight: 600; color: var(--text-primary); }

/* ── ABOUT SECTION ── */
.arch {
    background: var(--accent-dim);
    border: 1px solid rgba(108,159,255,.2);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    font-family: 'Cascadia Code', monospace !important;
    font-size: .82rem;
    line-height: 1.7;
    color: var(--text-secondary);
    overflow-x: auto;
}
.arch b { color: var(--accent); }
.info-table { width: 100%; border-collapse: collapse; }
.info-table th {
    padding: .55rem .75rem; text-align: left;
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; color: var(--text-dim);
    border-bottom: 2px solid var(--border-strong);
}
.info-table td {
    padding: .5rem .75rem; font-size: .84rem;
    color: var(--text-secondary); border-bottom: 1px solid var(--border);
}
.info-table .lvl { font-weight: 700; color: var(--accent); }
.info-table a { color: var(--accent); text-decoration: none; }
.info-table a:hover { text-decoration: underline; }

/* ── SECTION HEADING ── */
.sec-head {
    font-size: .95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.25rem 0 .6rem;
    padding-bottom: .35rem;
    border-bottom: 1px solid var(--border);
}

/* ── DOWNLOAD BUTTONS (override Streamlit) ── */
.stDownloadButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    transition: all .15s !important;
}
.stDownloadButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PIPELINE & HELPERS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def _load_pipeline():
    ocr = OCREngine()
    ext = FieldExtractor()
    feat = FeatureExtractor()
    det = AnomalyDetector()
    md = Path("tmp_work") / "model"
    if md.exists():
        feat.load_stats(str(md))
        det.load(str(md))
    return ocr, ext, feat, det


def _analyze(cv_img, ocr, ext, feat, det):
    res = ocr.run(cv_img)
    fields = ext.extract(res)
    vf = feat.visual_features(cv_img)
    sf = feat.statistical_features(fields, res)
    forged = det.predict_one({**vf, **sf})
    return res, fields, vf, sf, forged


def _annotate(rgb, ocr_res):
    out = rgb.copy()
    for _, box, conf in ocr_res.lines_top_to_bottom():
        pts = np.array(box, dtype=np.int32)
        c = (220, 60, 60) if conf < 0.4 else (240, 180, 40) if conf < 0.7 else (50, 190, 90)
        cv2.polylines(out, [pts], True, c, 2)
        tl = pts[0]
        cv2.putText(out, f"{conf:.0%}", (int(tl[0]), int(tl[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    return out


def _clevel(c):
    return "green" if c >= 0.7 else "yellow" if c >= 0.4 else "red"


def _field_conf(val, ocr_res):
    if not val or not ocr_res.texts:
        return None
    vl = str(val).lower()
    best = None
    for t, _, c in ocr_res.lines_top_to_bottom():
        if vl in t.lower() or t.lower() in vl:
            if best is None or c > best:
                best = c
    return best


def _fc(label, value, conf=None):
    v = value or "---"
    cls = badge = ""
    if conf is not None:
        cl = _clevel(conf)
        cls = f" c-{cl}"
        badge = f'<span class="fc-badge badge-{cl}">{conf:.0%}</span>'
    return f'<div class="fc{cls}">{badge}<div class="fc-label">{label}</div><div class="fc-value">{v}</div></div>'


def _fg(label, value):
    return f'<div class="fg-item"><div class="fg-label">{label}</div><div class="fg-val">{value}</div></div>'


def _page_header(icon, title, desc=""):
    d = f'<span class="ph-desc">{desc}</span>' if desc else ''
    st.markdown(
        f'<div class="page-header">'
        f'<span class="ph-icon">{icon}</span>'
        f'<span class="ph-title">{title}</span>'
        f'{d}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
PAGES = ["Analyze", "Batch", "EDA", "Datasets", "About"]
ICONS = {"Analyze": "🔍", "Batch": "📦", "EDA": "📊", "Datasets": "🗂", "About": "ℹ"}

with st.sidebar:
    st.markdown(
        '<div class="sb-header">'
        '<div class="sb-logo">DocFusion</div>'
        '<div class="sb-tagline">Intelligent Document Analyzer</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        PAGES,
        format_func=lambda p: f"{ICONS[p]}  {p}",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sb-footer">v2.0 &middot; 2026 ML Rihal CodeStacker</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: ANALYZE
# ═══════════════════════════════════════════════════════════════
if page == "Analyze":
    _page_header("🔍", "Receipt Analyzer", "Upload a receipt to extract fields and detect forgery")

    ocr, ext, feat, det = _load_pipeline()

    uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if not uploaded:
        st.markdown(
            '<div class="hero-empty">'
            '<div class="icon">📄</div>'
            '<h3>No document uploaded</h3>'
            '<p>Drag and drop a receipt image above, or click Browse Files</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        pil = Image.open(uploaded).convert("RGB")
        arr = np.array(pil)
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing document..."):
            ocr_res, fields, vf, sf, forged = _analyze(cv_img, ocr, ext, feat, det)

        avg_c = float(np.mean(ocr_res.confidences)) if ocr_res.confidences else 0.0

        # Verdict
        if forged:
            st.markdown(
                '<div class="verdict verdict-forged">'
                '<span class="verdict-icon">&#9888;</span>'
                'SUSPICIOUS'
                '<span class="verdict-sub">Potential forgery detected</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict verdict-genuine">'
                '<span class="verdict-icon">&#10003;</span>'
                'GENUINE'
                '<span class="verdict-sub">No anomalies detected</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        if avg_c < 0.6:
            st.markdown(
                f'<div class="conf-alert">'
                f'&#9888; Low OCR confidence ({avg_c:.0%}) &mdash; results may be unreliable. Try a clearer scan.'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        # Main two-column
        col_img, col_data = st.columns([5, 7], gap="large")

        with col_img:
            if ocr_res.texts:
                st.image(_annotate(arr, ocr_res), use_container_width=True, caption="Color = confidence: green > yellow > red")
            else:
                st.image(pil, use_container_width=True)

        with col_data:
            vc = _field_conf(fields.get("vendor"), ocr_res)
            dc = _field_conf(fields.get("date"), ocr_res)
            tc = _field_conf(fields.get("total"), ocr_res)

            f1, f2, f3 = st.columns(3)
            f1.markdown(_fc("Vendor", fields.get("vendor"), vc), unsafe_allow_html=True)
            f2.markdown(_fc("Date", fields.get("date"), dc), unsafe_allow_html=True)
            f3.markdown(_fc("Total", fields.get("total"), tc), unsafe_allow_html=True)

            st.write("")

            n_fields = sum(1 for v in fields.values() if v is not None)
            pills_html = '<div class="pills">'
            pills_html += f'<span class="pill">Confidence <b>{avg_c:.0%}</b></span>'
            pills_html += f'<span class="pill">Lines <b>{len(ocr_res.texts)}</b></span>'
            pills_html += f'<span class="pill">Chars <b>{sum(len(t) for t in ocr_res.texts)}</b></span>'
            pills_html += f'<span class="pill">Fields <b>{n_fields}/3</b></span>'
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)

            st.write("")

            tab_v, tab_s, tab_o = st.tabs(["Visual Analysis", "Statistics", "Raw OCR"])

            with tab_v:
                items = [
                    ("ELA Mean", f"{vf['ela_mean']:.2f}"), ("ELA Max", f"{vf['ela_max']:.2f}"),
                    ("ELA Std", f"{vf['ela_std']:.2f}"), ("High ELA", f"{vf['ela_high_ratio']:.3f}"),
                    ("Edge Density", f"{vf['edge_density']:.4f}"), ("Noise", f"{vf['noise_level']:.1f}"),
                    ("Brightness", f"{vf['brightness_mean']:.1f}"), ("Bright Std", f"{vf['brightness_std']:.1f}"),
                    ("LBP Uniform", f"{vf['lbp_uniformity']:.3f}"), ("LBP Entropy", f"{vf['lbp_entropy']:.2f}"),
                ]
                st.markdown('<div class="fg">' + "".join(_fg(l, v) for l, v in items) + '</div>', unsafe_allow_html=True)

            with tab_s:
                items = [
                    ("Completeness", f"{sf['field_completeness']:.0%}"), ("Total", f"${sf['total_value']:.2f}"),
                    ("Z-Score", f"{sf['total_zscore']:.2f}"), ("Round Amt", "Yes" if sf['total_is_round'] else "No"),
                    ("Conf Mean", f"{sf['ocr_conf_mean']:.2f}"), ("Conf Min", f"{sf['ocr_conf_min']:.2f}"),
                    ("Text Len", str(int(sf['text_length']))), ("Lines", str(int(sf['num_lines']))),
                ]
                st.markdown('<div class="fg">' + "".join(_fg(l, v) for l, v in items) + '</div>', unsafe_allow_html=True)

            with tab_o:
                if ocr_res.texts:
                    html = '<div class="ocr-block">'
                    for t, _, c in ocr_res.lines_top_to_bottom():
                        cl = _clevel(c)
                        html += f'<div class="ol ol-{cl}"><span class="ol-conf">{c:.0%}</span><span class="ol-text">{t}</span></div>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("No text detected.")

        st.write("")

        d1, d2, _ = st.columns([1, 1, 4])
        rj = {"vendor": fields.get("vendor"), "date": fields.get("date"), "total": fields.get("total"), "is_forged": int(forged)}
        d1.download_button("Download JSON", json.dumps(rj, indent=2), "result.json", "application/json", use_container_width=True)
        raw = "\n".join(f"[{c:.2f}] {t}" for t, _, c in ocr_res.lines_top_to_bottom())
        d2.download_button("Download OCR", raw, "ocr_output.txt", "text/plain", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: BATCH
# ═══════════════════════════════════════════════════════════════
elif page == "Batch":
    _page_header("📦", "Batch Processing", "Upload multiple receipt images for bulk analysis")

    ocr, ext, feat, det = _load_pipeline()
    files = st.file_uploader("Upload receipt images", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=True)

    if not files:
        st.markdown(
            '<div class="hero-empty">'
            '<div class="icon">📦</div>'
            '<h3>No documents uploaded</h3>'
            '<p>Upload one or more receipt images to begin batch analysis</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        results = []
        prog = st.progress(0, text="Starting...")
        for i, f in enumerate(files):
            prog.progress((i + 1) / len(files), text=f"Analyzing {f.name} ({i + 1}/{len(files)})")
            pil = Image.open(f).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            ocr_res, fields, vf, sf, forged = _analyze(cv_img, ocr, ext, feat, det)
            results.append({
                "filename": f.name,
                "vendor": fields.get("vendor") or "---",
                "date": fields.get("date") or "---",
                "total": fields.get("total") or "---",
                "is_forged": int(forged),
                "confidence": round(float(np.mean(ocr_res.confidences)), 3) if ocr_res.confidences else 0,
                "lines": len(ocr_res.texts),
            })
        prog.empty()

        nf = sum(r["is_forged"] for r in results)
        ng = len(results) - nf
        ac = np.mean([r["confidence"] for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_fc("Documents", str(len(results))), unsafe_allow_html=True)
        c2.markdown(f'<div class="fc c-green"><div class="fc-label">Genuine</div><div class="fc-value" style="color:var(--green)">{ng}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="fc c-red"><div class="fc-label">Suspicious</div><div class="fc-value" style="color:var(--red)">{nf}</div></div>', unsafe_allow_html=True)
        c4.markdown(_fc("Avg Confidence", f"{ac:.0%}"), unsafe_allow_html=True)

        st.write("")

        tbl = '<table class="dtable"><thead><tr>'
        for h in ["File", "Vendor", "Date", "Total", "Status", "Conf", "Lines"]:
            tbl += f'<th>{h}</th>'
        tbl += '</tr></thead><tbody>'
        for r in results:
            chip = '<span class="chip chip-ok">Genuine</span>' if not r["is_forged"] else '<span class="chip chip-bad">Suspicious</span>'
            tbl += f'<tr><td class="fname">{r["filename"]}</td><td>{r["vendor"]}</td><td>{r["date"]}</td><td>{r["total"]}</td><td>{chip}</td><td>{r["confidence"]:.0%}</td><td>{r["lines"]}</td></tr>'
        tbl += '</tbody></table>'
        st.markdown(tbl, unsafe_allow_html=True)

        st.write("")
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("Download JSONL", "\n".join(json.dumps(r) for r in results), "batch.jsonl", "application/jsonl", use_container_width=True)
        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=results[0].keys()).writeheader()
        csv.DictWriter(buf, fieldnames=results[0].keys()).writerows(results)
        d2.download_button("Download CSV", buf.getvalue(), "batch.csv", "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: EDA
# ═══════════════════════════════════════════════════════════════
elif page == "EDA":
    _page_header("📊", "Exploratory Data Analysis", "Dataset statistics and distributions")

    try:
        from data_loader import UnifiedDataLoader
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
            "axes.edgecolor": "#333", "axes.labelcolor": "#999",
            "text.color": "#999", "xtick.color": "#777", "ytick.color": "#777",
            "grid.color": "#222",
        })

        with st.sidebar:
            st.divider()
            st.markdown("**EDA Settings**")
            dr = st.text_input("Data directory", "data", key="eda_root")
            sp = st.selectbox("Split", ["train", "test", "validation"], key="eda_split")
            go = st.button("Load & Analyze", type="primary", use_container_width=True)

        if go:
            loader = UnifiedDataLoader(dr)
            with st.spinner("Loading dataset..."):
                recs = loader.load_all(sp)
            if not recs:
                st.warning(f"No records in `{dr}` for `{sp}`. Run `python download_datasets.py` first.")
            else:
                stats = loader.get_stats(recs)
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(_fc("Records", str(stats["total_records"])), unsafe_allow_html=True)
                c2.markdown(f'<div class="fc c-green"><div class="fc-label">Genuine</div><div class="fc-value" style="color:var(--green)">{stats["genuine_count"]}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="fc c-red"><div class="fc-label">Forged</div><div class="fc-value" style="color:var(--red)">{stats["forged_count"]}</div></div>', unsafe_allow_html=True)
                c4.markdown(_fc("Sources", str(len(stats["by_source"]))), unsafe_allow_html=True)

                st.write("")
                t1, t2, t3, t4 = st.tabs(["Distributions", "Field Coverage", "Sources", "Top Vendors"])

                with t1:
                    totals = []
                    for r in recs:
                        t = r.get("fields", {}).get("total")
                        if t:
                            try: totals.append(float(str(t).replace(",", "")))
                            except ValueError: pass
                    if totals:
                        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
                        ax[0].hist(totals, bins=30, color="#6C9FFF", edgecolor="#0e1117", alpha=.9)
                        ax[0].set_title("Total Distribution", fontweight="bold", fontsize=12)
                        ax[0].set_xlabel("Amount"); ax[0].set_ylabel("Freq"); ax[0].grid(True, alpha=.15)
                        bp = ax[1].boxplot(totals, vert=True, patch_artist=True)
                        bp["boxes"][0].set_facecolor("#6C9FFF"); bp["boxes"][0].set_alpha(.5)
                        ax[1].set_title("Box Plot", fontweight="bold", fontsize=12); ax[1].grid(True, alpha=.15)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    else: st.info("No totals found.")

                with t2:
                    fd = stats["fields_present"]
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    cs = ["#6C9FFF", "#4ADE80", "#FBBF24"]
                    bars = ax.bar(list(fd.keys()), list(fd.values()), color=cs, edgecolor="#0e1117", width=.5)
                    ax.set_title("Field Presence", fontweight="bold", fontsize=12); ax.set_ylabel("Count")
                    ax.axhline(stats["total_records"], color="#555", ls="--", alpha=.4, label="Total"); ax.legend(fontsize=9)
                    ax.grid(True, axis="y", alpha=.12)
                    for b, val in zip(bars, fd.values()):
                        pct = val / stats["total_records"] * 100 if stats["total_records"] else 0
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + stats["total_records"]*.02, f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold", color="#aaa")
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                with t3:
                    sd = stats["by_source"]
                    if sd:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        cs = ["#6C9FFF", "#4ADE80", "#F87171", "#FBBF24"]
                        w, t, a = ax.pie(list(sd.values()), labels=list(sd.keys()), autopct="%1.1f%%", colors=cs[:len(sd)], startangle=140, textprops={"color": "#ccc", "fontsize": 11})
                        for x in a: x.set_fontweight("bold"); x.set_color("#fff")
                        ax.set_title("By Source", fontweight="bold", fontsize=12); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                with t4:
                    vd = {}
                    for r in recs:
                        v = r.get("fields", {}).get("vendor")
                        if v: vd[v] = vd.get(v, 0) + 1
                    if vd:
                        top = sorted(vd.items(), key=lambda x: -x[1])[:15]
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh([v[0][:28] for v in reversed(top)], [v[1] for v in reversed(top)], color="#6C9FFF", edgecolor="#0e1117", height=.55)
                        ax.set_xlabel("Count"); ax.set_title("Top 15 Vendors", fontweight="bold", fontsize=12); ax.grid(True, axis="x", alpha=.12)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    else: st.info("No vendor data.")
        else:
            st.markdown(
                '<div class="hero-empty"><div class="icon">📊</div><h3>Ready to explore</h3>'
                '<p>Configure data directory and split in the sidebar, then click Load &amp; Analyze</p></div>',
                unsafe_allow_html=True,
            )
    except ImportError:
        st.error("Missing `data_loader.py` or `matplotlib`.")


# ═══════════════════════════════════════════════════════════════
#  PAGE: DATASETS
# ═══════════════════════════════════════════════════════════════
elif page == "Datasets":
    _page_header("🗂", "Dataset Browser", "Browse SROIE, CORD-v2, and Find-It-Again records")

    try:
        from data_loader import UnifiedDataLoader
        with st.sidebar:
            st.divider()
            st.markdown("**Browser Settings**")
            dr = st.text_input("Data directory", "data", key="ds_root")
            sp = st.selectbox("Split", ["train", "test", "validation"], key="ds_sp")
            sf = st.selectbox("Source", ["all", "sroie", "cord", "finditagain"])
            ps = st.slider("Per page", 5, 50, 10)
            go = st.button("Load Dataset", type="primary", use_container_width=True)

        if go:
            loader = UnifiedDataLoader(dr)
            with st.spinner("Loading..."):
                recs = loader.load_all(sp)
            if sf != "all":
                recs = [r for r in recs if r["source"] == sf]
            if not recs:
                st.warning("No records found. Run `python download_datasets.py`.")
            else:
                st.markdown(f'<div class="pills"><span class="pill">Loaded <b>{len(recs)}</b> records</span></div>', unsafe_allow_html=True)
                st.write("")
                tp = max(1, (len(recs) + ps - 1) // ps)
                cp = st.number_input("Page", 1, tp, 1) - 1
                pr = recs[cp * ps: cp * ps + ps]
                for rec in pr:
                    sc = {"sroie": "var(--accent)", "cord": "var(--green)", "finditagain": "var(--yellow)"}.get(rec["source"], "#888")
                    isf = rec.get("label", {}).get("is_forged", 0)
                    ch = '<span class="chip chip-bad">Forged</span>' if isf else '<span class="chip chip-ok">Genuine</span>'
                    with st.expander(f"{rec['id']}"):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            ip = rec.get("image_path", "")
                            if ip and os.path.exists(ip): st.image(ip, width=260)
                            else: st.info("Image not available")
                        with c2:
                            st.markdown(f'<div class="pills"><span class="pill" style="border-color:{sc}">{rec["source"]}</span>{ch}</div>', unsafe_allow_html=True)
                            st.write("")
                            r1, r2, r3 = st.columns(3)
                            r1.markdown(_fc("Vendor", rec["fields"].get("vendor")), unsafe_allow_html=True)
                            r2.markdown(_fc("Date", rec["fields"].get("date")), unsafe_allow_html=True)
                            r3.markdown(_fc("Total", rec["fields"].get("total")), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="hero-empty"><div class="icon">🗂</div><h3>Ready to browse</h3>'
                '<p>Configure settings in the sidebar, then click Load Dataset</p></div>',
                unsafe_allow_html=True,
            )
    except ImportError:
        st.error("Missing `data_loader.py`.")


# ═══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "About":
    _page_header("ℹ", "About DocFusion", "2026 ML Rihal CodeStacker Challenge")

    st.markdown('<div class="sec-head">Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="arch">'
        'Image &rarr; <b>Preprocess</b> (CLAHE + Denoise) &rarr; <b>EasyOCR</b> (1536px) &rarr; <b>Regex Extraction</b> &rarr; Fields<br>'
        '&nbsp; &darr;<br>'
        '&nbsp; ELA + Edge + LBP &rarr; <b>XGBoost Classifier</b> &rarr; is_forged<br>'
        '&nbsp; Statistical Features &nearr;'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-head">Challenge Levels</div>', unsafe_allow_html=True)
    tbl = '<table class="info-table"><thead><tr><th>Level</th><th>Task</th><th>Implementation</th></tr></thead><tbody>'
    for lvl, task, impl in [
        ("1", "Document Understanding & EDA", "<code>notebooks/eda.ipynb</code> + EDA page"),
        ("2", "Structured Extraction", "<code>field_extractor.py</code> &mdash; regex + spatial + adjacent-line"),
        ("3A", "Anomaly Detection", "<code>anomaly_detector.py</code> + <code>feature_extractor.py</code>"),
        ("3B", "Web UI", "This Streamlit dashboard"),
        ("4A", "Harness Integration", "<code>solution.py</code> &mdash; train() / predict()"),
        ("4B", "Efficiency", "CPU-only, ~300ms/doc, &lt;500MB RAM"),
        ("4C", "Reproducibility", "requirements.txt, seeds, Docker"),
    ]:
        tbl += f'<tr><td class="lvl">{lvl}</td><td>{task}</td><td>{impl}</td></tr>'
    tbl += '</tbody></table>'
    st.markdown(tbl, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Datasets</div>', unsafe_allow_html=True)
    tbl = '<table class="info-table"><thead><tr><th>Dataset</th><th>Records</th><th>Usage</th></tr></thead><tbody>'
    for n, c, u in [
        ('<a href="https://www.kaggle.com/datasets/urbikn/sroie-datasetv2">SROIE v2</a>', "~1,000", "Field extraction"),
        ('<a href="https://huggingface.co/datasets/naver-clova-ix/cord-v2">CORD-v2</a>', "~11,000", "Field extraction"),
        ('<a href="https://l3i-share.univ-lr.fr/2023Finditagain/index.html">Find-It-Again</a>', "~1,500", "Forgery detection"),
    ]:
        tbl += f'<tr><td>{n}</td><td>{c}</td><td>{u}</td></tr>'
    tbl += '</tbody></table>'
    st.markdown(tbl, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg">' + "".join(_fg(l, v) for l, v in [
        ("OCR Model", "~30 MB"), ("Anomaly Model", "<1 MB"), ("Peak Memory", "<500 MB"), ("Inference", "~300 ms/doc")
    ]) + '</div>', unsafe_allow_html=True)

    st.write("")
    st.divider()
    st.caption("Built for the 2026 ML Rihal CodeStacker Challenge")
