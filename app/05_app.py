# app/05_app.py
"""
PlantDx -- Streamlit Demo

All predictions go through InferenceEngine (core/inference_engine.py).
Models loaded once per session via @st.cache_resource.
Every upload is saved to disk and logged to logs/predictions/.

Run from the project root (PlantDx/):
    streamlit run app/05_app.py

Pin: streamlit>=1.40.0,<1.45.0
"""

import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import Paths, load_config
from utils.io_utils import save_uploaded_image, load_json, collect_images
from utils.model_utils import DEVICE, load_qwen_model, load_clip
from core.dataset import INSTRUCTION
from core.inference_engine import InferenceEngine


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PlantDx",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,600;1,300&family=DM+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');
  :root {
    --bg:#0d1a0f; --surface:#142016; --border:#1e3320;
    --accent:#4ade80; --accent2:#86efac; --text:#e2f5e5; --muted:#6b9e72;
  }
  html,body,[data-testid="stAppViewContainer"]{
    background-color:var(--bg)!important;color:var(--text)!important;font-family:'Inter',sans-serif}
  [data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)}
  h1{font-family:'Fraunces',serif;font-weight:300;color:var(--accent)!important}
  h2,h3,h4{font-family:'Fraunces',serif;color:var(--accent2)!important}
  p,li,label{color:var(--text)!important}
  .dx-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:1.5rem;margin-bottom:1rem}
  .pill{display:inline-block;padding:3px 14px;border-radius:20px;
    font-family:'DM Mono',monospace;font-size:.78rem;font-weight:500;letter-spacing:.04em}
  .conf-bar-bg{background:#1e3320;border-radius:6px;height:8px;width:100%;margin-top:4px}
  .conf-bar-fill{height:8px;border-radius:6px}
  .field-row{display:flex;gap:.5rem;align-items:baseline;margin-bottom:.4rem;
    border-bottom:1px solid var(--border);padding-bottom:.4rem}
  .field-label{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted);
    min-width:180px;text-transform:uppercase;letter-spacing:.06em}
  .field-value{font-size:.92rem;color:var(--text)}
  .ood-warn{background:#2d1a00;border:1px solid #b45309;border-radius:10px;
    padding:1.2rem 1.5rem;color:#fde68a}
  .ood-warn strong{color:#fbbf24}
  .raw-output{font-family:'DM Mono',monospace;font-size:.82rem;background:#0a1209;
    border:1px solid var(--border);border-radius:8px;padding:1rem;
    white-space:pre-wrap;color:var(--accent2)}
  [data-testid="stFileUploader"]{
    border:1.5px dashed var(--border)!important;border-radius:10px!important;
    background:var(--surface)!important}
  .stButton>button{background:var(--accent)!important;color:#0d1a0f!important;
    font-family:'DM Mono',monospace!important;font-weight:500!important;
    border-radius:8px!important;border:none!important}
  .stButton>button:hover{filter:brightness(1.1)}
  [data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'DM Mono',monospace!important}
  [data-testid="stMetricLabel"]{color:var(--muted)!important}
</style>
""", unsafe_allow_html=True)


# ── Severity palette ──────────────────────────────────────────────────────────

SEVERITY_STYLE = {
    "None":               ("background:#14532d;color:#4ade80;", "✅"),
    "Mild":               ("background:#365314;color:#a3e635;", "🟢"),
    "Mild to Moderate":   ("background:#422006;color:#fbbf24;", "🟡"),
    "Moderate":           ("background:#431407;color:#fb923c;", "🟠"),
    "Moderate to Severe": ("background:#450a0a;color:#f87171;", "🔴"),
    "Severe":             ("background:#3b0764;color:#e879f9;", "🟣"),
}


def severity_pill(sev: str) -> str:
    style, icon = SEVERITY_STYLE.get(sev, ("background:#1c2b1f;color:#d1d5db;", "❓"))
    return '<span class="pill" style="{}">{} {}</span>'.format(style, icon, sev)


def confidence_bar_html(pct: float) -> str:
    color = "#4ade80" if pct >= 80 else ("#fbbf24" if pct >= 60 else "#f87171")
    return (
        '<div style="margin-top:2px">'
        '<span style="font-family:DM Mono,monospace;font-size:.78rem;color:{0}">'
        "{1:.1f} %</span>"
        '<div class="conf-bar-bg">'
        '<div class="conf-bar-fill" style="width:{2:.0f}%;background:{0}"></div>'
        "</div></div>"
    ).format(color, pct, min(pct, 100))


# ── Cached resource loaders ───────────────────────────────────────────────────
# @st.cache_resource: each function runs once per Streamlit session.
# Models stay in VRAM; no reload on widget interaction.

@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_paths():
    return Paths(get_config())


@st.cache_resource(show_spinner=False)
def get_qwen_model():
    cfg, paths = get_config(), get_paths()
    if not paths.best_model.exists():
        return None, None
    try:
        return load_qwen_model(str(paths.best_model), cfg["model"]["name"])
    except Exception as exc:
        st.error(
            "Qwen2.5-VL failed to load: {}\n\n"
            "Check outputs/checkpoints/best/ — it may be missing or corrupted.\n"
            "CUDA OOM? Close other GPU applications and restart Streamlit.".format(exc)
        )
        return None, None


@st.cache_resource(show_spinner=False)
def get_clip():
    cfg = get_config()
    try:
        return load_clip(cfg["evaluation"]["clip_model"])
    except Exception as exc:
        st.error("CLIP failed to load: {}".format(exc))
        return None, None


@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[InferenceEngine]:
    """
    Build the single InferenceEngine instance for this session.
    All inference calls in the app go through this object — nothing
    else calls model_utils directly.
    """
    model, processor      = get_qwen_model()
    clip_model, clip_proc = get_clip()
    if model is None:
        return None
    return InferenceEngine(model, processor, clip_model, clip_proc, get_config())


@st.cache_data(show_spinner=False)
def get_eval_report() -> Optional[dict]:
    paths = get_paths()
    return load_json(paths.eval_report) if paths.eval_report.exists() else None


@st.cache_data(show_spinner=False)
def get_sample_images() -> dict:
    paths  = get_paths()
    result = {}
    if not paths.dataset_root.exists():
        return result
    for folder in sorted(f for f in paths.dataset_root.iterdir() if f.is_dir()):
        imgs = collect_images(folder)
        if imgs:
            result[folder.name] = str(imgs[0])
    return result


# ── Prediction logger ─────────────────────────────────────────────────────────

def log_prediction(
    image_record: dict,
    result:       dict,
    instruction:  str,
) -> None:
    """
    Write one structured JSON entry to logs/predictions/<uuid>.json.
    Called for every upload — whether it passed OOD or not.
    """
    paths = get_paths()
    paths.prediction_logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "log_id":         str(uuid.uuid4()),
        "timestamp":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "image":          image_record,
        "instruction":    instruction,
        "status":         result.get("status", ""),
        "is_plant":       result.get("is_plant", True),
        "ood_score":      result.get("ood_score", 0.0),
        "prediction":     result.get("prediction", ""),
        "confidence_pct": result.get("confidence", 0.0),
        "condition":      result.get("condition", ""),
        "severity":       result.get("severity", ""),
        "device":         DEVICE,
    }
    log_path = paths.prediction_logs_dir / "{}.json".format(entry["log_id"])
    with open(str(log_path), "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(cfg: dict) -> Tuple[str, bool]:
    with st.sidebar:
        st.markdown("## 🌿 PlantDx")
        st.markdown(
            "<span style='font-family:DM Mono,monospace;font-size:.74rem;color:#6b9e72'>"
            "Qwen2.5-VL-7B · LoRA r32 · PlantVillage 15-class · Device: {}"
            "</span>".format(DEVICE),
            unsafe_allow_html=True,
        )
        st.divider()

        ood_enabled = st.toggle(
            "OOD Rejection (CLIP)",
            value=cfg["app"]["ood_enabled"],
            help="Reject images that don't look like plant leaves.",
        )
        instruction = st.text_area("Diagnosis prompt", value=INSTRUCTION, height=100)

        st.divider()

        report = get_eval_report()
        if report:
            st.markdown("### 📊 Evaluation Metrics")
            cls = report["classification"]
            cap = report["caption"]
            sev = report["severity"]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", "{:.2%}".format(cls["accuracy"]))
                st.metric("Macro F1", "{:.4f}".format(cls["macro_f1"]))
                st.metric("BLEU-4",   "{:.4f}".format(cap["bleu4"]))
            with c2:
                st.metric("ROUGE-L",  "{:.4f}".format(cap["rougeL"]))
                st.metric("Sev. R\u00b2",  "{:.4f}".format(sev["r2"]))
                st.metric("Sev. MAE", "{:.4f}".format(sev["mae"]))
            st.divider()
            st.markdown("### Per-class F1")
            import pandas as pd
            items = sorted(
                report["classification"]["per_class_f1"].items(),
                key=lambda x: x[1], reverse=True,
            )
            df = pd.DataFrame(items, columns=["class", "f1"])
            st.bar_chart(df.set_index("class")["f1"], use_container_width=True, height=280)
        else:
            st.info("Run 04_evaluate.py to populate metrics here.")

    return instruction, ood_enabled


# ── Diagnosis card ────────────────────────────────────────────────────────────

def render_diagnosis(result: dict) -> None:
    """Render a structured diagnosis card from an InferenceEngine result dict."""

    def field(label, value):
        st.markdown(
            '<div class="field-row">'
            '<span class="field-label">{}</span>'
            '<span class="field-value">{}</span>'
            "</div>".format(label, value),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dx-card">', unsafe_allow_html=True)
    st.markdown("#### \U0001f52c Structured Diagnosis")

    field("Plant",     result.get("plant",     "Unknown"))
    field("Condition", result.get("condition", "Unknown"))

    st.markdown(
        '<div class="field-row">'
        '<span class="field-label">Severity</span>'
        '<span class="field-value">{}</span>'
        "</div>".format(severity_pill(result.get("severity", "Unknown"))),
        unsafe_allow_html=True,
    )

    conf = result.get("confidence", 0.0)
    if conf > 0:
        st.markdown(
            '<div class="field-row">'
            '<span class="field-label">Gen. Confidence Score</span>'
            '<span class="field-value" style="width:100%">{}</span>'
            "</div>".format(confidence_bar_html(conf)),
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Full model output**")
    st.markdown(
        '<div class="raw-output">{}</div>'.format(result.get("prediction", "")),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_ood_warning(ood_score: float) -> None:
    st.markdown(
        '<div class="ood-warn">'
        "<strong>\u26a0\ufe0f Image not recognised as a plant leaf</strong><br><br>"
        "This image does not appear to be a supported plant leaf "
        "(CLIP plant-similarity score: <strong>{:.3f}</strong>).<br><br>"
        "PlantDx supports tomato, potato, and pepper bell leaves from PlantVillage. "
        "Please upload a clear, close-up photo of a plant leaf."
        "</div>".format(ood_score),
        unsafe_allow_html=True,
    )


# ── Upload tab ────────────────────────────────────────────────────────────────

def tab_upload(
    engine:      Optional[InferenceEngine],
    instruction: str,
    ood_enabled: bool,
) -> None:
    st.markdown("### Upload a plant leaf image")
    st.markdown(
        '<span style="font-family:DM Mono,monospace;font-size:.8rem;color:#6b9e72">'
        "Supported: JPG · PNG · JPEG</span>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label="Drop image here", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    paths = get_paths()
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Uploaded image**")
        st.image(image, use_container_width=True)
        image_record = save_uploaded_image(image, uploaded.name, paths.raw_uploads)
        st.markdown(
            '<span style="font-family:DM Mono,monospace;font-size:.72rem;color:#6b9e72">'
            "Saved \u2192 data/raw/uploads/{}</span>".format(image_record["saved_as"]),
            unsafe_allow_html=True,
        )

    with col2:
        if engine is None:
            st.error("No trained model. Run `python scripts/02_train.py` first.")
            return

        with st.spinner("Running diagnosis\u2026"):
            result = engine.predict(image, instruction=instruction, check_ood=ood_enabled)

        if result["status"] == "ood":
            render_ood_warning(result["ood_score"])
        elif result["status"] == "error":
            st.error("Inference error: {}".format(result.get("error", "unknown")))
        else:
            render_diagnosis(result)

        log_prediction(image_record, result, instruction)


# ── Sample gallery tab ────────────────────────────────────────────────────────

def tab_samples(
    engine:      Optional[InferenceEngine],
    instruction: str,
) -> None:
    st.markdown("### Sample images from PlantVillage")
    samples = get_sample_images()
    if not samples:
        st.info("Dataset not found. Check paths.dataset_root in configs/config.yaml.")
        return

    selected = st.selectbox("Choose a class", list(samples.keys()))
    if not selected:
        return

    image = Image.open(samples[selected]).convert("RGB")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption=selected, use_container_width=True)
    with col2:
        if engine is None:
            st.error("No trained model. Run `python scripts/02_train.py` first.")
            return
        if st.button("Run diagnosis"):
            with st.spinner("Analyzing\u2026"):
                # check_ood=False: these are known PlantVillage images
                result = engine.predict(image, instruction=instruction, check_ood=False)
            if result["status"] == "error":
                st.error("Inference error: {}".format(result.get("error", "")))
            else:
                render_diagnosis(result)


# ── Prediction log tab ────────────────────────────────────────────────────────

def tab_logs() -> None:
    st.markdown("### Prediction Log")
    paths = get_paths()
    logs  = sorted(paths.prediction_logs_dir.glob("*.json"), reverse=True)

    if not logs:
        st.info("No predictions logged yet. Upload an image to start.")
        return

    st.markdown(
        '<span style="font-family:DM Mono,monospace;font-size:.8rem;color:#6b9e72">'
        "{} logged prediction(s) \u00b7 {}</span>".format(len(logs), paths.prediction_logs_dir),
        unsafe_allow_html=True,
    )
    st.divider()

    for log_path in logs[:20]:
        try:
            entry = load_json(log_path)
        except Exception:
            continue
        ts       = entry.get("timestamp", "\u2014")
        conf     = entry.get("confidence_pct", 0.0)
        is_pl    = entry.get("is_plant", False)
        pred     = entry.get("prediction", "")
        img_name = entry.get("image", {}).get("original_name", "\u2014")
        ood      = entry.get("ood_score", 0.0)
        status   = "\u2705 Plant" if is_pl else "\u26a0\ufe0f OOD ({:.3f})".format(ood)

        with st.expander(
            "{}  \u00b7  {}  \u00b7  {}".format(ts, img_name, status),
            expanded=False,
        ):
            if pred:
                st.markdown("**Gen. Confidence Score:** {:.1f} %".format(conf))
                st.markdown(
                    '<div class="raw-output">{}</div>'.format(pred),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("Rejected by OOD filter \u2014 no inference ran.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cfg = get_config()

    st.markdown(
        "<h1 style='margin-bottom:0'>\U0001f331 PlantDx</h1>"
        "<p style='color:#6b9e72;margin-top:0;font-size:1rem'>"
        "Vision-Language plant disease diagnostic system \u00b7 Qwen2.5-VL-7B-Instruct"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    instruction, ood_enabled = render_sidebar(cfg)

    paths = get_paths()
    if not paths.best_model.exists():
        st.warning(
            "No trained model at `{}`. "
            "Run `python scripts/02_train.py` first.".format(paths.best_model),
            icon="\u26a0\ufe0f",
        )

    with st.spinner("Loading models\u2026"):
        engine = get_engine()

    tab1, tab2, tab3 = st.tabs(
        ["\U0001f4e4 Upload Image", "\U0001f5bc\ufe0f Sample Gallery", "\U0001f4cb Prediction Log"]
    )
    with tab1:
        tab_upload(engine, instruction, ood_enabled)
    with tab2:
        tab_samples(engine, instruction)
    with tab3:
        tab_logs()

    st.divider()
    st.markdown(
        '<span style="font-family:DM Mono,monospace;font-size:.72rem;color:#4a7253">'
        "PlantDx v1.0 \u00b7 Qwen2.5-VL-7B-Instruct \u00b7 LoRA r32 \u00b7 "
        "PlantVillage 15-class \u00b7 OOD via CLIP ViT-B/32 \u00b7 Device: {}"
        "</span>".format(DEVICE),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
