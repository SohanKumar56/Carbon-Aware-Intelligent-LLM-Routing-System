"""
app.py — Carbon-Aware Intelligent LLM Routing System
=====================================================
Main Streamlit dashboard entry point.

Run with:
    streamlit run app.py

Architecture overview
---------------------
User Prompt
    ↓
complexity_classifier  (MiniLM fine-tuned, 93.2% accuracy)
    ├── small  → TinyLlama / DeepSeek-Coder 1.3B / Qwen2 1.5B
    ├── medium → Qwen2.5 3B / Phi-3
    └── large  → Qwen2.5 7B / Gemma3 4B
    ↓
energy_tracker  — CO₂ / kWh accounting vs always-large baseline
    ↓
Streamlit dashboard — routing result, energy savings, charts
"""

import streamlit as st

from config import PAGE_TITLE, PAGE_ICON, LAYOUT
from ollama_dashboard import render_dashboard
from prompt_router_dashboard import render_prompt_router_tab, render_router_stats

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark startup-dashboard aesthetic ─────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* ── Hide Streamlit chrome ────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Cards ────────────────────────────────────────── */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-accent { border-left: 4px solid #22c55e; }

    /* ── Hero header ──────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 60%);
        border: 1px solid #065f46;
        border-radius: 18px;
        padding: 2rem 2.4rem 1.6rem;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
        margin: 0 0 .4rem;
    }
    .hero p { color: #94a3b8; font-size: .95rem; margin: 0; }

    /* ── Stage badge ──────────────────────────────────── */
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: .25rem .85rem;
        font-size: .8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .04em;
    }
    .badge-green  { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
    .badge-blue   { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-orange { background: #431407; color: #fdba74; border: 1px solid #f97316; }

    /* ── Metric overrides ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem; }

    /* ── Table ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        background: #1e293b !important;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #ecfdf5;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        padding: .55rem 1.4rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }

    /* ── Text input ───────────────────────────────────── */
    .stTextArea textarea {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
    }

    /* ── Section titles ───────────────────────────────── */
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #4ade80;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: .4rem;
        border-bottom: 1px solid #1e3a2b;
    }

    /* ── Progress bar ─────────────────────────────────── */
    .stProgress > div > div { background: #22c55e !important; border-radius: 9px; }
    .stProgress { border-radius: 9px; }

    /* ── Selectbox ────────────────────────────────────── */
    .stSelectbox > div > div {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
    }

    /* ── Divider ──────────────────────────────────────── */
    hr { border-color: #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: .6rem 0 1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:700;color:#4ade80;">🌿 Carbon-Aware</div>
            <div style="font-size:.78rem;color:#64748b;">LLM Routing System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Dashboard",
        ["Prompt Router", "Ollama Compare"],
        help="Prompt Router: classify & route prompts to right-sized models. "
             "Ollama Compare: run the same prompt on 3 models side-by-side.",
    )
    st.markdown("---")
    render_router_stats()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "Prompt Router":
    render_prompt_router_tab()

elif app_mode == "Ollama Compare":
    render_dashboard()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;margin-top:3rem;padding:1rem;
                color:#334155;font-size:.78rem;font-family:'JetBrains Mono',monospace;">
        Carbon-Aware LLM Routing · Green AI ·
        Built with 🌿 Streamlit + HuggingFace Transformers + Ollama + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
