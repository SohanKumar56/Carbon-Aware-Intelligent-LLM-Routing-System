"""
app.py — Carbon-Aware Adaptive AI Inference System
===================================================
Main Streamlit dashboard entry point.

Run with:
    streamlit run app.py

Architecture overview
---------------------
User Input
    ↓
inference_pipeline.run_pipeline()
    ├── Stage 1: rule_engine      (near-zero energy)
    ├── Stage 2: RoBERTa          (small model)
    └── Stage 3: BERT             (large model)
    ↓
energy_tracker.estimate_energy() — CO₂ / kWh accounting
    ↓
Streamlit dashboard — charts, metrics, log table
"""

import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go

from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, EXAMPLE_TEXTS,
    ENERGY_LARGE_MODEL,
)
from inference_pipeline import run_pipeline
from ollama_integration import (
    get_ollama_models,
    run_multiple_ollama_models,
    check_ollama_availability,
)
from model_comparison import compare_models, get_best_model
from dashboard_utils import (
    energy_bar_chart,
    stage_distribution_pie,
    carbon_timeline,
    green_score_gauge,
    build_log_df,
    STAGE_COLOURS,
    _base_layout,
    CARD_COLOUR,
    TEXT_COLOUR,
    GRID_COLOUR,
)
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

    /* ── Selectbox / example picker ───────────────────── */
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

# ── Session state initialisation ──────────────────────────────────────────────
if "inference_log"    not in st.session_state:
    st.session_state["inference_log"]    = []
if "total_saved_kwh"  not in st.session_state:
    st.session_state["total_saved_kwh"]  = 0.0
if "last_result"      not in st.session_state:
    st.session_state["last_result"]      = None
if "last_ollama_results" not in st.session_state:
    st.session_state["last_ollama_results"] = []
if "selected_model_for_viz" not in st.session_state:
    st.session_state["selected_model_for_viz"] = "Traditional"

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Dashboard", 
        ["Sentiment Tracker", "Ollama Compare", "Prompt Router"],
        help="Choose between: Sentiment Analysis, Ollama Model Comparison, or Prompt Complexity Routing."
    )
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "Sentiment Tracker":
    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>🌿 Carbon-Aware AI Inference System</h1>
            <p>
                A Green AI demonstration that uses a <strong>three-stage adaptive pipeline</strong>
                to minimise energy consumption and carbon emissions during NLP inference.
                Simple inputs are resolved with near-zero energy; complex ones escalate to
                heavier models only when needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — INPUT
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📝 Input</p>', unsafe_allow_html=True)

    col_input, col_sidebar = st.columns([3, 1])

    with col_input:
        # Example sentence picker
        example_choice = st.selectbox(
            "💡 Load an example sentence",
            options=["— choose an example —"] + EXAMPLE_TEXTS,
            key="example_picker",
        )

        default_text = (
            "" if example_choice == "— choose an example —" else example_choice
        )
        user_text = st.text_area(
            "Enter text for sentiment analysis",
            value=default_text,
            height=110,
            placeholder="Type a sentence and click Run Inference…",
            label_visibility="collapsed",
        )

    with col_sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Ollama Models Dropdown
        st.markdown("**🤖 Ollama Models** (max 3)")
        ollama_available = check_ollama_availability()
        
        if ollama_available:
            ollama_models_list = get_ollama_models()
            model_options = [f"{m['name']} ({m['size']} GB)" for m in ollama_models_list]
            
            selected_ollama = st.multiselect(
                "Select Ollama models",
                options=model_options,
                max_selections=3,
                label_visibility="collapsed",
                help="Select up to 3 Ollama models to compare with traditional pipeline"
            )
            
            # Map back to model IDs
            selected_model_ids = []
            for selection in selected_ollama:
                for model in ollama_models_list:
                    if f"{model['name']} ({model['size']} GB)" == selection:
                        selected_model_ids.append(model['id'])
                        break
        else:
            st.warning("⚠️ Ollama not running")
            selected_model_ids = []
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("⚡ Run Inference", use_container_width=True)
        clear_log   = st.button("🗑 Clear Log",     use_container_width=True)
        if clear_log:
            st.session_state["inference_log"]   = []
            st.session_state["total_saved_kwh"] = 0.0
            st.session_state["last_result"]     = None
            st.rerun()

    # ── Run pipeline ──────────────────────────────────────────────────────────────
    if run_clicked:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            # Run traditional pipeline
            with st.spinner("Running traditional pipeline…"):
                result = run_pipeline(user_text.strip())
            
            result["text"] = user_text.strip()
            st.session_state["last_result"] = result
            st.session_state["total_saved_kwh"] += result["saved_kwh"]
            st.session_state["inference_log"].append(result)
            
            # Run Ollama models if selected
            ollama_results = []
            if selected_model_ids:
                with st.spinner(f"Running {len(selected_model_ids)} Ollama model(s)…"):
                    ollama_results = run_multiple_ollama_models(selected_model_ids, user_text.strip())
                
                st.session_state["last_ollama_results"] = ollama_results
            else:
                st.session_state["last_ollama_results"] = []

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PREDICTION RESULTS (Traditional + Ollama)
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">🎯 Prediction Results</p>', unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    ollama_results = st.session_state.get("last_ollama_results", [])

    if result is None:
        st.info("No inference run yet. Enter text above and click **Run Inference**.")
    else:
        # Display Traditional Pipeline Result
        st.markdown("### 🌿 Traditional Pipeline")
        stage = result["stage"]
        label = result["label"]
        conf  = result["confidence"]

        # Stage badge
        badge_class = {
            "Rule Engine": "badge-green",
            "RoBERTa":     "badge-blue",
            "BERT":        "badge-orange",
        }.get(stage, "badge-green")

        emoji_label = (
            "🟢 POSITIVE" if label == "POSITIVE" 
            else "🔴 NEGATIVE" if label == "NEGATIVE"
            else "🟡 NEUTRAL"
        )
        stage_icon  = {"Rule Engine": "🌿", "RoBERTa": "💙", "BERT": "🟠"}.get(stage, "")

        # Three result cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="card card-accent">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.5rem;">Prediction</div>
                    <div style="font-size:1.7rem;font-weight:700;
                                color:{'#4ade80' if label=='POSITIVE' else '#f87171' if label=='NEGATIVE' else '#eab308'};">
                        {emoji_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            bar_width = int(conf * 100)
            bar_col   = "#4ade80" if conf >= 0.8 else "#eab308"
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.6rem;">Confidence</div>
                    <div style="font-size:1.7rem;font-weight:700;
                                font-family:'JetBrains Mono',monospace;color:#f1f5f9;">
                        {conf:.1%}
                    </div>
                    <div style="background:#334155;border-radius:6px;height:6px;margin-top:.6rem;">
                        <div style="background:{bar_col};width:{bar_width}%;height:6px;
                                    border-radius:6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.6rem;">Inference Stage</div>
                    <div style="font-size:1.1rem;font-weight:600;margin-bottom:.5rem;">
                        {stage_icon} {stage}
                    </div>
                    <span class="badge {badge_class}">{stage.upper()}</span>
                    <div style="font-size:.78rem;color:#64748b;margin-top:.55rem;">
                        ⏱ {result.get('latency_ms', 0):.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rule engine keyword hint
        if stage == "Rule Engine" and "matched" in result:
            st.success(f"🔑 Matched keywords: `{', '.join(result['matched'])}`")
        
        # Display Ollama Results if available
        if ollama_results:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🤖 Ollama Models Results")
            
            for ollama_result in ollama_results:
                if ollama_result.get("error"):
                    st.error(f"❌ {ollama_result['model_name']}: {ollama_result['error']}")
                else:
                    o_label = ollama_result["label"]
                    o_conf = ollama_result["confidence"]
                    o_emoji = (
                        "🟢 POSITIVE" if o_label == "POSITIVE"
                        else "🔴 NEGATIVE" if o_label == "NEGATIVE"
                        else "🟡 NEUTRAL"
                    )
                    
                    o_col1, o_col2, o_col3 = st.columns(3)
                    with o_col1:
                        st.markdown(
                            f"""
                            <div class="card">
                                <div style="font-size:.75rem;color:#94a3b8;margin-bottom:.3rem;">
                                    {ollama_result['model_name']}
                                </div>
                                <div style="font-size:1.3rem;font-weight:700;
                                            color:{'#4ade80' if o_label=='POSITIVE' else '#f87171' if o_label=='NEGATIVE' else '#eab308'};">
                                    {o_emoji}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with o_col2:
                        st.markdown(
                            f"""
                            <div class="card">
                                <div style="font-size:.75rem;color:#94a3b8;margin-bottom:.3rem;">Confidence</div>
                                <div style="font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;">
                                    {o_conf:.1%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with o_col3:
                        st.markdown(
                            f"""
                            <div class="card">
                                <div style="font-size:.75rem;color:#94a3b8;margin-bottom:.3rem;">Latency</div>
                                <div style="font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;">
                                    {ollama_result['latency_ms']:.1f} ms
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        
        # Model Comparison Table
        if ollama_results:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Model Comparison")
            
            comparison_df = compare_models(result, ollama_results)
            
            # Add instruction
            st.markdown("**👇 Click on a row to view detailed metrics and visualizations below**")
            
            # Display the dataframe with selection
            event = st.dataframe(
                comparison_df, 
                use_container_width=True, 
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Handle row selection
            if event.selection and event.selection.rows:
                selected_row_idx = event.selection.rows[0]
                selected_model_name = comparison_df.iloc[selected_row_idx]["Model"]
                st.session_state["selected_model_for_viz"] = selected_model_name
            elif "selected_model_for_viz" not in st.session_state:
                # Default to Traditional if nothing selected
                st.session_state["selected_model_for_viz"] = comparison_df.iloc[0]["Model"]
            
            # Show which model is selected
            current_selection = st.session_state.get("selected_model_for_viz", comparison_df.iloc[0]["Model"])
            st.info(f"📊 Currently viewing metrics for: **{current_selection}**")
            
            # Show best model
            best_model = get_best_model(result, ollama_results)
            st.success(f"🏆 **Best Model**: {best_model['name']} (Score: {best_model['score']:.1f}/100) - {best_model['reason']}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ENERGY METRICS
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">⚡ Energy & Carbon Metrics</p>', unsafe_allow_html=True)

    if result:
        # Determine which model's data to display
        selected_viz_model = st.session_state.get("selected_model_for_viz", "Traditional")
        
        # Get the selected model's data
        # The model name format from comparison table is "Traditional (Stage)" or "Ollama (Model Name)"
        if selected_viz_model.startswith("Traditional"):
            viz_data = result
            model_display_name = selected_viz_model
        else:
            # Extract Ollama model name from format "Ollama (Model Name)"
            viz_data = None
            for ollama_result in ollama_results:
                ollama_display = f"Ollama ({ollama_result['model_name']})"
                if ollama_display == selected_viz_model:
                    viz_data = ollama_result
                    model_display_name = selected_viz_model
                    break
            
            # If not found or error, fall back to traditional
            if viz_data is None or viz_data.get("error"):
                viz_data = result
                model_display_name = f"Traditional ({result['stage']})"
                st.warning(f"⚠️ Selected model data unavailable. Showing Traditional model metrics.")
        
        # Display model name being visualized
        st.markdown(f"**Showing metrics for:** {model_display_name}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Energy Used",    f"{viz_data.get('energy_kwh', 0):.6f} kWh")
        m2.metric("🌍 CO₂ Emissions",  f"{viz_data.get('co2_kg', 0):.6f} kg")
        
        # Calculate saved percentage for Ollama models
        if not selected_viz_model.startswith("Traditional"):
            # Compare Ollama model energy to BERT baseline
            ollama_energy = viz_data.get('energy_kwh', 0)
            saved_pct = ((ENERGY_LARGE_MODEL - ollama_energy) / ENERGY_LARGE_MODEL) * 100
            viz_data['saved_pct'] = saved_pct
            viz_data['saved_kwh'] = ENERGY_LARGE_MODEL - ollama_energy
            
            # Calculate green score for Ollama
            viz_data['green_score'] = max(0, round(100 * (1 - ollama_energy / ENERGY_LARGE_MODEL), 1))
        
        m3.metric("💚 Energy Saved",   f"{viz_data.get('saved_pct', 0):.1f}%",
                  delta=f"vs always running BERT")
        m4.metric("🏭 Total Saved Today",
                  f"{st.session_state['total_saved_kwh'] * 1e6:.2f} µWh",
                  delta=f"{len(st.session_state['inference_log'])} inferences")

        st.markdown("<br>", unsafe_allow_html=True)

        # Green Score + Carbon Meter side by side
        gs_col, cm_col = st.columns([1, 1])

        with gs_col:
            st.plotly_chart(
                green_score_gauge(viz_data.get("green_score", 0)),
                use_container_width=True,
            )

        with cm_col:
            st.markdown(
                '<div class="card" style="margin-top:1rem;">'
                '<div style="font-size:.85rem;color:#94a3b8;margin-bottom:.8rem;">'
                '🌱 Carbon Efficiency Meter</div>',
                unsafe_allow_html=True,
            )
            efficiency = viz_data.get('saved_pct', 0) / 100
            colour = (
                "normal" if efficiency >= 0.6
                else "off"
            )
            st.progress(max(min(efficiency, 1.0), 0.0))
            saved_label = f"Saved {viz_data.get('saved_pct', 0):.1f}% compared to always running BERT"
            st.caption(saved_label)

            # Stage energy comparison table
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Add current model to comparison if it's Ollama
            if not selected_viz_model.startswith("Traditional"):
                stage_df = pd.DataFrame({
                    "Stage":      ["Rule Engine", "RoBERTa", "BERT", model_display_name],
                    "Energy (µWh)": [0.001, 0.120, 0.320, viz_data.get('energy_kwh', 0) * 1000],
                    "CO₂ (µg)":  [
                        round(0.000001 * 475_000, 4),
                        round(0.000120 * 475_000, 4),
                        round(0.000320 * 475_000, 4),
                        round(viz_data.get('co2_kg', 0) * 1_000_000, 4),
                    ],
                })
            else:
                stage_df = pd.DataFrame({
                    "Stage":      ["Rule Engine", "RoBERTa", "BERT"],
                    "Energy (µWh)": [0.001, 0.120, 0.320],
                    "CO₂ (µg)":  [
                        round(0.000001 * 475_000, 4),
                        round(0.000120 * 475_000, 4),
                        round(0.000320 * 475_000, 4),
                    ],
                })
            st.dataframe(stage_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Run an inference to see energy metrics.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — VISUALISATION DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📊 Visualisation Dashboard</p>', unsafe_allow_html=True)

    log        = st.session_state["inference_log"]
    log_df     = build_log_df(log)
    
    # Get selected model for visualization
    selected_viz_model = st.session_state.get("selected_model_for_viz", "Traditional")
    
    # Check if an Ollama model is selected
    if selected_viz_model.startswith("Ollama") and ollama_results:
        # Show Ollama model-specific visualizations
        st.info(f"📊 Showing visualizations for: **{selected_viz_model}**")
        
        # Find the selected Ollama model data
        selected_ollama_data = None
        for ollama_result in ollama_results:
            if f"Ollama ({ollama_result['model_name']})" == selected_viz_model:
                selected_ollama_data = ollama_result
                break
        
        if selected_ollama_data and not selected_ollama_data.get("error"):
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Model Performance Comparison Chart
                comparison_data = {
                    "Metric": ["Energy (kWh)", "CO₂ (kg)", "Latency (ms)", "Confidence"],
                    "Value": [
                        selected_ollama_data.get('energy_kwh', 0) * 1000,  # Convert to mWh for visibility
                        selected_ollama_data.get('co2_kg', 0) * 1000,  # Convert to g
                        selected_ollama_data.get('latency_ms', 0),
                        selected_ollama_data.get('confidence', 0) * 100,  # Convert to percentage
                    ],
                    "Unit": ["mWh", "g", "ms", "%"]
                }
                
                fig1 = go.Figure()
                colors = ['#22c55e', '#3b82f6', '#eab308', '#f97316']
                for i, (metric, value, unit, color) in enumerate(zip(
                    comparison_data["Metric"], 
                    comparison_data["Value"],
                    comparison_data["Unit"],
                    colors
                )):
                    fig1.add_trace(go.Bar(
                        name=metric,
                        x=[metric],
                        y=[value],
                        marker=dict(color=color),
                        text=[f"{value:.2f} {unit}"],
                        textposition="outside",
                        hovertemplate=f"<b>{metric}</b><br>Value: {value:.4f} {unit}<extra></extra>",
                    ))
                
                layout1 = _base_layout(f"📊 {selected_ollama_data['model_name']} Performance Metrics")
                layout1.update(
                    xaxis=dict(gridcolor=GRID_COLOUR, showgrid=False),
                    yaxis=dict(title="Value", gridcolor=GRID_COLOUR, showgrid=True),
                    showlegend=False,
                    height=300,
                )
                fig1.update_layout(**layout1)
                st.plotly_chart(fig1, use_container_width=True)
            
            with chart_col2:
                # Comparison with Traditional Pipeline
                trad_energy = result.get('energy_kwh', 0) * 1000
                ollama_energy = selected_ollama_data.get('energy_kwh', 0) * 1000
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    name="Traditional",
                    x=["Energy (mWh)"],
                    y=[trad_energy],
                    marker=dict(color='#3b82f6'),
                    text=[f"{trad_energy:.4f} mWh"],
                    textposition="outside",
                ))
                fig2.add_trace(go.Bar(
                    name=selected_ollama_data['model_name'],
                    x=["Energy (mWh)"],
                    y=[ollama_energy],
                    marker=dict(color='#22c55e'),
                    text=[f"{ollama_energy:.4f} mWh"],
                    textposition="outside",
                ))
                
                layout2 = _base_layout("⚡ Energy Comparison: Traditional vs Ollama")
                layout2.update(
                    xaxis=dict(gridcolor=GRID_COLOUR, showgrid=False),
                    yaxis=dict(title="Energy (mWh)", gridcolor=GRID_COLOUR, showgrid=True),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.15),
                    height=300,
                )
                fig2.update_layout(**layout2)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Model Details Table
            st.markdown("**Model Details:**")
            details_df = pd.DataFrame({
                "Metric": ["Label", "Confidence", "Latency", "Energy", "CO₂", "Model Size"],
                "Value": [
                    selected_ollama_data.get('label', 'N/A'),
                    f"{selected_ollama_data.get('confidence', 0):.1%}",
                    f"{selected_ollama_data.get('latency_ms', 0):.1f} ms",
                    f"{selected_ollama_data.get('energy_kwh', 0):.6f} kWh",
                    f"{selected_ollama_data.get('co2_kg', 0):.6f} kg",
                    f"{selected_ollama_data.get('model', 'N/A')}",
                ]
            })
            st.dataframe(details_df, hide_index=True, use_container_width=True)
        else:
            st.warning("⚠️ Selected Ollama model data unavailable. Showing traditional dashboard.")
            # Fall back to traditional charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(energy_bar_chart(result["stage"] if result else None), use_container_width=True)
            with chart_col2:
                st.plotly_chart(stage_distribution_pie(log_df), use_container_width=True)
            st.plotly_chart(carbon_timeline(log_df), use_container_width=True)
    else:
        # Show traditional pipeline visualizations
        highlight_stage = result["stage"] if result else None
        
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(
                energy_bar_chart(highlight_stage),
                use_container_width=True,
            )

        with chart_col2:
            st.plotly_chart(stage_distribution_pie(log_df), use_container_width=True)

        st.plotly_chart(carbon_timeline(log_df), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — INFERENCE LOG TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📋 Inference Log</p>', unsafe_allow_html=True)

    # Get selected model for visualization
    selected_viz_model = st.session_state.get("selected_model_for_viz", "Traditional")
    
    # Check if an Ollama model is selected
    if selected_viz_model.startswith("Ollama") and ollama_results:
        # Show Ollama model-specific log
        st.info(f"📋 Showing inference details for: **{selected_viz_model}**")
        
        # Find the selected Ollama model data
        selected_ollama_data = None
        for ollama_result in ollama_results:
            if f"Ollama ({ollama_result['model_name']})" == selected_viz_model:
                selected_ollama_data = ollama_result
                break
        
        if selected_ollama_data and not selected_ollama_data.get("error"):
            # Create a detailed log for the Ollama model
            ollama_log_data = {
                "Input": [result.get("text", "N/A")[:60] + ("…" if len(result.get("text", "")) > 60 else "")],
                "Prediction": [selected_ollama_data.get("label", "N/A")],
                "Model": [selected_ollama_data.get("model_name", "N/A")],
                "Confidence": [f"{selected_ollama_data.get('confidence', 0):.2%}"],
                "Latency (ms)": [f"{selected_ollama_data.get('latency_ms', 0):.1f}"],
                "Energy (kWh)": [f"{selected_ollama_data.get('energy_kwh', 0):.6f}"],
                "CO₂ (kg)": [f"{selected_ollama_data.get('co2_kg', 0):.6f}"],
            }
            ollama_log_df = pd.DataFrame(ollama_log_data)
            st.dataframe(ollama_log_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Selected Ollama model data unavailable.")
    else:
        # Show traditional pipeline log
        if log_df.empty:
            st.info("No inferences logged yet.")
        else:
            # Colour-code Stage column using pandas styler
            def colour_stage(val: str) -> str:
                colours = {
                    "Rule Engine": "background-color:#14532d;color:#4ade80",
                    "RoBERTa":     "background-color:#1e3a5f;color:#93c5fd",
                    "BERT":        "background-color:#431407;color:#fdba74",
                }
                return colours.get(val, "")

            styled = log_df.style.applymap(colour_stage, subset=["Stage"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;margin-top:3rem;padding:1rem;
                    color:#334155;font-size:.78rem;font-family:'JetBrains Mono',monospace;">
            Carbon-Aware AI Inference System · Green AI Demo ·
            Built with 🌿 Streamlit + HuggingFace Transformers + Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # ── OLLAMA COMPARE DASHBOARD ──────────────────────────────────────────────
    if app_mode == "Ollama Compare":
        render_dashboard()
    elif app_mode == "Prompt Router":
        # ── PROMPT ROUTER DASHBOARD ───────────────────────────────────────────
        render_prompt_router_tab()
        render_router_stats()
