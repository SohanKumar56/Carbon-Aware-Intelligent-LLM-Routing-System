"""
ollama_dashboard.py — UI Module for Multi-Model Comparison.
Renders the 3-panel dashboard and handles state for concurrent execution.
"""

import streamlit as st
import concurrent.futures
import time
from ollama_service import (
    is_ollama_running, 
    start_ollama, 
    get_installed_models, 
    run_ollama_inference
)
from ollama_integration import OLLAMA_MODELS, run_ollama_inference as ollama_run_inference

def generate_llm_explanation(prompt_text, model_id="tinyllama:latest"):
    """
    Generate explanation using LLM for more natural language.
    
    Parameters
    ----------
    prompt_text : str
        The prompt to send to the LLM
    model_id : str
        The model to use for explanation generation (default: tinyllama)
    
    Returns
    -------
    str
        The LLM-generated explanation
    """
    try:
        result = ollama_run_inference(model_id, prompt_text, timeout=60)
        if result.get("error"):
            # Fallback to simple explanation if LLM fails
            return None
        return result.get("raw_response", "").strip()
    except Exception as e:
        # Fallback if LLM call fails
        return None

def generate_efficiency_explanation(model_id, energy_kwh, latency_ms, response_length):
    """
    Generate an explanation of why this model performed efficiently or not using LLM.
    
    Parameters
    ----------
    model_id : str
        The model identifier
    energy_kwh : float
        Energy consumed in kWh
    latency_ms : float
        Inference latency in milliseconds
    response_length : int
        Length of the response in characters
    
    Returns
    -------
    str
        Efficiency explanation text
    """
    model_info = OLLAMA_MODELS.get(model_id, {})
    model_size = model_info.get('size', 0)
    model_name = model_info.get('name', model_id)
    
    # Calculate efficiency metrics
    energy_per_char = (energy_kwh * 1000) / response_length if response_length > 0 else 0
    energy_mwh = energy_kwh * 1000
    
    # Try LLM-generated explanation first
    llm_prompt = f"""
Generate a concise efficiency analysis for this model performance:

Model: {model_name} ({model_size:.1f}GB)
Energy consumed: {energy_mwh:.3f} mWh
Latency: {latency_ms:.0f}ms
Response length: {response_length} characters
Energy per character: {energy_per_char:.6f} mWh/char

Provide a 2-3 sentence explanation of why this model is efficient or not. Focus on:
1. Model size appropriateness
2. Energy efficiency 
3. Performance characteristics

Keep it brief and technical. Start with "Efficiency Analysis:" and then your explanation.
"""
    
    llm_explanation = generate_llm_explanation(llm_prompt)
    
    if llm_explanation:
        return f"**Efficiency Analysis for {model_name}**\n\n{llm_explanation}"
    
    # Fallback to hardcoded explanation if LLM fails
    explanation = f"**Efficiency Analysis for {model_name}**\n\n"
    
    # Size-based efficiency analysis
    if model_size < 1.0:
        explanation += f"🟢 **Compact Model Advantage**: {model_name} ({model_size:.1f}GB) is highly energy-efficient for this task.\n"
        explanation += f"- Energy per character: {energy_per_char:.6f} mWh/char\n"
        explanation += f"- Excellent for simple to moderate complexity prompts\n"
    elif model_size < 3.0:
        explanation += f"🟡 **Balanced Model**: {model_name} ({model_size:.1f}GB) offers good performance/efficiency balance.\n"
        explanation += f"- Energy per character: {energy_per_char:.6f} mWh/char\n"
        explanation += f"- Suitable for medium complexity tasks requiring some reasoning\n"
    else:
        explanation += f"🔴 **Large Model**: {model_name} ({model_size:.1f}GB) has higher energy requirements.\n"
        explanation += f"- Energy per character: {energy_per_char:.6f} mWh/char\n"
        explanation += f"- Appropriate for complex tasks requiring advanced reasoning\n"
    
    # Latency analysis
    if latency_ms < 500:
        explanation += f"⚡ **Fast Response**: {latency_ms:.0f}ms indicates efficient processing.\n"
    elif latency_ms < 2000:
        explanation += f"⏱️ **Moderate Response**: {latency_ms:.0f}ms is acceptable for this model size.\n"
    else:
        explanation += f"🐌 **Slower Response**: {latency_ms:.0f}ms suggests complex processing or resource constraints.\n"
    
    # Overall efficiency assessment
    if energy_per_char < 0.001:
        explanation += f"✅ **Highly Efficient**: Excellent energy-to-output ratio.\n"
    elif energy_per_char < 0.005:
        explanation += f"✅ **Efficient**: Good energy-to-output ratio.\n"
    else:
        explanation += f"⚠️ **Less Efficient**: Higher energy consumption per output character.\n"
    
    return explanation

def generate_comparison_explanation(current_model_id, compared_model_id, current_energy, compared_energy, current_latency, compared_latency):
    """
    Generate an explanation of why the compared model is less efficient than the current model using LLM.
    
    Parameters
    ----------
    current_model_id : str
        The current (better) model identifier
    compared_model_id : str
        The compared (less efficient) model identifier
    current_energy : float
        Current model energy in kWh
    compared_energy : float
        Compared model energy in kWh
    current_latency : float
        Current model latency in ms
    compared_latency : float
        Compared model latency in ms
    
    Returns
    -------
    str
        Comparison explanation text
    """
    current_info = OLLAMA_MODELS.get(current_model_id, {})
    compared_info = OLLAMA_MODELS.get(compared_model_id, {})
    
    current_name = current_info.get('name', current_model_id)
    compared_name = compared_info.get('name', compared_model_id)
    current_size = current_info.get('size', 0)
    compared_size = compared_info.get('size', 0)
    
    # Calculate differences
    energy_diff = compared_energy - current_energy
    energy_diff_pct = (energy_diff / current_energy * 100) if current_energy > 0 else 0
    latency_diff = compared_latency - current_latency
    latency_diff_pct = (latency_diff / current_latency * 100) if current_latency > 0 else 0
    
    # Try LLM-generated explanation first
    llm_prompt = f"""
Generate a concise comparison explanation for these two models:

More efficient model: {current_name} ({current_size:.1f}GB, {current_energy*1000:.3f} mWh, {current_latency:.0f}ms)
Less efficient model: {compared_name} ({compared_size:.1f}GB, {compared_energy*1000:.3f} mWh, {compared_latency:.0f}ms)

Energy difference: {energy_diff*1000:.3f} mWh ({energy_diff_pct:.1f}% increase)
Latency difference: {latency_diff:.0f}ms ({latency_diff_pct:.1f}% increase)

Explain in 2-3 sentences why {compared_name} is less efficient than {current_name}. Focus on:
1. Size differences and their impact
2. Energy consumption comparison
3. Performance trade-offs

Keep it brief and technical. Start with "Comparison:" and then your explanation.
"""
    
    llm_explanation = generate_llm_explanation(llm_prompt)
    
    if llm_explanation:
        return f"**Why {compared_name} is Less Efficient**\n\n{llm_explanation}"
    
    # Fallback to hardcoded explanation if LLM fails
    explanation = f"**Why {compared_name} is Less Efficient**\n\n"
    
    # Size comparison
    if compared_size > current_size:
        size_diff = compared_size - current_size
        explanation += f"📏 **Size Disadvantage**: {compared_name} is {size_diff:.1f}GB larger than {current_name}.\n"
        explanation += f"- Larger models consume more energy for the same task\n"
        explanation += f"- This size difference is not justified for this prompt's complexity\n"
    
    # Energy comparison
    if energy_diff > 0:
        explanation += f"⚡ **Energy Inefficiency**: {compared_name} consumed {energy_diff*1000:.3f} mWh more energy ({energy_diff_pct:.1f}% increase).\n"
        explanation += f"- Current model: {current_energy*1000:.3f} mWh\n"
        explanation += f"- Compared model: {compared_energy*1000:.3f} mWh\n"
    
    # Latency comparison
    if latency_diff > 0:
        explanation += f"⏱️ **Speed Disadvantage**: {compared_name} was {latency_diff:.0f}ms slower ({latency_diff_pct:.1f}% increase).\n"
        explanation += f"- Current model: {current_latency:.0f}ms\n"
        explanation += f"- Compared model: {compared_latency:.0f}ms\n"
    
    # Overall assessment
    explanation += f"\n**Conclusion**: {current_name} is more efficient because it provides similar or better results with significantly lower energy consumption and faster response times.\n"
    explanation += f"The prompt complexity doesn't justify the additional computational overhead of {compared_name}.\n"
    
    return explanation

def render_ollama_status():
    """Render the Ollama status badge and auto-start logic."""
    if "ollama_started" not in st.session_state:
        st.session_state.ollama_started = False

    is_running = is_ollama_running()
    
    if not is_running and not st.session_state.ollama_started:
        st.session_state.ollama_started = True # Mark as tried immediately
        with st.status("🚀 Ollama not running. Attempting auto-start...", expanded=False):
            success = start_ollama()
            if success:
                st.success("Ollama started successfully!")
            else:
                st.error("Failed to start Ollama automatically. Please ensure it is installed and running.")

    # Status indicator
    is_running = is_ollama_running() # Check again
    if is_running:
        st.markdown('<span class="badge badge-green">● OLLAMA RUNNING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">○ OLLAMA DISCONNECTED</span>', unsafe_allow_html=True)

def render_dashboard():
    """Main entry point for the Comparison Dashboard."""
    
    # Custom CSS for the Ollama panels (reusing existing theme patterns)
    st.markdown("""
        <style>
        .panel-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        .panel-header {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
        }
        .panel-output {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            flex-grow: 1;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.5;
            color: #f1f5f9;
            white-space: pre-wrap;
            border: 1px solid #1e293b;
        }
        .latency-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🤖 Multi-Model Compare</p>', unsafe_allow_html=True)
    
    # Init session state for panels
    if "ollama_responses" not in st.session_state:
        st.session_state.ollama_responses = [None, None, None]
    if "ollama_loading" not in st.session_state:
        st.session_state.ollama_loading = False

    # ── Top Section: Prompt ───────────────────────────────────────────────────
    col_prompt, col_status = st.columns([3, 1])
    
    with col_prompt:
        user_prompt = st.text_area(
            "Enter your prompt",
            placeholder="Ask something to compare models...",
            height=120,
            label_visibility="collapsed"
        )
    
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        render_ollama_status()
        
        models = get_installed_models()
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.rerun()
            
        run_btn = st.button("🚀 Run Models", use_container_width=True, type="primary", disabled=st.session_state.ollama_loading)
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.ollama_responses = [None, None, None]
            st.rerun()

    # ── Main Section: Panels ──────────────────────────────────────────────────
    cols = st.columns(3)
    
    # Model selection persist
    if "panel_models" not in st.session_state:
        st.session_state.panel_models = [
            models[0] if len(models) > 0 else "n/a",
            models[1] if len(models) > 1 else (models[0] if len(models) > 0 else "n/a"),
            models[2] if len(models) > 2 else (models[0] if len(models) > 0 else "n/a"),
        ]

    for i in range(3):
        with cols[i]:
            st.session_state.panel_models[i] = st.selectbox(
                f"Model {i+1}",
                options=models if models else ["No models found"],
                index=min(i, len(models)-1) if models else 0,
                key=f"model_select_{i}"
            )
            
            # Panel Container
            container = st.container()
            with container:
                resp = st.session_state.ollama_responses[i]
                
                # Header
                st.markdown(f"""
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; margin-bottom:5px;">
                        PANEL {i+1}
                    </div>
                """, unsafe_allow_html=True)
                
                # Output area
                if st.session_state.ollama_loading:
                    st.info("Thinking...")
                elif resp:
                    if resp["status"] == "success":
                        # Show the model response
                        st.markdown(f'<div class="panel-output">{resp["response"]}</div>', unsafe_allow_html=True)
                        
                        # Generate and show efficiency explanation using real energy data
                        model_id = resp["model"]
                        energy_kwh = resp.get("energy_kwh", 0.0)
                        latency_ms = resp["latency"] * 1000  # convert to ms
                        response_length = len(resp["response"])
                        
                        efficiency_explanation = generate_efficiency_explanation(
                            model_id, energy_kwh, latency_ms, response_length
                        )
                        
                        # Show energy metrics in the badge
                        energy_mwh = energy_kwh * 1000
                        co2_mg = resp.get("co2_kg", 0.0) * 1000
                        st.markdown(f'<div class="latency-badge">⏱ {resp["latency"]}s · {resp["model"]} · ⚡ {energy_mwh:.3f} mWh · 🌱 {co2_mg:.3f} mg CO₂</div>', unsafe_allow_html=True)
                        
                        # Show efficiency explanation in an expandable section
                        with st.expander("📊 Efficiency Analysis", expanded=False):
                            st.markdown(efficiency_explanation)
                        
                        if st.button(f"📋 Copy", key=f"copy_{i}"):
                            # Streamlit doesn't have a simple copy-to-clipboard, but we can show it
                            st.toast("Response copied to memory (conceptual)!")
                    else:
                        st.error(f"Error: {resp['message']}")
                else:
                    st.markdown('<div class="panel-output" style="color:#334155;">Idle... waiting for prompt.</div>', unsafe_allow_html=True)

    # ── Execution Logic ───────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        elif not models:
            st.error("❌ No models selected or installed.")
        else:
            st.session_state.ollama_loading = True
            st.rerun()

    # Triggered rerun handler
    if st.session_state.ollama_loading:
        selected_models = st.session_state.panel_models
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_panel = {
                executor.submit(run_ollama_inference, selected_models[i], user_prompt): i 
                for i in range(3)
            }
            
            for future in concurrent.futures.as_completed(future_to_panel):
                panel_idx = future_to_panel[future]
                try:
                    data = future.result()
                    st.session_state.ollama_responses[panel_idx] = data
                except Exception as exc:
                    st.session_state.ollama_responses[panel_idx] = {
                        "status": "error", 
                        "message": str(exc),
                        "model": selected_models[panel_idx]
                    }
        
        st.session_state.ollama_loading = False
        st.rerun()

    # ── Model Comparison Section ──────────────────────────────────────────────
    if all(st.session_state.ollama_responses):  # Only show if all panels have results
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Model Efficiency Comparison</p>', unsafe_allow_html=True)
        
        # Compare each model against the most efficient one
        responses = st.session_state.ollama_responses
        successful_responses = [resp for resp in responses if resp.get("status") == "success"]
        
        if len(successful_responses) >= 2:  # Need at least 2 successful responses for comparison
            # Find the most efficient model (lowest energy)
            efficient_model = min(successful_responses, key=lambda x: x.get("energy_kwh", float('inf')))
            
            # Compare other models against the efficient one
            for resp in successful_responses:
                if resp != efficient_model:
                    model_id = resp["model"]
                    efficient_id = efficient_model["model"]
                    
                    comparison = generate_comparison_explanation(
                        efficient_id,
                        model_id,
                        efficient_model.get("energy_kwh", 0),
                        resp.get("energy_kwh", 0),
                        efficient_model.get("latency", 0) * 1000,
                        resp.get("latency", 0) * 1000
                    )
                    
                    with st.expander(f"⚠️ Why {resp['model']} is less efficient than {efficient_model['model']}", expanded=False):
                        st.markdown(comparison)
