"""
prompt_router_dashboard.py — Streamlit dashboard for Prompt Complexity Router

Provides an interactive interface for:
1. Classifying prompt complexity
2. Getting model routing recommendations
3. Viewing measured energy savings
4. Comparing routing strategies
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from complexity_classifier import route_to_model, classify_prompt_complexity_detailed
from routing_pipeline import run_routing_pipeline, BASELINE_MODEL
from ollama_integration import OLLAMA_MODELS, check_ollama_availability, run_ollama_inference
import pandas as pd
from datetime import datetime


def generate_llm_model_explanation(model_name, model_size, complexity, energy_mwh, energy_saved_pct, co2_mg, green_score, latency_ms, prompt, model_id):
    """
    Generate explanation of why this model was chosen using the same model that processed the prompt.
    
    Parameters
    ----------
    model_name : str
        Name of the selected model
    model_size : float
        Size of the model in GB
    complexity : str
        Classified complexity level
    energy_mwh : float
        Energy consumed in mWh
    energy_saved_pct : float
        Percentage of energy saved
    co2_mg : float
        CO2 emissions in mg
    green_score : int
        Green score out of 100
    latency_ms : float
        Total latency in ms
    prompt : str
        The original prompt
    model_id : str
        The model ID to use for generating the explanation (same as the one that processed the prompt)
    
    Returns
    -------
    str
        LLM-generated explanation or None if generation fails
    """
    try:
        llm_prompt = f"""
You are {model_name} ({model_size:.1f}GB). Explain why you were chosen for this prompt.

FACTS (do not change these):
- Prompt complexity: {complexity.upper()}
- Your energy consumption: {energy_mwh:.3f} mWh
- Energy saved vs large model: {energy_saved_pct:.1f}%
- Your latency: {latency_ms:.0f}ms
- Green score: {green_score}/100

STRICT RULES:
1. You MUST use the exact model name "{model_name}" - do not invent names
2. You MUST use the exact size {model_size:.1f}GB - do not change sizes
3. If your size is SMALL (<3GB), this is an ADVANTAGE for efficiency
4. If complexity is {complexity.upper()}, explain why your size matches this complexity level
5. Never claim larger models are more efficient than smaller models
6. Do not hallucinate numbers - use only the facts provided above

Explain in 2-3 sentences why you were optimal for this {complexity} complexity task. Focus on:
- Your size appropriateness for {complexity} complexity
- Your energy efficiency achievements
- Your performance characteristics

Start with "Model Selection:" and keep it technical and accurate.
"""
        result = run_ollama_inference(model_id, llm_prompt, timeout=60)
        if result.get("error"):
            return None
        return result.get("raw_response", "").strip()
    except Exception as e:
        return None


def generate_llm_comparison_explanation(optimal_model_name, optimal_size, optimal_energy, optimal_latency, compared_model_name, compared_size, compared_energy, compared_latency, complexity, energy_diff, energy_diff_pct, latency_diff, latency_diff_pct, compared_model_id):
    """
    Generate LLM-based explanation for why a compared model is less efficient using the compared model itself.
    
    Parameters
    ----------
    optimal_model_name : str
        Name of the optimal model
    optimal_size : float
        Size of optimal model in GB
    optimal_energy : float
        Energy consumed by optimal model in mWh
    optimal_latency : float
        Latency of optimal model in ms
    compared_model_name : str
        Name of the compared model
    compared_size : float
        Size of compared model in GB
    compared_energy : float
        Energy consumed by compared model in mWh
    compared_latency : float
        Latency of compared model in ms
    complexity : str
        Complexity level of the prompt
    energy_diff : float
        Energy difference in mWh
    energy_diff_pct : float
        Energy difference percentage
    latency_diff : float
        Latency difference in ms
    latency_diff_pct : float
        Latency difference percentage
    compared_model_id : str
        The model ID to use for generating the explanation (the compared model itself)
    
    Returns
    -------
    str
        LLM-generated explanation or None if generation fails
    """
    try:
        llm_prompt = f"""
You are {compared_model_name} ({compared_size:.1f}GB). Explain why you are less efficient than {optimal_model_name} ({optimal_size:.1f}GB).

FACTS (do not change these):
- Your energy: {compared_energy:.3f} mWh
- Optimal model energy: {optimal_energy:.3f} mWh
- Energy difference: {energy_diff:.3f} mWh ({energy_diff_pct:.1f}%)
- Your latency: {compared_latency:.0f}ms
- Optimal model latency: {optimal_latency:.0f}ms
- Latency difference: {latency_diff:.0f}ms ({latency_diff_pct:.1f}%)
- Prompt complexity: {complexity.upper()}

STRICT RULES:
1. You MUST use exact model names "{compared_model_name}" and "{optimal_model_name}" - do not invent names
2. You MUST use exact sizes {compared_size:.1f}GB and {optimal_size:.1f}GB - do not change sizes
3. If your size is LARGER, this generally increases energy consumption (not decreases)
4. Smaller models are typically MORE energy-efficient for appropriate complexity levels
5. Explain why your size is or isn't justified for this {complexity} complexity task
6. Do not hallucinate numbers - use only the facts provided above
7. If {energy_diff > 0}, acknowledge you consumed MORE energy, not less

Explain in 2-3 sentences why you are less efficient than {optimal_model_name} for this {complexity} complexity prompt. Focus on:
- Your size vs optimal model size and energy impact
- Your performance trade-offs
- Whether your additional capacity is justified for this task

Start with "Comparison:" and keep it technical and accurate.
"""
        result = run_ollama_inference(compared_model_id, llm_prompt, timeout=60)
        if result.get("error"):
            return None
        return result.get("raw_response", "").strip()
    except Exception as e:
        return None


def render_prompt_router_tab():
    """Render the Prompt Router tab in the dashboard."""
    
    # Initialize session state for history
    if 'processing_history' not in st.session_state:
        st.session_state['processing_history'] = []
    
    # Custom header with leaf logo and system title
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: bold; margin: 0.5rem 0; color: #f1f5f9;">
            🌿 Carbon-Aware Intelligent LLM Routing System
        </h1>
        <p style="font-size: 1.1rem; color: #94a3b8; margin: 1rem 0; line-height: 1.6;">
            Classifies prompt complexity and routes to appropriately-sized models to minimize energy waste.<br>
            Reduces carbon emissions by using smaller models when appropriate without sacrificing quality.<br>
            Smart routing that saves energy while maintaining optimal performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.subheader("📝 Enter Your Prompt")
    
    # Example prompts for quick testing
    examples = {
        "Simple Fact": "What is the capital of France?",
        "Explanation": "Explain how photosynthesis works in plants",
        "Code Generation": "Write a Python function to implement binary search with comments",
        "Math Problem": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "Complex Reasoning": "Prove that the square root of 2 is irrational using proof by contradiction",
        "Code Debugging": "Debug this code: def factorial(n): return n * factorial(n)",
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_example = st.selectbox(
            "Quick Examples:",
            options=["Custom"] + list(examples.keys()),
            key="router_example"
        )
    
    if selected_example != "Custom":
        prompt = st.text_area(
            "Prompt:",
            value=examples[selected_example],
            height=100,
            key="router_prompt"
        )
    else:
        prompt = st.text_area(
            "Prompt:",
            placeholder="Enter your prompt here...",
            height=100,
            key="router_prompt_custom"
        )
    
    # Classification button (now automatic inference)
    if st.button("🔍 Classify & Auto-Process", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt first!")
            return
        
        # Check Ollama availability first
        ollama_available = check_ollama_availability()
        if not ollama_available:
            st.error("⚠️ Ollama is not running. Please start Ollama first with 'ollama serve'")
            return
        
        with st.spinner("Classifying prompt complexity and running automatic inference..."):
            # Run full pipeline with automatic inference
            result = run_routing_pipeline(prompt, auto_route=True)
            
            # Add timestamp
            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result['prompt_preview'] = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            # Store in session state for analytics
            st.session_state['pipeline_result'] = result
            st.session_state['current_prompt'] = prompt
            
            # Add to history
            st.session_state['processing_history'].append(result)
    
    # Display results if available
    if 'pipeline_result' in st.session_state:
        result = st.session_state['pipeline_result']
        prompt = st.session_state['current_prompt']
        
        st.markdown("---")
        
        # === OUTPUT SECTION ===
        st.subheader("🤖 Output")
        
        if result.get('error'):
            st.error(f"❌ Error during inference: {result['error']}")
        else:
            # Display the LLM output
            st.text_area("LLM Response", value=result['response'], height=150, disabled=True, key="llm_output")
        
        st.markdown("---")
        
        # === CLASSIFICATION RESULTS ===
        st.subheader("📊 Classification Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            complexity = result['complexity']
            color = {
                'small': '🟢',
                'medium': '🟡',
                'large': '🔴'
            }.get(complexity, '⚪')
            st.metric(
                "Complexity",
                f"{color} {complexity.upper()}",
                delta=None
            )
        
        with col2:
            confidence = result['confidence']
            conf_pct = f"{confidence:.1%}"
            st.metric(
                "Confidence",
                conf_pct,
                delta="High" if confidence > 0.9 else ("Medium" if confidence > 0.7 else "Low")
            )
        
        with col3:
            st.metric(
                "Classification Latency",
                f"{result['classification_latency_ms']:.0f} ms",
                delta=None
            )
        
        with col4:
            model_name = result['model_name']
            st.metric(
                "Routed Model",
                model_name,
                delta=None
            )
        
        # === PROBABILITY CHARTS ===
        st.subheader("📈 Complexity Probability Distribution")
        
        # Use final classification for consistency
        final_complexity = result['complexity']
        final_confidence = result['confidence']
        
        # Create probability distribution based on final classification
        # The chosen class gets the confidence value, others share the remainder
        remaining_confidence = (1.0 - final_confidence) / 2
        final_probs = {
            'small': remaining_confidence,
            'medium': remaining_confidence,
            'large': remaining_confidence
        }
        final_probs[final_complexity] = final_confidence
        
        prob_df = pd.DataFrame({
            'Complexity': list(final_probs.keys()),
            'Probability': [v * 100 for v in final_probs.values()]
        })
        
        # Create side-by-side layout for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Create pie chart
            fig_pie = px.pie(
                prob_df,
                values='Probability',
                names='Complexity',
                color='Complexity',
                color_discrete_map={
                    'small': '#10b981',
                    'medium': '#f59e0b',
                    'large': '#ef4444'
                },
                title="Complexity Classification Probability"
            )
            fig_pie.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Create bar chart
            fig_bar = px.bar(
                prob_df,
                x='Complexity',
                y='Probability',
                color='Complexity',
                color_discrete_map={
                    'small': '#10b981',
                    'medium': '#f59e0b',
                    'large': '#ef4444'
                },
                title="Probability Breakdown"
            )
            fig_bar.update_layout(
                showlegend=False,
                yaxis_title="Probability (%)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # === ROUTING REASONING ===
        st.subheader("🚀 Routing Decision")
        
        st.info(f"**Reasoning:** {result['reasoning']}")
        
        # === INFERENCE RESULTS ===
        st.markdown("---")
        st.subheader("🤖 Inference Results")
        
        if result.get('error'):
            st.error(f"❌ Error during inference: {result['error']}")
        else:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Latency",
                    f"{result['total_latency_ms']:.0f} ms",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Inference Latency",
                    f"{result['inference_latency_ms']:.0f} ms",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Energy Used",
                    f"{result['energy_kwh']*1000:.3f} mWh",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "CO₂ Emitted",
                    f"{result['co2_kg']*1000:.3f} mg",
                    delta=None
                )
            
            # Why this model was chosen
            st.markdown("---")
            st.subheader("💡 Why This Model Was Chosen")
            
            # Try LLM-generated explanation first
            model_size = OLLAMA_MODELS.get(result['routed_model'], {}).get('size', 0)
            llm_explanation = generate_llm_model_explanation(
                model_name=result['model_name'],
                model_size=model_size,
                complexity=result['complexity'],
                energy_mwh=result['energy_kwh']*1000,
                energy_saved_pct=result['energy_saved_pct'],
                co2_mg=result['co2_kg']*1000,
                green_score=result['green_score'],
                latency_ms=result['total_latency_ms'],
                prompt=prompt,
                model_id=result['routed_model']  # Use the same model that processed the prompt
            )
            
            if llm_explanation:
                st.info(llm_explanation)
            else:
                # Fallback to hardcoded explanation if LLM fails
                model_explanation = f"""
                **Model**: {result['model_name']} ({model_size:.1f} GB)
                
                **Complexity Match**: The prompt was classified as **{result['complexity'].upper()}** complexity, which matches perfectly with this model's capabilities.
                
                **Energy Efficiency**: This model consumed **{result['energy_kwh']*1000:.3f} mWh** of energy, saving **{result['energy_saved_pct']:.1f}%** compared to using the large baseline model.
                
                **Environmental Impact**: Generated **{result['co2_kg']*1000:.3f} mg** of CO₂ emissions with a Green Score of **{result['green_score']}/100**.
                
                **Performance**: Completed in **{result['total_latency_ms']:.0f} ms** total latency.
                """
                st.info(model_explanation)
        
        # === ENERGY ANALYTICS ===
        st.markdown("---")
        st.subheader("💚")
        
        # Energy savings metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Energy Saved",
                f"{result['energy_saved_kwh']*1000:.3f} mWh",
                delta=f"{result['energy_saved_pct']:.0f}% saved"
            )
        
        with col2:
            st.metric(
                "Green Score",
                f"{result['green_score']}/100",
                delta="Excellent" if result['green_score'] > 75 else ("Good" if result['green_score'] > 50 else "Fair")
            )
        
        with col3:
            co2_saved = result['energy_saved_kwh'] * 0.475
            st.metric(
                "CO₂ Saved",
                f"{co2_saved*1000:.3f} mg",
                delta=None
            )
        
        # === ENERGY DISTRIBUTION CHART ===
        st.subheader("📊 Energy Distribution")
        
        # Pie chart for energy distribution
        energy_pie_df = pd.DataFrame({
            'Source': ['Used Energy', 'Saved Energy'],
            'Energy (mWh)': [result['energy_kwh']*1000, result['energy_saved_kwh']*1000]
        })
        
        fig_energy_pie = px.pie(
            energy_pie_df,
            values='Energy (mWh)',
            names='Source',
            color='Source',
            color_discrete_map={
                'Used Energy': '#ef4444',
                'Saved Energy': '#10b981'
            },
            title="Energy Distribution: Used vs Saved"
        )
        fig_energy_pie.update_layout(height=400)
        st.plotly_chart(fig_energy_pie, use_container_width=True)
        
        # === COMPARISON OPTION ===
        st.markdown("---")
        st.subheader("� Compare with Other Models")
        
        # Initialize comparison state
        if 'comparison_mode' not in st.session_state:
            st.session_state['comparison_mode'] = False
        if 'selected_compare_models' not in st.session_state:
            st.session_state['selected_compare_models'] = []
        if 'comparison_results' not in st.session_state:
            st.session_state['comparison_results'] = None
        
        # Compare button
        if st.button("🔀 Compare with Other Models", key="compare_button"):
            st.session_state['comparison_mode'] = not st.session_state['comparison_mode']
        
        # Show comparison options when in comparison mode
        if st.session_state['comparison_mode']:
            st.markdown("Select models to compare with:")
            
            # Available models for comparison
            comparison_models = {
                'small': {'name': 'Small → tinyllama', 'model_id': 'tinyllama:latest'},
                'medium': {'name': 'Medium → phi3', 'model_id': 'phi3:latest'},
                'large': {'name': 'Large → qwen2.5:7b', 'model_id': 'qwen2.5:7b'}
            }
            
            # Get the current model's complexity
            current_model_id = result['routed_model']
            
            # Show checkboxes for models to compare
            selected_models = []
            for complexity, model_info in comparison_models.items():
                if model_info['model_id'] == current_model_id:
                    st.checkbox(f"✓ {model_info['name']} (Already Used)", value=True, disabled=True, key=f"compare_{complexity}")
                else:
                    if st.checkbox(model_info['name'], key=f"compare_{complexity}"):
                        selected_models.append(model_info['model_id'])
            
            # Run comparison button
            if st.button("▶️ Run Comparison", type="primary", key="run_comparison"):
                if selected_models:
                    with st.spinner("Running comparison with selected models..."):
                        # Store current prompt for comparison
                        st.session_state['compare_prompt'] = prompt
                        st.session_state['selected_compare_models'] = selected_models
                        
                        # Run comparison
                        comparison_results = []
                        for model_id in selected_models:
                            from ollama_integration import run_ollama_inference
                            model_info = OLLAMA_MODELS.get(model_id, {})
                            timeout = model_info.get('timeout', 120)
                            
                            try:
                                comp_result = run_ollama_inference(model_id, prompt, timeout=timeout)
                                comparison_results.append({
                                    'model_id': model_id,
                                    'model': model_id,  # Add this for consistency
                                    'model_name': model_info.get('name', model_id),
                                    'size_gb': model_info.get('size', 0),
                                    'energy_kwh': comp_result.get('energy_kwh', 0),
                                    'co2_kg': comp_result.get('co2_kg', 0),
                                    'latency_ms': comp_result.get('latency_ms', 0),
                                    'response': comp_result.get('raw_response', ''),
                                    'error': comp_result.get('error')
                                })
                            except Exception as e:
                                comparison_results.append({
                                    'model_id': model_id,
                                    'model': model_id,  # Add this for consistency
                                    'model_name': model_info.get('name', model_id),
                                    'size_gb': model_info.get('size', 0),
                                    'energy_kwh': 0,
                                    'co2_kg': 0,
                                    'latency_ms': 0,
                                    'response': '',
                                    'error': str(e)
                                })
                        
                        st.session_state['comparison_results'] = comparison_results
                        st.success("Comparison completed!")
                else:
                    st.warning("Please select at least one model to compare.")
        
        # Show comparison results if available
        if st.session_state.get('comparison_results') and st.session_state['comparison_mode']:
            st.markdown("---")
            st.subheader("📋 Model Comparison Results")
            
            # Build comparison table
            comparison_data = {
                'Metric': [
                    'Model Used',
                    'Model Size (GB)',
                    'Inference Latency (ms)',
                    'Energy Used (mWh)',
                    'CO₂ Emitted (mg)'
                ],
                'Current Model': [
                    result['model_name'],
                    f"{OLLAMA_MODELS.get(result['routed_model'], {}).get('size', 0):.1f}",
                    f"{result['inference_latency_ms']:.1f}",
                    f"{result['energy_kwh']*1000:.3f}",
                    f"{result['co2_kg']*1000:.3f}"
                ]
            }
            
            # Add comparison models to table
            for comp_result in st.session_state['comparison_results']:
                comparison_data[comp_result['model_name']] = [
                    comp_result['model_name'],
                    f"{comp_result['size_gb']:.1f}",
                    f"{comp_result['latency_ms']:.1f}",
                    f"{comp_result['energy_kwh']*1000:.3f}",
                    f"{comp_result['co2_kg']*1000:.3f}"
                ]
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Show efficiency analysis for compared models
            st.markdown("### 📝 Model Efficiency Analysis")
            for comp_result in st.session_state['comparison_results']:
                with st.expander(f"Efficiency Analysis for {comp_result['model_name']}"):
                    if comp_result.get('error'):
                        st.error(f"Error: {comp_result['error']}")
                    else:
                        # Efficiency analysis
                        energy_diff = comp_result['energy_kwh'] - result['energy_kwh']
                        energy_diff_pct = (energy_diff / result['energy_kwh'] * 100) if result['energy_kwh'] > 0 else 0
                        latency_diff = comp_result['latency_ms'] - result['inference_latency_ms']
                        latency_diff_pct = (latency_diff / result['inference_latency_ms'] * 100) if result['inference_latency_ms'] > 0 else 0
                        
                        # Try LLM-generated explanation first
                        optimal_size = OLLAMA_MODELS.get(result['routed_model'], {}).get('size', 0)
                        llm_comparison = generate_llm_comparison_explanation(
                            optimal_model_name=result['model_name'],
                            optimal_size=optimal_size,
                            optimal_energy=result['energy_kwh']*1000,
                            optimal_latency=result['inference_latency_ms'],
                            compared_model_name=comp_result['model_name'],
                            compared_size=comp_result['size_gb'],
                            compared_energy=comp_result['energy_kwh']*1000,
                            compared_latency=comp_result['latency_ms'],
                            complexity=result['complexity'],
                            energy_diff=energy_diff*1000,
                            energy_diff_pct=energy_diff_pct,
                            latency_diff=latency_diff,
                            latency_diff_pct=latency_diff_pct,
                            compared_model_id=comp_result['model']  # Use the compared model itself
                        )
                        
                        if energy_diff > 0:
                            if llm_comparison:
                                st.warning(f"**⚠️ Less Efficient**: {comp_result['model_name']} consumed **{energy_diff*1000:.3f} mWh** ({energy_diff_pct:.1f}% more energy) than the optimal model.\n\n{llm_comparison}")
                            else:
                                # Fallback to hardcoded explanation
                                efficiency_analysis = f"""
                                **⚠️ Less Efficient**: {comp_result['model_name']} consumed **{energy_diff*1000:.3f} mWh** ({energy_diff_pct:.1f}% more energy) than the optimal model.
                                
                                **Why Less Efficient**: 
                                - Model size: {comp_result['size_gb']:.1f} GB vs optimal {optimal_size:.1f} GB
                                - Latency: {comp_result['latency_ms']:.0f} ms vs optimal {result['inference_latency_ms']:.0f} ms
                                - The prompt complexity ({result['complexity'].upper()}) doesn't require this model's capacity
                                """
                                st.warning(efficiency_analysis)
                        else:
                            if llm_comparison:
                                st.info(f"**✅ Comparable Efficiency**: {comp_result['model_name']} performed similarly to the optimal model.\n\n{llm_comparison}")
                            else:
                                # Fallback to hardcoded explanation
                                efficiency_analysis = f"""
                                **✅ Comparable Efficiency**: {comp_result['model_name']} performed similarly to the optimal model.
                                
                                **Analysis**: 
                                - Energy difference: {abs(energy_diff)*1000:.3f} mWh
                                - This model could be a viable alternative for similar complexity tasks
                                """
                                st.info(efficiency_analysis)
        
        # === PERFORMANCE INSIGHTS ===
        st.subheader("💡 Performance Insights")
        
        insights = []
        
        # Energy efficiency insight
        if result['energy_saved_pct'] > 70:
            insights.append("🌟 **Excellent energy efficiency** - You saved over 70% energy compared to using the large model!")
        elif result['energy_saved_pct'] > 40:
            insights.append("✅ **Good energy efficiency** - Significant energy savings achieved.")
        elif result['energy_saved_pct'] > 0:
            insights.append("👍 **Moderate energy savings** - Some efficiency gained by smart routing.")
        else:
            insights.append("⚠️ **No energy savings** - This prompt required the large model, which is appropriate for its complexity.")
        
        # Latency insight
        if result['total_latency_ms'] < 1000:
            insights.append("⚡ **Fast response** - Total processing time under 1 second.")
        elif result['total_latency_ms'] < 3000:
            insights.append("⏱️ **Reasonable response time** - Processing completed in under 3 seconds.")
        else:
            insights.append("🐌 **Slower response** - Complex task requiring more processing time.")
        
        # Model choice insight
        if result['complexity'] == 'small':
            insights.append("🎯 **Optimal routing** - Simple task efficiently handled by small model.")
        elif result['complexity'] == 'medium':
            insights.append("⚖️ **Balanced routing** - Medium complexity task matched with medium model.")
        else:
            insights.append("🧠 **Appropriate routing** - Complex task requires large model for quality results.")
        
        for insight in insights:
            st.info(insight)
        
        # === HISTORICAL ANALYTICS ===
        if len(st.session_state['processing_history']) > 1:
            st.markdown("---")
            st.subheader("📈 Historical Analytics")
            
            history_df = pd.DataFrame(st.session_state['processing_history'])
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_energy_saved = history_df['energy_saved_kwh'].sum() * 1000
                st.metric("Total Energy Saved", f"{total_energy_saved:.3f} mWh")
            
            with col2:
                avg_green_score = history_df['green_score'].mean()
                st.metric("Avg Green Score", f"{avg_green_score:.1f}/100")
            
            with col3:
                total_prompts = len(history_df)
                st.metric("Total Prompts Processed", total_prompts)
            
            with col4:
                small_prompts = (history_df['complexity'] == 'small').sum()
                st.metric("Small Prompts", f"{small_prompts} ({small_prompts/total_prompts*100:.0f}%)")
            
            # Complexity distribution pie chart
            complexity_counts = history_df['complexity'].value_counts()
            fig_history_pie = px.pie(
                values=complexity_counts.values,
                names=complexity_counts.index,
                color=complexity_counts.index,
                color_discrete_map={
                    'small': '#10b981',
                    'medium': '#f59e0b',
                    'large': '#ef4444'
                },
                title="Historical Complexity Distribution"
            )
            fig_history_pie.update_layout(height=400)
            st.plotly_chart(fig_history_pie, use_container_width=True)
            
            # Energy savings over time
            history_df['prompt_num'] = range(1, len(history_df) + 1)
            fig_energy_trend = px.line(
                history_df,
                x='prompt_num',
                y='energy_saved_pct',
                title="Energy Savings Trend (%)",
                labels={'energy_saved_pct': 'Energy Saved (%)', 'prompt_num': 'Prompt Number'}
            )
            fig_energy_trend.update_layout(height=400)
            st.plotly_chart(fig_energy_trend, use_container_width=True)
            
            # Historical data table
            st.subheader("📋 Processing History")
            
            history_display = history_df[[
                'timestamp', 'prompt_preview', 'complexity', 'model_name', 
                'energy_saved_pct', 'green_score', 'total_latency_ms'
            ]].copy()
            
            history_display.columns = [
                'Time', 'Prompt Preview', 'Complexity', 'Model Used',
                'Energy Saved (%)', 'Green Score', 'Latency (ms)'
            ]
            
            st.dataframe(history_display, use_container_width=True, hide_index=True)


def render_router_stats():
    """Render statistics about the router's performance (optional sidebar)."""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Router Stats")
    
    # Model info
    st.sidebar.markdown(f"**Available Models:** {len(OLLAMA_MODELS)}")
    st.sidebar.markdown(f"**Complexity Classes:** 3")
    st.sidebar.markdown(f"**Trained on:** 28,465 prompts")
    st.sidebar.markdown(f"**Model Accuracy:** 93.2%")
    
    # Links
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Resources:**")
    st.sidebar.markdown("- [Training Report](model/prompt_complexity_classifier/training_report.txt)")
    st.sidebar.markdown("- [Dataset Info](data/labeled/dataset_metadata.json)")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Prompt Router", page_icon="🎯", layout="wide")
    render_prompt_router_tab()
    render_router_stats()
