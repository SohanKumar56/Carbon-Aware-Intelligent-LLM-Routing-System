"""
prompt_router_dashboard.py — Streamlit dashboard for Prompt Complexity Router

Provides an interactive interface for:
1. Classifying prompt complexity
2. Getting model routing recommendations
3. Viewing estimated energy savings
4. Comparing routing strategies
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from complexity_classifier import route_to_model, classify_prompt_complexity_detailed
from routing_pipeline import run_routing_pipeline, BASELINE_MODEL, BASELINE_ENERGY
from ollama_integration import OLLAMA_MODELS, check_ollama_availability
import pandas as pd


def render_prompt_router_tab():
    """Render the Prompt Router tab in the dashboard."""
    
    st.header("🎯 Prompt Complexity Router")
    st.markdown("""
    This router classifies prompt complexity **before** running inference,
    routing to appropriately-sized models to minimize energy waste.
    
    **Philosophy**: Don't waste energy routing every prompt to the biggest model
    when a smaller one would do the job.
    """)
    
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
    
    # Classification button
    if st.button("🔍 Classify & Route", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt first!")
            return
        
        with st.spinner("Classifying prompt complexity..."):
            # Get routing recommendation
            routing = route_to_model(prompt)
            
            # Store in session state for later use
            st.session_state['routing_result'] = routing
            st.session_state['current_prompt'] = prompt
    
    # Display results if available
    if 'routing_result' in st.session_state:
        routing = st.session_state['routing_result']
        prompt = st.session_state['current_prompt']
        
        st.markdown("---")
        
        # Classification results
        st.subheader("📊 Classification Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            complexity = routing['complexity']
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
            confidence = routing['confidence']
            conf_pct = f"{confidence:.1%}"
            st.metric(
                "Confidence",
                conf_pct,
                delta="High" if confidence > 0.9 else ("Medium" if confidence > 0.7 else "Low")
            )
        
        with col3:
            st.metric(
                "Latency",
                f"{routing['latency_ms']:.0f} ms",
                delta=None
            )
        
        with col4:
            recommended_model = routing['recommended'][0]
            model_info = OLLAMA_MODELS.get(recommended_model, {})
            model_size = model_info.get('size', 0)
            st.metric(
                "Recommended",
                f"{model_size:.1f}GB",
                delta=None
            )
        
        # Probability breakdown
        st.subheader("📈 Complexity Probability Distribution")
        
        probs = routing['all_probs']
        prob_df = pd.DataFrame({
            'Complexity': list(probs.keys()),
            'Probability': [v * 100 for v in probs.values()]
        })
        
        fig = px.bar(
            prob_df,
            x='Complexity',
            y='Probability',
            color='Complexity',
            color_discrete_map={
                'small': '#10b981',
                'medium': '#f59e0b',
                'large': '#ef4444'
            },
            title="Complexity Classification Confidence"
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Probability (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Routing recommendation
        st.subheader("🚀 Routing Recommendation")
        
        st.info(f"**Reasoning:** {routing['reasoning']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Primary Models:**")
            for model in routing['recommended'][:3]:
                model_info = OLLAMA_MODELS.get(model, {})
                st.markdown(f"- `{model_info.get('name', model)}` ({model_info.get('size', 0):.1f}GB)")
        
        with col2:
            st.markdown("**Fallback Models:**")
            for model in routing['fallback'][:3]:
                model_info = OLLAMA_MODELS.get(model, {})
                st.markdown(f"- `{model_info.get('name', model)}` ({model_info.get('size', 0):.1f}GB)")
        
        # Energy estimation
        st.subheader("💚 Estimated Energy Savings")
        
        recommended_model = routing['recommended'][0]
        model_info = OLLAMA_MODELS.get(recommended_model, {})
        routed_energy = model_info.get('size', 0) * 0.00005
        
        baseline_model_info = OLLAMA_MODELS.get(BASELINE_MODEL, {})
        baseline_energy = BASELINE_ENERGY
        
        energy_saved = baseline_energy - routed_energy
        energy_saved_pct = (energy_saved / baseline_energy * 100) if baseline_energy > 0 else 0
        green_score = max(0, min(100, round(100 * (1 - routed_energy / baseline_energy))))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Energy Saved",
                f"{energy_saved*1000:.3f} mWh",
                delta=f"{energy_saved_pct:.0f}% saved"
            )
        
        with col2:
            st.metric(
                "Green Score",
                f"{green_score}/100",
                delta="Excellent" if green_score > 75 else ("Good" if green_score > 50 else "Fair")
            )
        
        with col3:
            co2_saved = energy_saved * 0.475
            st.metric(
                "CO₂ Saved",
                f"{co2_saved*1000:.3f} mg",
                delta=None
            )
        
        # Comparison chart
        st.subheader("📊 Energy Comparison")
        
        comparison_df = pd.DataFrame({
            'Strategy': ['Smart Routing\n(Recommended)', 'Always Large\n(Baseline)'],
            'Energy (mWh)': [routed_energy * 1000, baseline_energy * 1000],
            'Model': [model_info.get('name', recommended_model), baseline_model_info.get('name', BASELINE_MODEL)]
        })
        
        fig = px.bar(
            comparison_df,
            x='Strategy',
            y='Energy (mWh)',
            color='Strategy',
            text='Model',
            color_discrete_map={
                'Smart Routing\n(Recommended)': '#10b981',
                'Always Large\n(Baseline)': '#ef4444'
            },
            title="Energy Usage: Smart Routing vs Always-Large Baseline"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Run inference section (optional)
        st.markdown("---")
        st.subheader("🤖 Run Live Inference (Optional)")
        
        ollama_available = check_ollama_availability()
        
        if not ollama_available:
            st.warning("⚠️ Ollama is not running. Start Ollama to test live inference.")
        else:
            st.info("✅ Ollama is running! You can test live inference with the recommended model.")
            
            if st.button("▶️ Run Inference on Recommended Model", use_container_width=True):
                with st.spinner(f"Running inference on {model_info.get('name', recommended_model)}..."):
                    result = run_routing_pipeline(prompt, auto_route=True)
                    
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✅ Inference complete!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Latency", f"{result['total_latency_ms']:.0f} ms")
                        with col2:
                            st.metric("Energy Used", f"{result['energy_kwh']*1000:.3f} mWh")
                        with col3:
                            st.metric("Energy Saved", f"{result['energy_saved_pct']:.0f}%")
                        with col4:
                            st.metric("Green Score", f"{result['green_score']}/100")
                        
                        # Response
                        st.markdown("**Model Response:**")
                        st.text_area("", value=result['response'], height=200, disabled=True)


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
