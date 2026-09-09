"""
routing_pipeline.py — Prompt-based routing pipeline with energy tracking

Classifies prompt complexity upfront and routes to an appropriately-sized
local Ollama LLM to minimise energy waste.

Green AI principle: Don't waste energy routing every prompt to the biggest
model when a smaller one would do the job.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Optional

from complexity_classifier import classify_prompt_complexity_detailed, get_recommended_models
from ollama_integration import run_ollama_inference, OLLAMA_MODELS
from config import CO2_INTENSITY

logger = logging.getLogger(__name__)

# Baseline for comparison (always using largest model)
BASELINE_MODEL = "qwen2.5:7b"
BASELINE_ENERGY = OLLAMA_MODELS.get(BASELINE_MODEL, {}).get("size", 4.7) * 0.00005


def run_routing_pipeline(
    prompt: str,
    selected_models: Optional[Dict[str, str]] = None,
    auto_route: bool = True
) -> Dict:
    """
    Execute the routing pipeline on a prompt.
    
    Workflow:
    1. Classify prompt complexity
    2. Route to appropriate model size
    3. Run inference
    4. Track energy vs baseline (always using large model)
    
    Parameters
    ----------
    prompt : str
        User input prompt
    selected_models : dict, optional
        Manual model selection per complexity level:
        {'small': 'tinyllama:latest', 'medium': 'qwen2.5:3b', 'large': 'qwen2.5:7b'}
        If None, uses recommended defaults
    auto_route : bool
        If True, automatically routes based on classification.
        If False, returns routing recommendation without running inference.
    
    Returns
    -------
    dict
        complexity       – classified complexity level
        confidence       – classification confidence
        routed_model     – model ID that was selected
        response         – model response (if auto_route=True)
        latency_ms       – total latency including classification
        energy_kwh       – energy consumed
        co2_kg           – CO2 emissions
        energy_saved     – energy saved vs baseline
        energy_saved_pct – percentage saved
        green_score      – 0-100 environmental efficiency
        reasoning        – routing decision reasoning
    """
    t0 = time.perf_counter()
    
    # Step 1: Classify prompt complexity
    logger.info(f"Classifying prompt: {prompt[:100]}...")
    classification = classify_prompt_complexity_detailed(prompt)
    
    complexity = classification['label']
    confidence = classification['confidence']
    classification_latency = classification['latency_ms']
    
    logger.info(f"Classified as '{complexity}' (confidence: {confidence:.2%})")
    
    # Step 2: Get model recommendation
    if selected_models and complexity in selected_models:
        routed_model = selected_models[complexity]
        logger.info(f"Using manually selected model: {routed_model}")
    else:
        recommendations = get_recommended_models(complexity)
        routed_model = recommendations['primary'][0]
        logger.info(f"Recommended model: {routed_model}")
    
    # Generate routing reasoning
    reasoning = _generate_routing_reasoning(complexity, confidence, routed_model)
    
    # If auto_route is False, return recommendation only
    if not auto_route:
        return {
            'complexity': complexity,
            'confidence': confidence,
            'routed_model': routed_model,
            'classification_latency_ms': classification_latency,
            'reasoning': reasoning,
            'auto_route': False,
        }
    
    # Step 3: Run inference on selected model
    logger.info(f"Running inference on {routed_model}...")
    
    # Get model-specific timeout
    model_info = OLLAMA_MODELS.get(routed_model, {})
    timeout = model_info.get('timeout', 120)
    
    inference_result = run_ollama_inference(routed_model, prompt, timeout=timeout)
    
    # Step 4: Calculate metrics
    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    # Energy calculation
    energy_kwh = inference_result.get('energy_kwh', 0)
    co2_kg = energy_kwh * CO2_INTENSITY
    
    # Energy savings vs baseline (always using large model)
    energy_saved = BASELINE_ENERGY - energy_kwh
    energy_saved_pct = (energy_saved / BASELINE_ENERGY * 100) if BASELINE_ENERGY > 0 else 0
    
    # Green score (0-100, higher is better)
    green_score = max(0, min(100, round(100 * (1 - energy_kwh / BASELINE_ENERGY))))
    
    # Compile results
    result = {
        'complexity': complexity,
        'confidence': confidence,
        'routed_model': routed_model,
        'model_name': model_info.get('name', routed_model),
        'response': inference_result.get('raw_response', ''),
        'error': inference_result.get('error'),
        'classification_latency_ms': classification_latency,
        'inference_latency_ms': inference_result.get('latency_ms', 0),
        'total_latency_ms': total_latency_ms,
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'energy_saved_kwh': max(0, energy_saved),
        'energy_saved_pct': max(0, energy_saved_pct),
        'green_score': green_score,
        'baseline_model': BASELINE_MODEL,
        'baseline_energy': BASELINE_ENERGY,
        'reasoning': reasoning,
        'all_probs': classification['all_probs'],
    }
    
    logger.info(
        f"Routing complete: {complexity} → {routed_model} | "
        f"Energy saved: {energy_saved_pct:.1f}% | Green score: {green_score}"
    )
    
    return result


def _generate_routing_reasoning(complexity: str, confidence: float, model: str) -> str:
    """Generate human-readable routing reasoning."""
    
    reasons = []
    
    # Complexity reasoning
    complexity_reasons = {
        'small': "Simple task → routed to small model for maximum efficiency",
        'medium': "Moderate complexity → routed to medium model for balance",
        'large': "Complex task → routed to large model for best quality"
    }
    reasons.append(complexity_reasons.get(complexity, "Unknown complexity"))
    
    # Confidence
    if confidence > 0.9:
        reasons.append(f"High confidence ({confidence:.0%})")
    elif confidence < 0.7:
        reasons.append(f"⚠️ Lower confidence ({confidence:.0%}) - monitor quality")
    
    # Model selection
    model_info = OLLAMA_MODELS.get(model, {})
    model_size = model_info.get('size', 0)
    reasons.append(f"Selected: {model_info.get('name', model)} ({model_size:.1f}GB)")
    
    return " | ".join(reasons)


def compare_routing_strategies(
    prompt: str,
    strategies: list = None
) -> Dict:
    """
    Compare multiple routing strategies for the same prompt.
    
    Useful for demonstrating energy savings.
    
    Parameters
    ----------
    prompt : str
        Prompt to test
    strategies : list, optional
        List of strategy names: ['smart', 'always_small', 'always_large']
        Default: all three strategies
    
    Returns
    -------
    dict
        Comparison results for each strategy
    """
    if strategies is None:
        strategies = ['smart', 'always_small', 'always_large']
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\nTesting strategy: {strategy}")
        
        if strategy == 'smart':
            # Smart routing (our classifier)
            result = run_routing_pipeline(prompt)
            
        elif strategy == 'always_small':
            # Always route to smallest model
            result = run_routing_pipeline(
                prompt,
                selected_models={
                    'small': 'tinyllama:latest',
                    'medium': 'tinyllama:latest',
                    'large': 'tinyllama:latest'
                }
            )
            
        elif strategy == 'always_large':
            # Always route to largest model (baseline)
            result = run_routing_pipeline(
                prompt,
                selected_models={
                    'small': BASELINE_MODEL,
                    'medium': BASELINE_MODEL,
                    'large': BASELINE_MODEL
                }
            )
        
        results[strategy] = result
    
    return results
