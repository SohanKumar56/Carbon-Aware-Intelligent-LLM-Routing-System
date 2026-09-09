"""
routing_pipeline.py — Prompt-based routing pipeline with energy tracking

Classifies prompt complexity upfront and routes to an appropriately-sized
local Ollama LLM to minimise energy waste.

Green AI principle: Don't waste energy routing every prompt to the biggest
model when a smaller one would do the job.

Uses CodeCarbon for real hardware energy measurements.
"""

from __future__ import annotations
import logging
import time
import requests
from typing import Dict, Optional

from complexity_classifier import classify_prompt_complexity_detailed, get_recommended_models
from ollama_integration import run_ollama_inference, OLLAMA_MODELS
from energy_tracker import calculate_metrics
from config import CO2_INTENSITY, DEFAULT_BASELINE_ENERGY
from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)

# Baseline for comparison (always using largest model)
BASELINE_MODEL = "qwen2.5:7b"


def measure_baseline_energy(model_id: str) -> float:
    """
    Measure actual baseline energy consumption using CodeCarbon.
    
    Parameters
    ----------
    model_id : str
        Model to measure (typically the large baseline model)
    
    Returns
    -------
    float
        Energy consumption in kWh, or 0.0 if measurement fails
    """
    try:
        # Check if Ollama is available first
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            logger.warning("Ollama not available - using default baseline energy")
            return DEFAULT_BASELINE_ENERGY
        
        # Simple test inference to measure baseline
        test_prompt = "Test prompt for baseline energy measurement"
        
        def baseline_inference():
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": test_prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        
        tracker = EmissionsTracker(
            project_name="baseline_measurement",
            measure_power_secs=1,
            log_level="error",
            save_to_file=False,
        )
        
        tracker.start()
        baseline_inference()
        emissions = tracker.stop()
        
        # Calculate energy from emissions
        co2_kg = emissions if emissions else 0.0
        energy_kwh = co2_kg / CO2_INTENSITY if co2_kg > 0 else 0.0
        
        logger.info(f"Baseline energy measured: {energy_kwh:.6f} kWh for {model_id}")
        return energy_kwh
        
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama - using default baseline energy")
        return DEFAULT_BASELINE_ENERGY
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timeout - using default baseline energy")
        return DEFAULT_BASELINE_ENERGY
    except Exception as e:
        logger.warning(f"Baseline measurement failed: {e} - using default baseline energy")
        return DEFAULT_BASELINE_ENERGY


# Cache baseline energy measurement
_baseline_energy_cache = None

def get_baseline_energy() -> float:
    """Get baseline energy consumption (cached or measured)."""
    global _baseline_energy_cache
    
    if _baseline_energy_cache is not None:
        return _baseline_energy_cache
    
    # Use default baseline initially to avoid blocking startup
    # Will be measured on first successful inference
    _baseline_energy_cache = DEFAULT_BASELINE_ENERGY
    logger.info(f"Using default baseline energy: {DEFAULT_BASELINE_ENERGY:.6f} kWh")
    return _baseline_energy_cache


def update_baseline_energy_if_successful() -> None:
    """Attempt to measure actual baseline energy if Ollama is available in background."""
    global _baseline_energy_cache
    
    # Only try to update if we're currently using the default
    if _baseline_energy_cache != DEFAULT_BASELINE_ENERGY:
        return  # Already have a measured value
    
    try:
        # Check if Ollama is available with very short timeout
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            logger.info("Ollama available - attempting to measure actual baseline energy")
            actual_baseline = measure_baseline_energy(BASELINE_MODEL)
            if actual_baseline > 0 and actual_baseline != DEFAULT_BASELINE_ENERGY:
                _baseline_energy_cache = actual_baseline
                logger.info(f"Updated baseline energy to measured value: {actual_baseline:.6f} kWh")
    except Exception as e:
        logger.debug(f"Could not measure actual baseline energy: {e}, using default")


def run_routing_pipeline(
    prompt: str,
    selected_models: Optional[Dict[str, str]] = None,
    auto_route: bool = True
) -> Dict:
    """
    Execute the routing pipeline on a prompt.
    
    Workflow:
    1. Classify prompt complexity
    2. Route to appropriate model size (forced mapping)
    3. Run inference automatically
    4. Track energy vs baseline (always using large model)
    
    Parameters
    ----------
    prompt : str
        User input prompt
    selected_models : dict, optional
        Manual model selection per complexity level:
        {'small': 'tinyllama:latest', 'medium': 'phi3:latest', 'large': 'qwen2.5:7b'}
        If None, uses forced defaults as per requirements
    auto_route : bool
        If True, automatically routes and runs inference.
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
    
    # Step 2: Get forced model recommendation (no manual override allowed)
    recommendations = get_recommended_models(complexity)
    routed_model = recommendations['primary'][0]
    logger.info(f"Routed to forced model: {routed_model}")
    
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
    
    # Update baseline energy measurement if inference was successful
    if not inference_result.get('error'):
        update_baseline_energy_if_successful()
    
    # Step 4: Calculate metrics using CodeCarbon measurements
    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    # Energy calculation (uses real hardware measurements from CodeCarbon)
    energy_kwh = inference_result.get('energy_kwh', 0)
    
    # Get baseline energy (measured or cached)
    baseline_energy = get_baseline_energy()
    
    # Calculate all energy/carbon metrics using the centralized function
    metrics = calculate_metrics(energy_kwh, baseline_energy)
    co2_kg = metrics['co2_kg']
    energy_saved = metrics['energy_saved_kwh']
    energy_saved_pct = metrics['energy_saved_pct']
    green_score = metrics['green_score']
    
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
        'baseline_energy': baseline_energy,  # Now using measured baseline
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
            # Smart routing (our classifier with forced models)
            result = run_routing_pipeline(prompt)
            
        elif strategy == 'always_small':
            # Always route to smallest model (manually override classification)
            # Temporarily modify recommendations for this test
            from complexity_classifier import get_recommended_models
            original_recs = get_recommended_models
            def always_small_recs(complexity):
                return {'primary': ['tinyllama:latest'], 'fallback': ['tinyllama:latest']}
            # Monkey patch for this test
            import complexity_classifier
            complexity_classifier.get_recommended_models = always_small_recs
            result = run_routing_pipeline(prompt)
            # Restore original
            complexity_classifier.get_recommended_models = original_recs
            
        elif strategy == 'always_large':
            # Always route to largest model (baseline)
            from complexity_classifier import get_recommended_models
            original_recs = get_recommended_models
            def always_large_recs(complexity):
                return {'primary': ['qwen2.5:7b'], 'fallback': ['qwen2.5:7b']}
            # Monkey patch for this test
            import complexity_classifier
            complexity_classifier.get_recommended_models = always_large_recs
            result = run_routing_pipeline(prompt)
            # Restore original
            complexity_classifier.get_recommended_models = original_recs
        
        results[strategy] = result
    
    return results
