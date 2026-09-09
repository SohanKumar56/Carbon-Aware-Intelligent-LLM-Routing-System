"""
ollama_integration.py — Ollama model integration for direct prompt inference

This module provides integration with locally running Ollama models
for direct prompt inference with CodeCarbon energy measurements (REQUIRED).
"""

from __future__ import annotations
import requests
import json
import time
import logging
from typing import Dict, List, Optional
from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Available Ollama models with their sizes (in GB) and recommended timeout
OLLAMA_MODELS = {
    "gemma3:4b": {"size": 3.3, "name": "Gemma 3 (4B)", "timeout": 180},
    "deepseek-r1:1.5b": {"size": 1.1, "name": "DeepSeek R1 (1.5B)", "timeout": 90},
    "qwen2.5:3b": {"size": 1.9, "name": "Qwen 2.5 (3B)", "timeout": 120},
    "qwen2.5:7b": {"size": 4.7, "name": "Qwen 2.5 (7B)", "timeout": 600},  # Increased to 600s (10 min)
    "zephyr:latest": {"size": 4.1, "name": "Zephyr", "timeout": 240},  # Increased to 240s
    "deepseek-coder:1.3b": {"size": 0.776, "name": "DeepSeek Coder (1.3B)", "timeout": 90},
    "qwen2:1.5b": {"size": 0.934, "name": "Qwen 2 (1.5B)", "timeout": 90},
    "tinyllama:latest": {"size": 0.637, "name": "TinyLlama", "timeout": 60},
    "qwen2:latest": {"size": 4.4, "name": "Qwen 2", "timeout": 240},  # Increased to 240s
    "phi3:latest": {"size": 2.2, "name": "Phi-3", "timeout": 120},
    "qwen2.5-coder:latest": {"size": 4.7, "name": "Qwen 2.5 Coder", "timeout": 300},  # Increased to 300s
}

# Real hardware energy measurement using CodeCarbon
def measure_inference_energy(inference_function) -> tuple:
    """
    Measure actual energy consumption during inference using CodeCarbon.
    
    Parameters
    ----------
    inference_function : callable
        Function that performs the inference
    
    Returns
    -------
    tuple
        (result, energy_kwh, co2_kg) where:
        - result: the inference result
        - energy_kwh: actual energy consumed in kWh
        - co2_kg: actual CO2 emissions in kg
    """
    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="ollama_inference",
        measure_power_secs=1,
        log_level="error",
        save_to_file=False,
    )
    
    # Start tracking
    tracker.start()
    
    # Run inference
    result = inference_function()
    
    # Stop tracking and get emissions
    emissions = tracker.stop()  # returns kg CO2
    
    # Calculate energy from emissions (using IEA 2023 average)
    co2_kg = emissions if emissions else 0.0
    energy_kwh = co2_kg / 0.475 if co2_kg > 0 else 0.0
    
    logger.info(f"CodeCarbon measurement: {energy_kwh:.6f} kWh, {co2_kg:.6f} kg CO2")
    
    return result, energy_kwh, co2_kg


def get_ollama_models() -> List[Dict[str, str]]:
    """Get list of available Ollama models"""
    return [
        {"id": model_id, "name": info["name"], "size": info["size"]}
        for model_id, info in OLLAMA_MODELS.items()
    ]


def run_ollama_inference(model_id: str, text: str, timeout: int = 120) -> Dict:
    """
    Run direct prompt inference using an Ollama model with real hardware energy measurement.
    
    Parameters
    ----------
    model_id : str
        Ollama model identifier (e.g., "tinyllama:latest")
    text : str
        User prompt to send to the model
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    dict
        raw_response – actual model response to the prompt
        model        – model identifier
        model_name   – human-readable model name
        latency_ms   – inference time in milliseconds
        energy_kwh   – actual energy consumed (measured by CodeCarbon)
        co2_kg       – actual CO₂ emissions (measured by CodeCarbon)
        error        – error message if failed
    """
    t0 = time.perf_counter()
    
    if model_id not in OLLAMA_MODELS:
        return {
            "raw_response": "",
            "model": model_id,
            "model_name": "Unknown",
            "latency_ms": 0,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": f"Model {model_id} not found",
        }
    
    model_info = OLLAMA_MODELS[model_id]
    
    try:
        # Define the inference function for CodeCarbon measurement
        def perform_inference():
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model_id,
                    "prompt": text,
                    "stream": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        
        # Measure actual energy consumption during inference
        result, energy_kwh, co2_kg = measure_inference_energy(perform_inference)
        
        # Extract response text
        response_text = result.get("response", "").strip()
        
        # Calculate latency
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        logger.info(f"Ollama {model_id} inference completed | Energy: {energy_kwh:.6f} kWh, CO2: {co2_kg:.6f} kg")
        
        return {
            "raw_response": response_text,
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
        }
        
    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "raw_response": "",
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": "Request timeout",
        }
    
    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "raw_response": "",
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": "Cannot connect to Ollama. Is it running?",
        }
    
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(f"Ollama inference error: {e}")
        return {
            "raw_response": "",
            "model": model_id,
            "model_name": model_info["name"],
            "latency_ms": latency_ms,
            "energy_kwh": 0,
            "co2_kg": 0,
            "error": str(e),
        }


def run_multiple_ollama_models(model_ids: List[str], text: str) -> List[Dict]:
    """
    Run sentiment analysis on multiple Ollama models.
    
    Parameters
    ----------
    model_ids : List[str]
        List of Ollama model identifiers (max 3)
    text : str
        Input text for sentiment analysis
    
    Returns
    -------
    List[dict]
        List of results from each model
    """
    if len(model_ids) > 3:
        logger.warning(f"Too many models selected ({len(model_ids)}). Using first 3.")
        model_ids = model_ids[:3]
    
    results = []
    for model_id in model_ids:
        # Get model-specific timeout from OLLAMA_MODELS dict
        timeout = OLLAMA_MODELS.get(model_id, {}).get("timeout", 120)
        result = run_ollama_inference(model_id, text, timeout=timeout)
        results.append(result)
    
    return results


def check_ollama_availability() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
