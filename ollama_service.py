"""
ollama_service.py — Backend service for Ollama model management and inference.
Handles process lifecycle, model discovery, and API communication.
"""

import subprocess
import platform
import time
import requests
import os
import logging
from codecarbon import EmissionsTracker

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"

logger = logging.getLogger(__name__)

def is_ollama_running() -> bool:
    """Check if the Ollama server is responsive."""
    try:
        response = requests.get(OLLAMA_API_TAGS, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_ollama_path():
    """Find the ollama binary path, prioritizing known locations on Windows."""
    if platform.system() == "Windows":
        # Check standard installation path
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        standard_path = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(standard_path):
            return standard_path
    return "ollama" # Fallback to PATH

def start_ollama():
    """Start the Ollama server in the background based on the OS."""
    if is_ollama_running():
        return True

    ollama_bin = get_ollama_path()
    try:
        if platform.system() == "Windows":
            # Quoting path for spaces safety and using shell=True for better execution
            cmd = f'"{ollama_bin}" serve'
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Start ollama serve on Mac/Linux
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp # ensures it keeps running independently
            )
        
        # Poll until ready
        max_retries = 10
        for _ in range(max_retries):
            time.sleep(2)
            if is_ollama_running():
                return True
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False

def get_installed_models() -> list[str]:
    """Fetch all installed Ollama models."""
    try:
        response = requests.get(OLLAMA_API_TAGS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # models is a list of dicts with 'name' key
            return [m['name'] for m in data.get('models', [])]
        return []
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return []

def run_ollama_inference(model: str, prompt: str) -> dict:
    """Run a non-streaming inference on a specific model with energy measurement."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    
    # Use CodeCarbon to measure energy consumption
    tracker = EmissionsTracker(
        project_name="ollama_dashboard",
        measure_power_secs=1,
        log_level="error",
        save_to_file=False,
    )
    
    try:
        tracker.start()
        response = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=60)
        emissions = tracker.stop()
        latency = time.time() - start_time
        
        # Calculate energy from emissions (using IEA 2023 average)
        co2_intensity = 0.475  # kg CO₂/kWh
        energy_kwh = emissions / co2_intensity if emissions else 0.0
        co2_kg = emissions if emissions else 0.0
        
        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "response": result.get("response", ""),
                "latency": round(latency, 2),
                "model": model,
                "energy_kwh": energy_kwh,
                "co2_kg": co2_kg
            }
        else:
            return {
                "status": "error",
                "message": f"API Error: {response.status_code}",
                "model": model,
                "energy_kwh": 0.0,
                "co2_kg": 0.0
            }
    except requests.Timeout:
        tracker.stop()
        return {
            "status": "error",
            "message": "Request timed out",
            "model": model,
            "energy_kwh": 0.0,
            "co2_kg": 0.0
        }
    except Exception as e:
        try:
            tracker.stop()
        except:
            pass
        return {
            "status": "error",
            "message": str(e),
            "model": model,
            "energy_kwh": 0.0,
            "co2_kg": 0.0
        }
