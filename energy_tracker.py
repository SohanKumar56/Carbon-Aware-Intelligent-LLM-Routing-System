"""
energy_tracker.py — Energy & carbon footprint estimation for LLM routing.

Provides CO₂ accounting for Ollama model inference based on model size.
Energy is estimated as: model_size_gb × 0.00005 kWh per inference.

Optional CodeCarbon integration: if the ``codecarbon`` library is installed,
real hardware measurements are used instead of estimates.

Public API
----------
estimate_routing_energy(model_size_gb, baseline_size_gb)
    → dict with energy_kwh, co2_kg, green_score, energy_saved_pct, energy_saved_kwh

format_energy(kwh)  → human-readable string
format_co2(kg)      → human-readable string
"""

from __future__ import annotations
import logging
from typing import Callable, Any

from config import CO2_INTENSITY

logger = logging.getLogger(__name__)

# ── Energy constant ────────────────────────────────────────────────────────────
# Estimated kWh per GB of model size for local Ollama inference
ENERGY_PER_GB = 0.00005  # kWh / GB

# ── Optional CodeCarbon import ─────────────────────────────────────────────────
try:
    from codecarbon import EmissionsTracker as _CCTracker
    _CODECARBON_AVAILABLE = True
    logger.info("CodeCarbon found — hardware tracking enabled.")
except ImportError:
    _CODECARBON_AVAILABLE = False
    logger.info("CodeCarbon not installed — using empirical energy estimates.")


def estimate_routing_energy(
    model_size_gb: float,
    baseline_size_gb: float,
) -> dict:
    """
    Return energy/carbon metrics for a routed Ollama inference.

    Parameters
    ----------
    model_size_gb : float
        Size of the model actually used (GB).
    baseline_size_gb : float
        Size of the always-large baseline model (GB) for savings comparison.

    Returns
    -------
    dict
        energy_kwh        – kilowatt-hours consumed
        co2_kg            – kilograms of CO₂ equivalent
        green_score       – 0-100 environmental efficiency score
        energy_saved_pct  – % energy saved vs always using the baseline
        energy_saved_kwh  – absolute kWh saved vs baseline
    """
    energy_kwh    = model_size_gb * ENERGY_PER_GB
    baseline_kwh  = baseline_size_gb * ENERGY_PER_GB
    co2_kg        = energy_kwh * CO2_INTENSITY

    saved_kwh     = max(0.0, baseline_kwh - energy_kwh)
    saved_pct     = round(saved_kwh / baseline_kwh * 100, 1) if baseline_kwh > 0 else 0.0
    green_score   = max(0, min(100, round(100 * (1 - energy_kwh / baseline_kwh)))) if baseline_kwh > 0 else 0

    return {
        "energy_kwh":       energy_kwh,
        "co2_kg":           co2_kg,
        "green_score":      green_score,
        "energy_saved_pct": saved_pct,
        "energy_saved_kwh": saved_kwh,
    }


def measure_with_tracker(fn: Callable[[], Any]) -> tuple[Any, dict]:
    """
    Execute *fn()* inside a CodeCarbon tracker (if available).

    Falls back gracefully when CodeCarbon is not installed.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable wrapping the inference call.

    Returns
    -------
    tuple
        (result_of_fn, {energy_kwh, co2_kg, source})
    """
    if _CODECARBON_AVAILABLE:
        tracker = _CCTracker(
            project_name="carbon_aware_llm_routing",
            measure_power_secs=1,
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        result = fn()
        emissions = tracker.stop()   # returns kg CO₂
        energy_kwh = emissions / CO2_INTENSITY if emissions else 0.0
        return result, {
            "energy_kwh": energy_kwh,
            "co2_kg":     emissions or 0.0,
            "source":     "CodeCarbon",
        }
    else:
        result = fn()
        return result, {"source": "empirical"}


def format_energy(kwh: float) -> str:
    """Human-readable kWh string."""
    return f"{kwh:.6f} kWh"


def format_co2(kg: float) -> str:
    """Human-readable CO₂ string."""
    return f"{kg:.6f} kg CO₂"
