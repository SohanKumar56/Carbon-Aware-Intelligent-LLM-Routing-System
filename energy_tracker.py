"""
energy_tracker.py — Energy & carbon footprint tracking for LLM routing.

Provides CO₂ accounting for Ollama model inference using CodeCarbon
for real hardware energy measurements.

CodeCarbon is REQUIRED for this system to function properly.

Public API
----------
measure_with_tracker(fn) → tuple(result, {energy_kwh, co2_kg, source})
calculate_metrics(energy_kwh, baseline_energy_kwh) → dict with green_score, savings
format_energy(kwh)  → human-readable string
format_co2(kg)      → human-readable string
"""

from __future__ import annotations
import logging
from typing import Callable, Any

from config import CO2_INTENSITY
from codecarbon import EmissionsTracker as _CCTracker

logger = logging.getLogger(__name__)


def calculate_metrics(
    energy_kwh: float,
    baseline_energy_kwh: float,
) -> dict:
    """
    Calculate energy/carbon metrics from measured energy values.

    Parameters
    ----------
    energy_kwh : float
        Actual energy consumed (measured by CodeCarbon).
    baseline_energy_kwh : float
        Baseline energy consumption (measured by CodeCarbon for large model).

    Returns
    -------
    dict
        co2_kg            – kilograms of CO₂ equivalent
        green_score       – 0-100 environmental efficiency score
        energy_saved_pct  – % energy saved vs always using the baseline
        energy_saved_kwh  – absolute kWh saved vs baseline
    """
    co2_kg = energy_kwh * CO2_INTENSITY

    saved_kwh = max(0.0, baseline_energy_kwh - energy_kwh)
    saved_pct = round(saved_kwh / baseline_energy_kwh * 100, 1) if baseline_energy_kwh > 0 else 0.0
    green_score = max(0, min(100, round(100 * (1 - energy_kwh / baseline_energy_kwh)))) if baseline_energy_kwh > 0 else 0

    return {
        "co2_kg": co2_kg,
        "green_score": green_score,
        "energy_saved_pct": saved_pct,
        "energy_saved_kwh": saved_kwh,
    }


def measure_with_tracker(fn: Callable[[], Any]) -> tuple[Any, dict]:
    """
    Execute *fn()* inside a CodeCarbon tracker for real hardware energy measurement.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable wrapping the inference call.

    Returns
    -------
    tuple
        (result_of_fn, {energy_kwh, co2_kg, source})
    """
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


def format_energy(kwh: float) -> str:
    """Human-readable kWh string."""
    return f"{kwh:.6f} kWh"


def format_co2(kg: float) -> str:
    """Human-readable CO₂ string."""
    return f"{kg:.6f} kg CO₂"
