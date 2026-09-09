"""
config.py — Central configuration for Carbon-Aware LLM Routing System
All constants live here for easy tuning.
"""

# ── CO₂ intensity (kg CO₂ per kWh) — global average grid ──────────────────────
CO2_INTENSITY = 0.475  # kg CO₂ / kWh  (IEA world average 2023)

# ── Default baseline energy (kWh) for qwen2.5:7b when Ollama is unavailable ──────
DEFAULT_BASELINE_ENERGY = 0.000235  # Default baseline energy in kWh

# ── Streamlit page settings ───────────────────────────────────────────────────
PAGE_TITLE = "Carbon-Aware LLM Routing"
PAGE_ICON  = "🌿"
LAYOUT     = "wide"
