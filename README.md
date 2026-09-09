# 🌿 Carbon-Aware Intelligent LLM Routing System

A Green AI system that classifies the **complexity of a user's prompt** before running any LLM inference, then routes it to the smallest model capable of answering it well — saving energy and reducing carbon emissions without sacrificing quality.

> **Core principle:** Don't waste energy sending every prompt to the biggest model when a smaller one would do the job.

---

## Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [The Complexity Classifier](#the-complexity-classifier)
- [Model Routing](#model-routing)
- [Energy & Carbon Tracking](#energy--carbon-tracking)
- [Dashboards](#dashboards)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Tech Stack](#tech-stack)

---

## How It Works

```
User enters a prompt
        ↓
MiniLM Classifier  (22M params, ~10ms, runs locally)
        ↓
  ┌─────┴──────────────┬──────────────────┐
  ▼                    ▼                  ▼
SMALL               MEDIUM             LARGE
(simple fact,    (moderate code,    (math proof,
 translation,     explanation,       complex code,
 quick lookup)    summary)           multi-step reasoning)
  ↓                    ↓                  ↓
TinyLlama /       Qwen2.5 3B /       Qwen2.5 7B /
DeepSeek 1.3B     Phi-3              Gemma3 4B
  ↓                    ↓                  ↓
        Response + Energy + CO₂ + Green Score
        displayed in the Streamlit dashboard
```

---

## System Architecture

### Key Files

| File | Role |
|---|---|
| `app.py` | Streamlit entry point — renders the two dashboards |
| `complexity_classifier.py` | Loads the fine-tuned MiniLM model, runs classification, applies 7 heuristic override rules |
| `routing_pipeline.py` | Orchestrates: classify → select Ollama model → run inference → compute energy savings |
| `ollama_integration.py` | Calls the local Ollama server, parses responses, estimates energy per model size |
| `ollama_service.py` | Manages the Ollama process (auto-start, health check, model list) |
| `energy_tracker.py` | Computes kWh, CO₂, green score, and energy saved vs baseline |
| `config.py` | Central constants: CO₂ intensity, page settings |
| `prompt_router_dashboard.py` | Streamlit UI for the Prompt Router tab |
| `ollama_dashboard.py` | Streamlit UI for the Multi-Model Compare tab |

### Model Files

```
model/
└── prompt_complexity_classifier/
    ├── model.safetensors       ← fine-tuned MiniLM weights (used at runtime)
    ├── config.json             ← label mappings: 0=small, 1=medium, 2=large
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_args.bin
    └── training_report.txt     ← accuracy, F1, confusion matrix
```

---

## The Complexity Classifier

### Model

- **Base model:** `microsoft/MiniLM-L12-H384-uncased` — 22M parameters, fast on CPU
- **Task:** 3-class sequence classification (`small` / `medium` / `large`)
- **Inference time:** ~10 ms on CPU

### Training

| Property | Value |
|---|---|
| Training data | 28,465 prompts |
| Sources | WildChat (20k), GSM8K (7.5k math), SupraLabs (992 gold-labelled) |
| Device | Tesla T4 (Google Colab) |
| Training time | 6.83 minutes |
| Epochs | 3, batch size 32, FP16 |

### Results

```
Accuracy : 93.20%
F1 macro : 0.9271

Confusion Matrix:
           small  medium  large
small   :  1441     83     21
medium  :    23   1287    124
large   :    35    101   2578
```

### Heuristic Override Rules

The ML model's prediction passes through 7 rule-based checks that catch common edge cases:

| Rule | Condition | Action |
|---|---|---|
| 1 | Math proof keywords (prove, theorem, induction…) | Force → `large` |
| 2 | Prompt > 120 tokens | Force → `large` |
| 3 | Prompt > 80 tokens AND predicted `small` | Upgrade → `medium` |
| 4 | 2+ complex reasoning signals (step by step, analyze…) | Upgrade `small` → `large` |
| 5 | Multi-step indicators (3+) AND long prompt | Upgrade `small` → `medium` |
| 6 | Code task + 5+ tech requirements (auth, DB, API…) | Force → `large` |
| 7 | Academic/research keywords AND long prompt | Upgrade `small` → `medium` |

When a rule fires, confidence is capped at 85% and a `rule_override` flag is set — visible in the dashboard as *"🎯 Enhanced by heuristic rules"*.

---

## Model Routing

### Complexity → Model Mapping

| Complexity | Model options | Size | Est. energy/inference |
|---|---|---|---|
| **small** | TinyLlama | 0.64 GB | 0.000032 kWh |
| **small** | DeepSeek-Coder 1.3B | 0.78 GB | 0.000039 kWh |
| **small** | Qwen2 1.5B | 0.93 GB | 0.000047 kWh |
| **medium** | Qwen2.5 3B | 1.9 GB | 0.000095 kWh |
| **medium** | Phi-3 | 2.2 GB | 0.000110 kWh |
| **large** | Qwen2.5 7B *(baseline)* | 4.7 GB | 0.000235 kWh |
| **large** | Gemma3 4B | 3.3 GB | 0.000165 kWh |

The **baseline** is always running `qwen2.5:7b`. All energy savings are computed relative to this.

### Energy Savings Examples

| Prompt | Class | Routed to | Energy saved vs baseline |
|---|---|---|---|
| "What is the capital of France?" | small | TinyLlama | ~87% |
| "Write a binary search in Python" | medium | Qwen2.5 3B | ~60% |
| "Prove √2 is irrational" | large | Qwen2.5 7B | 0% (appropriate) |

---

## Energy & Carbon Tracking

### Energy Estimation

Ollama model energy is estimated by model size:

```
energy_kwh = model_size_gb × 0.00005
```

### CO₂ Calculation

```
co2_kg = energy_kwh × 0.475
```

`0.475 kg CO₂/kWh` is the IEA 2023 global average grid intensity.

### Green Score

```
green_score = 100 × (1 − energy_used / baseline_energy)
```

- **100** = used a tiny model, maximum savings
- **0** = used the full 7B baseline

### CodeCarbon (optional)

If the `codecarbon` library is installed, real hardware measurements replace the estimates. The system detects it automatically and falls back to estimates if it's absent.

---

## Dashboards

### Prompt Router (main tab)

1. Enter any prompt or pick a quick example
2. Click **Classify & Route** — the local MiniLM model runs in ~10ms
3. See:
   - **Complexity tier** (🟢 SMALL / 🟡 MEDIUM / 🔴 LARGE)
   - **Confidence** and probability distribution chart
   - **Recommended Ollama models** (primary + fallback)
   - **Routing reasoning** (including any rule overrides)
   - **Energy savings** vs always-large baseline
   - **Green Score** and CO₂ saved
4. Optionally click **Run Inference** to call the Ollama model and see the actual response with live energy metrics

### Multi-Model Compare tab

- Run the same free-form prompt across 3 Ollama models simultaneously
- Results appear side-by-side with latency badges
- Useful for quality comparison across model sizes

---

## Project Structure

```
Carbon-Aware-Intelligent-LLM-Routing-System/
│
├── app.py                          # Streamlit entry point
├── config.py                       # CO₂ intensity, page settings
├── energy_tracker.py               # kWh / CO₂ / green score calculations
│
├── complexity_classifier.py        # MiniLM inference + heuristic rules
├── routing_pipeline.py             # classify → route → infer → metrics
│
├── ollama_integration.py           # Ollama API calls + energy estimates
├── ollama_service.py               # Ollama process management
│
├── prompt_router_dashboard.py      # Streamlit UI: Prompt Router tab
├── ollama_dashboard.py             # Streamlit UI: Multi-Model Compare tab
│
├── model/
│   ├── train_classifier.py         # Local training script
│   └── prompt_complexity_classifier/
│       ├── model.safetensors       # Trained weights (used at runtime)
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── training_report.txt
│
├── data/
│   ├── fetch_datasets.py           # Downloads raw datasets
│   ├── build_labeled_dataset.py    # Applies heuristic labels
│   ├── cache/                      # Raw parquet files
│   └── labeled/
│       ├── prompt_complexity_labeled.parquet   # Training data
│       └── llm_judge_eval_set.csv              # Quality spot-check
│
├── eval/
│   └── routing_validation.py       # Validates routing decisions vs Ollama
│
├── colab_train_classifier.py       # Google Colab training notebook script
├── test_ollama.py                  # Quick Ollama connectivity test
├── test_improvements.py            # Classifier smoke tests
│
└── requirements.txt
```

---

## Installation & Setup

### Requirements

- Python 3.8+
- 4 GB RAM minimum (8 GB recommended for 7B models)
- [Ollama](https://ollama.com/download) installed

### 1. Clone and create virtual environment

```powershell
git clone <repo-url>
cd Carbon-Aware-Intelligent-LLM-Routing-System
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install Ollama and pull models

Download from **https://ollama.com/download** and run the installer, then pull at least one model per tier:

```powershell
# Small tier (pick one — TinyLlama is fastest to download)
ollama pull tinyllama          # 0.6 GB

# Medium tier
ollama pull phi3               # 2.2 GB

# Large tier (the baseline — needed for energy comparison)
ollama pull qwen2.5:7b         # 4.7 GB
```

You don't need all models. The router will pick the best available model for each tier.

### 4. Verify the classifier loads correctly

```powershell
python -c "from complexity_classifier import test_classifier; test_classifier()"
```

You should see 6 prompts classified with complexity labels, confidence scores, and recommended models. If this works, everything is ready.

---

## Running the App

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

Ollama starts automatically if it's installed. If it's not running, the Ollama Compare tab will show a disconnected status but the Prompt Router's **Classify & Route** button (classification only) will still work without Ollama.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit 1.33+ |
| Classifier model | HuggingFace Transformers + PyTorch (CPU) |
| Local LLMs | Ollama |
| Visualisation | Plotly |
| Data | Pandas, Parquet |
| Carbon tracking (optional) | CodeCarbon |
| Model format | SafeTensors |

---

## Re-training the Classifier

The trained model is already included in `model/prompt_complexity_classifier/` — no re-training needed to run the system.

If you want to retrain (e.g., on new data):

**Option A — Google Colab (recommended, uses GPU):**
Open `Colab_Training_Notebook.ipynb` in Google Colab, run all cells, then download the output folder and replace `model/prompt_complexity_classifier/`.

**Option B — Local:**
```powershell
python model/train_classifier.py
```
Requires the labeled dataset at `data/labeled/prompt_complexity_labeled.parquet`. Takes ~7 minutes on a T4 GPU or significantly longer on CPU.
