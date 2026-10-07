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
- [Data Processing Pipeline](#data-processing-pipeline)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)

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
| `ollama_integration.py` | Calls the local Ollama server, parses responses, measures energy via CodeCarbon |
| `ollama_service.py` | Manages the Ollama process (auto-start, health check, model list) |
| `energy_tracker.py` | Computes kWh, CO₂, green score using CodeCarbon measurements |
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

### Energy Measurement

Ollama model energy consumption is measured using **CodeCarbon** for real hardware energy tracking. CodeCarbon is **required** for this system to function - it provides actual power consumption measurements from your hardware during inference.

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

### CodeCarbon (required)

The `codecarbon` library is **required** for energy and carbon measurements. It provides real hardware tracking instead of estimates, ensuring accurate carbon footprint calculations for your specific hardware setup.

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
├── app.py                          # Streamlit entry point with dual dashboards
├── config.py                       # Central configuration (CO₂ intensity, page settings)
├── energy_tracker.py               # Energy & carbon footprint calculations
│
├── complexity_classifier.py        # MiniLM inference + 7 heuristic override rules
├── routing_pipeline.py             # classify → route → infer → metrics pipeline
│
├── ollama_integration.py           # Ollama API calls + CodeCarbon energy measurements
├── ollama_service.py               # Ollama process management (auto-start, health check)
│
├── prompt_router_dashboard.py      # Streamlit UI: Prompt Router tab with LLM explanations
├── ollama_dashboard.py             # Streamlit UI: Multi-Model Compare tab
│
├── model/
│   ├── train_classifier.py         # Local training script
│   └── prompt_complexity_classifier/  # Trained MiniLM model (93.2% accuracy)
│       ├── model.safetensors       # Fine-tuned weights (133MB)
│       ├── config.json             # Label mappings (0=small, 1=medium, 2=large)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── training_report.txt     # Accuracy, F1, confusion matrix
│
├── data/
│   ├── fetch_datasets.py           # Downloads raw datasets (WildChat, GSM8K, SupraLabs)
│   ├── fetch_datasets_spark.py    # PySpark version for distributed downloading
│   ├── build_labeled_dataset.py    # Applies heuristic labels using Pandas
│   ├── build_labeled_dataset_spark.py  # PySpark version (5x faster)
│   ├── compare_pandas_vs_spark.py  # Benchmarking script
│   ├── spark_config.py             # PySpark configuration
│   ├── judge_rubric.md             # Classification criteria for labeling
│   ├── inspect_labeled_data.py     # Data inspection utilities
│   ├── inspect_supralabs.py        # SupraLabs dataset inspection
│   ├── cache/                      # Raw downloaded datasets (parquet format)
│   └── labeled/                     # Final labeled training data
│       ├── prompt_complexity_labeled.parquet   # 28,465 labeled prompts
│       └── dataset_metadata.json
│
├── eval/
│   └── routing_validation.py       # Validates routing decisions vs real Ollama models
│
├── colab_train_classifier.py       # Google Colab training script
├── Colab_Training_Notebook.ipynb   # Interactive GPU training notebook
│
├── test_ollama.py                  # Quick Ollama connectivity test
├── test_improvements.py            # Classifier smoke tests
├── test_pyspark_installation.py    # PySpark installation verification
│
├── requirements.txt                # Python dependencies
│
└── DOCUMENTATION/
    ├── QUICK_START_GUIDE.md        # Quick start instructions
    ├── OLLAMA_INTEGRATION.md       # Ollama integration guide
    ├── PROMPT_ROUTER_FEATURE.md    # Complete prompt router documentation
    ├── IMPLEMENTATION_COMPLETE.md  # Implementation summary
    ├── COLAB_TRAINING_INSTRUCTIONS.md  # GPU training guide
    ├── SPARK_PREPROCESSING_GUIDE.md   # PySpark preprocessing guide
    ├── CONFIDENCE_SCORING_UPDATE.md   # Confidence scoring details
    ├── CURRICULUM_IMPLEMENTATION_MAP.md  # Feature implementation roadmap
    ├── PYSPARK_IMPLEMENTATION_SUMMARY.md # PySpark integration summary
    ├── IMPROVEMENTS_APPLIED.md     # Applied improvements log
    ├── READY_TO_USE.md            # Ready-to-use status
    └── TEST_THE_IMPROVEMENTS.md   # Testing guide
```

---

## Installation & Setup

### Requirements

- Python 3.8+
- 4 GB RAM minimum (8 GB recommended for 7B models)
- [Ollama](https://ollama.com/download) installed
- **CodeCarbon** (required for energy/carbon measurements)
- **Optional:** PySpark 3.5+ for distributed preprocessing

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
# Standard installation
pip install -r requirements.txt

# Includes PySpark for distributed preprocessing
# (Already included in requirements.txt)
```

### 2a. Optional: Run PySpark Preprocessing

```powershell
# Run distributed data preprocessing (5x faster than Pandas)
cd data
python build_labeled_dataset_spark.py

# Benchmark Pandas vs PySpark
python compare_pandas_vs_spark.py

# See SPARK_PREPROCESSING_GUIDE.md for details
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
| **Distributed Processing** | **PySpark 3.5+ (NEW! 🚀)** |
| Carbon tracking (required) | CodeCarbon |
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

---

## Data Processing Pipeline

### Dataset Sources

The system uses three main datasets for training the complexity classifier:

1. **WildChat** (20,000 prompts) - Diverse real-world user queries from allenai/WildChat
2. **GSM8K** (7,473 prompts) - Grade-school math problems requiring multi-step reasoning
3. **SupraLabs** (992 prompts) - Pre-labeled routing examples from SupraLabs/Prompt-Routing-Dataset

### Data Processing Options

#### Option 1: Pandas (Standard)
```powershell
cd data
python fetch_datasets.py          # Download raw datasets
python build_labeled_dataset.py   # Apply heuristic labels
```

#### Option 2: PySpark (Recommended for Large Datasets)
```powershell
cd data
python fetch_datasets_spark.py         # Download with PySpark (faster)
python build_labeled_dataset_spark.py  # Process with PySpark (5x faster)
```

**PySpark Benefits:**
- 5x faster processing for large datasets
- Distributed computing capabilities
- Better memory management
- Scalable to millions of prompts

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Large | 14,061 | 49.4% |
| Medium | 8,232 | 28.9% |
| Small | 6,172 | 21.7% |

### Heuristic Labeling

The system uses heuristic rules to label prompts based on:
- Token count (short prompts → small, long prompts → large)
- Multi-step reasoning indicators (step by step, analyze, evaluate)
- Math/code content keywords
- Question complexity patterns

See `data/judge_rubric.md` for detailed classification criteria.

---

## Testing & Validation

### Quick Classifier Test

Test the classifier without Ollama:

```powershell
python complexity_classifier.py
```

This runs 6 built-in test prompts and displays classification results.

### Ollama Connectivity Test

Verify Ollama is running and accessible:

```powershell
python test_ollama.py
```

### Routing Validation

Validate routing decisions with real Ollama models:

```powershell
cd eval
python routing_validation.py
```

This script:
- Samples prompts from each complexity class
- Runs them through both small and large models
- Compares response quality
- Measures actual energy savings

### PySpark Installation Test

Verify PySpark is installed correctly:

```powershell
python test_pyspark_installation.py
```

### Benchmarking

Compare Pandas vs PySpark performance:

```powershell
cd data
python compare_pandas_vs_spark.py
```

---

## Troubleshooting

### Classifier Issues

**"Model not found" error**
- Ensure `model/prompt_complexity_classifier/` directory exists
- Check that `model.safetensors` file is present (133MB)
- Verify `config.json` contains correct label mappings

**Slow classification on first run**
- First load takes 1-2 seconds (model loading)
- Subsequent calls are ~10ms due to module-level caching
- This is normal behavior

**Low confidence scores**
- Confidence < 70% may indicate ambiguous prompts
- The system includes 7 heuristic override rules to handle edge cases
- Check the reasoning section for rule override notifications

### Ollama Issues

**"Ollama not running" error**
- Start Ollama: `ollama serve`
- Check if running: `curl http://localhost:11434/api/tags`
- Verify installation from https://ollama.com/download

**"Cannot connect to Ollama"**
- Ensure Ollama server is running on port 11434
- Check firewall settings
- Try restarting Ollama server

**"Request timeout"**
- Increase timeout in `ollama_integration.py` (model-specific timeouts)
- Use smaller models for faster inference
- Check system resources (RAM, CPU)

**Models not appearing in dropdown**
- Pull models first: `ollama pull <model-name>`
- List installed models: `ollama list`
- Check `OLLAMA_MODELS` dictionary in `ollama_integration.py`

### Energy Tracking Issues

**"CodeCarbon not installed"**
- Install with: `pip install codecarbon>=2.4.1`
- Required for accurate energy measurements
- The system uses real hardware energy tracking, not estimates

**Energy measurements showing 0**
- Ensure CodeCarbon is properly installed
- Check that hardware sensors are accessible
- May need administrator privileges on some systems

### PySpark Issues

**"PySpark not found"**
- Install with: `pip install pyspark>=3.5.0 pyarrow>=14.0.1`
- Already included in requirements.txt
- Optional: PySpark is not required for basic functionality

**Java not found error**
- PySpark requires Java (JRE 8 or higher)
- Install Java from https://www.java.com/download/
- Set JAVA_HOME environment variable

**Memory errors during processing**
- Increase Spark memory in `data/spark_config.py`
- Use Pandas version for smaller datasets
- Close other applications to free memory

### Dashboard Issues

**Streamlit not starting**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Try: `streamlit run app.py --server.port 8501`

**Dashboard not accessible**
- Check if firewall blocks port 8501
- Try accessing http://localhost:8501
- Check Streamlit logs for errors

**Sidebar navigation not working**
- Clear browser cache
- Refresh the page
- Check browser console for JavaScript errors

### General Issues

**Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python path configuration

**Performance issues**
- Close other applications
- Use smaller Ollama models
- Disable PySpark if not needed
- Check system resources

**For more help:**
- Check documentation files in the project root
- Review inline code documentation
- Check logs in the Streamlit dashboard
- See specific documentation files for detailed guides
