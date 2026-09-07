# 🌿 Carbon-Aware AI Inference System

A comprehensive AI system that demonstrates Green AI principles through three main features:
1. **Sentiment Analysis**: Adaptive three-stage pipeline with energy tracking
2. **Ollama Model Comparison**: Compare local LLMs side-by-side
3. **Prompt Complexity Router**: Intelligent routing to appropriately-sized models (93.2% accuracy)

The system tracks energy consumption, carbon emissions, and performance metrics while demonstrating how smart routing can save 30-60% energy without sacrificing quality.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dashboard 1: Sentiment Tracker](#dashboard-1-sentiment-tracker)
- [Dashboard 2: Ollama Compare](#dashboard-2-ollama-compare)
- [Dashboard 3: Prompt Router](#dashboard-3-prompt-router-new)
- [Energy & Carbon Calculations](#energy--carbon-calculations)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Examples](#examples)

---

## 🎯 Overview

This system demonstrates **Green AI** principles through three interactive dashboards:

### Dashboard 1: Sentiment Tracker
Traditional sentiment analysis (POSITIVE/NEGATIVE/NEUTRAL) using an adaptive three-stage pipeline:
- **Rule Engine** → **RoBERTa** → **BERT**
- Compares with local Ollama LLMs
- Tracks energy consumption and carbon emissions

### Dashboard 2: Ollama Compare  
Side-by-side comparison of multiple Ollama models:
- Compare up to 3 models simultaneously
- Performance, energy, and latency metrics
- Interactive visualizations

### Dashboard 3: Prompt Router (NEW! ✨)
Intelligent prompt complexity classification and routing:
- Classifies prompts as **SMALL** / **MEDIUM** / **LARGE**
- Routes to appropriately-sized models (1-2B / 3-4B / 7B+ parameters)
- **93.2% accuracy** with hybrid ML + heuristic rules
- **30-60% average energy savings** vs always using large models

The system tracks and compares:
- ⚡ Energy consumption (kWh)
- 🌍 Carbon emissions (kg CO₂)
- ⏱️ Inference latency (ms)
- 🎯 Prediction confidence (%)
- 💚 Green Score (energy efficiency)
- 📊 Overall Score (weighted performance)

---

## ✨ Key Features

### 1. **Three Interactive Dashboards**

#### Sentiment Tracker
- Adaptive three-stage pipeline (Rule Engine → RoBERTa → BERT)
- Compare with up to 3 Ollama models simultaneously
- Real-time energy and carbon tracking
- **60-99% energy savings** through intelligent escalation

#### Ollama Compare
- Side-by-side comparison of 11 Ollama models
- Support from TinyLlama (0.6GB) to Qwen 2.5 7B (4.7GB)
- Performance metrics and visualizations
- No hardcoded confidence values

#### Prompt Router (NEW! ✨)
- **Pre-inference complexity classification**
- Routes prompts BEFORE running any LLM
- **93.2% accuracy** (trained on 28,465 prompts)
- Hybrid ML + 7 heuristic rules for edge cases
- **30-60% average energy savings**
- Transparent rule override indicators

### 2. **Intelligent Routing**

**Philosophy**: Don't waste energy routing every prompt to the biggest model when a smaller one would do the job.

**How it works**:
```
User Prompt
    ↓
Complexity Classifier (MiniLM-L12, ~10ms)
    ↓
    ├─ SMALL (1-2B params)   → 87% energy saved
    ├─ MEDIUM (3-4B params)  → 50% energy saved
    └─ LARGE (7B+ params)    → Appropriate for complexity
```

**Examples**:
- "What is Python?" → SMALL → TinyLlama (0.6GB)
- "Write binary search function" → MEDIUM → Qwen 2.5 3B (1.9GB)
- "Prove √2 is irrational" → LARGE → Qwen 2.5 7B (4.7GB)

### 3. **Hybrid Classification Approach**

**ML Model** (93.2% baseline accuracy):
- Microsoft MiniLM-L12-H384-uncased (22M params)
- Trained on 28,465 prompts (WildChat, GSM8K, SupraLabs)
- Fast CPU inference (~10ms)

**7 Heuristic Rules** (catch edge cases):
1. Math proofs → LARGE
2. Very long prompts → upgrade appropriately
3. Complex reasoning indicators → LARGE
4. Multi-step tasks → detect and upgrade
5. Comprehensive code with requirements → LARGE
6. Research/academic tasks → upgrade
7. Keep simple prompts simple

**Result**: Better accuracy on edge cases while maintaining speed.

### 4. **Comprehensive Metrics**
- Energy consumption tracking
- Carbon footprint calculation
- Latency measurement
- Confidence scoring
- Green Score (0-100)
- Overall Score (weighted composite)

### 5. **Interactive Visualizations**
- Energy comparison bar charts
- Probability distribution charts
- Stage distribution pie charts
- Cumulative CO₂ timeline
- Green Score gauge
- Model-specific performance charts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌───────────────┐ ┌──────────────┐ ┌────────────────┐
│  SENTIMENT    │ │    OLLAMA    │ │  PROMPT        │
│  TRACKER      │ │   COMPARE    │ │  ROUTER        │
└───────┬───────┘ └──────┬───────┘ └────────┬───────┘
        │                │                   │
        ▼                ▼                   ▼
    3-Stage          Model           Complexity
    Pipeline       Comparison       Classification
        │                │                   │
        └────────────────┴───────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │    ENERGY TRACKING & ANALYSIS        │
        │  • kWh  • CO₂  • Green Score        │
        └─────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │      INTERACTIVE DASHBOARD           │
        │  • Metrics  • Charts  • Insights    │
        └─────────────────────────────────────┘
```

### Prompt Router Architecture

```
User Prompt (Before any LLM inference)
    ↓
Complexity Classifier (MiniLM-L12, 22M params)
    │
    ├─ ML Model Prediction
    │   └─ 93.2% accuracy
    │
    ├─ Heuristic Rules Check (7 rules)
    │   ├─ Math proofs → LARGE
    │   ├─ Long prompts → upgrade
    │   ├─ Complex reasoning → LARGE
    │   ├─ Multi-step tasks → upgrade
    │   ├─ Code with requirements → adjust
    │   ├─ Academic tasks → upgrade
    │   └─ Preserve simple prompts
    │
    └─ Final Classification
        ├─ SMALL (1-2B) → 87% energy saved
        ├─ MEDIUM (3-4B) → 50% energy saved
        └─ LARGE (7B+) → Appropriate for task
    ↓
Routing Recommendation
    ├─ Primary models
    ├─ Fallback models
    └─ Energy savings estimate
```

---

## 🎯 Dashboard 3: Prompt Router (NEW!)

### Overview

The **Prompt Complexity Router** classifies prompts **before** running any LLM inference and routes them to appropriately-sized models to minimize energy waste.

**Key Principle**: Don't waste energy routing every prompt to the biggest model when a smaller one would do the job.

### Three Complexity Classes

#### 🟢 SMALL (1-2B parameters)
**Best for**: Simple factual questions, basic translations, short lookups
- **Models**: TinyLlama (0.6GB), DeepSeek Coder 1.3B (0.8GB), Qwen 2 1.5B (0.9GB)
- **Energy**: ~0.00003 kWh/inference
- **Savings**: 87% vs always using large models
- **Example**: "What is the capital of France?"

#### 🟡 MEDIUM (3-4B parameters)
**Best for**: Moderate reasoning, code explanations, summaries
- **Models**: Qwen 2.5 3B (1.9GB), Phi-3 (2.2GB), Gemma 3 4B (3.3GB)
- **Energy**: ~0.00010-0.00017 kWh/inference
- **Savings**: 29-58% vs always using large models
- **Example**: "Write a Python function to implement binary search"

#### 🔴 LARGE (7B+ parameters)
**Best for**: Multi-step reasoning, complex math, intricate code
- **Models**: Qwen 2.5 7B (4.7GB), Zephyr (4.1GB), Qwen 2 (4.4GB)
- **Energy**: ~0.00024 kWh/inference
- **Savings**: 0-10% (appropriate for task complexity)
- **Example**: "Prove that the square root of 2 is irrational using proof by contradiction"

### Classifier Performance

**Model**: Microsoft MiniLM-L12-H384-uncased (22M parameters)

**Training**:
- **Dataset**: 28,465 prompts from 3 sources
  - WildChat: 20,000 diverse user prompts
  - GSM8K: 7,473 math reasoning problems
  - SupraLabs: 992 pre-labeled routing examples
- **Device**: Tesla T4 GPU (Google Colab)
- **Time**: 6.83 minutes
- **Batch size**: 32 with FP16 precision

**Results**:
```
Accuracy: 93.20%
F1 Score: 0.9271

Confusion Matrix:
           small  medium  large
small      1441     83     21   (93.2% correct)
medium       23   1287    124   (89.7% correct)
large        35    101   2578   (95.0% correct)
```

**Inference**: ~10ms on CPU (fast enough for real-time routing)

### Hybrid ML + Rules Approach

The classifier uses a **two-stage approach** for better accuracy:

#### Stage 1: ML Model
- MiniLM-L12 provides baseline classification
- 93.2% accuracy on diverse prompts
- Fast CPU inference

#### Stage 2: Heuristic Rules (7 rules)

1. **Math Proofs → LARGE**
   - Keywords: prove, proof, theorem, contradiction, induction
   - Math proofs always upgraded to LARGE

2. **Very Long Prompts**
   - >120 tokens → LARGE
   - >100 tokens + SMALL → LARGE
   - >80 tokens + SMALL → MEDIUM

3. **Complex Reasoning Indicators**
   - Keywords: explain in detail, comprehensive, step by step, analyze
   - 2+ signals → upgrade SMALL to LARGE
   - 3+ signals + MEDIUM → upgrade to LARGE

4. **Multi-Step Tasks**
   - Keywords: first, then, next, step 1, multiple
   - 3+ indicators → upgrade SMALL to MEDIUM

5. **Code with Requirements**
   - Detects: authentication, database, API, testing, deployment
   - 5+ tech requirements → LARGE
   - 3+ requirements → upgrade SMALL to LARGE/MEDIUM

6. **Research/Academic Tasks**
   - Keywords: research, analyze, compare and contrast, evaluate
   - Academic tasks + SMALL → MEDIUM

7. **Preserve Simple Prompts**
   - Questions like "What is...", "Define...", "Translate..." stay SMALL
   - Prevents false upgrades

**Transparency**:
- When rules trigger, reasoning shows: "🎯 Enhanced by heuristic rules for better accuracy"
- Confidence adjusted to ≤85% (honest about hybrid approach)
- `rule_override` flag available in API

### Dashboard Features

1. **Prompt Input**
   - Quick example selector
   - Custom prompt input
   - Real-time classification

2. **Classification Results**
   - Complexity label (SMALL/MEDIUM/LARGE)
   - Confidence score (0-100%)
   - Classification latency
   - Probability distribution chart

3. **Routing Recommendation**
   - Primary recommended models
   - Fallback models
   - Human-readable reasoning
   - Model size information

4. **Energy Estimation**
   - Energy saved vs baseline (always-large)
   - Green score (0-100)
   - CO₂ savings estimate
   - Visual comparison chart

5. **Live Inference (Optional)**
   - Test with real Ollama models
   - Real-time performance metrics
   - Response viewing

### Energy Savings Examples

**Small Prompt**: "What is Python?"
```
Classification: SMALL (98% confidence)
Recommended: TinyLlama (0.6GB)
Energy: 0.00003 kWh
Baseline (7B): 0.00024 kWh
Saved: 0.000203 kWh (87% saved)
Green Score: 87/100
```

**Medium Prompt**: "Write a Python function for binary search"
```
Classification: MEDIUM (90% confidence)
Recommended: Qwen 2.5 3B (1.9GB)
Energy: 0.00010 kWh
Baseline (7B): 0.00024 kWh
Saved: 0.000140 kWh (58% saved)
Green Score: 58/100
```

**Large Prompt**: "Prove √2 is irrational"
```
Classification: LARGE (85% confidence, rule override)
Recommended: Qwen 2.5 7B (4.7GB)
Energy: 0.00024 kWh
Baseline (7B): 0.00024 kWh
Saved: 0 kWh (0% saved - appropriate!)
Green Score: 0/100
```

### Usage Examples

**Command Line**:
```python
from complexity_classifier import classify_prompt_complexity, route_to_model

# Simple classification
complexity = classify_prompt_complexity("What is the capital of France?")
# Returns: 'small'

# Full routing decision
result = route_to_model("Explain quantum entanglement")
print(result)
# {
#   'complexity': 'large',
#   'confidence': 0.94,
#   'recommended': ['qwen2.5:7b', 'gemma3:4b'],
#   'reasoning': '...',
#   'rule_override': False
# }
```

**Dashboard**: Navigate to "Prompt Router" from sidebar, enter prompt, click "Classify & Route"

---

## 🌿 Dashboard 1: Sentiment Tracker

### Three-Stage Adaptive System

```
┌──────────────────┐
│   STAGE 1:       │
│  RULE ENGINE     │  ← Try first (near-zero energy)
│  (Keywords)      │
└────────┬─────────┘
         │ No match?
         ▼
┌──────────────────┐
│   STAGE 2:       │
│   RoBERTa        │  ← Try second (low energy)
│  (Small Model)   │
└────────┬─────────┘
         │ Low confidence?
         ▼
┌──────────────────┐
│   STAGE 3:       │
│     BERT         │  ← Final fallback (high energy)
│  (Large Model)   │
└──────────────────┘
```

### Stage 1: Rule Engine

**Purpose**: Catch obvious sentiments with zero ML overhead

**How it works**:
- Maintains lists of positive and negative keywords
- Performs simple keyword matching
- Returns immediately if match found

**Keywords**:
```python
Positive: ["excellent", "amazing", "love", "great", "fantastic", ...]
Negative: ["terrible", "awful", "hate", "worst", "horrible", ...]
```

**Energy**: 0.000001 kWh (essentially zero)

**When it triggers**:
- Text contains clear sentiment keywords
- Examples: "This is terrible!", "I love this product!"

**Confidence**: Always 100% (keyword match is certain)

### Stage 2: RoBERTa (Small Model)

**Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Purpose**: Handle most cases with low energy

**How it works**:
- Transformer-based model (125M parameters)
- Trained on Twitter data (good for informal text)
- Outputs: POSITIVE/NEGATIVE/NEUTRAL with confidence

**Energy**: 0.000120 kWh

**When it triggers**:
- Rule engine found no keywords
- Text requires ML understanding

**Confidence threshold**: 80%
- If confidence ≥ 80% → Accept result, stop here
- If confidence < 80% → Escalate to Stage 3

### Stage 3: BERT (Large Model)

**Model**: `nlptown/bert-base-multilingual-uncased-sentiment`

**Purpose**: Final fallback for complex cases

**How it works**:
- Larger transformer model (110M parameters)
- Trained on product reviews (5-star ratings)
- Outputs: 1-5 stars (mapped to sentiment)

**Mapping**:
```
1 STAR  → NEGATIVE
2 STARS → NEGATIVE
3 STARS → NEUTRAL
4 STARS → POSITIVE
5 STARS → POSITIVE
```

**Energy**: 0.000320 kWh (baseline for comparison)

**When it triggers**:
- RoBERTa confidence < 80%
- Complex or ambiguous text

**Confidence threshold**: 50% (always accepted as final answer)

### Energy Savings Calculation

```
Saved Energy = BERT Energy - Actual Energy Used
Saved % = (Saved Energy / BERT Energy) × 100

Example:
- Rule Engine used: 0.000001 kWh
- BERT baseline: 0.000320 kWh
- Saved: 0.000319 kWh
- Saved %: 99.7%
```

---

## 🤖 Dashboard 2: Ollama Compare

### Supported Models

| Model | Size | Timeout | Best For |
|-------|------|---------|----------|
| TinyLlama | 0.637 GB | 30s | Speed testing |
| DeepSeek Coder 1.3B | 0.776 GB | 40s | Fast inference |
| Qwen 2 1.5B | 0.934 GB | 45s | Balanced |
| DeepSeek R1 1.5B | 1.1 GB | 45s | Fast & accurate |
| Qwen 2.5 3B | 1.9 GB | 60s | Good balance |
| Phi-3 | 2.2 GB | 60s | Quality results |
| Gemma 3 4B | 3.3 GB | 120s | High quality |
| Zephyr | 4.1 GB | 150s | Complex text |
| Qwen 2 | 4.4 GB | 150s | Advanced |
| Qwen 2.5 7B | 4.7 GB | 180s | Best quality |
| Qwen 2.5 Coder | 4.7 GB | 180s | Code-focused |

### How Ollama Models Work

#### 1. Prompt Engineering

The system sends a structured prompt to Ollama models:

```
You must analyze sentiment and provide ONLY these two lines:

Sentiment: POSITIVE
Confidence: 85

Or:

Sentiment: NEGATIVE
Confidence: 75

Or:

Sentiment: NEUTRAL
Confidence: 60

Now analyze this text: "{user_input}"

Remember: Output ONLY two lines in the exact format shown above. No explanations.
```

#### 2. Response Parsing

The system extracts sentiment and confidence using multiple patterns:

**Pattern 1**: `Confidence: 85`
```python
regex: r'confidence[:\s]+(\d+\.?\d*)'
```

**Pattern 2**: `85%`
```python
regex: r'(\d+\.?\d*)%'
```

**Pattern 3**: Standalone numbers
```python
regex: r'\b(\d+\.?\d*)\b'
# Validates: 0 ≤ number ≤ 100
```

#### 3. Confidence Validation

**CRITICAL**: No hardcoded fallback values!

```python
if label is None or confidence is None:
    return ERROR  # Model didn't follow format
else:
    return (label, confidence, success=True)
```

If a model doesn't provide confidence, it shows as **ERROR** rather than using fake values.

#### 4. Energy Estimation

Ollama model energy is estimated based on model size:

```
Energy (kWh) = Model Size (GB) × 0.00005

Examples:
- TinyLlama (0.637 GB): 0.000032 kWh
- Gemma 3 4B (3.3 GB): 0.000165 kWh
- Qwen 2.5 7B (4.7 GB): 0.000235 kWh
```

**Note**: This is an estimate. Actual energy depends on hardware and inference time.

---

## 📊 Model Comparison

### Comparison Table

The system displays a comprehensive comparison table with these metrics:

| Metric | Description | Source |
|--------|-------------|--------|
| **Model** | Model name and type | System |
| **Label** | POSITIVE/NEGATIVE/NEUTRAL | Model output |
| **Confidence** | 0-100% certainty | Model-generated |
| **Latency (ms)** | Inference time | Measured |
| **Energy (kWh)** | Power consumption | Calculated |
| **CO₂ (kg)** | Carbon emissions | Calculated |
| **Green Score** | Energy efficiency (0-100) | Calculated |
| **Overall Score** | Weighted composite (0-100) | Calculated |

### Interactive Selection

Users can click on any row in the comparison table to:
- View detailed metrics for that model
- See model-specific visualizations
- Compare energy consumption
- Analyze performance characteristics

---

## ⚡ Energy & Carbon Calculations

### Energy Consumption

#### Traditional Models

**Measured empirically** on CPU inference:

```
Rule Engine:  0.000001 kWh  (1 µWh)
RoBERTa:      0.000120 kWh  (120 µWh)
BERT:         0.000320 kWh  (320 µWh)
```

#### Ollama Models

**Estimated** based on model size:

```
Energy (kWh) = Model Size (GB) × 0.00005 kWh/GB

Example (Gemma 3 4B):
Energy = 3.3 GB × 0.00005 = 0.000165 kWh
```

### Carbon Emissions

**Formula**:
```
CO₂ (kg) = Energy (kWh) × CO₂ Intensity (kg/kWh)

Where:
CO₂ Intensity = 0.475 kg/kWh (IEA world average 2023)
```

**Example**:
```
BERT Energy: 0.000320 kWh
CO₂ = 0.000320 × 0.475 = 0.000152 kg (0.152 g)
```

### Green Score

**Purpose**: Measure energy efficiency compared to BERT baseline

**Formula**:
```
Green Score = max(0, 100 × (1 - Model Energy / BERT Energy))

Where:
BERT Energy = 0.000320 kWh (baseline)
```

**Examples**:

```
Rule Engine:
Green Score = 100 × (1 - 0.000001 / 0.000320)
            = 100 × (1 - 0.003125)
            = 100 × 0.996875
            = 99.7 ≈ 100

RoBERTa:
Green Score = 100 × (1 - 0.000120 / 0.000320)
            = 100 × (1 - 0.375)
            = 100 × 0.625
            = 62.5 ≈ 62

BERT:
Green Score = 100 × (1 - 0.000320 / 0.000320)
            = 100 × 0
            = 0

TinyLlama (0.000032 kWh):
Green Score = 100 × (1 - 0.000032 / 0.000320)
            = 100 × (1 - 0.1)
            = 100 × 0.9
            = 90
```

**Interpretation**:
- **100**: Uses almost no energy (99%+ savings)
- **75**: Uses 75% less energy than BERT
- **50**: Uses 50% less energy than BERT
- **0**: Uses same or more energy than BERT

---

## 🎯 Scoring System

### Overall Score

**Purpose**: Composite metric combining multiple factors

**Formula**:
```
Overall Score = (Confidence × 40%) + (Energy × 30%) + (Speed × 20%) + (Carbon × 10%)
```

**Component Calculations**:

#### 1. Confidence Score (40%)
```
Confidence Score = Model Confidence × 40

Example:
Confidence = 0.85 (85%)
Confidence Score = 0.85 × 40 = 34 points
```

#### 2. Energy Score (30%)
```
Energy Score = (1 - min(Model Energy / BERT Energy, 1.0)) × 30

Example:
Model Energy = 0.000120 kWh
BERT Energy = 0.000320 kWh
Energy Score = (1 - 0.000120/0.000320) × 30
             = (1 - 0.375) × 30
             = 0.625 × 30
             = 18.75 points
```

#### 3. Speed Score (20%)
```
Speed Score = (1 - min(Latency / Max Latency, 1.0)) × 20

Where:
Max Latency = 5000 ms (baseline)

Example:
Latency = 1000 ms
Speed Score = (1 - 1000/5000) × 20
            = (1 - 0.2) × 20
            = 0.8 × 20
            = 16 points
```

#### 4. Carbon Score (10%)
```
Carbon Score = (1 - min(Model CO₂ / BERT CO₂, 1.0)) × 10

Where:
BERT CO₂ = 0.000152 kg (baseline)

Example:
Model CO₂ = 0.000057 kg
Carbon Score = (1 - 0.000057/0.000152) × 10
             = (1 - 0.375) × 10
             = 0.625 × 10
             = 6.25 points
```

### Complete Example

**RoBERTa Model**:
```
Confidence: 85% → 34 points
Energy: 0.000120 kWh → 18.75 points
Latency: 453 ms → 18.19 points
CO₂: 0.000057 kg → 6.25 points

Overall Score = 34 + 18.75 + 18.19 + 6.25 = 77.19 / 100
```

### Best Model Selection

The system automatically identifies the best model:

```python
Best Model = max(all_models, key=lambda x: x.overall_score)
```

**Reasoning includes**:
- "high confidence" (≥85%)
- "very low energy" (<0.0001 kWh)
- "low energy" (<0.0002 kWh)
- "fast response" (<500 ms)
- "best overall balance" (default)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Ollama (for LLM comparison)
- 4GB+ RAM recommended

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Carbon-Aware-AI-Inference-system
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

**Activate**:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies**:
```
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
plotly>=5.14.0
pandas>=2.0.0
requests>=2.31.0
```

### Step 4: Install Ollama (Optional)

**For model comparison features**:

1. Download from: https://ollama.ai
2. Install Ollama
3. Pull models:
```bash
ollama pull tinyllama
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

### Step 5: Run Application

```bash
streamlit run app.py
```

Or use the wrapper:
```bash
python run_app.py
```

**Access**: http://localhost:8502

---

## 🛠️ Tech Stack

### Core Technologies

#### Frontend & Dashboard
- **Streamlit** (1.28.0+)
  - Interactive web dashboard
  - Real-time visualizations
  - Session state management
  - Component-based UI

#### Machine Learning & NLP
- **HuggingFace Transformers** (4.30.0+)
  - Pre-trained sentiment models
  - Pipeline API for inference
  - Model caching and optimization
  
- **PyTorch** (2.0.0+)
  - Deep learning backend
  - CPU inference support
  - Model loading and execution

#### Data Visualization
- **Plotly** (5.14.0+)
  - Interactive charts and graphs
  - Gauge charts for metrics
  - Timeline visualizations
  - Bar and pie charts

#### Data Processing
- **Pandas** (2.0.0+)
  - DataFrame operations
  - Data aggregation
  - Metric calculations

#### API & Integration
- **Requests** (2.31.0+)
  - Ollama API communication
  - HTTP request handling
  - Timeout management

### Models & Services

#### Traditional ML Models
- **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
  - 125M parameters
  - Twitter-trained sentiment analysis
  - 3-class classification (POS/NEG/NEU)

- **BERT** (`nlptown/bert-base-multilingual-uncased-sentiment`)
  - 110M parameters
  - Multilingual support
  - 5-star rating system

#### LLM Integration
- **Ollama** (Local LLM Runtime)
  - 11 supported models (0.6GB - 4.7GB)
  - Local inference (no API costs)
  - Model management and caching

### Architecture Patterns

#### Design Patterns
- **Adaptive Pipeline**: Three-stage escalation strategy
- **Lazy Loading**: Models loaded on-demand and cached
- **Module-level Caching**: Singleton pattern for model instances
- **Strategy Pattern**: Different inference strategies per stage

#### Performance Optimizations
- **Model Caching**: Load once, reuse across requests
- **CPU Inference**: No GPU required (accessible)
- **Lazy Evaluation**: Only run models when needed
- **Timeout Management**: Model-specific timeout handling

### Development Tools

#### Python Environment
- **Python** 3.8+
- **Virtual Environment** (.venv)
- **pip** for dependency management

#### Code Organization
- **Modular Architecture**: Separation of concerns
- **Type Hints**: Python type annotations
- **Logging**: Structured logging with Python logging module
- **Error Handling**: Comprehensive exception management

### Deployment

#### Local Deployment
- **Streamlit Server**: Built-in development server
- **Port**: 8502 (default)
- **Hot Reload**: Automatic code reloading

#### Requirements
- **RAM**: 4GB+ recommended
- **Storage**: 2GB+ for models
- **CPU**: Multi-core recommended
- **OS**: Windows/Linux/macOS

---

## 📖 Usage

### Basic Workflow

#### Dashboard Selection
After starting the app, select from the sidebar:
1. **Sentiment Tracker** - Traditional pipeline + Ollama comparison
2. **Ollama Compare** - Side-by-side model comparison
3. **Prompt Router** - Complexity classification and routing

#### Sentiment Tracker Workflow
1. **Enter Text**: Type or select example text
2. **Select Ollama Models** (optional): Choose up to 3 models
3. **Run Inference**: Click "⚡ Run Inference"
4. **View Results**: See predictions, confidence, and metrics
5. **Compare Models**: Click rows in comparison table
6. **Analyze Visualizations**: Review charts and graphs

#### Prompt Router Workflow
1. **Enter Prompt**: Type or select example
2. **Classify**: Click "🔍 Classify & Route"
3. **View Classification**: See complexity (SMALL/MEDIUM/LARGE)
4. **Check Recommendations**: View suggested models
5. **See Energy Savings**: Compare vs always-large baseline
6. **Run Live Inference** (optional): Test with Ollama

### Example Prompts

#### Sentiment Analysis (Dashboard 1)
```
"This product is amazing and I love it!"
Expected: POSITIVE (Rule Engine, 100%, 0 ms)

"Setup was complicated, documentation is lacking, and support never 
responded. The core functionality works, but I'm actively looking 
for alternatives."
Expected: NEGATIVE (RoBERTa/BERT, 85-92%, ~450 ms)
```

#### Prompt Router (Dashboard 3)
```
SMALL: "What is the capital of France?"
→ TinyLlama (0.6GB), 87% energy saved

MEDIUM: "Write a Python function to implement binary search"
→ Qwen 2.5 3B (1.9GB), 58% energy saved

LARGE: "Prove that the square root of 2 is irrational using proof by contradiction"
→ Qwen 2.5 7B (4.7GB), 0% saved (appropriate for complexity)
```

---

## 📁 Project Structure

```
Carbon-Aware-AI-Inference-system/
├── app.py                          # Main Streamlit application (3 dashboards)
├── config.py                       # Configuration and constants
│
├── SENTIMENT TRACKER (Dashboard 1)
├── inference_pipeline.py           # Three-stage adaptive pipeline
├── model_loader.py                 # Model loading and caching
├── rule_engine.py                  # Keyword-based sentiment detection
│
├── OLLAMA COMPARE (Dashboard 2)
├── ollama_integration.py           # Ollama API integration
├── model_comparison.py             # Model comparison logic
├── ollama_dashboard.py             # Ollama-specific dashboard
├── ollama_service.py               # Ollama service utilities
├── test_ollama.py                  # Ollama testing script
│
├── PROMPT ROUTER (Dashboard 3 - NEW!)
├── complexity_classifier.py        # ML + heuristic classification
├── routing_pipeline.py             # Routing with energy tracking
├── prompt_router_dashboard.py      # Prompt router dashboard UI
├── test_improvements.py            # Classifier testing script
│
├── MODEL & DATA
├── model/
│   ├── train_classifier.py        # Training script
│   └── prompt_complexity_classifier/
│       ├── config.json             # Model configuration
│       ├── model.safetensors       # Trained model (133MB)
│       ├── tokenizer files         # Tokenizer config
│       └── training_report.txt    # Training metrics
├── data/
│   ├── fetch_datasets.py          # Dataset downloader
│   ├── build_labeled_dataset.py   # Labeling pipeline
│   ├── cache/                     # Raw datasets (28k prompts)
│   └── labeled/                   # Final labeled data
├── eval/
│   └── routing_validation.py      # Real-world validation
│
├── SHARED UTILITIES
├── energy_tracker.py              # Energy and carbon tracking
├── dashboard_utils.py             # Visualization utilities
│
├── CONFIGURATION
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── run_app.py                     # Application wrapper
└── .venv/                        # Virtual environment
```

---

## 🔧 Technical Details

### Confidence Scoring

#### Traditional Models (RoBERTa/BERT)

**Source**: Neural network softmax output

```python
# Model returns probabilities for each class
output = model(text)
# Example: {'label': 'POSITIVE', 'score': 0.923}

confidence = output['score']  # Real probability from softmax
```

**Characteristics**:
- ✅ Mathematical probability (0.0-1.0)
- ✅ Varies with input
- ✅ Reflects model certainty
- ✅ All probabilities sum to 1.0

#### Ollama Models (LLMs)

**Source**: Model self-assessment

```python
# Model is asked to provide confidence
prompt = "Analyze sentiment and provide confidence (0-100)"

# Model responds with its own confidence estimate
response = "Sentiment: POSITIVE\nConfidence: 87"

confidence = parse_confidence(response)  # 0.87
```

**Characteristics**:
- ✅ Self-assessed certainty
- ✅ Varies with input
- ✅ No hardcoded fallbacks
- ⚠️ Not a mathematical probability

**Validation**:
```python
if confidence is None:
    return ERROR  # No fake values!
```

### Latency Measurement

**Method**: Wall-clock time measurement

```python
import time

t0 = time.perf_counter()
result = run_inference(model, text)
latency_ms = (time.perf_counter() - t0) * 1000
```

**What it includes**:
- Model loading (if not cached)
- Tokenization
- Inference computation
- Post-processing
- Network overhead (for Ollama)

**System-specific**: Results vary based on:
- CPU/GPU performance
- Available RAM
- System load
- Model caching state

### Model Caching

**Traditional Models**:
```python
_cache = {}  # Module-level cache

def load_model(model_name):
    if model_name not in _cache:
        _cache[model_name] = pipeline(model=model_name)
    return _cache[model_name]
```

**Benefits**:
- First inference: Slow (model loading)
- Subsequent inferences: Fast (cached)
- Memory efficient (load once, reuse)

### Error Handling

**Ollama Connection Errors**:
```python
try:
    response = requests.post(OLLAMA_API_URL, ...)
except requests.exceptions.ConnectionError:
    return {"error": "Cannot connect to Ollama. Is it running?"}
except requests.exceptions.Timeout:
    return {"error": "Request timeout"}
```

**Confidence Parsing Failures**:
```python
label, confidence, success = parse_sentiment(response)
if not success:
    return {"error": "Model did not provide confidence in expected format"}
```

---

## 📊 Examples

### Example 1: Rule Engine Success

**Input**: "This product is terrible!"

**Result**:
```
Traditional Pipeline:
├─ Stage: Rule Engine
├─ Label: NEGATIVE
├─ Confidence: 100%
├─ Latency: 0.0 ms
├─ Energy: 0.000001 kWh
├─ CO₂: 0.000000 kg
├─ Green Score: 100
└─ Matched Keywords: ["terrible"]
```

**Why Rule Engine**:
- Keyword "terrible" found in negative list
- Instant match, no ML needed
- Maximum energy savings (99.7%)

### Example 2: RoBERTa Success

**Input**: "The package arrived on time and everything was in order."

**Result**:
```
Traditional Pipeline:
├─ Stage: RoBERTa
├─ Label: POSITIVE
├─ Confidence: 87.3%
├─ Latency: 453 ms
├─ Energy: 0.000120 kWh
├─ CO₂: 0.000057 kg
└─ Green Score: 62
```

**Why RoBERTa**:
- No obvious keywords (Rule Engine skipped)
- RoBERTa confidence 87.3% > 80% threshold
- No need for BERT (energy saved)

### Example 3: BERT Fallback

**Input**: "It's okay, I guess. Not great, not terrible."

**Result**:
```
Traditional Pipeline:
├─ Stage: BERT
├─ Label: NEUTRAL
├─ Confidence: 68.2%
├─ Latency: 892 ms
├─ Energy: 0.000320 kWh
├─ CO₂: 0.000152 kg
└─ Green Score: 0
```

**Why BERT**:
- No keywords (Rule Engine skipped)
- RoBERTa confidence 65% < 80% threshold
- Ambiguous text requires larger model

### Example 4: Model Comparison

**Input**: "I've been using this for 6 months. Setup was complicated, documentation is lacking, support never responded. Core functionality works, but I'm actively looking for alternatives."

**Results**:
```
┌─────────────────────┬──────────┬────────┬─────────┬────────┬───────┐
│ Model               │ Label    │ Conf   │ Latency │ Energy │ Score │
├─────────────────────┼──────────┼────────┼─────────┼────────┼───────┤
│ Traditional (BERT)  │ NEGATIVE │ 92.3%  │ 892 ms  │ 0.320  │ 71.2  │
│ Ollama (Qwen 2.5)   │ NEGATIVE │ 85.0%  │ 63354ms │ 0.235  │ 52.8  │
│ Ollama (DeepSeek)   │ NEGATIVE │ 88.0%  │ 38097ms │ 0.055  │ 68.4  │
└─────────────────────┴──────────┴────────┴─────────┴────────┴───────┘

Best Model: Traditional (BERT)
Reason: high confidence, best overall balance
```

**Analysis**:
- All models correctly identified NEGATIVE sentiment
- BERT: Highest confidence, but most energy
- DeepSeek: Best energy efficiency, good confidence
- Qwen 2.5: Slowest, but still accurate

---

## 🎓 Key Learnings

### 1. Energy Efficiency Matters
- Rule Engine saves 99.7% energy when applicable
- Adaptive pipeline reduces average energy by 60-80%
- Small models (RoBERTa) handle 70% of cases efficiently

### 2. Confidence is Critical
- High confidence (>90%) = trustworthy prediction
- Low confidence (<70%) = needs human review
- Model-generated confidence > hardcoded values

### 3. Model Selection Trade-offs
- **Speed**: TinyLlama (fast) vs Qwen 7B (slow)
- **Accuracy**: Larger models better for complex text
- **Energy**: Smaller models more efficient
- **Sarcasm**: Requires larger models (4B+)

### 4. Green AI Principles
- Start with simplest solution (Rule Engine)
- Escalate only when necessary (Adaptive Pipeline)
- Measure and optimize (Energy Tracking)
- Compare alternatives (Model Comparison)

---

## 🔮 Future Enhancements

### Potential Improvements

1. **Parallel Ollama Execution**
   - Run multiple models simultaneously
   - Reduce total inference time
   - Requires careful resource management

2. **GPU Support**
   - Faster inference for traditional models
   - Lower latency for large models
   - More accurate energy measurement

3. **Custom Model Training**
   - Fine-tune on domain-specific data
   - Improve accuracy for specific use cases
   - Optimize for energy efficiency

4. **Real-time Energy Monitoring**
   - Hardware-level power measurement
   - More accurate energy tracking
   - Per-component breakdown

5. **Advanced Visualizations**
   - Model performance over time
   - Energy consumption trends
   - Confidence distribution analysis

6. **API Endpoint**
   - REST API for programmatic access
   - Batch processing support
   - Integration with other systems

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **HuggingFace Transformers**: Pre-trained models
- **Ollama**: Local LLM infrastructure
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Visualization library
- **CodeCarbon**: Energy tracking inspiration

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on the repository.

---

**Built with 🌿 for Green AI and Sustainable Computing**
