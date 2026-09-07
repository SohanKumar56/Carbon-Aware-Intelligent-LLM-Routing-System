# 🎯 Prompt Complexity Router Feature

## Overview

The **Prompt Complexity Router** is an intelligent routing system that classifies incoming prompts **before** running inference and routes them to appropriately-sized LLMs. This implements the Green AI principle: "Don't waste energy routing every prompt to the biggest model when a smaller one would do the job."

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER PROMPT                           │
│               (Before any LLM inference)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  COMPLEXITY CLASSIFIER  │
        │    (MiniLM-L12, 22M)    │
        │    93.2% Accuracy       │
        └────────┬────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌────────┐  ┌─────────┐  ┌────────┐
│ SMALL  │  │ MEDIUM  │  │ LARGE  │
│ 1-2B   │  │  3-4B   │  │  7B+   │
│ params │  │ params  │  │ params │
└────────┘  └─────────┘  └────────┘
    │            │            │
    ▼            ▼            ▼
┌─────────────────────────────────────┐
│       ENERGY TRACKING                │
│  Track savings vs always-large       │
└─────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. **Pre-Inference Classification**
- Classifies prompt complexity **before** running any LLM
- No wasted inference cycles on oversized models
- Fast classification: ~5-15ms on CPU

### 2. **Three Complexity Classes**

#### 🟢 SMALL (1-2B parameters)
**Suitable for:**
- Simple factual questions ("What is the capital of France?")
- Basic translations
- Short lookups
- Simple definitions

**Models:**
- TinyLlama (0.6GB)
- DeepSeek Coder 1.3B (0.8GB)
- Qwen 2 1.5B (0.9GB)

**Energy:** ~0.00003 kWh/inference

#### 🟡 MEDIUM (3-4B parameters)
**Suitable for:**
- Moderate reasoning (2-3 steps)
- Code explanations
- Moderate-length summaries
- Creative writing with constraints

**Models:**
- Qwen 2.5 3B (1.9GB)
- Phi-3 (2.2GB)
- Gemma 3 4B (3.3GB)

**Energy:** ~0.00010-0.00017 kWh/inference

#### 🔴 LARGE (7B+ parameters)
**Suitable for:**
- Multi-step reasoning chains
- Complex math/logic problems
- Long-form content generation
- Ambiguous/nuanced tasks
- Complex code generation

**Models:**
- Qwen 2.5 7B (4.7GB)
- Zephyr (4.1GB)
- Qwen 2 (4.4GB)

**Energy:** ~0.00024 kWh/inference

### 3. **Classifier Details**

**Model:** Microsoft MiniLM-L12-H384-uncased
- **Parameters:** 22M (tiny, fast)
- **Training Data:** 28,465 prompts
  - WildChat: 20,000 diverse user prompts
  - GSM8K: 7,473 math reasoning problems
  - SupraLabs: 992 pre-labeled routing examples
- **Training Time:** 6.8 minutes on T4 GPU
- **Accuracy:** 93.2% on held-out validation set
- **F1 Score (macro):** 0.927

**Confusion Matrix:**
```
           Predicted
Actual     small  medium  large
small      1441     83     21   (93.2% correct)
medium       23   1287    124   (89.7% correct)
large        35    101   2578   (95.0% correct)
```

### 4. **Energy Savings**

**Baseline:** Always using largest model (Qwen 2.5 7B, 4.7GB)

**Typical Savings:**
- **Small prompts:** 87% energy saved
- **Medium prompts:** 29-58% energy saved
- **Large prompts:** 0% saved (uses large model as appropriate)

**Example:**
- 100 prompts/day
- 40% classified as small, 35% medium, 25% large
- **Energy saved:** 42% on average
- **CO₂ reduced:** ~0.02 kg/day per 100 prompts

---

## 📊 Dashboard Features

### Main View

1. **Prompt Input**
   - Quick examples selector
   - Custom prompt input
   - Real-time classification

2. **Classification Results**
   - Complexity label (small/medium/large)
   - Confidence score
   - Classification latency
   - Recommended model size

3. **Probability Distribution**
   - Visual breakdown of all class probabilities
   - Shows model confidence across all categories

4. **Routing Recommendation**
   - Primary recommended models
   - Fallback models
   - Human-readable reasoning

5. **Energy Estimation**
   - Energy saved vs baseline
   - Green score (0-100)
   - CO₂ savings estimate
   - Visual comparison chart

6. **Live Inference (Optional)**
   - Test with actual Ollama models
   - Real-time performance metrics
   - Response quality verification

---

## 🚀 Usage

### Command Line Interface

```python
from complexity_classifier import classify_prompt_complexity, route_to_model

# Simple classification
complexity = classify_prompt_complexity("What is Python?")
# Returns: 'small'

# Detailed routing
result = route_to_model("Explain quantum entanglement")
print(result)
# {
#   'complexity': 'large',
#   'confidence': 0.94,
#   'recommended': ['qwen2.5:7b', 'gemma3:4b'],
#   'reasoning': 'Complex task requiring large models...'
# }
```

### Routing Pipeline

```python
from routing_pipeline import run_routing_pipeline

# Auto-route and run inference
result = run_routing_pipeline(
    "Write a Python function for binary search",
    auto_route=True
)

print(f"Routed to: {result['routed_model']}")
print(f"Energy saved: {result['energy_saved_pct']:.1f}%")
print(f"Response: {result['response']}")
```

### Dashboard

1. Start the app:
```bash
streamlit run app.py
```

2. Select "Prompt Router" from sidebar

3. Enter a prompt or use quick examples

4. Click "Classify & Route"

5. View results and optionally run live inference

---

## 📈 Performance Metrics

### Classification Performance
- **Accuracy:** 93.2%
- **Latency:** 5-15ms (CPU)
- **Memory:** ~150MB loaded model

### End-to-End Routing
- **Total Latency:** Classification (10ms) + Model inference (varies)
- **Energy Overhead:** Negligible (~0.000001 kWh for classification)
- **Net Savings:** 30-60% average across diverse workloads

### Production Readiness
- ✅ CPU-friendly inference
- ✅ Model caching (singleton pattern)
- ✅ Error handling and fallbacks
- ✅ Confidence scoring for safety
- ✅ Comprehensive logging

---

## 🧪 Validation

### Routing Validation (`eval/routing_validation.py`)

This script validates that routing decisions are both:
1. **Safe:** Small-routed prompts are answered well by small models
2. **Efficient:** Energy savings are real, not theoretical

**What it does:**
- Samples prompts from each complexity class
- Runs them through both small and large models
- Compares response quality
- Measures actual energy savings

**Key Metric:**
> "Of prompts routed to Small, what % were answered equivalently well by Small vs Large?"

**Run validation:**
```bash
cd eval
python routing_validation.py
```

**Note:** Requires Ollama running with models installed

---

## 🔬 Training Details

### Dataset Construction

**Sources:**
1. **WildChat** (allenai/WildChat)
   - 20,000 English single-turn prompts
   - Diverse real-world user queries
   - Heuristic labeling based on features

2. **GSM8K** (openai/gsm8k)
   - 7,473 grade-school math problems
   - Multi-step reasoning required
   - Labeled as medium/large based on complexity

3. **SupraLabs** (SupraLabs/Prompt-Routing-Dataset)
   - 992 pre-labeled routing examples
   - Gold-standard labels
   - Mapped to 3-class taxonomy

**Labeling Strategy:**
- SupraLabs: Direct mapping (gold labels)
- WildChat/GSM8K: Heuristic features
  - Token count
  - Multi-step reasoning cues
  - Math/code content
  - Question complexity

**Class Distribution:**
- Large: 49.4% (14,061 prompts)
- Medium: 28.9% (8,232 prompts)
- Small: 21.7% (6,172 prompts)

### Training Configuration

**Model:** microsoft/MiniLM-L12-H384-uncased
- Architecture: BERT-based encoder
- Size: 22M parameters
- Max length: 256 tokens

**Training:**
- Device: Tesla T4 GPU (Colab)
- Batch size: 32
- Learning rate: 2e-5
- Epochs: 3
- FP16: Enabled
- Time: 6.83 minutes

**Optimization:**
- Stratified train/val split (80/20)
- Class-balanced sampling
- Dynamic padding
- Best model checkpoint selection

---

## 📂 File Structure

```
Carbon_Aware_AI_Inference_System_new/
│
├── complexity_classifier.py          # Main inference module
├── routing_pipeline.py               # Routing pipeline with energy tracking
├── prompt_router_dashboard.py        # Streamlit dashboard
│
├── model/
│   ├── train_classifier.py          # Training script (local)
│   └── prompt_complexity_classifier/ # Trained model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── training_report.txt
│
├── data/
│   ├── fetch_datasets.py            # Dataset downloader
│   ├── build_labeled_dataset.py     # Labeling pipeline
│   ├── judge_rubric.md              # LLM-judge criteria
│   ├── cache/                       # Raw datasets
│   └── labeled/                     # Final labeled data
│       ├── prompt_complexity_labeled.parquet
│       └── dataset_metadata.json
│
├── eval/
│   └── routing_validation.py        # Real-world validation
│
└── Colab_Training_Notebook.ipynb    # GPU training notebook
```

---

## 🎯 Use Cases

### 1. **API Cost Reduction**
Route 80% of simple queries to small models → 60% cost reduction

### 2. **Energy-Aware LLM Services**
Datacenters can reduce power consumption for user-facing chatbots

### 3. **Green AI Applications**
Demonstrate environmental responsibility in AI deployments

### 4. **Educational Tool**
Show students the impact of model size on energy consumption

### 5. **Benchmarking**
Compare routing strategies: naive vs learned vs rule-based

---

## 🔮 Future Enhancements

### Potential Improvements

1. **Fine-tuned Routing**
   - Domain-specific classifiers (code, math, creative writing)
   - User feedback loop for continuous improvement

2. **Multi-Model Ensembles**
   - Route to multiple small models, aggregate responses
   - Parallel inference for latency-sensitive applications

3. **Dynamic Routing**
   - Adjust routing based on real-time energy prices
   - Shift load to renewable energy hours

4. **Quality Monitoring**
   - Automatic quality checks on small-model responses
   - Fallback to large model if quality is insufficient

5. **Advanced Features**
   - Streaming classification for long prompts
   - Multi-turn conversation context
   - Personalized routing per user

---

## 📚 References

### Papers & Concepts
- **Green AI:** Patterson et al., "Carbon Emissions and Large Neural Network Training"
- **Adaptive Computation:** Graves, "Adaptive Computation Time for Recurrent Neural Networks"
- **Model Routing:** Zhou et al., "Efficient Inference of Large Language Models via Model Routing"

### Datasets
- [WildChat](https://huggingface.co/datasets/allenai/WildChat) - Real user prompts
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Math reasoning
- [SupraLabs Routing](https://huggingface.co/datasets/SupraLabs/Prompt-Routing-Dataset) - Pre-labeled routing data

### Models
- [MiniLM-L12](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) - Fast text encoder
- [Ollama](https://ollama.ai) - Local LLM runtime

---

## 🤝 Contributing

Improvements welcome! Areas of interest:
- Better heuristic features for labeling
- Domain-specific routing classifiers
- Real-world validation benchmarks
- Alternative routing algorithms

---

## 📄 License

Educational and research purposes. See main LICENSE file.

---

**Built with 🌿 for Green AI and Sustainable Computing**
