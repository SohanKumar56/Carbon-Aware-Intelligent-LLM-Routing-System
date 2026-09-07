# 🚀 Quick Start Guide - Carbon-Aware AI Inference System

## ✅ Project Status: **COMPLETE & READY TO USE**

All implementation tasks have been successfully completed. You now have a fully functional Carbon-Aware AI Inference System with three dashboards:

1. **Sentiment Tracker** - Traditional 3-stage adaptive pipeline
2. **Ollama Compare** - Compare multiple LLM models
3. **Prompt Router** - NEW! Intelligent prompt complexity routing (93.2% accuracy)

---

## 🎯 What's Been Built

### The Prompt Complexity Router Feature

**The Problem**: Running every prompt through the largest LLM model wastes 30-60% of energy.

**The Solution**: Classify prompt complexity BEFORE inference and route to appropriately-sized models.

**The Results**:
- ⚡ **93.2% Classification Accuracy**
- 🌿 **30-60% Average Energy Savings**
- ⏱️ **~10ms Classification Time** (CPU)
- 🎯 **Three Complexity Classes**: Small (1-2B params), Medium (3-4B params), Large (7B+ params)

---

## 🚀 How to Run

### 1. Start the Dashboard

```bash
streamlit run app.py
```

### 2. Navigate to Prompt Router

In the sidebar, select:
- **"Prompt Router"** from the navigation menu

### 3. Test the Classifier

**Option A: Use Quick Examples**
- Select from dropdown: "Simple Fact", "Code Generation", "Complex Reasoning", etc.
- Click "🔍 Classify & Route"

**Option B: Enter Custom Prompt**
- Select "Custom" from dropdown
- Type your own prompt
- Click "🔍 Classify & Route"

### 4. View Results

You'll see:
- **Complexity Classification**: Small/Medium/Large with confidence score
- **Routing Recommendation**: Which models to use
- **Energy Estimation**: How much energy you'll save vs. always-large baseline
- **Green Score**: Overall efficiency rating (0-100)
- **Probability Distribution**: Visual breakdown of classification confidence

### 5. (Optional) Run Live Inference

If you have Ollama running:
- Click "Run Inference on Recommended Model"
- See actual response and real-time metrics

---

## 📊 Understanding the Results

### Complexity Classes

#### 🟢 SMALL (1-2B parameters)
**Best for:**
- Simple factual questions ("What is the capital of France?")
- Basic translations
- Short lookups
- Simple definitions

**Energy Usage**: ~0.00003 kWh/inference  
**Example Models**: TinyLlama, Qwen 2 1.5B

#### 🟡 MEDIUM (3-4B parameters)
**Best for:**
- Moderate reasoning (2-3 steps)
- Code explanations
- Moderate-length summaries
- Creative writing with constraints

**Energy Usage**: ~0.00010-0.00017 kWh/inference  
**Example Models**: Qwen 2.5 3B, Phi-3

#### 🔴 LARGE (7B+ parameters)
**Best for:**
- Multi-step reasoning chains
- Complex math/logic problems
- Long-form content generation
- Complex code generation

**Energy Usage**: ~0.00024 kWh/inference  
**Example Models**: Qwen 2.5 7B, Gemma 3 4B

---

## 📈 Performance Metrics

### Classifier Performance
```
Accuracy:       93.20%
F1 Score:       0.9271
Training Time:  6.83 minutes (T4 GPU)
Inference Time: ~10ms (CPU)
Model Size:     22M parameters (~90MB)
```

### Confusion Matrix
```
           Predicted
Actual     small  medium  large
small      1441     83     21   (93.2% correct)
medium       23   1287    124   (89.7% correct)
large        35    101   2578   (95.0% correct)
```

### Energy Savings
- **Small prompts**: 87% energy saved vs. always-large
- **Medium prompts**: 29-58% energy saved
- **Large prompts**: 0% saved (uses large as appropriate)
- **Average**: 30-60% across diverse workloads

---

## 🗂️ Project Structure

```
Carbon_Aware_AI_Inference_System_new/
│
├── app.py                             ← Main dashboard (updated with Router tab)
├── complexity_classifier.py           ← Main classification API
├── routing_pipeline.py                ← Routing with energy tracking
├── prompt_router_dashboard.py         ← Streamlit Router UI
│
├── model/
│   ├── train_classifier.py           ← Training script
│   └── prompt_complexity_classifier/  ← Trained model (93.2% accuracy) ✅
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── training_report.txt
│
├── data/
│   ├── fetch_datasets.py             ← Dataset downloader
│   ├── build_labeled_dataset.py      ← Labeling pipeline
│   ├── judge_rubric.md               ← Classification criteria
│   ├── cache/                        ← Raw datasets (28,465 prompts)
│   └── labeled/                      ← Final labeled data
│       ├── prompt_complexity_labeled.parquet
│       └── dataset_metadata.json
│
├── eval/
│   └── routing_validation.py         ← Real-world validation
│
├── Colab_Training_Notebook.ipynb     ← GPU training notebook
├── colab_train_classifier.py         ← Colab training script
│
└── DOCUMENTATION/
    ├── PROMPT_ROUTER_FEATURE.md      ← Complete feature docs
    ├── IMPLEMENTATION_COMPLETE.md    ← Full implementation summary
    ├── COLAB_TRAINING_INSTRUCTIONS.md ← Training guide
    └── QUICK_START_GUIDE.md          ← This file
```

---

## 🧪 Testing the System

### Quick Test (Command Line)
```bash
python complexity_classifier.py
```
This runs built-in tests with 6 example prompts.

### Full Validation (Requires Ollama)
```bash
cd eval
python routing_validation.py
```
Validates routing decisions with real Ollama models.

### Dashboard Test
```bash
streamlit run prompt_router_dashboard.py
```
Runs the router dashboard in standalone mode.

---

## 💡 Example Use Cases

### 1. API Cost Reduction
Route 80% of simple queries to small models → **60% cost reduction**

### 2. Energy-Aware LLM Services
Datacenters can reduce power consumption for user-facing chatbots

### 3. Green AI Applications
Demonstrate environmental responsibility in AI deployments

### 4. Educational Tool
Show students the impact of model size on energy consumption

---

## 🔍 How It Works

```
User Prompt
     ↓
Complexity Classifier (MiniLM-L12, 22M params)
     ↓
     ├── SMALL  → Route to 1-2B models (87% energy saved)
     ├── MEDIUM → Route to 3-4B models (29-58% saved)
     └── LARGE  → Route to 7B+ models (appropriate for task)
     ↓
Run Inference + Track Energy Savings
     ↓
Display Results + Green Score
```

---

## 📚 Documentation Files

1. **PROMPT_ROUTER_FEATURE.md** - Complete feature documentation
2. **IMPLEMENTATION_COMPLETE.md** - Full implementation summary
3. **COLAB_TRAINING_INSTRUCTIONS.md** - How to retrain the model on GPU
4. **data/judge_rubric.md** - Classification criteria used for labeling
5. **model/prompt_complexity_classifier/training_report.txt** - Training metrics

---

## 🎓 Training Details

### Dataset
- **Total Prompts**: 28,465
- **Sources**: 
  - WildChat: 20,000 diverse user prompts
  - GSM8K: 7,473 math reasoning problems
  - SupraLabs: 992 pre-labeled routing examples
  
- **Distribution**:
  - Large: 49.4% (14,061)
  - Medium: 28.9% (8,232)
  - Small: 21.7% (6,172)

### Model
- **Base**: Microsoft MiniLM-L12-H384-uncased
- **Parameters**: 22M (tiny, fast)
- **Training Device**: Tesla T4 GPU (Google Colab)
- **Training Time**: 6.83 minutes
- **Batch Size**: 32
- **FP16**: Enabled

---

## 🔮 Future Enhancements (Optional)

1. **Domain-Specific Classifiers**
   - Separate classifiers for code, math, creative writing
   
2. **Quality Monitoring**
   - Automatic quality checks on responses
   - Fallback if quality insufficient

3. **Advanced Features**
   - Streaming classification
   - Multi-turn conversation context
   - Personalized routing per user

4. **Optimization**
   - Model quantization (reduce size)
   - ONNX export (platform-agnostic)

---

## 🐛 Troubleshooting

### "Model not found" error
Make sure the model exists at:
```
model/prompt_complexity_classifier/
```

### Slow inference
First load takes ~1-2 seconds (model loading). Subsequent calls are ~10ms due to caching.

### Ollama not available
The classifier works standalone. Ollama is only needed for live inference testing.

---

## 🌿 Key Achievements

✅ **93.2% Accuracy** - High-quality classification  
✅ **Fast Inference** - ~10ms on CPU  
✅ **Energy Efficient** - 30-60% average savings  
✅ **Production Ready** - Cached, error-handled, well-documented  
✅ **Fully Integrated** - Seamlessly added to existing dashboard  
✅ **Validated** - Real-world validation script included  
✅ **Reproducible** - Training notebook + scripts provided  

---

## 📧 Need Help?

Check these files:
- `PROMPT_ROUTER_FEATURE.md` - Complete feature guide
- `IMPLEMENTATION_COMPLETE.md` - Implementation details
- Code docstrings - Inline documentation
- Training report - `model/prompt_complexity_classifier/training_report.txt`

---

## 🎉 Ready to Use!

The Prompt Complexity Router is **complete and production-ready**!

### Quick Start:
1. Run `streamlit run app.py`
2. Select "Prompt Router" from sidebar
3. Enter a prompt or use quick examples
4. Click "Classify & Route"
5. See complexity, recommendations, and energy savings!

---

**Built with 🌿 for Green AI and Sustainable Computing**

*Last Updated: September 7, 2026*
