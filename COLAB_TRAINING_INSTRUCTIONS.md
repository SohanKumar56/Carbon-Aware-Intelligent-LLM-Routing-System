# 🚀 Colab Training Instructions

The CPU training was taking **~22 hours**. Let's use Colab's free T4 GPU instead — it'll finish in **5-10 minutes**.

---

## Quick Start (3 Easy Steps)

### Step 1: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Upload notebook**
3. Upload `Colab_Training_Notebook.ipynb` from this directory

### Step 2: Enable GPU

1. In Colab: **Runtime → Change runtime type**
2. Select: **T4 GPU**
3. Click **Save**

### Step 3: Run All Cells

1. **Runtime → Run all** (or Ctrl+F9)
2. Wait ~5-10 minutes
3. Download `prompt_complexity_classifier.zip` from Files panel

---

## What the Notebook Does

1. ✅ **Installs dependencies** (transformers, datasets, etc.)
2. ✅ **Verifies GPU** is available (T4 check)
3. ✅ **Fetches datasets** from Hugging Face Hub:
   - WildChat (20k prompts)
   - GSM8K (7.5k prompts)
   - SupraLabs (1k prompts)
4. ✅ **Applies heuristic labels** (small/medium/large)
5. ✅ **Trains MiniLM classifier**:
   - Batch size: 32 (GPU-optimized)
   - FP16 mixed precision
   - 3 epochs (~5-10 min)
6. ✅ **Evaluates & saves model**
7. ✅ **Zips for download**

---

## After Training

### Download the Model

1. In Colab, click **Files** (folder icon on left)
2. Find `prompt_complexity_classifier.zip`
3. Right-click → **Download**

### Extract to Your Local Project

```bash
# Unzip to your local project
cd C:\Projects\Carbon_Aware_AI_Inference_System_new
unzip path/to/prompt_complexity_classifier.zip -d model/
```

Or manually extract the ZIP to:
```
model/
  └── prompt_complexity_classifier/
      ├── config.json
      ├── model.safetensors
      ├── tokenizer_config.json
      ├── vocab.txt
      └── training_report.txt
```

---

## Expected Results

**Training Time:** 5-10 minutes on T4 GPU  
**Accuracy:** ~80-85%  
**F1 Score:** ~0.78-0.83  

The model will be ready for CPU inference in your local pipeline!

---

## Troubleshooting

### ⚠️ "CUDA not available"
- Go to **Runtime → Change runtime type**
- Select **T4 GPU** (not None)
- Click **Save** and re-run

### ⚠️ "Out of memory"
- Reduce `per_device_train_batch_size` from 32 to 16
- In the training cell, find: `per_device_train_batch_size=32`
- Change to: `per_device_train_batch_size=16`

### ⚠️ Dataset fetch is slow
- This is normal — first time downloads from Hugging Face Hub
- WildChat is large, streaming helps
- Total fetch time: ~3-5 minutes

---

## Alternative: Python Script Version

If you prefer a single Python script instead of Jupyter notebook:

1. Upload `colab_train_classifier.py` to Colab
2. Enable T4 GPU
3. Create a new code cell:
```python
!python colab_train_classifier.py
```

---

## Next Steps After Download

Once you have the trained model locally, continue with:

1. ✅ **Step 4**: Real-world routing validation (`eval/routing_validation.py`)
2. ✅ **Step 5**: Integration into pipeline (`complexity_classifier.py`)
3. ✅ **Step 6**: Streamlit dashboard tab

See main README for full implementation guide.
