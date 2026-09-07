"""
Colab Training Script for Prompt Complexity Classifier
Run this in Google Colab with GPU (T4) enabled

Instructions:
1. Runtime → Change runtime type → T4 GPU
2. Run all cells in order
3. Download the trained model at the end
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════

print("Installing dependencies...")
!pip install -q transformers datasets accelerate scikit-learn torch

print("\n✅ Dependencies installed!")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: Verify GPU
# ══════════════════════════════════════════════════════════════════════════════

import torch
import os

print("="*80)
print("GPU CHECK")
print("="*80)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
else:
    print("⚠️ WARNING: GPU not detected! Training will be very slow.")
    print("Go to Runtime → Change runtime type → T4 GPU")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: Fetch Datasets
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("FETCHING DATASETS")
print("="*80)

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

# Create directories
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_wildchat(sample_size=20000):
    print("Fetching WildChat...")
    dataset = load_dataset("allenai/WildChat", split="train", streaming=True)
    
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= sample_size:
            break
        if example.get("language") != "English":
            continue
        conversation = example.get("conversation", [])
        if conversation and conversation[0].get("role") == "user":
            prompt_text = conversation[0].get("content", "").strip()
            if 10 <= len(prompt_text) <= 2000:
                prompts.append({"prompt": prompt_text, "source_dataset": "wildchat"})
    
    df = pd.DataFrame(prompts)
    df.to_parquet(CACHE_DIR / "wildchat_sampled.parquet", index=False)
    print(f"✅ Collected {len(df)} WildChat prompts")
    return df

def fetch_gsm8k():
    print("Fetching GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [{"prompt": ex["question"], "source_dataset": "gsm8k", "answer": ex.get("answer", "")} 
               for ex in dataset]
    df = pd.DataFrame(prompts)
    df.to_parquet(CACHE_DIR / "gsm8k.parquet", index=False)
    print(f"✅ Collected {len(df)} GSM8K prompts")
    return df

def fetch_supralabs():
    print("Fetching SupraLabs...")
    dataset = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train")
    df = dataset.to_pandas()
    df["source_dataset"] = "supralabs"
    if "text" in df.columns:
        df = df.rename(columns={"text": "prompt"})
    if "label" in df.columns:
        df["original_label"] = df["label"]
    df.to_parquet(CACHE_DIR / "supralabs_routing.parquet", index=False)
    print(f"✅ Collected {len(df)} SupraLabs prompts")
    return df

# Fetch all datasets
wildchat_df = fetch_wildchat()
gsm8k_df = fetch_gsm8k()
supralabs_df = fetch_supralabs()

print(f"\n✅ Total prompts fetched: {len(wildchat_df) + len(gsm8k_df) + len(supralabs_df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4: Build Labeled Dataset
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("BUILDING LABELED DATASET")
print("="*80)

import re
import numpy as np

# Label mapping for SupraLabs
def map_supralabs_labels(df):
    def assign_label(row):
        if row['routing_choice'] == 'small model':
            return 'small'
        elif row['routing_choice'] == 'big model':
            return 'medium' if row['complexity_score'] <= 2 else 'large'
        return 'medium'
    df['complexity_label'] = df.apply(assign_label, axis=1)
    return df

# Heuristics
class ComplexityHeuristics:
    REASONING_KEYWORDS = {'step by step', 'explain why', 'prove that', 'calculate', 
                          'solve for', 'derive', 'analyze', 'evaluate'}
    MATH_SYMBOLS = {'=', '+', '-', '*', '/', '^', '√', '∫', '∑', '\\frac', '\\int'}
    CODE_MARKERS = {'def ', 'class ', 'import ', 'function', 'return', '```', 
                    'python', 'algorithm', 'implement'}
    SIMPLE_PATTERNS = {'what is', 'who is', 'when was', 'define', 'meaning of'}
    
    @staticmethod
    def heuristic_label(prompt, source):
        text_lower = prompt.lower()
        token_count = len(prompt) // 4
        
        if source == 'gsm8k':
            return 'medium' if token_count < 40 else 'large'
        
        has_reasoning = any(kw in text_lower for kw in ComplexityHeuristics.REASONING_KEYWORDS)
        has_math = any(sym in prompt for sym in ComplexityHeuristics.MATH_SYMBOLS)
        has_code = any(marker in text_lower for marker in ComplexityHeuristics.CODE_MARKERS)
        is_simple = any(pat in text_lower[:50] for pat in ComplexityHeuristics.SIMPLE_PATTERNS)
        
        if is_simple and token_count < 50:
            return 'small'
        if token_count < 30 and not (has_reasoning or has_math or has_code):
            return 'small'
        if has_reasoning or has_math or '```' in prompt:
            return 'large'
        if token_count > 300:
            return 'large'
        if has_code or (100 <= token_count <= 300):
            return 'medium'
        return 'medium'

# Apply labels
supralabs_labeled = map_supralabs_labels(supralabs_df)
wildchat_df['complexity_label'] = wildchat_df['prompt'].apply(
    lambda x: ComplexityHeuristics.heuristic_label(x, 'wildchat'))
gsm8k_df['complexity_label'] = gsm8k_df['prompt'].apply(
    lambda x: ComplexityHeuristics.heuristic_label(x, 'gsm8k'))

# Combine
full_df = pd.concat([
    supralabs_labeled[['prompt', 'source_dataset', 'complexity_label']],
    wildchat_df[['prompt', 'source_dataset', 'complexity_label']],
    gsm8k_df[['prompt', 'source_dataset', 'complexity_label']]
], ignore_index=True)

print(f"✅ Combined dataset: {len(full_df):,} samples")
print(f"\nClass distribution:")
print(full_df['complexity_label'].value_counts())

# Save
LABELED_DIR = Path("data/labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)
full_df.to_parquet(LABELED_DIR / "prompt_complexity_labeled.parquet", index=False)
print(f"\n✅ Saved to {LABELED_DIR / 'prompt_complexity_labeled.parquet'}")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5: Train Classifier (GPU-Optimized)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING CLASSIFIER (GPU)")
print("="*80)

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset

# Load data
df = pd.read_parquet(LABELED_DIR / "prompt_complexity_labeled.parquet")

# Train/val split
df['strat_key'] = df['complexity_label'] + "_" + df['source_dataset']
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['strat_key'])
train_df = train_df.drop('strat_key', axis=1)
val_df = val_df.drop('strat_key', axis=1)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

# Label mapping
LABEL2ID = {"small": 0, "medium": 1, "large": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Tokenize
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_dataset(df, tokenizer):
    df['label'] = df['complexity_label'].map(LABEL2ID)
    dataset = Dataset.from_pandas(df[['prompt', 'label']])
    def tokenize_fn(examples):
        return tokenizer(examples['prompt'], truncation=True, max_length=256, padding=False)
    return dataset.map(tokenize_fn, batched=True, remove_columns=['prompt'])

train_dataset = tokenize_dataset(train_df, tokenizer)
val_dataset = tokenize_dataset(val_df, tokenizer)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

# Training args (GPU-optimized)
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Bumped for GPU
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    fp16=True,  # Mixed precision for speed
    dataloader_pin_memory=True,
    report_to="none",
)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
    }

# Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
print("\n🚀 Starting training...")
train_start = time.time()
trainer.train()
train_time = time.time() - train_start

print(f"\n✅ Training completed in {train_time:.2f}s ({train_time/60:.2f} min)")

# Evaluate
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

accuracy = accuracy_score(true_labels, pred_labels)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
cm = confusion_matrix(true_labels, pred_labels)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 (macro):   {f1_macro:.4f}")
print(f"\nConfusion Matrix:")
print(f"           small  medium  large")
for i, label in enumerate(['small', 'medium', 'large']):
    print(f"{label:8s}: " + "  ".join(f"{v:5d}" for v in cm[i]))

# Save model
MODEL_DIR = Path("prompt_complexity_classifier")
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print(f"\n✅ Model saved to {MODEL_DIR}")

# Save report
with open(MODEL_DIR / "training_report.txt", "w") as f:
    f.write(f"Training Time: {train_time:.2f}s ({train_time/60:.2f} min)\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 (macro): {f1_macro:.4f}\n")
    f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6: Download Model
# ══════════════════════════════════════════════════════════════════════════════

print("\n📦 Zipping model for download...")
!zip -r prompt_complexity_classifier.zip prompt_complexity_classifier/

print("\n✅ Model zipped! Download 'prompt_complexity_classifier.zip' from Files panel")
print("   Then extract it to your local: model/prompt_complexity_classifier/")
