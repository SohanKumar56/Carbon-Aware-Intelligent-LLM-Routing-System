"""
train_classifier.py — Train the Prompt Complexity Classifier

Fine-tunes a small, fast text encoder (DistilBERT or MiniLM) as a 3-class
sequence classifier (small/medium/large) for CPU-friendly inference.

Follows the existing repo's design philosophy:
- No GPU required (but will use if available)
- Fast inference on CPU
- Model caching for reuse
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "labeled"
MODEL_OUTPUT_DIR = Path(__file__).parent / "prompt_complexity_classifier"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
# Using MiniLM for fast, efficient inference (22M parameters, CPU-friendly)
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"  # Alternative: "distilbert-base-uncased"

# Label mapping
LABEL2ID = {"small": 0, "medium": 1, "large": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_dataset() -> pd.DataFrame:
    """Load the labeled dataset."""
    logger.info("Loading labeled dataset...")
    df = pd.read_parquet(DATA_DIR / "prompt_complexity_labeled.parquet")
    logger.info(f"Loaded {len(df):,} samples")
    logger.info(f"Class distribution:\n{df['complexity_label'].value_counts()}")
    return df


def prepare_train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val split.
    
    Stratify by both label AND source to ensure representation.
    """
    logger.info("Creating train/val split...")
    
    # Create stratification column: label + source
    df['strat_key'] = df['complexity_label'] + "_" + df['source_dataset']
    
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['strat_key']
    )
    
    train_df = train_df.drop('strat_key', axis=1)
    val_df = val_df.drop('strat_key', axis=1)
    
    logger.info(f"Train set: {len(train_df):,} samples")
    logger.info(f"Val set:   {len(val_df):,} samples")
    
    logger.info("\nTrain class distribution:")
    logger.info(train_df['complexity_label'].value_counts())
    logger.info("\nVal class distribution:")
    logger.info(val_df['complexity_label'].value_counts())
    
    return train_df, val_df


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 256
) -> Dataset:
    """Tokenize prompts and convert to HuggingFace Dataset."""
    
    # Convert labels to numeric
    df['label'] = df['complexity_label'].map(LABEL2ID)
    
    # Create HF dataset
    dataset = Dataset.from_pandas(df[['prompt', 'label']])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['prompt'],
            truncation=True,
            max_length=max_length,
            padding=False  # Pad dynamically during training
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['prompt']
    )
    
    return tokenized


def compute_metrics(eval_pred):
    """Compute evaluation metrics during training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> Trainer:
    """Train the sequence classification model."""
    
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    train_start = time.time()
    
    trainer.train()
    
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    return trainer


def evaluate_model(trainer: Trainer, val_dataset: Dataset) -> Dict:
    """Evaluate the trained model and generate detailed metrics."""
    
    logger.info("Evaluating model on validation set...")
    
    # Get predictions
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(
        true_labels,
        pred_labels,
        target_names=['small', 'medium', 'large'],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:     {accuracy:.4f}")
    logger.info(f"F1 (macro):   {f1_macro:.4f}")
    logger.info(f"F1 (weighted):{f1_weighted:.4f}")
    logger.info(f"\nPer-class F1 scores:")
    for label in ['small', 'medium', 'large']:
        f1 = class_report[label]['f1-score']
        support = class_report[label]['support']
        logger.info(f"  {label:8s}: {f1:.4f} (n={support})")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"           small  medium  large")
    for i, row_label in enumerate(['small', 'medium', 'large']):
        row_str = f"{row_label:8s}: " + "  ".join(f"{val:5d}" for val in cm[i])
        logger.info(row_str)
    logger.info(f"{'='*60}")
    
    return results


def save_training_report(results: Dict, metadata: Dict):
    """Save training report as markdown."""
    
    report_file = MODEL_OUTPUT_DIR / "training_report.md"
    
    cm = np.array(results['confusion_matrix'])
    class_report = results['classification_report']
    
    with open(report_file, 'w') as f:
        f.write("# Prompt Complexity Classifier - Training Report\n\n")
        
        # Metadata
        f.write("## Training Configuration\n\n")
        f.write(f"- **Model**: {metadata['model_name']}\n")
        f.write(f"- **Training samples**: {metadata['train_size']:,}\n")
        f.write(f"- **Validation samples**: {metadata['val_size']:,}\n")
        f.write(f"- **Epochs**: {metadata['num_epochs']}\n")
        f.write(f"- **Batch size**: {metadata['batch_size']}\n")
        f.write(f"- **Learning rate**: {metadata['learning_rate']}\n")
        f.write(f"- **Device**: {metadata['device']}\n")
        f.write(f"- **Training time**: {metadata['train_time']:.2f}s ({metadata['train_time']/60:.2f} min)\n\n")
        
        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write(f"- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"- **F1 Score (macro)**: {results['f1_macro']:.4f}\n")
        f.write(f"- **F1 Score (weighted)**: {results['f1_weighted']:.4f}\n\n")
        
        # Per-class metrics
        f.write("## Per-Class Performance\n\n")
        f.write("| Class  | Precision | Recall | F1-Score | Support |\n")
        f.write("|--------|-----------|--------|----------|----------|\n")
        for label in ['small', 'medium', 'large']:
            metrics = class_report[label]
            f.write(f"| {label:6s} | {metrics['precision']:.4f}    | {metrics['recall']:.4f} | {metrics['f1-score']:.4f}   | {int(metrics['support']):8d} |\n")
        f.write("\n")
        
        # Confusion matrix
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write("Predicted →  small  medium  large\n")
        f.write("Actual ↓\n")
        for i, row_label in enumerate(['small', 'medium', 'large']):
            row_str = f"{row_label:8s}:  " + "  ".join(f"{val:5d}" for val in cm[i])
            f.write(row_str + "\n")
        f.write("```\n\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        
        # Check for common issues
        if results['accuracy'] < 0.7:
            f.write("⚠️ **Warning**: Accuracy below 70%. Consider:\n")
            f.write("- Increasing training data\n")
            f.write("- Adjusting hyperparameters\n")
            f.write("- Reviewing label quality\n\n")
        elif results['accuracy'] >= 0.85:
            f.write("✅ **Excellent**: Model achieves >85% accuracy on validation set.\n\n")
        elif results['accuracy'] >= 0.75:
            f.write("✅ **Good**: Model achieves >75% accuracy, suitable for production.\n\n")
        else:
            f.write("⚠️ **Moderate**: Accuracy 70-75%. Consider further tuning.\n\n")
        
        # Class balance check
        f1_scores = [class_report[label]['f1-score'] for label in ['small', 'medium', 'large']]
        f1_std = np.std(f1_scores)
        if f1_std > 0.1:
            f.write("⚠️ **Class imbalance detected**: F1 scores vary significantly across classes.\n")
            f.write("   Consider collecting more data for underperforming classes.\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review confusion matrix for common misclassifications\n")
        f.write("2. Run `eval/routing_validation.py` for real-world validation\n")
        f.write("3. Integrate into production pipeline via `complexity_classifier.py`\n")
    
    logger.info(f"Saved training report to {report_file}")


def main():
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("PROMPT COMPLEXITY CLASSIFIER TRAINING")
    logger.info("="*80)
    
    train_start_time = time.time()
    
    # 1. Load data
    df = load_dataset()
    
    # 2. Train/val split
    train_df, val_df = prepare_train_val_split(df, test_size=0.2)
    
    # 3. Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 4. Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize_dataset(val_df, tokenizer)
    
    # 5. Train model
    trainer = train_model(
        train_dataset,
        val_dataset,
        tokenizer,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    total_train_time = time.time() - train_start_time
    
    # 6. Evaluate
    results = evaluate_model(trainer, val_dataset)
    
    # 7. Save model
    logger.info("Saving model and tokenizer...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    logger.info(f"Model saved to {MODEL_OUTPUT_DIR}")
    
    # 8. Save training report
    metadata = {
        'model_name': MODEL_NAME,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'num_epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'train_time': total_train_time,
    }
    
    save_training_report(results, metadata)
    
    # 9. Save results as JSON
    results_file = MODEL_OUTPUT_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        results_json = {
            'accuracy': float(results['accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'confusion_matrix': results['confusion_matrix'],
            'classification_report': results['classification_report'],
        }
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved evaluation results to {results_file}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
    logger.info(f"Final accuracy: {results['accuracy']:.4f}")
    logger.info(f"Final F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"\nModel and reports saved to: {MODEL_OUTPUT_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
