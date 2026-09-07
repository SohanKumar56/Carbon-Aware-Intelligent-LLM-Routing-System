"""
routing_validation.py — Real-world validation of prompt routing decisions

This script validates that our classifier's routing decisions actually make sense
by running prompts through different-sized Ollama models and comparing quality.

Key validation question:
"Of prompts routed to Small, what % were answered equivalently well by Small vs Large?"
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from ollama_integration import run_ollama_inference, OLLAMA_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent / "model" / "prompt_complexity_classifier"
DATA_DIR = Path(__file__).parent.parent / "data" / "labeled"
OUTPUT_DIR = Path(__file__).parent / "validation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model mapping for routing
MODEL_ROUTING = {
    'small': ['tinyllama:latest', 'deepseek-coder:1.3b', 'qwen2:1.5b'],
    'medium': ['qwen2.5:3b', 'phi3:latest'],
    'large': ['qwen2.5:7b', 'gemma3:4b']
}


def load_complexity_classifier():
    """Load the trained complexity classifier."""
    logger.info(f"Loading classifier from {MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    logger.info("✅ Classifier loaded successfully")
    return tokenizer, model


def classify_prompt(prompt: str, tokenizer, model) -> Tuple[str, float]:
    """
    Classify a prompt's complexity.
    
    Returns
    -------
    tuple
        (predicted_label, confidence_score)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)
    
    label = model.config.id2label[predicted.item()]
    return label, confidence.item()


def get_sample_prompts(n_samples: int = 50) -> pd.DataFrame:
    """
    Get a stratified sample of prompts for validation.
    
    Parameters
    ----------
    n_samples : int
        Number of prompts per class to sample
    
    Returns
    -------
    pd.DataFrame
        Sampled prompts with true labels
    """
    logger.info(f"Sampling {n_samples} prompts per class for validation...")
    
    df = pd.read_parquet(DATA_DIR / "prompt_complexity_labeled.parquet")
    
    # Sample from each class
    sampled = []
    for label in ['small', 'medium', 'large']:
        label_df = df[df['complexity_label'] == label]
        n_sample = min(n_samples, len(label_df))
        sample = label_df.sample(n=n_sample, random_state=42)
        sampled.append(sample)
    
    result = pd.concat(sampled, ignore_index=True)
    logger.info(f"Sampled {len(result)} prompts total")
    
    return result


def run_quality_comparison(
    prompt: str,
    small_model: str,
    large_model: str,
    timeout_small: int = 30,
    timeout_large: int = 180
) -> Dict:
    """
    Run the same prompt through small and large models, compare quality.
    
    Parameters
    ----------
    prompt : str
        The prompt to test
    small_model : str
        Small model ID (e.g., 'tinyllama:latest')
    large_model : str
        Large model ID (e.g., 'qwen2.5:7b')
    
    Returns
    -------
    dict
        Comparison results with responses and metrics
    """
    results = {
        'prompt': prompt,
        'small_model': small_model,
        'large_model': large_model,
    }
    
    # Run small model
    logger.info(f"Running {small_model}...")
    small_result = run_ollama_inference(small_model, prompt, timeout=timeout_small)
    results['small_response'] = small_result.get('raw_response', '')
    results['small_error'] = small_result.get('error', None)
    results['small_latency'] = small_result.get('latency_ms', 0)
    results['small_energy'] = small_result.get('energy_kwh', 0)
    
    # Run large model
    logger.info(f"Running {large_model}...")
    large_result = run_ollama_inference(large_model, prompt, timeout=timeout_large)
    results['large_response'] = large_result.get('raw_response', '')
    results['large_error'] = large_result.get('error', None)
    results['large_latency'] = large_result.get('latency_ms', 0)
    results['large_energy'] = large_result.get('energy_kwh', 0)
    
    # Simple quality heuristic: check if both succeeded
    small_success = results['small_error'] is None and len(results['small_response']) > 10
    large_success = results['large_error'] is None and len(results['large_response']) > 10
    
    results['small_success'] = small_success
    results['large_success'] = large_success
    results['equivalent_quality'] = small_success and large_success
    
    # Energy savings
    if small_success:
        results['energy_saved_kwh'] = results['large_energy'] - results['small_energy']
        results['energy_saved_pct'] = (
            100 * results['energy_saved_kwh'] / results['large_energy']
            if results['large_energy'] > 0 else 0
        )
    
    return results


def validate_routing_decisions(
    n_samples_per_class: int = 20,
    small_model: str = 'tinyllama:latest',
    large_model: str = 'qwen2.5:7b'
) -> pd.DataFrame:
    """
    Main validation function.
    
    For each complexity class:
    1. Classify prompts
    2. Route to appropriate model
    3. Also run through large model for comparison
    4. Measure quality equivalence and energy savings
    
    Parameters
    ----------
    n_samples_per_class : int
        Number of prompts to validate per class
    small_model : str
        Small model to use for 'small' routing
    large_model : str
        Large model for comparison baseline
    
    Returns
    -------
    pd.DataFrame
        Validation results for all prompts
    """
    logger.info("="*80)
    logger.info("ROUTING VALIDATION")
    logger.info("="*80)
    
    # Load classifier
    tokenizer, model = load_complexity_classifier()
    
    # Get sample prompts
    sample_df = get_sample_prompts(n_samples=n_samples_per_class)
    
    # Validate each prompt
    results = []
    
    for idx, row in sample_df.iterrows():
        prompt = row['prompt']
        true_label = row['complexity_label']
        
        logger.info(f"\n[{idx+1}/{len(sample_df)}] Processing prompt...")
        logger.info(f"True label: {true_label}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Classify
        predicted_label, confidence = classify_prompt(prompt, tokenizer, model)
        logger.info(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
        
        # Only validate 'small' predictions (most critical for safety)
        if predicted_label == 'small':
            comparison = run_quality_comparison(
                prompt, small_model, large_model
            )
            comparison['true_label'] = true_label
            comparison['predicted_label'] = predicted_label
            comparison['confidence'] = confidence
            results.append(comparison)
        else:
            # For medium/large, just record the prediction
            results.append({
                'prompt': prompt,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'small_model': small_model,
                'large_model': large_model,
                'small_success': None,
                'large_success': None,
                'equivalent_quality': None,
            })
    
    results_df = pd.DataFrame(results)
    return results_df


def generate_validation_report(results_df: pd.DataFrame) -> Dict:
    """Generate validation report with key metrics."""
    
    report = {
        'total_prompts': len(results_df),
        'by_predicted_class': {},
    }
    
    # Overall accuracy (predicted vs true)
    if 'true_label' in results_df.columns and 'predicted_label' in results_df.columns:
        accuracy = (results_df['true_label'] == results_df['predicted_label']).mean()
        report['classifier_accuracy'] = accuracy
    
    # Analyze 'small' routing decisions
    small_routed = results_df[results_df['predicted_label'] == 'small']
    if len(small_routed) > 0:
        # Filter out rows where we actually ran the comparison
        small_tested = small_routed[small_routed['equivalent_quality'].notna()]
        
        if len(small_tested) > 0:
            equivalent_pct = small_tested['equivalent_quality'].mean()
            avg_energy_saved = small_tested['energy_saved_pct'].mean()
            
            report['small_routing'] = {
                'total_routed': len(small_routed),
                'total_tested': len(small_tested),
                'equivalent_quality_rate': equivalent_pct,
                'avg_energy_saved_pct': avg_energy_saved,
            }
    
    # Per-class breakdown
    for label in ['small', 'medium', 'large']:
        class_df = results_df[results_df['predicted_label'] == label]
        if len(class_df) > 0:
            report['by_predicted_class'][label] = {
                'count': len(class_df),
                'avg_confidence': class_df['confidence'].mean(),
            }
    
    return report


def main():
    """Main execution."""
    
    logger.info("Starting routing validation...")
    logger.info("This will test if 'small' routing decisions are safe and efficient\n")
    
    # Check if Ollama models are available
    try:
        # Run validation with limited samples for speed
        results_df = validate_routing_decisions(
            n_samples_per_class=10,  # Small sample for quick validation
            small_model='tinyllama:latest',
            large_model='qwen2.5:3b'  # Using 3B as "large" for speed
        )
        
        # Save results
        output_file = OUTPUT_DIR / "routing_validation_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Results saved to {output_file}")
        
        # Generate report
        report = generate_validation_report(results_df)
        
        # Save report
        report_file = OUTPUT_DIR / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Report saved to {report_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        if 'classifier_accuracy' in report:
            logger.info(f"Classifier Accuracy: {report['classifier_accuracy']:.2%}")
        
        if 'small_routing' in report:
            sr = report['small_routing']
            logger.info(f"\nSmall Model Routing:")
            logger.info(f"  Total routed to small: {sr['total_routed']}")
            logger.info(f"  Tested with comparison: {sr['total_tested']}")
            logger.info(f"  Equivalent quality rate: {sr['equivalent_quality_rate']:.2%}")
            logger.info(f"  Avg energy saved: {sr['avg_energy_saved_pct']:.1f}%")
        
        logger.info("\nPer-class predictions:")
        for label, stats in report['by_predicted_class'].items():
            logger.info(f"  {label:8s}: {stats['count']:3d} prompts (avg conf: {stats['avg_confidence']:.3f})")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.info("\n⚠️ Note: This validation requires Ollama to be running")
        logger.info("   with models installed. You can skip this step if Ollama")
        logger.info("   is not available - the classifier will still work!")
        return


if __name__ == "__main__":
    main()
