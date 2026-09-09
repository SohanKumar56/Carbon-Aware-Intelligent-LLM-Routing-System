"""
complexity_classifier.py — Production inference module for prompt complexity classification

Provides a fast interface for classifying prompt complexity and routing
to appropriate Ollama model sizes (small / medium / large).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Dict
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Model path
MODEL_DIR = Path(__file__).parent / "model" / "prompt_complexity_classifier"

# Module-level cache (singleton pattern — avoids reloading on every call)
_classifier_cache: Dict[str, any] = {}


def load_classifier():
    """
    Load the complexity classifier model and tokenizer.
    
    Uses module-level caching to avoid reloading on every call.
    
    Returns
    -------
    tuple
        (tokenizer, model)
    """
    global _classifier_cache
    
    if 'model' in _classifier_cache and 'tokenizer' in _classifier_cache:
        return _classifier_cache['tokenizer'], _classifier_cache['model']
    
    logger.info(f"Loading complexity classifier from {MODEL_DIR}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()  # Set to evaluation mode
        
        _classifier_cache['tokenizer'] = tokenizer
        _classifier_cache['model'] = model
        
        logger.info("✅ Complexity classifier loaded and cached")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load complexity classifier: {e}")
        raise


def classify_prompt_complexity(prompt: str) -> str:
    """
    Classify a prompt's complexity for routing.
    
    This is the main public API — returns a simple string label.
    
    Parameters
    ----------
    prompt : str
        The prompt text to classify
    
    Returns
    -------
    str
        One of: 'small' | 'medium' | 'large'
    
    Examples
    --------
    >>> classify_prompt_complexity("What is the capital of France?")
    'small'
    
    >>> classify_prompt_complexity("Explain quantum entanglement in detail")
    'large'
    """
    result = classify_prompt_complexity_detailed(prompt)
    return result['label']


def classify_prompt_complexity_detailed(prompt: str) -> Dict:
    """
    Classify prompt complexity with full details.
    
    Uses hybrid approach: ML model + heuristic rules for edge cases.
    
    Parameters
    ----------
    prompt : str
        The prompt text to classify
    
    Returns
    -------
    dict
        label         – 'small' | 'medium' | 'large'
        confidence    – float 0-1, model confidence
        latency_ms    – inference time in milliseconds
        all_probs     – dict with probability for each class
        rule_override – bool, whether heuristic rules adjusted the classification
    """
    t0 = time.perf_counter()
    
    # Load model (cached after first call)
    tokenizer, model = load_classifier()
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)
    
    # Extract results
    predicted_idx = predicted.item()
    predicted_label = model.config.id2label[predicted_idx]
    confidence_score = confidence.item()
    
    # Get all class probabilities
    all_probs = {
        model.config.id2label[i]: probs[0][i].item()
        for i in range(len(probs[0]))
    }
    
    # Apply heuristic rules to catch edge cases
    original_label = predicted_label
    predicted_label, rule_applied = _apply_heuristic_rules(
        prompt, predicted_label, all_probs, confidence_score
    )
    
    # Adjust confidence if rule override happened
    if rule_applied:
        # Moderate confidence when rules override ML
        confidence_score = min(confidence_score, 0.85)
        logger.info(f"Heuristic rule upgraded: {original_label} → {predicted_label}")
    
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    logger.info(
        f"Classified prompt as '{predicted_label}' "
        f"(confidence: {confidence_score:.3f}, latency: {latency_ms}ms, "
        f"rule_override: {rule_applied})"
    )
    
    return {
        'label': predicted_label,
        'confidence': confidence_score,
        'latency_ms': latency_ms,
        'all_probs': all_probs,
        'rule_override': rule_applied,
    }


def get_recommended_models(complexity: str) -> Dict[str, list]:
    """
    Get recommended Ollama models for a given complexity level.
    
    Parameters
    ----------
    complexity : str
        One of: 'small', 'medium', 'large'
    
    Returns
    -------
    dict
        'primary'   – list of primary recommended models
        'fallback'  – list of fallback models if primary unavailable
    
    Examples
    --------
    >>> get_recommended_models('small')
    {
        'primary': ['tinyllama:latest', 'deepseek-coder:1.3b'],
        'fallback': ['qwen2:1.5b']
    }
    """
    recommendations = {
        'small': {
            'primary': ['tinyllama:latest', 'deepseek-coder:1.3b', 'qwen2:1.5b'],
            'fallback': ['deepseek-r1:1.5b'],
        },
        'medium': {
            'primary': ['qwen2.5:3b', 'phi3:latest'],
            'fallback': ['gemma3:4b'],
        },
        'large': {
            'primary': ['qwen2.5:7b', 'gemma3:4b'],
            'fallback': ['qwen2:latest', 'zephyr:latest'],
        },
    }
    
    return recommendations.get(complexity, recommendations['medium'])


def route_to_model(prompt: str) -> Dict:
    """
    Full routing decision: classify complexity and recommend models.
    
    Parameters
    ----------
    prompt : str
        The prompt to route
    
    Returns
    -------
    dict
        complexity        – 'small' | 'medium' | 'large'
        confidence        – model confidence 0-1
        recommended       – list of recommended model IDs
        fallback          – list of fallback model IDs
        latency_ms        – classification time
        reasoning         – human-readable explanation
        rule_override     – whether heuristic rules adjusted classification
    """
    # Classify
    result = classify_prompt_complexity_detailed(prompt)
    complexity = result['label']
    
    # Get recommendations
    models = get_recommended_models(complexity)
    
    # Generate reasoning
    conf = result['confidence']
    reasoning = _generate_reasoning(
        complexity, conf, prompt, result.get('rule_override', False)
    )
    
    return {
        'complexity': complexity,
        'confidence': conf,
        'recommended': models['primary'],
        'fallback': models['fallback'],
        'latency_ms': result['latency_ms'],
        'all_probs': result['all_probs'],
        'reasoning': reasoning,
        'rule_override': result.get('rule_override', False),
    }


def _apply_heuristic_rules(
    prompt: str, 
    predicted: str, 
    all_probs: Dict[str, float],
    confidence: float
) -> tuple[str, bool]:
    """
    Apply heuristic rules to catch edge cases the ML model might miss.
    
    Parameters
    ----------
    prompt : str
        The input prompt
    predicted : str
        ML model's prediction
    all_probs : dict
        Probability distribution across classes
    confidence : float
        Model confidence score
    
    Returns
    -------
    tuple
        (final_label, rule_applied)
    """
    prompt_lower = prompt.lower()
    tokens = prompt.split()
    token_count = len(tokens)
    
    rule_applied = False
    final_label = predicted
    
    # Rule 1: Mathematical proofs are almost always LARGE
    math_proof_keywords = [
        'prove', 'proof', 'theorem', 'lemma', 'corollary',
        'contradiction', 'induction', 'mathematical proof',
        'rigorous proof', 'formal proof', 'qed'
    ]
    if any(kw in prompt_lower for kw in math_proof_keywords):
        if predicted == 'small':
            # Math proofs need large models regardless of probabilities
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: Math proof detected, upgraded small → large")
        elif predicted == 'medium' and token_count > 20:
            # Long math proofs should use large models
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: Complex math proof detected, upgraded medium → large")
    
    # Rule 2: Very long prompts (>100 tokens) are rarely SMALL or MEDIUM
    if token_count > 120:
        if predicted != 'large':
            final_label = 'large'
            rule_applied = True
            logger.info(f"Rule: Very long prompt ({token_count} tokens), upgraded {predicted} → large")
    elif token_count > 100 and predicted == 'small':
        final_label = 'large'
        rule_applied = True
        logger.info(f"Rule: Very long prompt ({token_count} tokens), upgraded small → large")
    elif token_count > 80 and predicted == 'small':
        final_label = 'medium'
        rule_applied = True
        logger.info(f"Rule: Long prompt ({token_count} tokens), upgraded small → medium")
    
    # Rule 3: Complex reasoning indicators
    complex_reasoning_keywords = [
        'explain in detail', 'comprehensive', 'thorough analysis',
        'step by step', 'detailed explanation', 'in-depth',
        'complex', 'advanced', 'sophisticated', 'nuanced',
        'implications', 'analyze', 'evaluate'
    ]
    complex_reasoning_count = sum(1 for kw in complex_reasoning_keywords if kw in prompt_lower)
    if complex_reasoning_count >= 2:
        if predicted == 'small':
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: Multiple complexity signals detected, upgraded small → large")
        elif predicted == 'medium' and complex_reasoning_count >= 3:
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: High complexity signals detected, upgraded medium → large")
    
    # Rule 4: Multi-step tasks
    multi_step_indicators = [
        'first', 'then', 'next', 'finally', 'step 1', 'step 2',
        'multiple', 'several', 'various', 'different aspects'
    ]
    multi_step_count = sum(1 for ind in multi_step_indicators if ind in prompt_lower)
    if multi_step_count >= 3 and predicted == 'small' and token_count > 30:
        if all_probs.get('medium', 0) > 0.1:
            final_label = 'medium'
            rule_applied = True
            logger.info("Rule: Multi-step task detected, upgraded small → medium")
    
    # Rule 5: Code generation with requirements
    code_indicators = ['function', 'class', 'algorithm', 'implement', 'code', 'program', 'application', 'system', 'create']
    requirement_indicators = ['with', 'including', 'that', 'using', 'should', 'must', 'comprehensive', 'complete']
    
    has_code = any(ind in prompt_lower for ind in code_indicators)
    requirement_count = sum(1 for ind in requirement_indicators if ind in prompt_lower)
    
    # Count specific technical requirements
    tech_requirements = ['authentication', 'database', 'api', 'frontend', 'backend', 'error handling', 
                        'logging', 'testing', 'tests', 'validation', 'documentation', 'deployment']
    tech_count = sum(1 for req in tech_requirements if req in prompt_lower)
    
    if has_code and tech_count >= 5:
        # Very comprehensive technical requirements
        if predicted != 'large':
            final_label = 'large'
            rule_applied = True
            logger.info(f"Rule: Comprehensive technical task with {tech_count} requirements, upgraded {predicted} → large")
    elif has_code and requirement_count >= 3 and token_count > 20:
        if predicted == 'small':
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: Complex code generation with multiple requirements, upgraded small → large")
        elif predicted == 'medium' and requirement_count >= 5:
            final_label = 'large'
            rule_applied = True
            logger.info("Rule: Very complex code generation, upgraded medium → large")
    elif has_code and requirement_count >= 2 and token_count > 15:
        if predicted == 'small':
            final_label = 'medium'
            rule_applied = True
            logger.info("Rule: Code generation with requirements, upgraded small → medium")
    
    # Rule 6: Research/academic tasks
    academic_keywords = [
        'research', 'analyze', 'compare and contrast', 'evaluate',
        'critique', 'synthesis', 'literature review', 'hypothesis'
    ]
    if any(kw in prompt_lower for kw in academic_keywords) and token_count > 20:
        if predicted == 'small':
            final_label = 'medium'
            rule_applied = True
            logger.info("Rule: Academic/research task detected, upgraded small → medium")
    
    # Rule 7: Very simple prompts should stay SMALL (prevent false upgrades)
    very_simple_patterns = [
        r'^what is ', r'^define ', r'^translate ', 
        r'^what are ', r'^who is ', r'^when was '
    ]
    import re
    if any(re.match(pattern, prompt_lower) for pattern in very_simple_patterns):
        if token_count < 10 and predicted != 'small':
            # Let simple questions stay simple
            pass  # Don't override if model says small
    
    return final_label, rule_applied


def _generate_reasoning(complexity: str, confidence: float, prompt: str, rule_override: bool = False) -> str:
    """Generate human-readable routing reasoning."""
    
    prompt_len = len(prompt)
    token_est = prompt_len // 4
    
    reasons = []
    
    # Rule override notification
    if rule_override:
        reasons.append("🎯 Enhanced by heuristic rules for better accuracy")
    
    # Complexity-specific reasoning
    if complexity == 'small':
        reasons.append("Simple task suitable for small models (1-2B params)")
        if confidence > 0.9:
            reasons.append("High confidence - clear factual/simple prompt")
        if token_est < 50:
            reasons.append("Short prompt, likely simple query")
    
    elif complexity == 'medium':
        reasons.append("Moderate complexity requiring medium models (3-4B params)")
        if token_est > 100:
            reasons.append("Moderate length suggests some context needed")
        reasons.append("May involve multi-step reasoning or moderate code")
    
    elif complexity == 'large':
        reasons.append("Complex task requiring large models (7B+ params)")
        if token_est > 200:
            reasons.append("Long prompt with detailed requirements")
        if confidence > 0.85:
            reasons.append("High confidence - clear complexity signals detected")
        reasons.append("Likely multi-step reasoning, math, or complex code")
    
    # Confidence caveat
    if confidence < 0.7:
        reasons.append(f"⚠️ Moderate confidence ({confidence:.0%}) - consider fallback")
    
    return " • ".join(reasons)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience functions for quick testing
# ══════════════════════════════════════════════════════════════════════════════

def test_classifier():
    """Quick test of the classifier with sample prompts."""
    
    test_prompts = [
        "What is the capital of France?",
        "Explain how a car engine works",
        "Write a Python function to solve the traveling salesman problem using dynamic programming",
        "Translate 'hello' to Spanish",
        "Debug this code and explain what's wrong: def foo(x): return x + y",
        "Prove that the square root of 2 is irrational using proof by contradiction",
    ]
    
    print("="*80)
    print("COMPLEXITY CLASSIFIER TEST")
    print("="*80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt[:70]}...")
        
        result = route_to_model(prompt)
        
        print(f"    Complexity: {result['complexity']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Recommended: {result['recommended'][0]}")
        print(f"    Reasoning: {result['reasoning']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run test when script is executed directly
    test_classifier()
