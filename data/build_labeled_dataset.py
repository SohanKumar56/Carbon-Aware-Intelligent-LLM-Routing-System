"""
build_labeled_dataset.py — Build unified labeled dataset for Prompt Complexity Classifier

This script combines three data sources and applies:
1. Gold labels from SupraLabs (mapped to small/medium/large)
2. Heuristic weak-labeling for WildChat and GSM8K
3. LLM-judge spot-check for quality validation

Output: unified CSV/Parquet with columns: prompt, source_dataset, complexity_label
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Dict, Tuple
import json

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CACHE_DIR = Path(__file__).parent / "cache"
OUTPUT_DIR = Path(__file__).parent / "labeled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: SupraLabs Gold Label Mapping
# ══════════════════════════════════════════════════════════════════════════════

def map_supralabs_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map SupraLabs routing_choice and complexity_score to our 3-class taxonomy.
    
    Mapping strategy:
    - routing_choice == "small model" → small
    - routing_choice == "big model" + complexity_score <= 2 → medium
    - routing_choice == "big model" + complexity_score >= 3 → large
    
    Parameters
    ----------
    df : pd.DataFrame
        SupraLabs dataset with routing_choice and complexity_score
    
    Returns
    -------
    pd.DataFrame
        DataFrame with complexity_label column added
    """
    logger.info("Mapping SupraLabs labels to small/medium/large taxonomy...")
    
    def assign_label(row):
        if row['routing_choice'] == 'small model':
            return 'small'
        elif row['routing_choice'] == 'big model':
            if row['complexity_score'] <= 2:
                return 'medium'
            else:
                return 'large'
        else:
            # Fallback (shouldn't happen with this dataset)
            return 'medium'
    
    df['complexity_label'] = df.apply(assign_label, axis=1)
    
    # Log distribution
    label_dist = df['complexity_label'].value_counts()
    logger.info(f"SupraLabs label distribution:\n{label_dist}")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Heuristic Weak-Labeling
# ══════════════════════════════════════════════════════════════════════════════

class ComplexityHeuristics:
    """Heuristic features for prompt complexity classification."""
    
    # Multi-step reasoning cues
    REASONING_KEYWORDS = {
        'step by step', 'explain why', 'prove that', 'demonstrate',
        'calculate', 'solve for', 'derive', 'show that',
        'reason about', 'think through', 'work through',
        'justify', 'analyze', 'evaluate', 'compare and contrast',
        'let\'s think', 'chain of thought', 'reasoning'
    }
    
    # Math/logic indicators
    MATH_SYMBOLS = {
        '=', '+', '-', '*', '/', '^', '√', '∫', '∑', '∏',
        '>', '<', '≥', '≤', '≠', '≈', '∞', 'π', 
        '\\frac', '\\int', '\\sum', '\\prod', '\\sqrt'
    }
    
    MATH_KEYWORDS = {
        'equation', 'formula', 'theorem', 'proof', 'calculate',
        'compute', 'integral', 'derivative', 'matrix', 'vector',
        'probability', 'statistics', 'algebra', 'geometry',
        'trigonometry', 'calculus', 'logarithm'
    }
    
    # Code indicators
    CODE_MARKERS = {
        'def ', 'class ', 'import ', 'function', 'return',
        'for ', 'while ', 'if ', 'else:', 'try:', 'except:',
        '```', 'python', 'javascript', 'java', 'c++',
        'algorithm', 'implement', 'debug', 'code'
    }
    
    # Simple factual/lookup indicators
    SIMPLE_PATTERNS = {
        'what is', 'who is', 'when was', 'where is',
        'define', 'meaning of', 'translate',
        'list', 'name', 'tell me about'
    }
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token count approximation (chars / 4)."""
        return len(text) // 4
    
    @staticmethod
    def has_code_blocks(text: str) -> bool:
        """Check for code blocks (markdown or obvious code)."""
        return '```' in text or bool(re.search(r'\n\s{4,}', text))
    
    @staticmethod
    def count_questions(text: str) -> int:
        """Count question marks (multi-part questions)."""
        return text.count('?')
    
    @staticmethod
    def has_multi_step_reasoning(text: str) -> bool:
        """Check for multi-step reasoning cues."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ComplexityHeuristics.REASONING_KEYWORDS)
    
    @staticmethod
    def has_math_content(text: str) -> bool:
        """Check for mathematical content."""
        # Check symbols
        if any(symbol in text for symbol in ComplexityHeuristics.MATH_SYMBOLS):
            return True
        # Check keywords
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ComplexityHeuristics.MATH_KEYWORDS)
    
    @staticmethod
    def has_code_content(text: str) -> bool:
        """Check for coding-related content."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in ComplexityHeuristics.CODE_MARKERS)
    
    @staticmethod
    def is_simple_factual(text: str) -> bool:
        """Check if prompt is a simple factual question."""
        text_lower = text.lower()
        # Short + starts with simple pattern
        if len(text) < 100 and any(pattern in text_lower[:50] for pattern in ComplexityHeuristics.SIMPLE_PATTERNS):
            return True
        return False
    
    @staticmethod
    def compute_features(text: str) -> Dict:
        """Compute all heuristic features for a prompt."""
        return {
            'token_count': ComplexityHeuristics.count_tokens(text),
            'char_count': len(text),
            'has_code_blocks': ComplexityHeuristics.has_code_blocks(text),
            'question_count': ComplexityHeuristics.count_questions(text),
            'has_multi_step': ComplexityHeuristics.has_multi_step_reasoning(text),
            'has_math': ComplexityHeuristics.has_math_content(text),
            'has_code': ComplexityHeuristics.has_code_content(text),
            'is_simple': ComplexityHeuristics.is_simple_factual(text),
        }


def heuristic_label_prompt(prompt: str, source: str) -> str:
    """
    Assign complexity label using heuristics.
    
    Parameters
    ----------
    prompt : str
        The prompt text
    source : str
        Source dataset (wildchat, gsm8k)
    
    Returns
    -------
    str
        Predicted label: small, medium, or large
    """
    features = ComplexityHeuristics.compute_features(prompt)
    
    # GSM8K: default to large (multi-step math reasoning)
    if source == 'gsm8k':
        # GSM8K is multi-step math, but simpler problems might be medium
        if features['token_count'] < 40 and not features['has_multi_step']:
            return 'medium'
        return 'large'
    
    # WildChat: use heuristics
    # Rule 1: Simple factual questions → small
    if features['is_simple'] and features['token_count'] < 50:
        return 'small'
    
    # Rule 2: Very short, no complexity indicators → small
    if features['token_count'] < 30 and not any([
        features['has_multi_step'],
        features['has_math'],
        features['has_code'],
        features['has_code_blocks']
    ]):
        return 'small'
    
    # Rule 3: Multi-step reasoning, math, or complex code → large
    if features['has_multi_step'] or features['has_math'] or features['has_code_blocks']:
        return 'large'
    
    # Rule 4: Long prompts (>300 tokens) with multiple questions → large
    if features['token_count'] > 300 or features['question_count'] >= 3:
        return 'large'
    
    # Rule 5: Medium-length with some complexity (code keywords) → medium
    if features['has_code'] or (100 <= features['token_count'] <= 300):
        return 'medium'
    
    # Default: medium for ambiguous cases
    return 'medium'


def apply_heuristic_labels(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply heuristic labeling to a dataset."""
    logger.info(f"Applying heuristic labels to {source} dataset...")
    
    df['complexity_label'] = df['prompt'].apply(
        lambda x: heuristic_label_prompt(x, source)
    )
    
    # Log distribution
    label_dist = df['complexity_label'].value_counts()
    logger.info(f"{source} heuristic label distribution:\n{label_dist}")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: LLM-as-Judge Spot Check
# ══════════════════════════════════════════════════════════════════════════════

# LLM Judge rubric
LLM_JUDGE_RUBRIC = """You are an expert at classifying the complexity of LLM prompts for routing purposes.

Classify the following prompt into one of three complexity classes:

**SMALL**: Simple, straightforward tasks that small LLMs (1-2B parameters) can handle well
- Simple factual questions with clear answers
- Basic formatting/rewriting tasks
- Short translations
- Simple definitions or explanations
- Single-step reasoning
Examples: "What is the capital of France?", "Translate 'hello' to Spanish", "Define photosynthesis"

**MEDIUM**: Moderate complexity requiring medium LLMs (3-4B parameters)
- Multi-turn context understanding
- Moderate code explanations or simple debugging
- Summarization of moderate-length content
- Creative writing with some constraints
- Questions requiring 2-3 steps of reasoning
Examples: "Explain how a car engine works", "Write a short poem about autumn", "Debug this Python function"

**LARGE**: Complex tasks requiring large LLMs (7B+ parameters)
- Multi-step reasoning chains (math, logic)
- Complex code generation or architecture design
- Long-form content with nuanced requirements
- Ambiguous or open-ended problems requiring judgment
- Tasks requiring deep domain knowledge
Examples: "Prove this mathematical theorem", "Design a distributed system architecture", "Solve this multi-step word problem"

Prompt to classify:
"{prompt}"

Respond with ONLY one word: SMALL, MEDIUM, or LARGE"""


def save_llm_judge_rubric():
    """Save the LLM judge rubric to a markdown file."""
    rubric_file = OUTPUT_DIR.parent / "judge_rubric.md"
    with open(rubric_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Judge Rubric for Prompt Complexity Classification\n\n")
        f.write(LLM_JUDGE_RUBRIC)
    logger.info(f"Saved LLM judge rubric to {rubric_file}")


def llm_judge_classify(prompt: str, use_anthropic: bool = False) -> str | None:
    """
    Use LLM-as-judge to classify prompt complexity.
    
    Parameters
    ----------
    prompt : str
        The prompt to classify
    use_anthropic : bool
        Whether to use Anthropic API (requires API key in env)
    
    Returns
    -------
    str | None
        Classification: 'small', 'medium', 'large', or None if failed
    """
    if not use_anthropic:
        # Skip LLM judge if API not configured
        return None
    
    try:
        import anthropic
        import os
        
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        rubric_prompt = LLM_JUDGE_RUBRIC.format(prompt=prompt[:1000])  # Truncate very long prompts
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": rubric_prompt}]
        )
        
        response = message.content[0].text.strip().upper()
        
        # Parse response
        if 'SMALL' in response:
            return 'small'
        elif 'MEDIUM' in response:
            return 'medium'
        elif 'LARGE' in response:
            return 'large'
        else:
            logger.warning(f"Unexpected LLM judge response: {response}")
            return None
            
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return None


def create_llm_judge_evaluation_set(
    df: pd.DataFrame,
    sample_size: int = 800,
    use_anthropic: bool = False
) -> pd.DataFrame:
    """
    Create LLM-judged evaluation set for quality checking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full labeled dataset
    sample_size : int
        Number of samples to judge (default 800)
    use_anthropic : bool
        Whether to actually call Anthropic API
    
    Returns
    -------
    pd.DataFrame
        Evaluation set with LLM judge labels
    """
    logger.info(f"Creating LLM-judge evaluation set (n={sample_size})...")
    
    # Stratified sampling across sources and heuristic labels
    eval_samples = []
    
    for source in df['source_dataset'].unique():
        source_df = df[df['source_dataset'] == source]
        
        # Sample proportionally from each label class
        for label in ['small', 'medium', 'large']:
            label_df = source_df[source_df['complexity_label'] == label]
            if len(label_df) > 0:
                n_sample = min(
                    len(label_df),
                    max(1, int(sample_size * len(label_df) / len(source_df)))
                )
                sampled = label_df.sample(n=n_sample, random_state=42)
                eval_samples.append(sampled)
    
    eval_df = pd.concat(eval_samples, ignore_index=True)
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=42)
    
    logger.info(f"Sampled {len(eval_df)} prompts for LLM judge evaluation")
    logger.info(f"Distribution: {eval_df['complexity_label'].value_counts().to_dict()}")
    
    # Apply LLM judge if API is available
    if use_anthropic:
        logger.info("Applying LLM judge to evaluation samples...")
        eval_df['llm_judge_label'] = eval_df['prompt'].apply(
            lambda x: llm_judge_classify(x, use_anthropic=True)
        )
        
        # Calculate agreement
        valid_judgments = eval_df[eval_df['llm_judge_label'].notna()]
        if len(valid_judgments) > 0:
            agreement = (
                valid_judgments['complexity_label'] == valid_judgments['llm_judge_label']
            ).mean()
            logger.info(f"Heuristic-LLM agreement: {agreement:.2%}")
    else:
        logger.info("Skipping LLM judge (no API configured) - saving samples for manual review")
        eval_df['llm_judge_label'] = None
    
    # Save evaluation set
    eval_file = OUTPUT_DIR / "llm_judge_eval_set.parquet"
    eval_df.to_parquet(eval_file, index=False)
    logger.info(f"Saved LLM judge evaluation set to {eval_file}")
    
    # Also save as CSV for easy manual inspection
    eval_csv = OUTPUT_DIR / "llm_judge_eval_set.csv"
    eval_df[['prompt', 'source_dataset', 'complexity_label', 'llm_judge_label']].to_csv(
        eval_csv, index=False
    )
    logger.info(f"Saved evaluation set CSV to {eval_csv}")
    
    return eval_df


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Class Balancing
# ══════════════════════════════════════════════════════════════════════════════

def balance_classes(df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
    """
    Balance class distribution to prevent model bias.
    
    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataset
    method : str
        Balancing method: 'undersample' or 'none'
    
    Returns
    -------
    pd.DataFrame
        Balanced dataset
    """
    label_counts = df['complexity_label'].value_counts()
    logger.info(f"Original class distribution:\n{label_counts}")
    
    # Check for severe imbalance (max:min ratio > 5:1)
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 3:
        logger.info("Classes are reasonably balanced, no action needed")
        return df
    
    if method == 'undersample':
        logger.info("Applying undersampling to majority classes...")
        
        # Target: balance to ~2x the minority class
        target_count = min(int(min_count * 2.5), max_count)
        
        balanced_dfs = []
        for label in df['complexity_label'].unique():
            label_df = df[df['complexity_label'] == label]
            if len(label_df) > target_count:
                sampled = label_df.sample(n=target_count, random_state=42)
                balanced_dfs.append(sampled)
            else:
                balanced_dfs.append(label_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        new_counts = balanced_df['complexity_label'].value_counts()
        logger.info(f"Balanced class distribution:\n{new_counts}")
        
        return balanced_df
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    
    logger.info("="*80)
    logger.info("Building Unified Labeled Dataset")
    logger.info("="*80)
    
    # Save LLM judge rubric
    save_llm_judge_rubric()
    
    # ── Step 1: Load datasets ─────────────────────────────────────────────────
    logger.info("\n[1/5] Loading cached datasets...")
    
    wildchat_df = pd.read_parquet(CACHE_DIR / "wildchat_sampled.parquet")
    gsm8k_df = pd.read_parquet(CACHE_DIR / "gsm8k.parquet")
    supralabs_df = pd.read_parquet(CACHE_DIR / "supralabs_routing.parquet")
    
    logger.info(f"Loaded: WildChat={len(wildchat_df)}, GSM8K={len(gsm8k_df)}, SupraLabs={len(supralabs_df)}")
    
    # ── Step 2: Apply labels ──────────────────────────────────────────────────
    logger.info("\n[2/5] Applying labels...")
    
    # SupraLabs: gold labels
    supralabs_labeled = map_supralabs_labels(supralabs_df)
    
    # WildChat: heuristic labels
    wildchat_labeled = apply_heuristic_labels(wildchat_df, 'wildchat')
    
    # GSM8K: heuristic labels (biased toward large)
    gsm8k_labeled = apply_heuristic_labels(gsm8k_df, 'gsm8k')
    
    # ── Step 3: Combine datasets ──────────────────────────────────────────────
    logger.info("\n[3/5] Combining datasets...")
    
    # Keep only relevant columns
    supralabs_clean = supralabs_labeled[['prompt', 'source_dataset', 'complexity_label']].copy()
    wildchat_clean = wildchat_labeled[['prompt', 'source_dataset', 'complexity_label']].copy()
    gsm8k_clean = gsm8k_labeled[['prompt', 'source_dataset', 'complexity_label']].copy()
    
    # Combine
    full_df = pd.concat([
        supralabs_clean,
        wildchat_clean,
        gsm8k_clean
    ], ignore_index=True)
    
    logger.info(f"Combined dataset size: {len(full_df)}")
    logger.info(f"Overall class distribution:\n{full_df['complexity_label'].value_counts()}")
    
    # ── Step 4: Class balancing ───────────────────────────────────────────────
    logger.info("\n[4/5] Checking class balance...")
    
    balanced_df = balance_classes(full_df, method='undersample')
    
    # ── Step 5: LLM judge evaluation set ──────────────────────────────────────
    logger.info("\n[5/5] Creating LLM-judge evaluation set...")
    
    # Check if Anthropic API is available
    import os
    use_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    if use_anthropic:
        logger.info("Anthropic API key found - will perform LLM judging")
    else:
        logger.info("No Anthropic API key - saving samples for manual review only")
    
    eval_df = create_llm_judge_evaluation_set(
        balanced_df,
        sample_size=800,
        use_anthropic=use_anthropic
    )
    
    # ── Save final dataset ────────────────────────────────────────────────────
    logger.info("\n[FINAL] Saving labeled dataset...")
    
    # Save as Parquet (efficient)
    output_parquet = OUTPUT_DIR / "prompt_complexity_labeled.parquet"
    balanced_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved Parquet: {output_parquet}")
    
    # Save as CSV (human-readable)
    output_csv = OUTPUT_DIR / "prompt_complexity_labeled.csv"
    balanced_df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV: {output_csv}")
    
    # Save metadata
    metadata = {
        "total_samples": len(balanced_df),
        "class_distribution": balanced_df['complexity_label'].value_counts().to_dict(),
        "source_distribution": balanced_df['source_dataset'].value_counts().to_dict(),
        "original_size": len(full_df),
        "balancing_applied": len(balanced_df) < len(full_df),
        "eval_set_size": len(eval_df),
        "llm_judge_used": use_anthropic,
    }
    
    metadata_file = OUTPUT_DIR / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_file}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("="*80)
    logger.info(f"Final dataset: {len(balanced_df):,} samples")
    logger.info(f"Class distribution:")
    for label, count in balanced_df['complexity_label'].value_counts().items():
        pct = 100 * count / len(balanced_df)
        logger.info(f"  {label:8s}: {count:5d} ({pct:5.1f}%)")
    logger.info(f"\nFiles saved to: {OUTPUT_DIR}")
    logger.info("="*80)
    
    return balanced_df, eval_df


if __name__ == "__main__":
    main()
