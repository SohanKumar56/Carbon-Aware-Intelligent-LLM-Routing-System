"""
fetch_datasets.py — Download and inspect datasets for Prompt Complexity Classifier

This script fetches three datasets from Hugging Face Hub:
1. allenai/WildChat - diverse real user prompts (will sample 15-30k English single-turn)
2. openai/gsm8k - grade-school math problems (multi-step reasoning)
3. SupraLabs/Prompt-Routing-Dataset - already-labeled routing dataset

The script caches datasets locally and prints statistics for manual inspection.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any
import json

from datasets import load_dataset
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory for cached datasets
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_wildchat(sample_size: int = 20000) -> pd.DataFrame:
    """
    Fetch and sample WildChat dataset from Hugging Face.
    
    Filter criteria:
    - English language only
    - Single-turn conversations (first turn only)
    
    Parameters
    ----------
    sample_size : int
        Target number of prompts to sample (default 20000)
    
    Returns
    -------
    pd.DataFrame
        Sampled prompts with columns: prompt, source_dataset
    """
    logger.info("Fetching allenai/WildChat dataset...")
    
    try:
        # Load WildChat - it's a large dataset, so we'll stream and sample
        dataset = load_dataset(
            "allenai/WildChat",
            split="train",
            streaming=True,
        )
        
        prompts = []
        target_count = sample_size
        processed_count = 0
        
        logger.info(f"Sampling up to {target_count} English single-turn prompts...")
        
        for example in dataset:
            processed_count += 1
            
            # Progress logging every 5000 samples
            if processed_count % 5000 == 0:
                logger.info(f"Processed {processed_count} samples, collected {len(prompts)} valid prompts...")
            
            # Stop if we have enough
            if len(prompts) >= target_count:
                break
            
            # Filter for English and single-turn
            if example.get("language") != "English":
                continue
                
            # Get the conversation - WildChat stores as list of turns
            conversation = example.get("conversation", [])
            if not conversation or len(conversation) == 0:
                continue
            
            # Extract first user message only (single-turn)
            first_turn = conversation[0]
            if first_turn.get("role") == "user":
                prompt_text = first_turn.get("content", "").strip()
                
                # Skip empty or very short prompts
                if len(prompt_text) < 10:
                    continue
                
                # Skip extremely long prompts (> 2000 chars) for computational efficiency
                if len(prompt_text) > 2000:
                    continue
                    
                prompts.append({
                    "prompt": prompt_text,
                    "source_dataset": "wildchat"
                })
        
        logger.info(f"Collected {len(prompts)} prompts from WildChat after processing {processed_count} examples")
        
        df = pd.DataFrame(prompts)
        
        # Save to cache
        cache_file = CACHE_DIR / "wildchat_sampled.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached WildChat data to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching WildChat: {e}")
        # Try to load from cache if available
        cache_file = CACHE_DIR / "wildchat_sampled.parquet"
        if cache_file.exists():
            logger.info(f"Loading WildChat from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        raise


def fetch_gsm8k() -> pd.DataFrame:
    """
    Fetch GSM8K math reasoning dataset from Hugging Face.
    
    Returns
    -------
    pd.DataFrame
        Math problems with columns: prompt, source_dataset
    """
    logger.info("Fetching openai/gsm8k dataset...")
    
    try:
        # Load GSM8K train split
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        
        prompts = []
        for example in dataset:
            question = example.get("question", "").strip()
            if question:
                prompts.append({
                    "prompt": question,
                    "source_dataset": "gsm8k",
                    "answer": example.get("answer", "")  # Keep answer for potential validation
                })
        
        logger.info(f"Collected {len(prompts)} prompts from GSM8K")
        
        df = pd.DataFrame(prompts)
        
        # Save to cache
        cache_file = CACHE_DIR / "gsm8k.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached GSM8K data to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching GSM8K: {e}")
        # Try to load from cache if available
        cache_file = CACHE_DIR / "gsm8k.parquet"
        if cache_file.exists():
            logger.info(f"Loading GSM8K from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        raise


def fetch_supralabs_routing() -> pd.DataFrame:
    """
    Fetch SupraLabs Prompt-Routing-Dataset from Hugging Face.
    
    This is our gold-standard labeled dataset for routing/complexity.
    
    Returns
    -------
    pd.DataFrame
        Labeled prompts with columns: prompt, source_dataset, original_label, [other fields]
    """
    logger.info("Fetching SupraLabs/Prompt-Routing-Dataset...")
    
    try:
        # Load the SupraLabs routing dataset
        dataset = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train")
        
        # Convert to pandas for inspection
        df = dataset.to_pandas()
        
        # Add source identifier
        df["source_dataset"] = "supralabs"
        
        # Rename 'text' column to 'prompt' if it exists
        if "text" in df.columns:
            df = df.rename(columns={"text": "prompt"})
        
        # Store original label for mapping later
        if "label" in df.columns:
            df["original_label"] = df["label"]
        
        logger.info(f"Collected {len(df)} prompts from SupraLabs")
        logger.info(f"SupraLabs columns: {list(df.columns)}")
        
        # Save to cache
        cache_file = CACHE_DIR / "supralabs_routing.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached SupraLabs data to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching SupraLabs: {e}")
        # Try to load from cache if available
        cache_file = CACHE_DIR / "supralabs_routing.parquet"
        if cache_file.exists():
            logger.info(f"Loading SupraLabs from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        raise


def print_dataset_stats(df: pd.DataFrame, name: str) -> None:
    """Print statistics and examples for a dataset."""
    
    print(f"\n{'='*80}")
    print(f"{name} Dataset Statistics")
    print(f"{'='*80}")
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for labels if they exist
    if "label" in df.columns or "original_label" in df.columns:
        label_col = "original_label" if "original_label" in df.columns else "label"
        print(f"\nLabel distribution:")
        print(df[label_col].value_counts())
    
    # Prompt length statistics
    if "prompt" in df.columns:
        df["prompt_length"] = df["prompt"].str.len()
        print(f"\nPrompt length statistics:")
        print(df["prompt_length"].describe())
        
        # Token count approximation (rough: chars / 4)
        df["approx_tokens"] = df["prompt_length"] / 4
        print(f"\nApproximate token count statistics:")
        print(df["approx_tokens"].describe())
    
    # Show examples
    print(f"\n{'-'*80}")
    print(f"Example prompts from {name}:")
    print(f"{'-'*80}")
    
    for i, row in df.head(3).iterrows():
        prompt = row.get("prompt", "")
        label_info = ""
        if "original_label" in row:
            label_info = f" [Label: {row['original_label']}]"
        elif "label" in row:
            label_info = f" [Label: {row['label']}]"
        
        print(f"\nExample {i+1}:{label_info}")
        print(f"{prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("Fetching datasets for Prompt Complexity Classifier")
    print("="*80 + "\n")
    
    # Fetch all datasets
    datasets = {}
    
    try:
        # 1. WildChat
        logger.info("\n[1/3] Fetching WildChat...")
        wildchat_df = fetch_wildchat(sample_size=20000)
        datasets["wildchat"] = wildchat_df
        print_dataset_stats(wildchat_df, "WildChat")
        
    except Exception as e:
        logger.error(f"Failed to fetch WildChat: {e}")
        datasets["wildchat"] = None
    
    try:
        # 2. GSM8K
        logger.info("\n[2/3] Fetching GSM8K...")
        gsm8k_df = fetch_gsm8k()
        datasets["gsm8k"] = gsm8k_df
        print_dataset_stats(gsm8k_df, "GSM8K")
        
    except Exception as e:
        logger.error(f"Failed to fetch GSM8K: {e}")
        datasets["gsm8k"] = None
    
    try:
        # 3. SupraLabs
        logger.info("\n[3/3] Fetching SupraLabs Routing Dataset...")
        supralabs_df = fetch_supralabs_routing()
        datasets["supralabs"] = supralabs_df
        print_dataset_stats(supralabs_df, "SupraLabs Routing")
        
    except Exception as e:
        logger.error(f"Failed to fetch SupraLabs: {e}")
        datasets["supralabs"] = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_prompts = 0
    for name, df in datasets.items():
        if df is not None:
            count = len(df)
            total_prompts += count
            print(f"{name:20s}: {count:,} prompts")
        else:
            print(f"{name:20s}: FAILED")
    
    print(f"\n{'TOTAL':20s}: {total_prompts:,} prompts")
    print("="*80 + "\n")
    
    # Save summary metadata
    metadata = {
        "total_prompts": total_prompts,
        "datasets": {
            name: len(df) if df is not None else 0
            for name, df in datasets.items()
        },
        "cache_dir": str(CACHE_DIR),
    }
    
    metadata_file = CACHE_DIR / "fetch_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")
    logger.info("\n✅ Dataset fetching complete!")
    
    return datasets


if __name__ == "__main__":
    main()
