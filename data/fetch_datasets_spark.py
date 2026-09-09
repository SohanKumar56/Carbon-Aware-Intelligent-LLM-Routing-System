"""
fetch_datasets_spark.py — Distributed dataset fetching and preprocessing with PySpark

Fetches datasets from Hugging Face and applies distributed filtering and sampling
using Spark for scalability.

This is the PySpark equivalent of fetch_datasets.py, demonstrating distributed
data processing at scale.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, length, trim, lower, lit, 
    when, size, split, regexp_extract,
    udf, explode, first, struct
)
from pyspark.sql.types import StringType, IntegerType, BooleanType, StructType, StructField

from datasets import load_dataset
from spark_config import create_spark_session_preset, get_optimal_partitions, stop_spark_session

logger = logging.getLogger(__name__)

# Cache directory for intermediate results
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_wildchat_spark(
    spark: SparkSession,
    sample_size: int = 20000,
    cache: bool = True
) -> DataFrame:
    """
    Fetch and preprocess WildChat dataset using PySpark for distributed processing.
    
    This demonstrates Spark's ability to handle streaming datasets and apply
    distributed transformations.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    sample_size : int
        Target number of prompts to sample
    cache : bool
        Whether to cache results to disk
    
    Returns
    -------
    DataFrame
        Spark DataFrame with columns: prompt, source_dataset
    """
    cache_file = CACHE_DIR / "wildchat_sampled_spark.parquet"
    
    # Check cache first
    if cache and cache_file.exists():
        logger.info(f"Loading WildChat from Spark cache: {cache_file}")
        return spark.read.parquet(str(cache_file))
    
    logger.info("Fetching allenai/WildChat dataset...")
    
    # Step 1: Load dataset using Hugging Face (this part is sequential)
    # In production, this could be loaded from HDFS/S3
    dataset = load_dataset(
        "allenai/WildChat",
        split="train",
        streaming=True,
    )
    
    # Step 2: Extract prompts into Pandas (for conversion to Spark)
    logger.info(f"Sampling up to {sample_size} English single-turn prompts...")
    
    prompts_data = []
    processed_count = 0
    target_count = sample_size * 2  # Over-sample, then filter
    
    for example in dataset:
        processed_count += 1
        
        if processed_count % 5000 == 0:
            logger.info(f"Processed {processed_count} samples, collected {len(prompts_data)}...")
        
        if len(prompts_data) >= target_count:
            break
        
        # Filter for English
        if example.get("language") != "English":
            continue
        
        # Get conversation
        conversation = example.get("conversation", [])
        if not conversation:
            continue
        
        # Extract first user message
        first_turn = conversation[0]
        if first_turn.get("role") == "user":
            prompt_text = first_turn.get("content", "").strip()
            
            if prompt_text:  # Not empty
                prompts_data.append({
                    "prompt": prompt_text,
                    "source_dataset": "wildchat",
                    "raw_length": len(prompt_text)
                })
    
    logger.info(f"Collected {len(prompts_data)} raw prompts from WildChat")
    
    # Step 3: Convert to Spark DataFrame for distributed processing
    schema = StructType([
        StructField("prompt", StringType(), False),
        StructField("source_dataset", StringType(), False),
        StructField("raw_length", IntegerType(), False)
    ])
    
    pandas_df = pd.DataFrame(prompts_data)
    spark_df = spark.createDataFrame(pandas_df, schema=schema)
    
    # Calculate optimal partitions
    estimated_size_mb = len(prompts_data) * 0.5 / 1024  # Rough estimate
    num_partitions = get_optimal_partitions(spark, estimated_size_mb)
    spark_df = spark_df.repartition(num_partitions)
    
    logger.info(f"Created Spark DataFrame with {num_partitions} partitions")
    
    # Step 4: Apply distributed filtering
    logger.info("Applying distributed filters...")
    
    filtered_df = (
        spark_df
        # Filter 1: Remove too short prompts (< 10 chars)
        .filter(col("raw_length") >= 10)
        # Filter 2: Remove too long prompts (> 2000 chars)
        .filter(col("raw_length") <= 2000)
        # Filter 3: Remove whitespace-only prompts
        .filter(trim(col("prompt")) != "")
        # Drop the temporary length column
        .drop("raw_length")
    )
    
    # Step 5: Sample to target size if needed
    total_count = filtered_df.count()
    logger.info(f"After filtering: {total_count} prompts")
    
    if total_count > sample_size:
        # Distributed random sampling
        fraction = sample_size / total_count
        result_df = filtered_df.sample(withReplacement=False, fraction=fraction, seed=42)
        result_df = result_df.limit(sample_size)
    else:
        result_df = filtered_df
    
    final_count = result_df.count()
    logger.info(f"Final WildChat dataset: {final_count} prompts")
    
    # Step 6: Cache results
    if cache:
        logger.info(f"Caching to {cache_file}")
        result_df.write.mode("overwrite").parquet(str(cache_file))
    
    return result_df


def fetch_gsm8k_spark(
    spark: SparkSession,
    cache: bool = True
) -> DataFrame:
    """
    Fetch GSM8K math reasoning dataset using PySpark.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    cache : bool
        Whether to cache results
    
    Returns
    -------
    DataFrame
        Spark DataFrame with columns: prompt, source_dataset
    """
    cache_file = CACHE_DIR / "gsm8k_spark.parquet"
    
    if cache and cache_file.exists():
        logger.info(f"Loading GSM8K from Spark cache: {cache_file}")
        return spark.read.parquet(str(cache_file))
    
    logger.info("Fetching openai/gsm8k dataset...")
    
    # Load from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # Extract questions
    prompts_data = [
        {
            "prompt": example["question"].strip(),
            "source_dataset": "gsm8k"
        }
        for example in dataset
        if example["question"].strip()
    ]
    
    logger.info(f"Loaded {len(prompts_data)} GSM8K math problems")
    
    # Convert to Spark DataFrame
    pandas_df = pd.DataFrame(prompts_data)
    spark_df = spark.createDataFrame(pandas_df)
    
    # Apply distributed filtering
    result_df = (
        spark_df
        .filter(length(col("prompt")) >= 10)
        .filter(length(col("prompt")) <= 2000)
    )
    
    final_count = result_df.count()
    logger.info(f"Final GSM8K dataset: {final_count} prompts")
    
    if cache:
        logger.info(f"Caching to {cache_file}")
        result_df.write.mode("overwrite").parquet(str(cache_file))
    
    return result_df


def fetch_supralabs_spark(
    spark: SparkSession,
    cache: bool = True
) -> DataFrame:
    """
    Fetch SupraLabs routing dataset using PySpark.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    cache : bool
        Whether to cache results
    
    Returns
    -------
    DataFrame
        Spark DataFrame with columns: prompt, source_dataset, original_category
    """
    cache_file = CACHE_DIR / "supralabs_spark.parquet"
    
    if cache and cache_file.exists():
        logger.info(f"Loading SupraLabs from Spark cache: {cache_file}")
        return spark.read.parquet(str(cache_file))
    
    logger.info("Fetching SupraLabs/Prompt-Routing-Dataset...")
    
    # Load from Hugging Face
    dataset = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train")
    
    # Extract prompts and categories
    prompts_data = [
        {
            "prompt": example["prompt"].strip(),
            "source_dataset": "supralabs",
            "original_category": example.get("category", "unknown")
        }
        for example in dataset
        if example.get("prompt", "").strip()
    ]
    
    logger.info(f"Loaded {len(prompts_data)} SupraLabs routing examples")
    
    # Convert to Spark DataFrame
    pandas_df = pd.DataFrame(prompts_data)
    spark_df = spark.createDataFrame(pandas_df)
    
    final_count = spark_df.count()
    logger.info(f"Final SupraLabs dataset: {final_count} prompts")
    
    if cache:
        logger.info(f"Caching to {cache_file}")
        spark_df.write.mode("overwrite").parquet(str(cache_file))
    
    return spark_df


def fetch_all_datasets_spark(
    spark: SparkSession,
    wildchat_size: int = 20000,
    cache: bool = True
) -> DataFrame:
    """
    Fetch all datasets and combine them using Spark's distributed union.
    
    This demonstrates Spark's ability to combine multiple data sources
    at scale with proper partitioning.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    wildchat_size : int
        Number of WildChat prompts to sample
    cache : bool
        Whether to use caching
    
    Returns
    -------
    DataFrame
        Combined Spark DataFrame with all datasets
    """
    logger.info("=" * 70)
    logger.info("FETCHING ALL DATASETS WITH PYSPARK")
    logger.info("=" * 70)
    
    # Fetch each dataset (distributed processing)
    wildchat_df = fetch_wildchat_spark(spark, wildchat_size, cache)
    gsm8k_df = fetch_gsm8k_spark(spark, cache)
    supralabs_df = fetch_supralabs_spark(spark, cache)
    
    # Standardize schemas (add missing columns with nulls)
    wildchat_df = wildchat_df.withColumn("original_category", lit(None).cast(StringType()))
    gsm8k_df = gsm8k_df.withColumn("original_category", lit(None).cast(StringType()))
    
    # Union all datasets (distributed)
    logger.info("Combining datasets with distributed union...")
    combined_df = wildchat_df.unionByName(gsm8k_df).unionByName(supralabs_df)
    
    # Get counts per source (distributed aggregation)
    logger.info("Computing dataset statistics...")
    source_counts = combined_df.groupBy("source_dataset").count().collect()
    
    total_count = combined_df.count()
    
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY (PySpark)")
    logger.info("=" * 70)
    logger.info(f"Total prompts: {total_count:,}")
    logger.info("\nBy source:")
    for row in source_counts:
        logger.info(f"  - {row['source_dataset']}: {row['count']:,}")
    logger.info("=" * 70 + "\n")
    
    return combined_df


def print_spark_dataset_stats(df: DataFrame, name: str):
    """
    Print detailed statistics about a Spark DataFrame.
    
    Demonstrates Spark's distributed aggregation capabilities.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Dataset: {name}")
    logger.info(f"{'=' * 60}")
    
    # Basic stats (distributed computation)
    total_count = df.count()
    logger.info(f"Total rows: {total_count:,}")
    logger.info(f"Partitions: {df.rdd.getNumPartitions()}")
    
    # Prompt length statistics (distributed)
    length_stats = df.select(
        length(col("prompt")).alias("length")
    ).summary("count", "mean", "min", "max", "25%", "50%", "75%")
    
    logger.info("\nPrompt length statistics:")
    length_stats.show()
    
    # Source distribution (if available)
    if "source_dataset" in df.columns:
        logger.info("\nSource distribution:")
        df.groupBy("source_dataset").count().orderBy(col("count").desc()).show()
    
    logger.info(f"{'=' * 60}\n")


# Main execution
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("PYSPARK DATA FETCHING DEMONSTRATION")
    print("=" * 70 + "\n")
    
    # Create Spark session
    spark = create_spark_session_preset('local_full')
    
    try:
        # Fetch all datasets
        combined_df = fetch_all_datasets_spark(
            spark,
            wildchat_size=20000,
            cache=True
        )
        
        # Show sample
        print("\nSample data:")
        combined_df.show(10, truncate=50)
        
        # Detailed statistics
        print_spark_dataset_stats(combined_df, "Combined Dataset")
        
        # Save combined dataset
        output_file = CACHE_DIR / "combined_raw_spark.parquet"
        logger.info(f"Saving combined dataset to {output_file}")
        combined_df.write.mode("overwrite").parquet(str(output_file))
        
        logger.info("\n✓ Data fetching with PySpark completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data fetching: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Clean up
        stop_spark_session(spark)
