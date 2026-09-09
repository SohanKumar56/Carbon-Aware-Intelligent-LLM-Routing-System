"""
build_labeled_dataset_spark.py — Distributed feature engineering and labeling with PySpark

Applies heuristic complexity labeling at scale using Spark UDFs (User Defined Functions)
and distributed transformations.

This demonstrates:
- Spark UDFs for custom feature extraction
- Distributed data transformations
- Scalable feature engineering
- Parallel label assignment
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, length, lower, trim, lit, when,
    udf, struct, array, concat_ws
)
from pyspark.sql.types import (
    StringType, IntegerType, BooleanType, FloatType,
    StructType, StructField, MapType
)

from spark_config import create_spark_session_preset, stop_spark_session
from fetch_datasets_spark import fetch_all_datasets_spark

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
LABELED_DIR = Path(__file__).parent / "labeled"
LABELED_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SPARK UDFs FOR FEATURE EXTRACTION (Distributed)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_udf(prompt: str) -> Dict[str, any]:
    """
    UDF to extract complexity features from a prompt.
    
    This function will be distributed across Spark workers for parallel execution.
    
    Returns
    -------
    dict
        Feature dictionary with keys:
        - token_count: int
        - char_count: int
        - has_code_blocks: bool
        - question_count: int
        - has_multi_step: bool
        - has_math: bool
        - has_code: bool
        - is_simple: bool
    """
    if not prompt:
        return {
            'token_count': 0,
            'char_count': 0,
            'has_code_blocks': False,
            'question_count': 0,
            'has_multi_step': False,
            'has_math': False,
            'has_code': False,
            'is_simple': False
        }
    
    prompt_lower = prompt.lower()
    
    # Feature 1: Token count (rough approximation)
    token_count = len(prompt) // 4
    
    # Feature 2: Character count
    char_count = len(prompt)
    
    # Feature 3: Code blocks
    has_code_blocks = '```' in prompt or bool(re.search(r'\n\s{4,}', prompt))
    
    # Feature 4: Question count
    question_count = prompt.count('?')
    
    # Feature 5: Multi-step reasoning keywords
    reasoning_keywords = {
        'step by step', 'explain why', 'prove that', 'demonstrate',
        'calculate', 'solve for', 'derive', 'show that',
        'reason about', 'think through', 'work through',
        'justify', 'analyze', 'evaluate', 'compare and contrast',
        "let's think", 'chain of thought', 'reasoning'
    }
    has_multi_step = any(keyword in prompt_lower for keyword in reasoning_keywords)
    
    # Feature 6: Math content
    math_symbols = {'=', '+', '-', '*', '/', '^', '√', '∫', '∑', '∏'}
    math_keywords = {
        'equation', 'formula', 'theorem', 'proof', 'calculate',
        'compute', 'integral', 'derivative', 'matrix', 'vector',
        'probability', 'statistics', 'algebra', 'geometry'
    }
    has_math = (
        any(symbol in prompt for symbol in math_symbols) or
        any(keyword in prompt_lower for keyword in math_keywords)
    )
    
    # Feature 7: Code content
    code_markers = {
        'def ', 'class ', 'import ', 'function', 'return',
        'for ', 'while ', 'if ', 'else:', 'try:', 'except:',
        'python', 'javascript', 'java', 'c++',
        'algorithm', 'implement', 'debug', 'code'
    }
    has_code = any(marker in prompt_lower for marker in code_markers)
    
    # Feature 8: Simple factual question
    simple_patterns = {
        'what is', 'who is', 'when was', 'where is',
        'define', 'meaning of', 'translate',
        'list', 'name', 'tell me about'
    }
    is_simple = (
        len(prompt) < 100 and
        any(pattern in prompt_lower[:50] for pattern in simple_patterns)
    )
    
    return {
        'token_count': token_count,
        'char_count': char_count,
        'has_code_blocks': has_code_blocks,
        'question_count': question_count,
        'has_multi_step': has_multi_step,
        'has_math': has_math,
        'has_code': has_code,
        'is_simple': is_simple
    }


def heuristic_label_udf(prompt: str, source: str, features: Dict) -> str:
    """
    UDF to assign complexity label based on heuristics.
    
    This implements the same logic as the Pandas version but runs distributed.
    
    Parameters
    ----------
    prompt : str
        The prompt text
    source : str
        Source dataset (wildchat, gsm8k, supralabs)
    features : dict
        Extracted features
    
    Returns
    -------
    str
        Complexity label: 'small', 'medium', or 'large'
    """
    # GSM8K: default to large (multi-step math reasoning)
    if source == 'gsm8k':
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


def map_supralabs_category_udf(category: str) -> str:
    """
    UDF to map SupraLabs categories to complexity labels.
    
    Parameters
    ----------
    category : str
        Original SupraLabs category
    
    Returns
    -------
    str
        Mapped complexity: 'small', 'medium', or 'large'
    """
    if not category:
        return 'medium'
    
    category_lower = category.lower()
    
    # Simple/easy categories → small
    if any(keyword in category_lower for keyword in ['simple', 'easy', 'basic', 'quick']):
        return 'small'
    
    # Complex/hard categories → large
    if any(keyword in category_lower for keyword in ['complex', 'hard', 'advanced', 'difficult']):
        return 'large'
    
    # Default → medium
    return 'medium'


# Register UDFs with Spark
# Schema for features
features_schema = MapType(StringType(), StringType())

# Schema for feature extraction (returns dict)
feature_extraction_schema = StructType([
    StructField("token_count", IntegerType(), False),
    StructField("char_count", IntegerType(), False),
    StructField("has_code_blocks", BooleanType(), False),
    StructField("question_count", IntegerType(), False),
    StructField("has_multi_step", BooleanType(), False),
    StructField("has_math", BooleanType(), False),
    StructField("has_code", BooleanType(), False),
    StructField("is_simple", BooleanType(), False),
])


def register_udfs(spark: SparkSession):
    """Register all UDFs with the Spark session."""
    
    # Feature extraction UDF
    spark.udf.register("extract_features", extract_features_udf, feature_extraction_schema)
    
    # Labeling UDF
    spark.udf.register("heuristic_label", heuristic_label_udf, StringType())
    
    # Category mapping UDF
    spark.udf.register("map_supralabs_category", map_supralabs_category_udf, StringType())
    
    logger.info("✓ Registered Spark UDFs for distributed feature extraction")


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED FEATURE ENGINEERING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def apply_feature_engineering_spark(df: DataFrame) -> DataFrame:
    """
    Apply feature engineering using distributed Spark transformations.
    
    This extracts features from prompts in parallel across all Spark workers.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with columns: prompt, source_dataset
    
    Returns
    -------
    DataFrame
        DataFrame with added feature columns
    """
    logger.info("Applying distributed feature engineering...")
    
    # Create UDF from Python function
    extract_features_spark = udf(extract_features_udf, feature_extraction_schema)
    
    # Apply feature extraction (distributed across all partitions)
    df_with_features = df.withColumn("features", extract_features_spark(col("prompt")))
    
    # Expand feature struct into separate columns
    df_expanded = df_with_features.select(
        col("prompt"),
        col("source_dataset"),
        col("original_category"),
        col("features.token_count").alias("token_count"),
        col("features.char_count").alias("char_count"),
        col("features.has_code_blocks").alias("has_code_blocks"),
        col("features.question_count").alias("question_count"),
        col("features.has_multi_step").alias("has_multi_step"),
        col("features.has_math").alias("has_math"),
        col("features.has_code").alias("has_code"),
        col("features.is_simple").alias("is_simple")
    )
    
    logger.info("✓ Feature engineering completed (distributed)")
    
    return df_expanded


def apply_heuristic_labeling_spark(df: DataFrame) -> DataFrame:
    """
    Apply heuristic complexity labeling using distributed Spark UDFs.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with extracted features
    
    Returns
    -------
    DataFrame
        DataFrame with 'complexity' column added
    """
    logger.info("Applying distributed heuristic labeling...")
    
    # Create struct of features for UDF
    features_struct = struct(
        col("token_count"),
        col("char_count"),
        col("has_code_blocks"),
        col("question_count"),
        col("has_multi_step"),
        col("has_math"),
        col("has_code"),
        col("is_simple")
    )
    
    # Define UDF for labeling
    heuristic_label_spark = udf(
        lambda prompt, source, features: heuristic_label_udf(prompt, source, features),
        StringType()
    )
    
    # Apply labeling (distributed)
    df_labeled = df.withColumn(
        "complexity_heuristic",
        heuristic_label_spark(col("prompt"), col("source_dataset"), features_struct)
    )
    
    # For SupraLabs, use category mapping instead
    map_category_spark = udf(map_supralabs_category_udf, StringType())
    
    df_final = df_labeled.withColumn(
        "complexity",
        when(
            col("source_dataset") == "supralabs",
            map_category_spark(col("original_category"))
        ).otherwise(col("complexity_heuristic"))
    ).drop("complexity_heuristic")
    
    logger.info("✓ Heuristic labeling completed (distributed)")
    
    return df_final


def build_labeled_dataset_spark(
    spark: SparkSession,
    wildchat_size: int = 20000,
    use_cache: bool = True
) -> DataFrame:
    """
    Build complete labeled dataset using PySpark distributed processing.
    
    This is the main pipeline that demonstrates scalable ML data preprocessing:
    1. Fetch datasets (distributed loading)
    2. Feature extraction (distributed UDFs)
    3. Heuristic labeling (distributed transformations)
    4. Dataset balancing (distributed sampling)
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    wildchat_size : int
        Number of WildChat samples
    use_cache : bool
        Whether to use cached results
    
    Returns
    -------
    DataFrame
        Labeled Spark DataFrame ready for training
    """
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING LABELED DATASET WITH PYSPARK")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Fetch all datasets (distributed)
    logger.info("Step 1/4: Fetching datasets...")
    raw_df = fetch_all_datasets_spark(spark, wildchat_size, use_cache)
    
    # Step 2: Feature engineering (distributed)
    logger.info("\nStep 2/4: Extracting features (distributed)...")
    featured_df = apply_feature_engineering_spark(raw_df)
    
    # Cache for performance (stores in memory/disk across cluster)
    featured_df = featured_df.cache()
    
    # Step 3: Apply labeling (distributed)
    logger.info("\nStep 3/4: Applying heuristic labels (distributed)...")
    labeled_df = apply_heuristic_labeling_spark(featured_df)
    
    # Step 4: Compute statistics (distributed aggregation)
    logger.info("\nStep 4/4: Computing dataset statistics...")
    
    total_count = labeled_df.count()
    logger.info(f"Total labeled prompts: {total_count:,}")
    
    # Class distribution (distributed group by)
    logger.info("\nComplexity distribution:")
    class_dist = labeled_df.groupBy("complexity").count().orderBy("complexity")
    class_dist.show()
    
    # Source distribution (distributed group by)
    logger.info("\nSource distribution:")
    source_dist = labeled_df.groupBy("source_dataset").count().orderBy(col("count").desc())
    source_dist.show()
    
    # Cross-tabulation: source vs complexity (distributed pivot)
    logger.info("\nCross-tabulation (Source × Complexity):")
    crosstab = labeled_df.groupBy("source_dataset").pivot("complexity").count()
    crosstab.show()
    
    logger.info("\n" + "=" * 70)
    logger.info("LABELED DATASET READY FOR TRAINING")
    logger.info("=" * 70 + "\n")
    
    return labeled_df


def save_labeled_dataset_spark(df: DataFrame, output_dir: Path):
    """
    Save labeled dataset in multiple formats.
    
    Demonstrates Spark's ability to write large datasets efficiently.
    
    Parameters
    ----------
    df : DataFrame
        Labeled Spark DataFrame
    output_dir : Path
        Output directory
    """
    logger.info("Saving labeled dataset...")
    
    # Save as Parquet (efficient columnar format, partitioned)
    parquet_path = output_dir / "prompt_complexity_labeled_spark.parquet"
    logger.info(f"Saving Parquet to {parquet_path}")
    
    df.write.mode("overwrite").parquet(str(parquet_path))
    
    # Save as CSV for human inspection (single partition for single file)
    csv_path = output_dir / "prompt_complexity_labeled_spark.csv"
    logger.info(f"Saving CSV to {csv_path}")
    
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_path))
    
    # Save metadata
    metadata = {
        "total_samples": df.count(),
        "source": "PySpark distributed processing",
        "features": [
            "token_count", "char_count", "has_code_blocks",
            "question_count", "has_multi_step", "has_math",
            "has_code", "is_simple"
        ]
    }
    
    import json
    metadata_path = output_dir / "dataset_metadata_spark.json"
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Dataset saved successfully")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("PYSPARK FEATURE ENGINEERING & LABELING DEMONSTRATION")
    print("=" * 70 + "\n")
    
    # Create Spark session with optimized settings
    spark = create_spark_session_preset('local_full')
    
    try:
        # Build labeled dataset (distributed)
        labeled_df = build_labeled_dataset_spark(
            spark,
            wildchat_size=20000,
            use_cache=True
        )
        
        # Show sample
        print("\nSample labeled data:")
        labeled_df.select(
            "prompt",
            "source_dataset",
            "complexity",
            "token_count",
            "has_math",
            "has_code"
        ).show(10, truncate=50)
        
        # Save results
        save_labeled_dataset_spark(labeled_df, LABELED_DIR)
        
        # Show final statistics
        print("\nFinal dataset summary:")
        print(f"Total rows: {labeled_df.count():,}")
        print(f"Partitions: {labeled_df.rdd.getNumPartitions()}")
        print(f"Columns: {len(labeled_df.columns)}")
        
        logger.info("\n✓ PySpark preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Clean up
        stop_spark_session(spark)
