"""
compare_pandas_vs_spark.py — Performance benchmark: Pandas vs PySpark preprocessing

Demonstrates the scalability benefits of distributed processing by comparing
execution time and resource usage between Pandas and PySpark implementations.

This script provides empirical evidence for:
- When to use distributed computing (PySpark)
- When traditional single-machine processing (Pandas) is sufficient
- Trade-offs between simplicity and scalability
"""

from __future__ import annotations
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Tuple
import psutil

import pandas as pd

# Import both implementations
from fetch_datasets import fetch_wildchat, fetch_gsm8k, fetch_supralabs_routing
from build_labeled_dataset import (
    apply_heuristic_labels,
    ComplexityHeuristics
)

# Import Spark versions
try:
    from pyspark.sql import SparkSession
    from spark_config import create_spark_session_preset, stop_spark_session
    from fetch_datasets_spark import fetch_all_datasets_spark
    from build_labeled_dataset_spark import (
        apply_feature_engineering_spark,
        apply_heuristic_labeling_spark
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark not installed. Install with: pip install pyspark>=3.5.0")

logger = logging.getLogger(__name__)

# Benchmark configuration
BENCHMARK_SIZES = [1000, 5000, 10000, 20000]  # Sample sizes to test
CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_pandas_pipeline(sample_size: int) -> Dict:
    """
    Benchmark Pandas preprocessing pipeline.
    
    Parameters
    ----------
    sample_size : int
        Number of samples to process
    
    Returns
    -------
    dict
        Benchmark results with timing and memory metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PANDAS BENCHMARK - {sample_size:,} samples")
    logger.info(f"{'='*60}")
    
    results = {
        'method': 'pandas',
        'sample_size': sample_size,
        'times': {},
        'memory': {}
    }
    
    start_memory = get_memory_usage()
    total_start = time.perf_counter()
    
    # Step 1: Fetch datasets
    logger.info("Step 1: Fetching datasets...")
    step_start = time.perf_counter()
    
    try:
        # Scale down each dataset proportionally
        wildchat_portion = int(sample_size * 0.70)  # 70%
        gsm8k_df = fetch_gsm8k()
        gsm8k_portion = min(len(gsm8k_df), int(sample_size * 0.26))  # 26%
        supralabs_df = fetch_supralabs_routing()
        supralabs_portion = min(len(supralabs_df), int(sample_size * 0.04))  # 4%
        
        wildchat_df = fetch_wildchat(sample_size=wildchat_portion)
        gsm8k_df = gsm8k_df.head(gsm8k_portion)
        supralabs_df = supralabs_df.head(supralabs_portion)
        
        # Combine
        combined_df = pd.concat([wildchat_df, gsm8k_df, supralabs_df], ignore_index=True)
        
        results['times']['fetch'] = time.perf_counter() - step_start
        results['memory']['after_fetch'] = get_memory_usage()
        logger.info(f"✓ Fetched {len(combined_df):,} samples in {results['times']['fetch']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None
    
    # Step 2: Feature extraction
    logger.info("\nStep 2: Extracting features...")
    step_start = time.perf_counter()
    
    feature_rows = []
    for _, row in combined_df.iterrows():
        features = ComplexityHeuristics.compute_features(row['prompt'])
        feature_row = {**row.to_dict(), **features}
        feature_rows.append(feature_row)
    
    featured_df = pd.DataFrame(feature_rows)
    
    results['times']['features'] = time.perf_counter() - step_start
    results['memory']['after_features'] = get_memory_usage()
    logger.info(f"✓ Extracted features in {results['times']['features']:.2f}s")
    
    # Step 3: Apply labeling
    logger.info("\nStep 3: Applying labels...")
    step_start = time.perf_counter()
    
    labeled_df = apply_heuristic_labels(featured_df, "combined")
    
    results['times']['labeling'] = time.perf_counter() - step_start
    results['memory']['after_labeling'] = get_memory_usage()
    logger.info(f"✓ Applied labels in {results['times']['labeling']:.2f}s")
    
    # Step 4: Aggregations
    logger.info("\nStep 4: Computing statistics...")
    step_start = time.perf_counter()
    
    complexity_dist = labeled_df['complexity'].value_counts()
    source_dist = labeled_df['source_dataset'].value_counts()
    
    results['times']['aggregations'] = time.perf_counter() - step_start
    logger.info(f"✓ Computed statistics in {results['times']['aggregations']:.2f}s")
    
    # Total time and memory
    results['times']['total'] = time.perf_counter() - total_start
    results['memory']['peak'] = get_memory_usage()
    results['memory']['delta'] = results['memory']['peak'] - start_memory
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PANDAS TOTAL TIME: {results['times']['total']:.2f}s")
    logger.info(f"PEAK MEMORY: {results['memory']['peak']:.1f} MB")
    logger.info(f"MEMORY DELTA: {results['memory']['delta']:.1f} MB")
    logger.info(f"{'='*60}")
    
    return results


def benchmark_spark_pipeline(sample_size: int, spark: SparkSession) -> Dict:
    """
    Benchmark PySpark preprocessing pipeline.
    
    Parameters
    ----------
    sample_size : int
        Number of samples to process
    spark : SparkSession
        Active Spark session
    
    Returns
    -------
    dict
        Benchmark results with timing and memory metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PYSPARK BENCHMARK - {sample_size:,} samples")
    logger.info(f"{'='*60}")
    
    results = {
        'method': 'pyspark',
        'sample_size': sample_size,
        'times': {},
        'memory': {},
        'partitions': {}
    }
    
    start_memory = get_memory_usage()
    total_start = time.perf_counter()
    
    # Step 1: Fetch datasets (distributed)
    logger.info("Step 1: Fetching datasets (distributed)...")
    step_start = time.perf_counter()
    
    try:
        # Scale wildchat to match sample size
        wildchat_size = int(sample_size * 0.70)
        combined_df = fetch_all_datasets_spark(spark, wildchat_size, cache=False)
        
        # Limit to target sample size
        combined_df = combined_df.limit(sample_size)
        
        results['times']['fetch'] = time.perf_counter() - step_start
        results['memory']['after_fetch'] = get_memory_usage()
        results['partitions']['initial'] = combined_df.rdd.getNumPartitions()
        
        actual_count = combined_df.count()
        logger.info(f"✓ Fetched {actual_count:,} samples in {results['times']['fetch']:.2f}s")
        logger.info(f"  Partitions: {results['partitions']['initial']}")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None
    
    # Step 2: Feature extraction (distributed)
    logger.info("\nStep 2: Extracting features (distributed)...")
    step_start = time.perf_counter()
    
    featured_df = apply_feature_engineering_spark(combined_df)
    featured_df.cache()  # Cache for reuse
    _ = featured_df.count()  # Force execution
    
    results['times']['features'] = time.perf_counter() - step_start
    results['memory']['after_features'] = get_memory_usage()
    results['partitions']['after_features'] = featured_df.rdd.getNumPartitions()
    logger.info(f"✓ Extracted features in {results['times']['features']:.2f}s")
    
    # Step 3: Apply labeling (distributed)
    logger.info("\nStep 3: Applying labels (distributed)...")
    step_start = time.perf_counter()
    
    labeled_df = apply_heuristic_labeling_spark(featured_df)
    labeled_df.cache()
    _ = labeled_df.count()  # Force execution
    
    results['times']['labeling'] = time.perf_counter() - step_start
    results['memory']['after_labeling'] = get_memory_usage()
    logger.info(f"✓ Applied labels in {results['times']['labeling']:.2f}s")
    
    # Step 4: Aggregations (distributed)
    logger.info("\nStep 4: Computing statistics (distributed)...")
    step_start = time.perf_counter()
    
    complexity_dist = labeled_df.groupBy("complexity").count().collect()
    source_dist = labeled_df.groupBy("source_dataset").count().collect()
    
    results['times']['aggregations'] = time.perf_counter() - step_start
    logger.info(f"✓ Computed statistics in {results['times']['aggregations']:.2f}s")
    
    # Clean up
    featured_df.unpersist()
    labeled_df.unpersist()
    
    # Total time and memory
    results['times']['total'] = time.perf_counter() - total_start
    results['memory']['peak'] = get_memory_usage()
    results['memory']['delta'] = results['memory']['peak'] - start_memory
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PYSPARK TOTAL TIME: {results['times']['total']:.2f}s")
    logger.info(f"PEAK MEMORY: {results['memory']['peak']:.1f} MB")
    logger.info(f"MEMORY DELTA: {results['memory']['delta']:.1f} MB")
    logger.info(f"{'='*60}")
    
    return results


def print_comparison_table(pandas_results: list, spark_results: list):
    """
    Print formatted comparison table of benchmark results.
    
    Parameters
    ----------
    pandas_results : list
        List of Pandas benchmark results
    spark_results : list
        List of PySpark benchmark results
    """
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON: Pandas vs PySpark")
    print("=" * 90)
    
    # Header
    print(f"\n{'Sample Size':<15} {'Method':<10} {'Total Time':<15} {'Speedup':<12} {'Memory (MB)':<15}")
    print("-" * 90)
    
    # Compare each size
    for i, size in enumerate(BENCHMARK_SIZES):
        if i >= len(pandas_results) or i >= len(spark_results):
            break
        
        pandas = pandas_results[i]
        spark = spark_results[i]
        
        if pandas and spark:
            speedup = pandas['times']['total'] / spark['times']['total']
            
            # Pandas row
            print(f"{size:<15,} {'Pandas':<10} {pandas['times']['total']:<15.2f}s "
                  f"{'—':<12} {pandas['memory']['peak']:<15.1f}")
            
            # Spark row
            print(f"{'':<15} {'PySpark':<10} {spark['times']['total']:<15.2f}s "
                  f"{speedup:<12.2f}x {spark['memory']['peak']:<15.1f}")
            
            print()
    
    print("=" * 90)


def save_results_csv(pandas_results: list, spark_results: list):
    """Save benchmark results to CSV for analysis."""
    
    rows = []
    for pandas, spark in zip(pandas_results, spark_results):
        if pandas and spark:
            rows.append({
                'sample_size': pandas['sample_size'],
                'pandas_total_time': pandas['times']['total'],
                'pandas_fetch_time': pandas['times']['fetch'],
                'pandas_features_time': pandas['times']['features'],
                'pandas_labeling_time': pandas['times']['labeling'],
                'pandas_agg_time': pandas['times']['aggregations'],
                'pandas_memory_mb': pandas['memory']['peak'],
                'spark_total_time': spark['times']['total'],
                'spark_fetch_time': spark['times']['fetch'],
                'spark_features_time': spark['times']['features'],
                'spark_labeling_time': spark['times']['labeling'],
                'spark_agg_time': spark['times']['aggregations'],
                'spark_memory_mb': spark['memory']['peak'],
                'spark_partitions': spark['partitions']['initial'],
                'speedup': pandas['times']['total'] / spark['times']['total'],
            })
    
    df = pd.DataFrame(rows)
    output_file = RESULTS_DIR / "pandas_vs_spark_benchmark.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Results saved to: {output_file}")


def main():
    """Run complete benchmark suite."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 90)
    print("PANDAS VS PYSPARK PREPROCESSING BENCHMARK")
    print("=" * 90)
    print("\nThis benchmark compares preprocessing performance between:")
    print("  - Pandas: Single-threaded, in-memory processing")
    print("  - PySpark: Distributed, parallel processing")
    print("\nTest sizes:", BENCHMARK_SIZES)
    print("=" * 90)
    
    if not SPARK_AVAILABLE:
        print("\n⚠️  PySpark not available. Install with:")
        print("    pip install pyspark>=3.5.0 pyarrow>=14.0.1")
        sys.exit(1)
    
    # Create Spark session
    print("\nInitializing Spark session...")
    spark = create_spark_session_preset('local_full')
    print(f"✓ Spark version: {spark.version}")
    print(f"✓ Available cores: {spark.sparkContext.defaultParallelism}")
    
    pandas_results = []
    spark_results = []
    
    try:
        # Run benchmarks for each size
        for size in BENCHMARK_SIZES:
            print(f"\n{'#'*90}")
            print(f"BENCHMARKING: {size:,} samples")
            print(f"{'#'*90}")
            
            # Pandas benchmark
            pandas_result = benchmark_pandas_pipeline(size)
            if pandas_result:
                pandas_results.append(pandas_result)
            
            # PySpark benchmark
            spark_result = benchmark_spark_pipeline(size, spark)
            if spark_result:
                spark_results.append(spark_result)
            
            # Show comparison
            if pandas_result and spark_result:
                speedup = pandas_result['times']['total'] / spark_result['times']['total']
                print(f"\n🎯 SPEEDUP: {speedup:.2f}x faster with PySpark")
        
        # Print final comparison
        print_comparison_table(pandas_results, spark_results)
        
        # Save results
        save_results_csv(pandas_results, spark_results)
        
        print("\n" + "=" * 90)
        print("✓ BENCHMARK COMPLETE")
        print("=" * 90)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
    finally:
        # Clean up
        print("\nCleaning up...")
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
