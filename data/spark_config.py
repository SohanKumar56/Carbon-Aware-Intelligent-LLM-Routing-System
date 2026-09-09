"""
spark_config.py — Centralized Spark session configuration
Handles both local development and cluster deployment modes.
"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import logging

logger = logging.getLogger(__name__)


def create_spark_session(
    app_name: str = "Carbon_Aware_LLM_Preprocessing",
    mode: str = "local",
    cores: str = "*",
    memory: str = "4g",
    **extra_configs
) -> SparkSession:
    """
    Create and configure a Spark session for data preprocessing.
    
    Parameters
    ----------
    app_name : str
        Name of the Spark application
    mode : str
        Deployment mode: 'local', 'local[*]', 'yarn', 'standalone'
    cores : str
        Number of cores for local mode (default '*' = all available)
    memory : str
        Driver memory allocation (default '4g')
    extra_configs : dict
        Additional Spark configuration parameters
    
    Returns
    -------
    SparkSession
        Configured Spark session
        
    Examples
    --------
    >>> # Local mode with all cores
    >>> spark = create_spark_session(mode='local', cores='*')
    
    >>> # Local mode with specific cores
    >>> spark = create_spark_session(mode='local', cores='4')
    
    >>> # Cluster mode
    >>> spark = create_spark_session(mode='yarn')
    """
    
    # Build master URL
    if mode == 'local':
        master = f"local[{cores}]"
    else:
        master = mode
    
    logger.info(f"Creating Spark session: {app_name} on {master}")
    
    # Base configuration
    conf = SparkConf()
    conf.set("spark.app.name", app_name)
    conf.set("spark.driver.memory", memory)
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")  # Pandas interop
    conf.set("spark.sql.adaptive.enabled", "true")  # Adaptive query execution
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")  # Auto-coalesce
    
    # Serialization optimization
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    # Memory management
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.sql.shuffle.partitions", "200")  # Default shuffle partitions
    
    # UI and logging
    conf.set("spark.ui.showConsoleProgress", "true")
    
    # Apply extra configs
    for key, value in extra_configs.items():
        conf.set(key, value)
    
    # Create session
    spark = (
        SparkSession.builder
        .master(master)
        .config(conf=conf)
        .getOrCreate()
    )
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")  # Reduce verbosity
    
    logger.info(f"Spark session created: {spark.version}")
    logger.info(f"Master: {spark.sparkContext.master}")
    logger.info(f"Available cores: {spark.sparkContext.defaultParallelism}")
    
    return spark


def get_optimal_partitions(spark: SparkSession, dataset_size_mb: float) -> int:
    """
    Calculate optimal number of partitions based on dataset size.
    
    Rule of thumb: 128-256 MB per partition for optimal performance.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    dataset_size_mb : float
        Estimated dataset size in megabytes
    
    Returns
    -------
    int
        Recommended number of partitions
    """
    cores = spark.sparkContext.defaultParallelism
    
    # Calculate partitions (aim for ~200MB per partition)
    size_based_partitions = max(1, int(dataset_size_mb / 200))
    
    # Ensure at least 2x cores for parallelism
    min_partitions = cores * 2
    
    optimal = max(min_partitions, size_based_partitions)
    
    logger.info(f"Optimal partitions: {optimal} (cores={cores}, size={dataset_size_mb}MB)")
    
    return optimal


def stop_spark_session(spark: SparkSession):
    """Stop the Spark session and release resources."""
    if spark:
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Spark session stopped successfully")


# Preset configurations for common scenarios
PRESET_CONFIGS = {
    "local_dev": {
        "mode": "local",
        "cores": "2",
        "memory": "2g",
        "spark.executor.memory": "2g",
    },
    "local_full": {
        "mode": "local",
        "cores": "*",
        "memory": "4g",
        "spark.executor.memory": "4g",
    },
    "local_large": {
        "mode": "local",
        "cores": "*",
        "memory": "8g",
        "spark.executor.memory": "8g",
        "spark.sql.shuffle.partitions": "400",
    },
    "cluster": {
        "mode": "yarn",
        "memory": "4g",
        "spark.executor.memory": "8g",
        "spark.executor.cores": "4",
        "spark.executor.instances": "10",
        "spark.sql.shuffle.partitions": "800",
    }
}


def create_spark_session_preset(preset: str = "local_full") -> SparkSession:
    """
    Create Spark session using a preset configuration.
    
    Parameters
    ----------
    preset : str
        Preset name: 'local_dev', 'local_full', 'local_large', 'cluster'
    
    Returns
    -------
    SparkSession
        Configured Spark session
        
    Examples
    --------
    >>> # Development (2 cores, 2GB)
    >>> spark = create_spark_session_preset('local_dev')
    
    >>> # Full local power (all cores, 4GB)
    >>> spark = create_spark_session_preset('local_full')
    
    >>> # Large dataset local (all cores, 8GB)
    >>> spark = create_spark_session_preset('local_large')
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset]
    logger.info(f"Using preset configuration: {preset}")
    
    return create_spark_session(**config)


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Spark configurations...\n")
    
    # Test local configuration
    print("=" * 60)
    print("Creating Spark session with 'local_full' preset...")
    print("=" * 60)
    
    spark = create_spark_session_preset('local_full')
    
    # Display session info
    print(f"\n✓ Spark version: {spark.version}")
    print(f"✓ Master: {spark.sparkContext.master}")
    print(f"✓ App name: {spark.sparkContext.appName}")
    print(f"✓ Available parallelism: {spark.sparkContext.defaultParallelism}")
    
    # Test optimal partitions calculation
    print(f"\n✓ Optimal partitions for 100MB dataset: {get_optimal_partitions(spark, 100)}")
    print(f"✓ Optimal partitions for 1GB dataset: {get_optimal_partitions(spark, 1000)}")
    print(f"✓ Optimal partitions for 10GB dataset: {get_optimal_partitions(spark, 10000)}")
    
    # Create a test DataFrame
    print("\nCreating test DataFrame...")
    test_data = [(i, f"prompt_{i}", "small" if i % 3 == 0 else "medium") 
                 for i in range(100)]
    test_df = spark.createDataFrame(test_data, ["id", "prompt", "complexity"])
    
    print(f"✓ Test DataFrame created with {test_df.count()} rows")
    print(f"✓ Partitions: {test_df.rdd.getNumPartitions()}")
    
    test_df.show(5, truncate=False)
    
    # Stop session
    stop_spark_session(spark)
    
    print("\n✓ Spark configuration test completed successfully!")
