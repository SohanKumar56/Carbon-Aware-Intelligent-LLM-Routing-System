"""
test_pyspark_installation.py — Quick test to verify PySpark installation

Run this to ensure PySpark is properly configured before running
the full preprocessing pipeline.
"""

import sys

def test_pyspark_import():
    """Test if PySpark can be imported."""
    print("Testing PySpark import...")
    try:
        import pyspark
        print(f"✓ PySpark version: {pyspark.__version__}")
        return True
    except ImportError:
        print("✗ PySpark not installed")
        print("  Install with: pip install pyspark>=3.5.0")
        return False

def test_spark_session():
    """Test if Spark session can be created."""
    print("\nTesting Spark session creation...")
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .master("local[2]") \
            .appName("InstallationTest") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()
        
        print(f"✓ Spark session created")
        print(f"  Version: {spark.version}")
        print(f"  Master: {spark.sparkContext.master}")
        print(f"  App Name: {spark.sparkContext.appName}")
        
        # Test simple operation
        data = [(1, "test"), (2, "data")]
        df = spark.createDataFrame(data, ["id", "value"])
        count = df.count()
        
        print(f"✓ Test DataFrame created with {count} rows")
        
        spark.stop()
        print("✓ Spark session stopped cleanly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating Spark session: {e}")
        return False

def test_spark_config_module():
    """Test if custom Spark config module works."""
    print("\nTesting custom Spark configuration...")
    try:
        sys.path.insert(0, 'data')
        from spark_config import create_spark_session_preset, stop_spark_session
        
        spark = create_spark_session_preset('local_dev')
        print(f"✓ Created Spark session with preset: 'local_dev'")
        print(f"  Available cores: {spark.sparkContext.defaultParallelism}")
        
        stop_spark_session(spark)
        print("✓ Custom Spark configuration working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with custom config: {e}")
        return False

def test_pyarrow():
    """Test if PyArrow is installed (needed for Pandas-Spark interop)."""
    print("\nTesting PyArrow installation...")
    try:
        import pyarrow
        print(f"✓ PyArrow version: {pyarrow.__version__}")
        return True
    except ImportError:
        print("✗ PyArrow not installed")
        print("  Install with: pip install pyarrow>=14.0.1")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("PYSPARK INSTALLATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test 1: PySpark import
    results.append(("PySpark Import", test_pyspark_import()))
    
    # Test 2: PyArrow (required for Pandas interop)
    results.append(("PyArrow Import", test_pyarrow()))
    
    # Test 3: Spark session
    if results[0][1]:  # Only if PySpark imported successfully
        results.append(("Spark Session", test_spark_session()))
    
    # Test 4: Custom config module
    if results[0][1]:  # Only if PySpark imported successfully
        results.append(("Custom Config", test_spark_config_module()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou're ready to run the PySpark preprocessing pipeline:")
        print("  cd data")
        print("  python build_labeled_dataset_spark.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install pyspark>=3.5.0 pyarrow>=14.0.1")
        return 1

if __name__ == "__main__":
    sys.exit(main())
