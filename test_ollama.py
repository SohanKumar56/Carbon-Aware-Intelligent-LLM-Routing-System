"""
Test script for Ollama integration
"""

from ollama_integration import check_ollama_availability, get_ollama_models, run_ollama_inference

# Test Ollama availability
print("Testing Ollama availability...")
is_available = check_ollama_availability()
print(f"Ollama available: {is_available}")

if is_available:
    # Test getting models
    print("\nAvailable Ollama models:")
    models = get_ollama_models()
    for model in models:
        print(f"  - {model['name']} ({model['size']} GB) [{model['id']}]")

    # Test inference with a small model
    print("\nTesting inference with tinyllama...")
    test_prompt = "What is the capital of France?"
    result = run_ollama_inference("tinyllama:latest", test_prompt)

    print(f"Result:    {result}")
    print(f"Latency:   {result.get('latency_ms')} ms")
    print(f"Energy:    {result.get('energy_kwh')} kWh")
else:
    print("\n⚠️ Ollama is not running. Please start Ollama to test.")
    print("Run: ollama serve")
