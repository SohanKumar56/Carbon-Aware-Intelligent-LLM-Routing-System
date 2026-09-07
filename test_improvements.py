"""Quick test script for improved complexity classifier"""

from complexity_classifier import route_to_model

# Test cases that should trigger different classifications
test_prompts = [
    ("Simple fact", "What is the capital of France?"),
    ("Math proof", "Prove that the square root of 2 is irrational using proof by contradiction"),
    ("Code with requirements", "Write a Python function to implement binary search with detailed comments and error handling"),
    ("Very long task", "Create a complete web application with user authentication, database integration, RESTful API endpoints, responsive frontend using React, comprehensive error handling, logging system, rate limiting, input validation, unit tests with 90% coverage, integration tests, documentation, and deployment instructions for AWS"),
    ("Complex reasoning", "Explain in detail the implications of Gödel's incompleteness theorems on computational theory with step by step analysis"),
]

print("=" * 80)
print("TESTING IMPROVED COMPLEXITY CLASSIFIER")
print("=" * 80)

for name, prompt in test_prompts:
    print(f"\n[{name}]")
    print(f"Prompt: {prompt[:70]}...")
    
    result = route_to_model(prompt)
    
    print(f"  Complexity: {result['complexity'].upper()}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Rule Override: {'YES ✓' if result['rule_override'] else 'No'}")
    print(f"  Primary Model: {result['recommended'][0]}")
    print(f"  Reasoning: {result['reasoning'][:100]}...")

print("\n" + "=" * 80)
print("Test complete!")
