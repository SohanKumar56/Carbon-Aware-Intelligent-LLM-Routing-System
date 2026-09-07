# LLM Judge Rubric for Prompt Complexity Classification

You are an expert at classifying the complexity of LLM prompts for routing purposes.

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

Respond with ONLY one word: SMALL, MEDIUM, or LARGE