import os
import re
from TinyAgent.llm import LLM
from .base import Benchmark

# Default LLM judge using gpt-oss:20b from ollama_cloud
judge_llm = LLM(
    model="gpt-oss:20b",
    provider="ollama_cloud",
    api_key=os.getenv("OLLAMA_CLOUD_API_KEY", os.getenv("LLM_API_KEY", "ollama")),
)


def judge_scorer(prediction: str, example: dict, judge: LLM = judge_llm) -> float:
    """The LLM-as-a-judge scorer scoring a prediction from 0.0 to 1.0."""
    prompt = f"""
Score the response from 0.0 to 1.0.

Expected: {example["expected"]}
Response: {prediction}

Reply with only a single number.
"""
    response = judge.generate([{"role": "user", "content": prompt}])
    text = response.content.strip()

    # Extract first float or integer found in the reply
    match = re.search(r"([0-1](?:\.\d+)?|\d+)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return 0.0


# Benchmark examples for LLM-as-a-judge evaluation
judge_benchmark = Benchmark(
    name="LLM-as-a-Judge",
    examples=[
        {
            "task": "Explain what an API is in simple terms for a beginner.",
            "expected": "An API (Application Programming Interface) allows two software programs to communicate with each other, like a waiter taking orders from a customer to the kitchen.",
        },
        {
            "task": "Summarize Newton's first law of motion.",
            "expected": "An object at rest stays at rest, and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force.",
        },
        {
            "task": "What causes the changing of seasons on Earth?",
            "expected": "Earth's axial tilt of approximately 23.5 degrees as it orbits the Sun causes varying angles and durations of sunlight throughout the year.",
        },
    ],
    scorer=judge_scorer,
)
