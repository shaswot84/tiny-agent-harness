import json
import re
from TinyAgent.llm import LLM
from .base import Benchmark
from .judge import judge_llm

RUBRIC_CRITERIA = ["Fluency", "Correctness", "Completeness", "Groundedness"]


def rubric_scorer(
    prediction: str,
    example: dict,
    judge: LLM = judge_llm,
) -> float:
    """The Rubric-based LLM-as-a-judge scorer.

    Evaluates:
    1. Fluency: Grammatical clarity, natural language flow, and coherence.
    2. Correctness: Factual and conceptual accuracy according to the expected answer/context.
    3. Completeness: Thoroughness in addressing all constraints and requirements of the task.
    4. Groundedness: Whether assertions are backed by provided reference facts without hallucination.

    Returns the averaged overall score (0.0 to 1.0).
    """
    context = example.get("context", "None provided.")
    expected = example.get("expected", "")

    prompt = f"""You are an expert evaluator. Evaluate the model response on a 0.0 to 1.0 scale across 4 rubrics:
1. Fluency: Grammatical clarity, style, and natural flow.
2. Correctness: Factual accuracy compared to expected/ground truth.
3. Completeness: Thorough coverage of all parts of the user request.
4. Groundedness: Faithfulness to the reference context without hallucination.

Task: {example.get("task", "")}
Reference Context: {context}
Expected Answer: {expected}
Model Response: {prediction}

Provide your scores as a JSON object strictly in this format:
{{
  "fluency": <number 0.0 to 1.0>,
  "correctness": <number 0.0 to 1.0>,
  "completeness": <number 0.0 to 1.0>,
  "groundedness": <number 0.0 to 1.0>
}}
Reply with only the JSON object.
"""
    response = judge.generate([{"role": "user", "content": prompt}])
    text = response.content.strip()

    # Try parsing JSON object
    try:
        # Match json block if wrapped in markdown
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            scores = [
                float(data.get("fluency", 0.0)),
                float(data.get("correctness", 0.0)),
                float(data.get("completeness", 0.0)),
                float(data.get("groundedness", 0.0)),
            ]
            # Average of 4 criteria
            return sum(scores) / len(scores)
    except Exception:
        pass

    # Fallback: extract any floating numbers found
    numbers = [float(n) for n in re.findall(r"([0-1](?:\.\d+)?|\d+)", text)]
    if len(numbers) >= 4:
        return sum(numbers[:4]) / 4.0
    elif numbers:
        return min(1.0, max(0.0, numbers[0]))

    return 0.0


# Rubric benchmark dataset testing Fluency, Correctness, Completeness, and Groundedness
rubric_benchmark = Benchmark(
    name="Rubric-Evaluation",
    examples=[
        {
            "task": "Explain the photosynthesis process and list its two main stages.",
            "context": (
                "Photosynthesis is the biochemical process by which green plants and certain other "
                "organisms convert light energy into chemical energy stored in glucose. It consists "
                "of two main sequential stages: 1) Light-dependent reactions taking place in the thylakoid "
                "membranes, and 2) Light-independent reactions (the Calvin cycle) occurring in the stroma."
            ),
            "expected": (
                "Photosynthesis converts light energy into chemical energy in glucose. "
                "The two main stages are: 1) Light-dependent reactions (in thylakoids) and "
                "2) Light-independent reactions or Calvin cycle (in stroma)."
            ),
        },
        {
            "task": "Describe the function of the human heart valves and why backflow prevention is critical.",
            "context": (
                "The heart has four valves: tricuspid, pulmonary, mitral (bicuspid), and aortic. "
                "Their primary mechanical function is ensuring unidirectional blood flow through the heart chambers. "
                "Preventing retrograde regurgitation (backflow) maintains cardiac output and prevents pulmonary congestion or heart failure."
            ),
            "expected": (
                "Heart valves (tricuspid, pulmonary, mitral, and aortic) enforce one-way blood flow. "
                "Preventing backflow ensures efficient oxygenated/deoxygenated circulation and prevents ventricular overload."
            ),
        },
        {
            "task": "What is the difference between synchronous and asynchronous programming in Python?",
            "context": (
                "Synchronous programming executes tasks sequentially, where each operation blocks execution until completion. "
                "Asynchronous programming (e.g. using asyncio, async/await, and event loops) allows the runtime to handle other tasks "
                "while waiting for non-blocking I/O operations (like network requests or disk reads) to complete."
            ),
            "expected": (
                "Synchronous code runs sequentially and blocks until each line finishes. "
                "Asynchronous code uses an event loop (async/await) allowing non-blocking I/O operations to run concurrently without blocking."
            ),
        },
    ],
    scorer=rubric_scorer,
)
