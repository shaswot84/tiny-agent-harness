import re
from .base import Benchmark
from .judge import judge_scorer


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Return True if the answer matches the prediction, False otherwise."""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match.group(1) == example["expected"] if match else False


# Three examples adapted from MMLU Pro for LLM judge evaluation
mmlu_pro = Benchmark(
    name="MMLU Pro",
    examples=[
        {
            "task": "Which body cavity contains the pituitary gland?",
            "expected": "the cranial cavity",
        },
        {
            "task": "What is the approximate mean cranial capacity of Homo erectus?",
            "expected": "just under 1000 cc",
        },
        {
            "task": "According to Moore's 'ideal utilitarianism,' the right action is the one that brings about the greatest amount of what?",
            "expected": "good",
        },
    ],
    scorer=judge_scorer,
)


# Multiple-choice benchmark preset from MMLU Pro
mmlu_pro_mcq = Benchmark(
    name="MMLU Pro MCQ",
    examples=[
        {
            "task": """Which of the following is the body cavity that contains
the pituitary gland?
A) Ventral B) Dorsal C) Buccal D) Thoracic E) Pericardial F) Abdominal 
G) Spinal H) Pelvic I) Pleural J) Cranial
Answer with only the letter.""",
            "expected": "J",
        },
        {
            "task": """What is the approximate mean cranial capacity of 
Homo erectus?
A) 1200 cc B) under 650 cc C) 1700 cc D) 1350 cc E) just under 1000 cc 
F) 1500 cc G) under 500 cc H) about 800 cc I) just over 1100 cc J) about 900 cc
Answer with only the letter.""",
            "expected": "E",
        },
        {
            "task": """According to Moore’s "ideal utilitarianism," the right action is the one that
brings about the greatest amount of:
A) wealth. B) virtue. C) fairness. D) pleasure. E) peace. F) justice. 
G) happiness. H) power. I) good. J) knowledge.
Answer with only the letter.""",
            "expected": "I",
        },
    ],
    scorer=exact_match_scorer,
)

