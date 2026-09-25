import re
from dataclasses import dataclass
from typing import Callable, Any

# Type hint for the scorers: (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Callable[[str, dict], bool | float]


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Return True if the answer matches the prediction, False otherwise."""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match.group(1) == example["expected"] if match else False


# Three representative examples from MMLU Pro
mmlu_pro = Benchmark(
    name="MMLU Pro",
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


def programmatic_scorer(prediction: str, example: dict) -> bool:
    """Check a prediction against its related check function."""
    return bool(example["check"](prediction))


# Three representative examples from IFEval
ifeval = Benchmark(
    name="IFEval",
    examples=[
        {
            "task": (
                "Write me a funny song with less than 10 sentences for a "
                "proposal to build a new playground at my local elementary school."
            ),
            "check": lambda text: sum(1 for c in text if c in ".!?") < 10,
        },
        {
            "task": (
                "Write an ad copy for a new product, a digital photo frame "
                "that connects to your social media accounts and displays your photos. "
                "Respond with at most 150 words."
            ),
            "check": lambda text: len(text.split()) <= 150,
        },
        {
            "task": (
                "I am planning a trip to Japan, and I would like thee to "
                "write an itinerary for my journey in a Shakespearean style. "
                "You are not allowed to use any commas in your response."
            ),
            "check": lambda text: "," not in text,
        },
    ],
    scorer=programmatic_scorer,
)


