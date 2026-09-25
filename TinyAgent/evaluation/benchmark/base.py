from dataclasses import dataclass
from typing import Callable

# Type hint for the scorers: (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    """A benchmark suite of evaluation examples and an evaluation scoring function."""
    name: str
    examples: list[dict]
    scorer: Callable[[str, dict], bool | float]
