from dataclasses import dataclass
from typing import Callable, Any

# Type hint for the scorers: (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Callable[[str, dict], bool | float]
