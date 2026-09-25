import pytest
from TinyAgent import Benchmark, Scorer
from TinyAgent.evaluation import Benchmark as EvalBenchmark


def test_benchmark_dataclass():
    examples = [
        {"input": "2 + 2", "expected": "4"},
        {"input": "3 * 3", "expected": "9"},
    ]

    def exact_match_scorer(pred: str, example: dict) -> bool:
        return pred.strip() == example["expected"].strip()

    bench = Benchmark(
        name="basic_math",
        examples=examples,
        scorer=exact_match_scorer,
    )

    assert bench.name == "basic_math"
    assert len(bench.examples) == 2
    assert bench.scorer("4", examples[0]) is True
    assert bench.scorer("5", examples[0]) is False
    assert Benchmark is EvalBenchmark
