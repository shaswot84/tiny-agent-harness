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


def test_evaluator_run():
    from TinyAgent import Evaluator, TinyAgent, Memory, NativeTools, NativeReAct, Response

    class MockAgentLLM:
        def generate(self, messages, tools=None):
            last_msg = messages[-1]["content"]
            if "2 + 2" in last_msg:
                return Response(content="4")
            return Response(content="wrong")

    def create_agent():
        return TinyAgent(
            llm=MockAgentLLM(),
            memory=Memory(),
            tools=NativeTools(),
            planner=NativeReAct(),
        )

    examples = [
        {"task": "What is 2 + 2?", "expected": "4"},
        {"task": "What is 3 + 3?", "expected": "6"},
    ]

    benchmark = Benchmark(
        name="arithmetic_test",
        examples=examples,
        scorer=lambda pred, ex: pred.strip() == ex["expected"].strip(),
    )

    evaluator = Evaluator(create_agent=create_agent)
    results = evaluator.run(benchmark)

    assert results["name"] == "arithmetic_test"
    assert results["pass_rate"] == 0.5
    assert len(results["results"]) == 2
    assert results["results"][0]["passed"] is True
    assert results["results"][1]["passed"] is False


def test_native_react_planner():
    from TinyAgent import NativeReAct, Response

    planner = NativeReAct(max_steps=5)
    assert planner.prompt == ""
    assert planner.max_steps == 5

    raw_response = Response(content="Final result", reasoning="Native reasoning")
    parsed = planner.parse(raw_response)
    assert parsed.content == "Final result"
    assert parsed.reasoning == "Native reasoning"

