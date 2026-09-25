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


def test_mmlu_pro_exact_match_scorer():
    from TinyAgent import exact_match_scorer, mmlu_pro

    assert len(mmlu_pro.examples) == 3
    assert mmlu_pro.name == "MMLU Pro"

    # Test exact_match_scorer with various formats
    ex = {"expected": "J"}
    assert exact_match_scorer("J", ex) is True
    assert exact_match_scorer("The correct answer is J.", ex) is True
    assert exact_match_scorer("Answer: (j)", ex) is True
    assert exact_match_scorer("A", ex) is False
    assert exact_match_scorer("No valid option", ex) is False


def test_ifeval_programmatic_scorer():
    from TinyAgent import programmatic_scorer, ifeval

    assert len(ifeval.examples) == 3
    assert ifeval.name == "IFEval"

    # Example 1: sentence punctuation check (< 10)
    ex1 = ifeval.examples[0]
    assert programmatic_scorer("A short song with three lines! Fun times ahead. The end.", ex1) is True
    assert programmatic_scorer(". . . . . . . . . .", ex1) is False

    # Example 2: word count check (<= 150 words)
    ex2 = ifeval.examples[1]
    assert programmatic_scorer("A snappy modern digital photo frame.", ex2) is True
    assert programmatic_scorer("word " * 151, ex2) is False

    # Example 3: no commas check
    ex3 = ifeval.examples[2]
    assert programmatic_scorer("Hark traveller go unto the mountains with great haste", ex3) is True
    assert programmatic_scorer("Hark, traveller!", ex3) is False


def test_judge_scorer_and_benchmark():
    from TinyAgent import judge_scorer, judge_benchmark, Response

    class MockJudgeLLM:
        def __init__(self, score_str: str):
            self.score_str = score_str

        def generate(self, messages, tools=None):
            return Response(content=self.score_str)

    example = {
        "task": "Explain API.",
        "expected": "An API is a interface between two apps.",
    }

    # Test exact score parsing
    judge_perfect = MockJudgeLLM("1.0")
    assert judge_scorer("An API is an interface.", example, judge=judge_perfect) == 1.0

    judge_partial = MockJudgeLLM("0.85\nExplanation follows...")
    assert judge_scorer("An API connects services.", example, judge=judge_partial) == 0.85

    judge_fallback = MockJudgeLLM("Cannot evaluate this.")
    assert judge_scorer("irrelevant", example, judge=judge_fallback) == 0.0

    assert len(judge_benchmark.examples) == 3
    assert judge_benchmark.name == "LLM-as-a-Judge"




