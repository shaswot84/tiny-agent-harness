from .base import Benchmark


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
