from dataclasses import dataclass

from .llm import Response


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        parts = []
        if self.thought:
            parts.append(f"💭 Thought:\n{self._indent(self.thought, 4)}")
        if self.action:
            parts.append(f"🛠️ Action:\n{self._indent(str(self.action), 4)}")
        if self.observation:
            parts.append(f"👁️ Observation:\n{self._indent(self.observation, 4)}")
        if self.answer:
            parts.append(f"💬 Answer:\n{self._indent(self.answer, 4)}")
        return "\n".join(parts) if parts else "  (Empty Step)"

    @staticmethod
    def _indent(text: str, spaces: int = 4) -> str:
        prefix = " " * spaces
        return "\n".join(f"{prefix}{line}" for line in text.strip().splitlines())


class Trajectory:
    """Records agent execution as a sequence of runs/steps."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)

    def __str__(self) -> str:
        if not self.runs:
            return "No trajectory recorded."

        lines = []
        for i, run in enumerate(self.runs, 1):
            lines.append(f"\n┌{'─' * 50}")
            lines.append(f"│ 📍 Run {i}: \"{run['query']}\"")
            lines.append(f"└{'─' * 50}")
            for j, step in enumerate(run["steps"], 1):
                lines.append(f"  ▶ Step {j}:")
                step_str = "\n".join(f"    {line}" for line in str(step).splitlines())
                lines.append(step_str)
        return "\n".join(lines)