from dataclasses import dataclass

from ..llm import Response


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

    @staticmethod
    def _display_width(text: str) -> int:
        """Calculate visual display width in terminal, handling emojis and wide chars."""
        import unicodedata

        width = 0
        for c in text:
            if unicodedata.combining(c) or c == "\ufe0f":
                continue
            if unicodedata.east_asian_width(c) in ("W", "F"):
                width += 2
            else:
                width += 1
        return width

    @classmethod
    def _pad_line(cls, text: str, target_width: int) -> str:
        """Pad or truncate a string so its visual display width matches target_width."""
        import unicodedata

        cur_w = cls._display_width(text)
        if cur_w == target_width:
            return text
        if cur_w < target_width:
            return text + (" " * (target_width - cur_w))

        # Truncate if visual width exceeds target
        res = []
        w = 0
        for c in text:
            cw = 0 if (unicodedata.combining(c) or c == "\ufe0f") else (2 if unicodedata.east_asian_width(c) in ("W", "F") else 1)
            if w + cw > target_width - 3:
                break
            res.append(c)
            w += cw
        res.append("...")
        w += 3
        return "".join(res) + (" " * max(0, target_width - w))

    def format_latest_run(self, width: int = 70) -> str:
        """Format the most recent run into a clean, perfectly aligned CLI box."""
        import json
        import textwrap

        if not self.runs:
            return ""

        run = self.runs[-1]
        run_idx = len(self.runs)
        inner_w = width - 4  # content width between borders ("│ " and " │")

        lines = []
        border_top = f"┌{'─' * (width - 2)}┐"
        border_sep = f"├{'─' * (width - 2)}┤"
        border_bot = f"└{'─' * (width - 2)}┘"

        lines.append(border_top)
        title_text = f"📍 Trajectory Run #{run_idx}"
        title_w = self._display_width(title_text)
        left_pad = max(0, (width - 2 - title_w) // 2)
        right_pad = max(0, width - 2 - title_w - left_pad)
        lines.append(f"│{' ' * left_pad}{title_text}{' ' * right_pad}│")
        lines.append(border_sep)

        # Query section
        q_label = "Query: "
        avail_q = inner_w - len(q_label)
        q_wrapped = textwrap.wrap(run["query"], width=avail_q) or [""]
        for idx, q_line in enumerate(q_wrapped):
            prefix = q_label if idx == 0 else (" " * len(q_label))
            lines.append(f"│ {self._pad_line(prefix + q_line, inner_w)} │")
        lines.append(border_sep)

        if not run["steps"]:
            lines.append(f"│ {self._pad_line('(No steps recorded)', inner_w)} │")
        else:
            for s_idx, step in enumerate(run["steps"], 1):
                step_title = f"▶ Step {s_idx}"
                lines.append(f"│ {self._pad_line(step_title, inner_w)} │")

                # Thought
                if step.thought:
                    lines.append(f"│ {self._pad_line('  💭 Thought:', inner_w)} │")
                    for t_line in textwrap.wrap(step.thought, width=max(10, inner_w - 6)) or [""]:
                        lines.append(f"│ {self._pad_line(f'      {t_line}', inner_w)} │")

                # Action
                if step.action:
                    lines.append(f"│ {self._pad_line('  🛠️  Action:', inner_w)} │")
                    action_str = json.dumps(step.action) if isinstance(step.action, dict) else str(step.action)
                    for a_line in textwrap.wrap(action_str, width=max(10, inner_w - 6)) or [""]:
                        lines.append(f"│ {self._pad_line(f'      {a_line}', inner_w)} │")

                # Observation
                if step.observation:
                    lines.append(f"│ {self._pad_line('  👁️  Observation:', inner_w)} │")
                    for o_line in textwrap.wrap(str(step.observation), width=max(10, inner_w - 6)) or [""]:
                        lines.append(f"│ {self._pad_line(f'      {o_line}', inner_w)} │")

                # Answer
                if step.answer:
                    lines.append(f"│ {self._pad_line('  💬 Answer:', inner_w)} │")
                    for ans_line in textwrap.wrap(step.answer, width=max(10, inner_w - 6)) or [""]:
                        lines.append(f"│ {self._pad_line(f'      {ans_line}', inner_w)} │")

                if s_idx < len(run["steps"]):
                    lines.append(f"│ {' ' * inner_w} │")

        lines.append(border_bot)
        return "\n".join(lines)


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

