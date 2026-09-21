from ..llm import LLM
from ..trajectory import Trajectory
from ..memory import Memory
from ..tools import Tools


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools | None = None,
        record_trajectory: bool = False,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None

        self.trajectory = Trajectory() if record_trajectory else None

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.memory.add("user", task)
        if self.trajectory:
            self.trajectory.initialize(task)
 
        return self._step()
 
    def _step(self) -> str:
        """Perform a single step."""
        # Generate response and add to memory
        tools = self.tools.schemas if self.tools else None
        response = self.llm.generate(self.memory.get_messages(), tools=tools)
        self.memory.add("assistant", response.content)

        if self.trajectory:
            self.trajectory.add(response)
        return response.content
 
    def _execute_action(self, action: str) -> str | None:
        """Execute a tool action."""
        # Placeholder - will be implemented in later chapters
        return f"Executed action: {action}"
