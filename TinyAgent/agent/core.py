from typing import Any

from ..llm import LLM, Response
from ..trajectory import Trajectory
from ..memory import Memory
from ..tools import Tools
from ..planner import ReAct


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools | None = None,
        planner: ReAct | None = None,
        record_trajectory: bool = True,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner

        self.trajectory = Trajectory() if record_trajectory else None

        # Build system prompt with all components
        # Note: If LLM native thinking mode is enabled (llm.think=True), we do not inject
        # prompt-based ReAct instructions to allow native model reasoning.
        use_prompt_planner = self.planner and not getattr(self.llm, "think", False)
        if use_prompt_planner or self.tools:
            system_prompt = "You are a helpful assistant.\n\n"
            if use_prompt_planner:
                system_prompt += self.planner.prompt
            if self.tools:
                system_prompt += self.tools.prompt
            self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.memory.add("user", task)
        if self.trajectory:
            self.trajectory.initialize(task)

        # *Autonomy* loop
        max_steps = self.planner.max_steps if self.planner else 10
        for step in range(max_steps):
            result = self._step()
            if result is not None:
                return result

        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """Perform a single step."""
        # THOUGHT: Generate response and add to memory
        schemas = self.tools.schemas if self.tools else None
        response = self.llm.generate(
            self.memory.get_messages(), tools=schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )

        # Tool parsing
        # When native thinking is enabled, bypass ReAct regex parsing to preserve native reasoning
        if self.planner and not getattr(self.llm, "think", False):
            response = self.planner.parse(response)
        if self.tools:
            response = self.tools.parse(response)

        # ANSWER: Stopping mechanism
        if not self.tools or self.tools.is_done(response):
            if self.trajectory:
                self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> None:
        """Execute a tool action."""

        # ACTION: execute tools
        result = self.tools.execute(response)

        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        if self.trajectory:
            self.trajectory.add(response, observation)

        return None
