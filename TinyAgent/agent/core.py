from ..llm import LLM
from ..trajectory import Trajectory
from ..memory import Memory
from ..tools import Tools


class TinyAgent:
    """A minimal, modular, and educational agent framework.

    Coordinates the core components of the agent lifecycle:
    - LLM: Model provider interface for generating completions.
    - Memory: Conversation state management (short-term, long-term RAG, summarization).
    - Tools: Tool registry for schema declarations, text parsing, and execution.
    - Trajectory: Optional step-by-step telemetry and interaction logging.
    """

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools | None = None,
        record_trajectory: bool = False,
    ):
        """Initialize the TinyAgent with its core runtime dependencies.

        Args:
            llm: Language model wrapper implementing generate(messages, tools=...).
            memory: Memory store holding the conversation history turns.
            tools: Optional Tools registry containing callable tools and schemas.
            record_trajectory: When True, logs steps, queries, and observations to Trajectory.
        """
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None

        # Trajectory tracker logs runs and execution steps for auditing or evaluation
        self.trajectory = Trajectory() if record_trajectory else None

    def run(self, task: str) -> str:
        """Run the agent to complete a user task.

        Lifecycle:
        1. Injects tool prompt into system message if prompt-based tools are registered.
        2. Records the initial user prompt in memory.
        3. Initializes a new run in trajectory tracker (if enabled).
        4. Triggers the step execution loop.

        Args:
            task: The user query or task instruction.

        Returns:
            The agent's text response.
        """
        # For text/JSON-prompted models (non-native tool calling), inject tool
        # descriptions and instructions into the system prompt if not present.
        if self.tools and not self.tools.native and self.tools.descriptions:
            existing_messages = self.memory.get_messages()
            has_system = any(msg.get("role") == "system" for msg in existing_messages)
            if not has_system:
                self.memory.add("system", self.tools.prompt)

        # Store user query into conversation history
        self.memory.add("user", task)

        # Start a new trace run for this query in trajectory
        if self.trajectory:
            self.trajectory.initialize(task)

        return self._step()

    def _step(self) -> str:
        """Perform a single step of the agent execution.

        1. Fetches available native tool schemas (if configured).
        2. Queries the LLM with all past messages in memory.
        3. Records the assistant's reply into conversation memory.
        4. Logs the step response to trajectory (if enabled).

        Returns:
            The raw text content produced by the assistant.
        """
        # Pass native function schemas to the LLM if available
        tools = self.tools.schemas if self.tools else None

        # Request completion with full conversation history
        response = self.llm.generate(self.memory.get_messages(), tools=tools)

        # Append assistant's answer to memory history
        self.memory.add("assistant", response.content)

        # Record step in trajectory for observability
        if self.trajectory:
            self.trajectory.add(response)

        return response.content

    def _execute_action(self, action: str) -> str | None:
        """Execute a tool action.

        Args:
            action: Description or serialized name of the action to invoke.

        Returns:
            Observation result from executing the action.
        """
        # Placeholder - will be implemented in later chapters
        return f"Executed action: {action}"
