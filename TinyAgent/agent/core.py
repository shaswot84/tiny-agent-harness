from ..llm import LLM, Response
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

        # Execute agent steps until a final answer is produced
        max_steps = 10
        for _ in range(max_steps):
            response = self._step()
            if not self.tools:
                return response.content
            if self.tools.is_done(response):
                return response.content

        return response.content

    def _step(self) -> Response:
        """Perform a single step of the agent execution.

        1. Fetches available native tool schemas (if configured).
        2. Queries the LLM with all past messages in memory.
        3. Parses prompt-based tool calls from text if native calling is not used.
        4. If a tool call is present:
           - Executes the tool.
           - Records the step with observation into trajectory.
           - Appends assistant action and tool observation to memory.
        5. If no tool call is present:
           - Records the final answer into memory and trajectory.

        Returns:
            The Response object produced for this step.
        """
        # Pass native function schemas to the LLM if available
        schemas = self.tools.schemas if self.tools else None

        # Request completion with full conversation history
        response = self.llm.generate(self.memory.get_messages(), tools=schemas)

        # Parse text-based tool calls if native calling wasn't used or produced no tool_call
        if self.tools and not response.tool_call and not self.tools.native:
            response = self.tools.parse(response)

        # If a tool call is present, execute it and feed observation back
        if self.tools and response.tool_call:
            # Check if this tool call is a stopping final_answer
            if self.tools.is_done(response):
                self.memory.add("assistant", response.content)
                if self.trajectory:
                    self.trajectory.add(response)
                return response

            observation = self.tools.execute(response)
            obs_str = str(observation)

            # Record step with action and observation into trajectory
            if self.trajectory:
                self.trajectory.add(response, observation=obs_str)

            # Record assistant turn (with tool call info) and observation turn into memory
            self.memory.add(
                "assistant",
                response.content,
                tool_call=response.tool_call,
            )
            obs_role, obs_content = self.tools.observation(obs_str)
            self.memory.add(obs_role, obs_content)
        else:
            # Final text response without tool calls
            self.memory.add("assistant", response.content)
            if self.trajectory:
                self.trajectory.add(response)

        return response

    def _execute_action(self, action: str) -> str | None:
        """Execute a tool action.

        Args:
            action: Description or serialized name of the action to invoke.

        Returns:
            Observation result from executing the action.
        """
        # Placeholder - will be implemented in later chapters
        return f"Executed action: {action}"

