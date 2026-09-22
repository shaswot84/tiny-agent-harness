import inspect
import json
from typing import Any, Callable

from TinyAgent.llm import Response

# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)

    # Extract metadata
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema


class Tools:
    """Tool registry for the Agent."""

    def __init__(
        self,
        requires_approval: list[str] | None = None,
        native: bool = False,
    ):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = list(requires_approval) if requires_approval is not None else []
        self.native = native

    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        schema: dict | None = None,
    ) -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool. If omitted, uses func's docstring.
            schema: Optional OpenAI-compatible tool schema definition. If omitted,
                    it is automatically inferred from func's signature and type annotations.
        """
        resolved_desc = description or (inspect.getdoc(func) or "")
        resolved_schema = schema if schema is not None else tool_to_schema(func)

        self.registry[name] = {
            "function": func,
            "description": resolved_desc,
            "schema": resolved_schema,
        }


    @property
    def schemas(self) -> list[dict] | None:
        """Collect registered tool schemas for native function calling.

        Returns:
            A list of OpenAI-style function calling dictionaries, or None if empty.
        """
        schemas = [
            tool["schema"]
            for tool in self.registry.values()
            if tool.get("schema") is not None
        ]
        return schemas if schemas else None

    @property
    def descriptions(self) -> str:
        """Format all registered tools into a human-readable list for prompting."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}"
            for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        """Generate the system instructions explaining tool usage via JSON."""
        return f"""
# Tools
 
If needed, you can only use the following tools to assist you 
in completing tasks:
 
{self.descriptions}
 
To use a tool, respond with JSON: 
{{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """Extract a JSON tool call from model generated text.

        Looks for a JSON block containing "tool" key and populates `response.tool_call`.

        Args:
            response: Response object containing generated text in `content`.

        Returns:
            Response object updated with parsed `tool_call` dict if present.
        """
        text = response.content

        # Locate substring starting from first '{' and ending at last '}'
        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start:end])

            # Return updated Response with populated tool_call
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
            )

        return response

    def execute(self, response: Response) -> Any:
        """Execute the function corresponding to a parsed tool call.

        Handles human-in-the-loop approval confirmation for sensitive tools
        before triggering execution.

        Args:
            response: Response object containing `tool_call` with 'tool' and 'kwargs'.

        Returns:
            The output returned by the tool function, or an error/denial string.
        """
        tool_call = response.tool_call
        if not tool_call:
            return "No tool call to execute."

        # Handle both OpenAI native format ({"function": {"name": ..., "arguments": ...}})
        # and prompt-based format ({"tool": ..., "kwargs": ...})
        if "function" in tool_call:
            name = tool_call["function"].get("name", "")
            raw_args = tool_call["function"].get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    kwargs = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    kwargs = {}
            elif isinstance(raw_args, dict):
                kwargs = raw_args
            else:
                kwargs = {}
        else:
            name = tool_call.get("tool", "")
            kwargs = tool_call.get("kwargs", {})
            if isinstance(kwargs, str):
                try:
                    kwargs = json.loads(kwargs)
                except Exception:
                    pass

        # Human-in-the-loop: ask user for confirmation before dangerous actions
        if name in self.registry and name in self.requires_approval:
            approval = input(f"Allow {name}? [y/N] ").strip().lower()
            if approval not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # Dispatch call to the registered Python callable
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."


    def observation(self, result: str, role: str | None = None) -> tuple[str, str]:
        """Format an execution observation into a (role, content) message tuple.

        - For native tool calling: uses role 'tool' and raw result.
        - For prompt-based tools: uses role 'user' and 'OBSERVATION: <result>'.

        Args:
            result: The stringified output from tool execution.
            role: Optional role override ('user' or 'tool').

        Returns:
            A tuple of (message_role, message_content).
        """
        if role is not None:
            assigned_role = role
        elif self.native:
            assigned_role = "tool"
        else:
            assigned_role = "user"

        if assigned_role == "tool":
            return "tool", str(result)
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """Check whether the agent has reached a completion state.

        Stopping conditions:
        1. The model output contains no tool call (regular conversational answer).
        2. The model explicitly invoked the 'final_answer' tool, in which case
           its arguments are unpacked into `response.content`.

        Args:
            response: Response object with generated text and optional tool_call.

        Returns:
            True if agent execution should stop, False if another tool step is required.
        """
        if not response.tool_call:
            return True

        tool_call = response.tool_call
        if "function" in tool_call:
            tool_name = tool_call["function"].get("name", "")
            raw_args = tool_call["function"].get("arguments", "")
        else:
            tool_name = tool_call.get("tool", "")
            raw_args = tool_call.get("kwargs", "")

        if tool_name == "final_answer":
            if isinstance(raw_args, dict) and "answer" in raw_args:
                response.content = raw_args["answer"]
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    response.content = parsed.get("answer", raw_args) if isinstance(parsed, dict) else str(parsed)
                except Exception:
                    response.content = raw_args
            else:
                response.content = str(raw_args)
            return True

        return False



