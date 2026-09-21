from typing import Callable


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] | None = None):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = list(requires_approval) if requires_approval is not None else []

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
            description: A description of the tool.
            schema: Optional OpenAI-compatible tool schema definition.
        """
        self.registry[name] = {
            "function": func,
            "description": description,
            "schema": schema,
        }

    @property
    def schemas(self) -> list[dict] | None:
        """Used only for native tool-calling."""
        schemas = [
            tool["schema"]
            for tool in self.registry.values()
            if tool.get("schema") is not None
        ]
        return schemas if schemas else None


    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}"
            for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools
 
If needed, you can only use the following tools to assist you 
in completing tasks:
 
{self.descriptions}
 
To use a tool, respond with JSON: 
{{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

