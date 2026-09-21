from typing import Callable


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] | None = None):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = list(requires_approval) if requires_approval is not None else []

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None
