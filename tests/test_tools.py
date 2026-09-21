from TinyAgent import Tools, TinyAgent, Memory
from TinyAgent.llm import Response


class MockLLM:
    def generate(self, messages, tools=None):
        return Response(content="Response from LLM")


def test_tools_initialization():
    tools = Tools()
    assert tools.registry == {}
    assert tools.requires_approval == []
    assert tools.schemas is None

    tools_with_approval = Tools(requires_approval=["execute_command", "delete_file"])
    assert tools_with_approval.requires_approval == ["execute_command", "delete_file"]


def test_tools_add_tool():
    tools = Tools()

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tools.add_tool("calculator_add", add, "Adds two numbers together")

    assert "calculator_add" in tools.registry
    tool_entry = tools.registry["calculator_add"]
    assert tool_entry["description"] == "Adds two numbers together"
    assert tool_entry["function"] == add
    assert tool_entry["function"](2, 3) == 5


def test_agent_with_tools():
    tools = Tools(requires_approval=["danger_tool"])

    def dummy_tool() -> str:
        return "dummy"

    tools.add_tool("dummy", dummy_tool, "Dummy tool")

    agent = TinyAgent(
        llm=MockLLM(),
        memory=Memory(),
        tools=tools,
    )
    assert agent.tools is tools
    assert "dummy" in agent.tools.registry
