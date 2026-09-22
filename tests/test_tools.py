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


def test_multiply_tool_registration():
    def multiply(a: str, b: str) -> str:
        return str(float(a) * float(b))

    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)",
    )

    assert "multiply" in tools.registry
    assert tools.registry["multiply"]["description"] == "Multiplies two numbers: multiply(a: str, b: str)"
    assert tools.registry["multiply"]["function"]("3", "4") == "12.0"


def test_tools_descriptions_and_prompt():
    tools = Tools()
    assert tools.descriptions == ""
    assert "# Tools" in tools.prompt

    tools.add_tool("tool_a", lambda: None, "Does action A")
    tools.add_tool("tool_b", lambda: None, "Does action B")

    assert tools.descriptions == "`tool_a`: Does action A\n`tool_b`: Does action B"
    prompt = tools.prompt
    assert "# Tools" in prompt
    assert "`tool_a`: Does action A" in prompt
    assert '{"tool": "name", "kwargs": {"param": "value"}}' in prompt


def test_tools_schemas_and_llm_forwarding():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    }

    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=lambda a, b: str(float(a) * float(b)),
        description="Multiply two numbers",
        schema=tool_schema,
    )

    assert tools.schemas == [tool_schema]

    received_tools = []

    class SpyLLM:
        def generate(self, messages, tools=None):
            nonlocal received_tools
            received_tools = tools
            return Response(content="Calculated answer")

    agent = TinyAgent(llm=SpyLLM(), memory=Memory(), tools=tools)
    agent.run("Calculate 2 * 3")

    assert received_tools == [tool_schema]


def test_tools_parse():
    tools = Tools()

    # Response with tool call in text
    resp = Response(
        content='I will calculate this: {"tool": "add", "kwargs": {"a": 2, "b": 3}}',
        reasoning="Need to add numbers",
    )
    parsed = tools.parse(resp)
    assert parsed.tool_call == {"tool": "add", "kwargs": {"a": 2, "b": 3}}
    assert parsed.content == resp.content
    assert parsed.reasoning == "Need to add numbers"

    # Response without tool call
    plain_resp = Response(content="Just a regular message")
    assert tools.parse(plain_resp).tool_call is None


def test_tools_execute(monkeypatch):
    tools = Tools(requires_approval=["delete_db"])
    tools.add_tool("add", lambda a, b: a + b, "Add")
    tools.add_tool("delete_db", lambda: "Database dropped", "Delete DB")

    # Normal execution
    resp = Response(tool_call={"tool": "add", "kwargs": {"a": 5, "b": 10}})
    assert tools.execute(resp) == 15

    # Unknown tool
    unknown_resp = Response(tool_call={"tool": "nonexistent"})
    assert tools.execute(unknown_resp) == "Tool 'nonexistent' not found."

    # Human-in-the-loop: approved
    dangerous_resp = Response(tool_call={"tool": "delete_db"})
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert tools.execute(dangerous_resp) == "Database dropped"

    # Human-in-the-loop: denied
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert tools.execute(dangerous_resp) == "Tool 'delete_db' was denied by the user."
