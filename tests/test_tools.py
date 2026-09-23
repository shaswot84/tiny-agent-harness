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
    # Auto-inferred schema
    assert tool_entry["schema"]["function"]["name"] == "add"
    assert tool_entry["schema"]["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert tool_entry["schema"]["function"]["parameters"]["properties"]["b"]["type"] == "integer"
    assert tool_entry["schema"]["function"]["parameters"]["required"] == ["a", "b"]


def test_tool_to_schema():
    from TinyAgent.tools import tool_to_schema

    def search(query: str, limit: int = 5, verbose: bool = False) -> list:
        """Search the web for results."""
        return []

    schema = tool_to_schema(search)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search the web for results."
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["limit"]["type"] == "integer"
    assert params["properties"]["verbose"]["type"] == "boolean"
    assert params["required"] == ["query"]



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

    # Response with trailing comma in kwargs and tool call object (LLM quirk)
    resp_trailing = Response(
        content='ACTION:\n{\n    "tool": "add",\n    "kwargs": {"a": 4.6, "b": 6.685,},\n}',
        reasoning="Handling addition with trailing comma",
    )
    parsed_trailing = tools.parse(resp_trailing)
    assert parsed_trailing.tool_call == {"tool": "add", "kwargs": {"a": 4.6, "b": 6.685}}

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


def test_tools_observation():
    # Prompt-based LLM (default)
    tools = Tools()
    role, message = tools.observation("42")
    assert role == "user"
    assert message == "OBSERVATION: 42"

    # Native tool-calling LLM
    native_tools = Tools(native=True)
    n_role, n_message = native_tools.observation("42")
    assert n_role == "tool"
    assert n_message == "42"

    # Explicit role override
    custom_role, custom_msg = tools.observation("42", role="tool")
    assert custom_role == "tool"
    assert custom_msg == "42"


def test_tools_is_done():
    tools = Tools()

    # No tool call -> is done
    no_tool_resp = Response(content="Final thought")
    assert tools.is_done(no_tool_resp) is True

    # Tool call with final_answer -> is done and updates content
    final_resp = Response(
        content='{"tool": "final_answer", "kwargs": "The result is 42"}',
        tool_call={"tool": "final_answer", "kwargs": "The result is 42"},
    )
    assert tools.is_done(final_resp) is True
    assert final_resp.content == "The result is 42"

    # Regular tool call -> not done
    regular_resp = Response(
        content='{"tool": "calculator", "kwargs": {"a": 1}}',
        tool_call={"tool": "calculator", "kwargs": {"a": 1}},
    )
    assert tools.is_done(regular_resp) is False


def test_native_tools():
    from TinyAgent.tools import NativeTools

    native_tools = NativeTools()
    def add(a: int, b: int) -> int:
        """Add two ints."""
        return a + b

    native_tools.add_tool("add", add)

    # schemas & prompt
    assert native_tools.prompt == ""
    assert len(native_tools.schemas) == 1
    assert native_tools.schemas[0]["function"]["name"] == "add"

    # parse
    raw_resp = Response(
        content="",
        tool_call={
            "id": "1",
            "type": "function",
            "function": {
                "name": "add",
                "arguments": '{"a": 10, "b": 20}',
            },
        },
    )
    parsed = native_tools.parse(raw_resp)
    assert parsed.tool_call == {"tool": "add", "kwargs": {"a": 10, "b": 20}}

    # observation
    role, obs = native_tools.observation("30")
    assert role == "tool"
    assert obs == "30"

    # is_done
    assert native_tools.is_done(Response(content="Done without tool")) is True
    assert native_tools.is_done(raw_resp) is False


def test_toolbox():
    from TinyAgent.tools import toolbox

    # Math helpers
    assert toolbox.add(2, 3) == "5.0"
    assert toolbox.multiply(4, 5) == "20.0"
    assert toolbox.subtract(10, 4) == "6.0"
    assert toolbox.divide(20, 4) == "5.0"
    assert "Error" in toolbox.divide(10, 0)
    assert toolbox.power(2, 3) == "8.0"

    # Control helper
    assert toolbox.final_answer("Done!") == "Done!"

    # Command execution
    out = toolbox.execute_command("echo 'hello toolbox'")
    assert "hello toolbox" in out



