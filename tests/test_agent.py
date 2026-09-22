from TinyAgent.agent import TinyAgent
from TinyAgent.memory import Memory
from TinyAgent.llm import Response


class MockLLM:
    def generate(self, messages, tools=None):
        return Response(content="I am a tiny agent response.")


def test_agent_run_and_memory():
    agent = TinyAgent(llm=MockLLM(), memory=Memory(), record_trajectory=True)
    assert agent.trajectory is not None

    answer = agent.run("Hello world")
    assert answer == "I am a tiny agent response."

    messages = agent.memory.get_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello world"}
    assert messages[1] == {"role": "assistant", "content": "I am a tiny agent response."}

    # Trajectory check
    assert len(agent.trajectory.runs) == 1
    assert agent.trajectory.runs[0]["query"] == "Hello world"
    assert agent.trajectory.runs[0]["steps"][0].answer == "I am a tiny agent response."


def test_agent_disable_trajectory():
    agent = TinyAgent(llm=MockLLM(), memory=Memory(), record_trajectory=False)
    assert agent.trajectory is None

    answer = agent.run("Test task")
    assert answer == "I am a tiny agent response."


def test_agent_tools_system_prompt():
    from TinyAgent.tools import Tools

    tools = Tools()
    tools.add_tool("get_weather", lambda city: "Sunny", "Get the weather for a city")

    agent = TinyAgent(llm=MockLLM(), memory=Memory(), tools=tools)
    agent.run("What is the weather?")

    messages = agent.memory.get_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert "# Tools" in messages[0]["content"]
    assert "get_weather" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_agent_tool_execution_loop():
    from TinyAgent.tools import Tools

    tools = Tools()
    tools.add_tool("multiply", lambda a, b: str(float(a) * float(b)), "Multiply two numbers")

    class ToolUsingLLM:
        def __init__(self):
            self.turn = 0

        def generate(self, messages, tools=None):
            self.turn += 1
            if self.turn == 1:
                # First turn: returns tool call in OpenAI native format
                return Response(
                    content="",
                    tool_call={
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"a": 5, "b": 6}',
                        },
                    },
                )
            else:
                # Second turn: after seeing observation "30.0", returns final answer
                return Response(content="5 times 6 is 30.0")

    agent = TinyAgent(
        llm=ToolUsingLLM(),
        memory=Memory(),
        tools=tools,
        record_trajectory=True,
    )
    result = agent.run("What is 5 times 6?")
    assert result == "5 times 6 is 30.0"

    messages = agent.memory.get_messages()
    # Expect: system prompt, user query, assistant tool_call turn, observation turn, final assistant turn
    assert any(msg.get("role") == "system" for msg in messages)
    assert any(msg.get("tool_calls") for msg in messages)
    assert any("30.0" in str(msg.get("content")) for msg in messages)

    # Check trajectory has both the tool call step and final answer
    assert len(agent.trajectory.runs) == 1
    assert len(agent.trajectory.runs[0]["steps"]) == 2
    assert agent.trajectory.runs[0]["steps"][0].observation == "30.0"
    assert agent.trajectory.runs[0]["steps"][1].answer == "5 times 6 is 30.0"


