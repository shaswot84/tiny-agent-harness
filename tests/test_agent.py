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

