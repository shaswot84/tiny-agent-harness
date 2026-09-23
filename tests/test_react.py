from TinyAgent import ReAct, TinyAgent, Memory, Tools, Response
from TinyAgent.planner import ReAct as PlannerReAct


def test_react_initialization():
    react = ReAct()
    assert react.max_steps == 10

    custom_react = ReAct(max_steps=5)
    assert custom_react.max_steps == 5
    assert ReAct is PlannerReAct


def test_react_prompt():
    react = ReAct()
    prompt = react.prompt
    assert "# ReAct (Reason and Act)" in prompt
    assert "THOUGHT: [Your reasoning about what to do next]" in prompt
    assert "ACTION:" in prompt
    assert '"tool": "final_answer"' in prompt


def test_react_parse():
    react = ReAct()
    raw_content = """THOUGHT: I should multiply 6 by 7 to get the answer.
ACTION:
{
    "tool": "multiply",
    "kwargs": {"a": 6, "b": 7}
}"""
    response = Response(content=raw_content)
    parsed = react.parse(response)

    assert parsed.reasoning == "I should multiply 6 by 7 to get the answer."
    assert '"tool": "multiply"' in parsed.content
    assert '"kwargs": {"a": 6, "b": 7}' in parsed.content


def test_react_parse_final_answer():
    react = ReAct()
    raw_content = """THOUGHT: I have the final calculation ready.
ACTION:
{
    "tool": "final_answer",
    "kwargs": "42"
}"""
    response = Response(content=raw_content)
    parsed = react.parse(response)

    assert parsed.reasoning == "I have the final calculation ready."
    assert '"tool": "final_answer"' in parsed.content
    assert '"kwargs": "42"' in parsed.content


def test_tinyagent_react_integration():
    class ScriptedLLM:
        def __init__(self):
            self.turn = 0

        def generate(self, messages, tools=None):
            self.turn += 1
            if self.turn == 1:
                return Response(
                    content="""THOUGHT: I need to multiply 3 and 4.
ACTION:
{
    "tool": "multiply",
    "kwargs": {"a": 3, "b": 4}
}"""
                )
            return Response(
                content="""THOUGHT: The result of 3 * 4 is 12.
ACTION:
{
    "tool": "final_answer",
    "kwargs": "12"
}"""
            )

    tools = Tools()
    tools.add_tool("multiply", lambda a, b: a * b, "Multiplies two numbers")

    agent = TinyAgent(
        llm=ScriptedLLM(),
        memory=Memory(),
        tools=tools,
        planner=ReAct(max_steps=5),
        record_trajectory=True,
    )

    result = agent.run("What is 3 * 4?")
    assert result == "12"

    messages = agent.memory.get_messages()
    assert messages[0]["role"] == "system"
    assert "# ReAct (Reason and Act)" in messages[0]["content"]
    assert "# Tools" in messages[0]["content"]

    # Trajectory check
    assert len(agent.trajectory.runs) == 1
    steps = agent.trajectory.runs[0]["steps"]
    assert len(steps) == 2
    assert steps[0].thought == "I need to multiply 3 and 4."
    assert steps[0].action == {"tool": "multiply", "kwargs": {"a": 3, "b": 4}}
    assert "12" in str(steps[0].observation)
    assert steps[1].thought == "The result of 3 * 4 is 12."
