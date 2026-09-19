from TinyAgent.trajectory import Trajectory, Step
from TinyAgent.llm import Response


def test_step_formatting():
    step = Step(
        thought="Thinking about Paris",
        action={"tool": "search", "query": "Paris weather"},
        observation="Sunny, 22C",
        answer="It is sunny in Paris.",
    )
    s = str(step)
    assert "💭 Thought:" in s
    assert "Thinking about Paris" in s
    assert "🛠️ Action:" in s
    assert "👁️ Observation:" in s
    assert "💬 Answer:" in s


def test_trajectory_recording():
    traj = Trajectory()
    assert str(traj) == "No trajectory recorded."

    traj.initialize("What is the capital of France?")
    traj.add(Response(content="Paris", reasoning="General knowledge"))

    assert len(traj.runs) == 1
    assert traj.runs[0]["query"] == "What is the capital of France?"
    assert len(traj.runs[0]["steps"]) == 1
    assert traj.runs[0]["steps"][0].answer == "Paris"
    assert "📍 Run 1:" in str(traj)
