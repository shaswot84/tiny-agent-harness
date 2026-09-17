from TinyAgent.llm import LLM
from TinyAgent.agent import TinyAgent
from TinyAgent.trajectory import Trajectory


def main():
    print("Hello from tiny-agent!")

    llm = LLM("gemma4:e4b")
    agent = TinyAgent(llm)

    result = agent.run("What is the capital of France?")
    print("Final Answer:", result)

    print("\nTrajectory:")
    print(agent.trajectory)


if __name__ == "__main__":
    main()
