from TinyAgent.llm import LLM
from TinyAgent.agent import TinyAgent
from TinyAgent.memory import Memory, TrimmingMemory, SummarizationMemory
from TinyAgent.embedding import EmbeddingModel
from TinyAgent.tools import Tools


def multiply(a: str, b: str) -> str:
    """Multiplies two numbers: multiply(a: str, b: str)"""
    return str(float(a) * float(b))


def main():
    # --- Example 1: Local Ollama ---
    # llm = LLM(model="gemma4:e4b", provider="ollama")

    # --- Example 2: Groq (Fast cloud inference) ---
    # export GROQ_API_KEY="your-key"
    # llm = LLM(model="llama-3.3-70b-versatile", provider="groq")

    # --- Example 3: OpenAI ---
    # export OPENAI_API_KEY="your-key"
    # llm = LLM(model="gpt-4o-mini", provider="openai")

    # --- Example 4: OpenRouter (Access any cloud model) ---
    # export OPENROUTER_API_KEY="your-key"
    # llm = LLM(model="meta-llama/llama-3.1-8b-instruct:free", provider="openrouter")

    # Default to local Ollama or custom base_url
    llm = LLM(model="gemma4:31b", provider="ollama_cloud")

    # Register tool
    tool_schema = {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": [
                    "a",
                    "b",
                ],
            },
        },
    }

    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)",
        schema=tool_schema,
    )


    agent = TinyAgent(
        llm=llm,
        memory=SummarizationMemory(llm=llm),
        tools=tools,
        record_trajectory=True,
    )

    print(f"Agent initialized with provider: {llm.base_url} (model: {llm.model})")
    print("Type your message below. Type '/exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        result = agent.run(user_input)
        print(f"Agent: {result}\n")

        # Display trajectory box if recording is enabled
        if agent.trajectory:
            print(agent.trajectory.format_latest_run())
            print()



if __name__ == "__main__":
    main()
