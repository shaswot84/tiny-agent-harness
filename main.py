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
    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)",
    )

    agent = TinyAgent(
        llm=llm,
        memory=TrimmingMemory(),
        tools=tools,
        record_trajectory=True,
    )


    print(f"Agent initialized with provider: {llm.base_url} (model: {llm.model})")
    # Quick single turn test (no heavy loops)
    result = agent.run("Hello! Introduce yourself briefly.")
    print("Agent:", result)


if __name__ == "__main__":
    main()
