from TinyAgent.llm import LLM
from TinyAgent.agent import TinyAgent
from TinyAgent.memory import Memory, TrimmingMemory, SummarizationMemory
from TinyAgent.embedding import EmbeddingModel

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

    agent = TinyAgent(
        llm=llm,
        memory=TrimmingMemory(),
        record_trajectory=True
    )

    print(f"Agent initialized with provider: {llm.base_url} (model: {llm.model})")
    # Quick single turn test (no heavy loops)
    result = agent.run("Hello! Introduce yourself briefly.")
    print("Agent:", result)

    print("\n--- Embeddings Test ---")
    embedding_model = EmbeddingModel(model="all-minilm:22m", provider="ollama")

    # Create embeddings
    embedding_a = embedding_model.embed("I love flamingos.")
    embedding_b = embedding_model.embed("Dolphins use echolocation.")
    embedding_c = embedding_model.embed("Flamingos are pink birds.")
     
    # Calculate cosine similarity between A and B
    dot_ab = sum(x * y for x, y in zip(embedding_a, embedding_b))
    norm_a = sum(x * x for x in embedding_a) ** 0.5
    norm_b = sum(x * x for x in embedding_b) ** 0.5
    similarity_ab = dot_ab / (norm_a * norm_b)
     
    # Calculate cosine similarity between A and C
    dot_ac = sum(x * y for x, y in zip(embedding_a, embedding_c))
    norm_c = sum(x * x for x in embedding_c) ** 0.5
    similarity_ac = dot_ac / (norm_a * norm_c)
     
    print(f"Similarity between A and B: {similarity_ab}")
    print(f"Similarity between A and C: {similarity_ac}")


if __name__ == "__main__":
    main()
