import pytest
from TinyAgent.memory import Memory, TrimmingMemory, SummarizationMemory, RAGMemory
from TinyAgent.llm import Response


def test_base_memory():
    mem = Memory()
    assert len(mem.get_messages()) == 0

    mem.add("user", "Hello")
    mem.add("assistant", "Hi there!")

    messages = mem.get_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there!"}


def test_trimming_memory():
    # Keep only the last 4 non-system messages (2 turns)
    mem = TrimmingMemory(recent=4)
    mem.add("system", "You are a helpful assistant.")

    # 3 user/assistant turns = 6 messages + 1 system
    for i in range(1, 4):
        mem.add("user", f"Turn {i} question")
        mem.add("assistant", f"Turn {i} answer")

    messages = mem.get_messages()
    # Should retain 1 system message + last 4 turn messages
    assert len(messages) == 5
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Turn 2 question"
    assert messages[4]["content"] == "Turn 3 answer"


def test_summarization_memory():
    # Mock LLM to test summarization without external network calls
    class MockLLM:
        def generate(self, messages):
            return Response(content="Summary of user facts.")

    mem = SummarizationMemory(llm=MockLLM())
    mem.add("user", "My name is Alice.")
    mem.add("assistant", "Nice to meet you Alice!")

    messages = mem.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Summary of user facts."


def test_rag_memory():
    # Deterministic mock embedder mapping keywords to orthogonal vectors
    class MockEmbeddingModel:
        def embed(self, text: str):
            text_lower = text.lower()
            if "dog" in text_lower or "puppy" in text_lower:
                return [1.0, 0.0]
            elif "cat" in text_lower or "kitten" in text_lower:
                return [0.0, 1.0]
            return [0.5, 0.5]

    docs = [
        "Dogs love running in the park.",
        "Cats enjoy sleeping on the couch.",
    ]
    rag = RAGMemory(embedding_model=MockEmbeddingModel(), documents=docs, top_k=1)

    # Search for dog
    results = rag.search("puppy")
    assert results == ["Dogs love running in the park."]

    # Search for cat
    results_cat = rag.search("kitten")
    assert results_cat == ["Cats enjoy sleeping on the couch."]

    # Verify add() augments user prompt
    rag.add("user", "Tell me about puppies")
    messages = rag.get_messages()
    assert len(messages) == 1
    assert "Context:\nDogs love running in the park." in messages[0]["content"]
    assert "Question: Tell me about puppies" in messages[0]["content"]
