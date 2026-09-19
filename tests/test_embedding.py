from TinyAgent.embedding import EmbeddingModel


def test_embedding_model_init():
    model = EmbeddingModel(model="all-minilm:22m", provider="ollama")
    assert model.model == "all-minilm:22m"
    assert "11434" in model.base_url


def test_cosine_similarity_logic():
    # Test deterministic vector cosine similarity calculation
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_c = [0.9, 0.1, 0.0]

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    assert sim_ab == 0.0
    assert sim_ac > 0.9
    assert sim_ac > sim_ab
