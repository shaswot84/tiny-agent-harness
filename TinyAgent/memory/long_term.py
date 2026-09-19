from typing import TYPE_CHECKING

from .base import Memory

if TYPE_CHECKING:
    from ...embedding import EmbeddingModel


class RAGMemory(Memory):
    """Long-term Memory with RAG using vector embeddings."""

    def __init__(self, embedding_model: "EmbeddingModel", documents: list[str], top_k: int = 3):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.top_k = top_k
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        # Augment user queries with retrieved context before storing
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        super().add(role, content, **kwargs)

    def search(self, query: str) -> list[str]:
        """Return the top-k documents most similar to the query."""
        query_embed = self.embedding_model.embed(query)
        scores = [self._cosine(query_embed, embed) for embed in self.embeddings]
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self.documents[index] for index in ranked[: self.top_k]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        if norm == 0:
            return 0.0
        return dot / norm
