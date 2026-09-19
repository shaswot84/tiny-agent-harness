import json
import os
import urllib.request
from dotenv import load_dotenv

from ..llm import PROVIDER_BASE_URLS

load_dotenv()


class EmbeddingModel:
    """Generate embeddings from text using OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the embedding model with model name and optional provider/endpoint."""
        self.model = model

        # Resolve base_url: explicit > provider preset > env > default local Ollama
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider and provider.lower() in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[provider.lower()]
        else:
            self.base_url = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")

        # Resolve api_key: explicit > provider-specific env var > generic LLM_API_KEY > fallback
        if api_key:
            self.api_key = api_key
        elif provider:
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var, os.getenv("LLM_API_KEY", "ollama"))
        else:
            self.api_key = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "ollama"))

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        # POST to the OpenAI-compatible /embeddings endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read().decode("utf-8"))

        # Extract and return the embedding vector
        return response["data"][0]["embedding"]
