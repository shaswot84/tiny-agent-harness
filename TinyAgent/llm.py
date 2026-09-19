import json
import os
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


# Standard base URLs for OpenAI-compatible cloud providers
PROVIDER_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "ollama_cloud": "https://ollama.com/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "together": "https://api.together.xyz/v1",
}


class LLM:
    """A minimal OpenAI-compatible LLM wrapper supporting local and cloud models."""

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        think: bool = False,
    ):
        self.model = model
        self.think = think

        # Resolve base_url: explicit > provider preset > OLLAMA_BASE_URL > default local
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider and provider.lower() in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[provider.lower()]
        else:
            self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")

        # Resolve api_key: explicit > provider-specific env var > generic LLM_API_KEY > fallback
        if api_key:
            self.api_key = api_key
        elif provider:
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var, os.getenv("LLM_API_KEY", "ollama"))
        else:
            self.api_key = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "ollama"))

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Response:
        """Generate a response from the LLM."""

        payload = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools

        if not self.think:
            payload["reasoning_effort"] = "none"

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        choice = result["choices"][0]
        message = choice["message"]

        tool_call = None
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]

        return Response(
            content=message.get("content", ""),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=result,
        )