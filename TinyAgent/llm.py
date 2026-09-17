import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


class LLM:
    """A simple LLM wrapper."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        think: bool = False,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think

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