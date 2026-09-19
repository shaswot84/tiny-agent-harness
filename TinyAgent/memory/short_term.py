from typing import TYPE_CHECKING

from .base import Memory

if TYPE_CHECKING:
    from ...llm import LLM


class TrimmingMemory(Memory):
    """Short-term memory that keeps only the last recent user/assistant turns."""

    def __init__(self, recent: int = 4):
        super().__init__()
        self.recent = recent

    def add(self, role: str, content: str, **kwargs) -> None:
        # Add the new message first using the parent class
        super().add(role, content, **kwargs)

        # Retain system prompt plus the most recent turns
        system = [msg for msg in self.messages if msg["role"] == "system"]
        turns = [msg for msg in self.messages if msg["role"] != "system"]
        self.messages = system + turns[-self.recent:]


class SummarizationMemory(Memory):
    """Short-term memory that compresses past turns into a running summary via an LLM."""

    def __init__(self, llm: "LLM"):
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)

        # After each completed turn, update the running summary
        if role == "assistant":
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            self.messages = [{"role": "system", "content": response.content}]
