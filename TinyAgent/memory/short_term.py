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
            # Extract existing system sections (instructions, tools, previous summary)
            instructions = []
            tools_section = []
            previous_summary = ""

            for message in self.messages:
                if message["role"] == "system":
                    sys_content = message["content"]
                    if "[Conversation Summary]" in sys_content:
                        # Split by section headers if formatted previously
                        parts = sys_content.split("[Conversation Summary]")
                        prefix = parts[0].strip()
                        previous_summary = parts[1].strip() if len(parts) > 1 else ""
                        if prefix:
                            instructions.append(prefix)
                    elif "# Tools" in sys_content or "[Tools]" in sys_content:
                        tools_section.append(sys_content)
                    else:
                        instructions.append(sys_content)

            conversation = ""
            for message in self.messages:
                if message["role"] != "system":
                    conversation += f"{message['role']}: {message['content']}\n"

            prompt = f"""Update the summary with the new conversation.

Summary: {previous_summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            updated_summary = response.content.strip()

            # Compose sections cleanly
            system_parts = []
            if instructions:
                system_parts.append("\n\n".join(instructions).strip())
            if tools_section:
                system_parts.append("\n\n".join(tools_section).strip())

            if system_parts:
                full_system_prompt = (
                    "\n\n".join(system_parts)
                    + f"\n\n[Conversation Summary]\n{updated_summary}"
                )
            else:
                full_system_prompt = updated_summary

            self.messages = [{"role": "system", "content": full_system_prompt}]

