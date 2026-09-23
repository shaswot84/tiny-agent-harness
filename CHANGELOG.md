# Changelog

All notable changes to the `tiny-agent` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Tool Parsing, Execution & Native Support (`TinyAgent.tools`)**:
  - Added `Tools.parse()` to extract JSON tool calls from free-form model text.
  - Added `Tools.execute()` supporting both prompt-based (`tool`, `kwargs`) and OpenAI-compatible native (`function`, `arguments`) tool calls, with human-in-the-loop approval confirmation.
  - Added `Tools.observation()` supporting role formatting (`"tool"` for native tool calls, `"user"` for prompt-based tool calls).
  - Added `Tools.is_done()` stopping mechanism supporting `final_answer` unpacking and natural completions.
  - Added `NativeTools(Tools)` subclass for models supporting native function calling with automatic schema mapping, zero system prompt stuffing, and `role: "tool"` observations.
  - Added `tool_to_schema()` utility using Python's `inspect` to automatically convert Python functions, docstrings, and type annotations into OpenAI-compatible JSON tool schemas.
  - Added `TinyAgent.tools.toolbox` containing standard built-in functions: `add`, `multiply`, `subtract`, `divide`, `power`, `execute_command`, and `final_answer`.
  - Added automatic tool schema generation fallback in `Tools.add_tool(..., schema=None)`.
  - Exported `NativeTools`, `tool_to_schema`, and `toolbox` in `TinyAgent` root package.

- **ReAct Planner (`TinyAgent.planner`)**:
  - Introduced `ReAct` planner module with configurable `max_steps`.
  - Added ReAct system prompt template specifying `THOUGHT`, `ACTION` JSON block, and `final_answer` completion semantics.
  - Implemented `ReAct.parse()` extracting `THOUGHT` reasoning and `ACTION` tool invocations using regex patterns into `Response.reasoning` and `Response.content`.
  - Integrated `planner` argument into `TinyAgent` constructor, automatically prepending `planner.prompt` into system memory and parsing step responses.
  - Exported `ReAct` in `TinyAgent` and `TinyAgent.planner`.
  - Added unit and agent integration tests in `tests/test_react.py`.

- **Iterative Tool Execution Loop (`TinyAgent.agent`)**:
  - Implemented multi-step loop in `TinyAgent.run()` / `_step()` to execute tools and feed observation responses back to the LLM until `is_done()` or safety step limit is reached.
  - Added automatic tools system prompt injection for prompt-based models when tools are registered.


- **Section-Aware Summarization Memory (`TinyAgent.memory`)**:
  - Enhanced `SummarizationMemory` to separate and preserve system instructions and `[Tools]` definitions while updating only the `[Conversation Summary]` section.

- **CLI Trajectory Box Formatting (`TinyAgent.trajectory`)**:
  - Added `Trajectory.format_latest_run()` rendering clean Unicode bordered boxes (`┌─┐`, `│`, `└─┘`) with accurate display widths for terminal emojis and multi-line word wrapping.
  - Added interactive CLI chat loop in `main.py` with `/exit` command and automatic trajectory box display after each turn.

- **Comprehensive Test Coverage**:
  - Added test cases in `tests/test_tools.py` for `parse`, `execute`, `observation`, `is_done`, `tool_to_schema`, `NativeTools`, and `toolbox`.
  - Added `test_agent_tool_execution_loop` and `test_agent_tools_system_prompt` in `tests/test_agent.py`.
  - Added `test_summarization_memory_preserves_tools_and_instructions` in `tests/test_memory.py`.
  - Added `test_trajectory_format_latest_run` in `tests/test_trajectory.py`.
## [0.1.0] - 2026-09-19

### Added
- **Modular Memory Architecture**:
  - `Memory` (Base): Fundamental in-memory message history container.
  - `TrimmingMemory` (Short-term): Configurable sliding window that keeps system messages plus the last $N$ turns (defaults to 4 messages).
  - `SummarizationMemory` (Short-term): Periodically compresses previous conversation history into a running system summary using an LLM.
  - `RAGMemory` (Long-term): Vector retrieval-augmented generation memory that indexes documents, computes cosine similarities against incoming queries, and augments user prompts with the top-$k$ relevant context snippets.
- **Embedding Support (`TinyAgent.embedding`)**:
  - Introduced `EmbeddingModel` supporting OpenAI-compatible `/embeddings` endpoints.
  - Built-in provider resolution for local Ollama (e.g., `all-minilm:22m`, `nomic-embed-text`) and cloud providers.
  - Integrated vector cosine similarity calculation logic.
- **Multi-Provider Cloud & Local LLM Support (`TinyAgent.llm`)**:
  - Added preset base URLs and automatic environment variable resolution for `ollama`, `ollama_cloud`, `openai`, `groq`, `openrouter`, `deepseek`, `gemini`, and `together`.
  - Added `.env` support via `python-dotenv` with zero-leak protection (added `.env` to `.gitignore`).
  - Added `.env.example` template with configuration keys for supported providers.
- **Trajectory Formatting & Toggle (`TinyAgent.trajectory`)**:
  - Formatted multi-line trajectory outputs with boxed run headers (`┌─── 📍 Run N ───┘`) and labeled step cards (`💭 Thought`, `🛠️ Action`, `👁️ Observation`, `💬 Answer`).
  - Added `record_trajectory: bool = False` flag to `TinyAgent` constructor to cleanly enable/disable trajectory recording.
- **Test Suite (`tests/`)**:
  - Added `pytest` setup in `pyproject.toml`.
  - Added `tests/test_agent.py` covering core run cycle, message persistence, and trajectory toggles.
  - Added `tests/test_memory.py` covering Base, Trimming, Summarization, and RAG memory strategies using deterministic mocks.
  - Added `tests/test_trajectory.py` covering step formatting and trajectory tracking.
  - Added `tests/test_embedding.py` covering embedding initialization and cosine similarity calculations.
  - Added `.pytest_cache/` to `.gitignore`.

### Changed
- **Architectural Modularization**:
  - Restructured `TinyAgent/` from flat files into dedicated subpackages:
    - `TinyAgent/agent/` (`core.py`)
    - `TinyAgent/llm/` (`client.py`)
    - `TinyAgent/embedding/` (`client.py`)
    - `TinyAgent/trajectory/` (`tracker.py`)
    - `TinyAgent/memory/` (`base.py`, `short_term.py`, `long_term.py`)
  - Configured top-level exports in `TinyAgent/__init__.py` to maintain ergonomic imports: `from TinyAgent import TinyAgent, LLM, EmbeddingModel, ...`.
- **Main Script Simplification**:
  - Cleaned up `main.py` into a lightweight, configurable entry point.
  - Shifted scratch embedding tests into official unit test suites.
