# Changelog

All notable changes to the `tiny-agent` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Tool Registry (`TinyAgent.tools`)**:
  - Introduced `Tools` registry supporting tool registration (`add_tool`) with functions and descriptions.
  - Added support for specifying tools requiring human approval (`requires_approval`).
  - Added placeholder property for native tool-calling schemas (`schemas`).
  - Integrated `tools` parameter into `TinyAgent` constructor.
  - Added unit test coverage in `tests/test_tools.py`.

---


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
