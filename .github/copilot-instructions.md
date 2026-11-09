# DUM-E project rules (read before every task)

Project: DUM-E, a RAG chatbot backend. Python 3.11+, Windows, CPU only, 8 GB RAM.

Flow:
user question -> exact-match response cache
  hit  -> return cached answer
  miss -> ChromaDB retrieval -> CrossEncoder rerank -> LLM
          -> store Q&A in active window (max 6 turns)
          -> if window overflows, oldest turn -> Sumy summary -> long-term buffer

Rules:
- Code lives in src/dume/. Use type hints, dataclasses and docstrings. Use the logging module; print() only in cli.py.
- All settings come from src/dume/config.py (reads .env). Never hardcode paths, keys or model names.
- Lazy-load heavy models (embedder, CrossEncoder) on first use and reuse them.
- CPU only. Never add CUDA packages. Batch sizes 32 or less.
- LLM access goes only through the LLMClient interface in src/dume/llm/, using the openai SDK against an OpenAI-compatible base URL.
- Keep pure logic testable without network or models: inject dependencies, write pytest tests with fakes.
- Use pathlib for paths. Keep functions small. Raise clear errors, never swallow exceptions silently.
- Do not add dependencies without telling me, and keep requirements.txt updated.
- After each task, tell me exactly how to run and test it.