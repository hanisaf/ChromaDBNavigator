# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: PyQt5 desktop app entry point (UI + wiring).
- `chroma_manager.py`: ChromaDB client, sync, search, and stats.
- `pdf_processor.py`: PDF parsing, page-based chunking, APA reference.
- `config.py`: Tunables (collection name, chunk sizes, UI bounds).
- `tests/`: Runnable test scripts (`test_installation.py`, `test_metal.py`, `test_stats_fix.py`).
- `dev/`: Developer helpers (installer, launcher, version checks).
- `fixes/`: Compatibility scripts (e.g., HuggingFace, Pydantic).
- `research-assistant-mcp/`: Optional MCP server utilities.

## Build, Test, and Development Commands
- Setup venv: 
  - `python3 -m venv venv && source venv/bin/activate`
- Install deps: 
  - `pip install -r requirements.txt`
- Run app: 
  - `python main.py` or `python dev/run.py`
- Sanity tests: 
  - `python tests/test_installation.py`
  - `python tests/test_metal.py` (GPU/MPS perf)
  - `python tests/test_stats_fix.py` (large-file stats)

## Coding Style & Naming Conventions
- Indentation: 4 spaces; follow PEP 8.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE` (see `config.py`).
- Type hints: preferred for public APIs (`chroma_manager.py`, `pdf_processor.py`).
- Imports: stdlib → third-party → local, grouped with blank lines.
- Formatting: keep lines readable; `black`/`isort` may be used locally (not enforced).

## Testing Guidelines
- Tests live in `tests/` and are runnable scripts.
- Add new tests as `tests/test_*.py`; prefer small, focused checks.
- GPU paths: validate via `tests/test_metal.py` on macOS (MPS) or CUDA.
- Goal: cover critical flows (PDF parse → chunk → add/query → stats).

## Commit & Pull Request Guidelines
- Commits: imperative, concise summaries (≤72 chars). Example: 
  - `Add APA extraction to PDFProcessor`
- PRs: include description, steps to reproduce, screenshots for UI changes, and linked issues. Note environment (OS, Python) and any perf impact.
- Keep changes scoped; update docs when behavior or config changes.

## Security & Configuration Tips
- No secrets stored; avoid committing local DBs or PDFs.
- Default DB path: `~/.chromadb/db` (override via UI/args).
- Compatibility: use pinned deps in `requirements.txt`; fix scripts in `fixes/` can help resolve version issues.
- Large downloads (embeddings/model deps) occur on first run; ensure disk space.
