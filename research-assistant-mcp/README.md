# Research Assistant MCP Server

An MCP (Model Context Protocol) server that exposes a PDF library as searchable resources to AI assistants like Claude. It combines ChromaDB vector search with PDF extraction tools so Claude can find, read, and cite documents from your collection.

## Features

- **Semantic search** over PDF content via ChromaDB embeddings
- **Filename search** for finding papers by author or title tokens
- **PDF reading** with page-range selection for efficient access
- **OCR support** for scanned documents via Tesseract
- **Image extraction** from PDF pages
- **Document structure analysis** and metadata inspection
- **APA citation guidance** baked into tool descriptions
- Each PDF in your library is registered as a `library://` MCP resource

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — required only for `read_pdf_with_ocr`
- A ChromaDB database (created by the ChromaDB Navigator GUI app, or by running the server with `--update_db True`)

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd ChromaDBNavigator/research-assistant-mcp
```

### 2. Install dependencies

Using `uv` (recommended):

```bash
uv venv
uv pip install -r requirements.txt
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Tesseract (optional, for OCR)

- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

## Running the Server

```bash
uv run python research-assistant.py \
  --library_directory /path/to/your/pdfs \
  --chroma_db_path /path/to/your/chroma_db
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--library_directory` | `~/Downloads/pdfs` | Directory containing your PDF files |
| `--chroma_db_path` | `~/Downloads/pdfs_db` | Path to the ChromaDB persistent database |
| `--update_db` | `False` | Set to `True` to sync new/modified PDFs into ChromaDB on startup |
| `--limit_text` | `-1` | Truncate extracted text to N characters (`-1` = unlimited) |

### First run (building the database)

If you do not have an existing ChromaDB database, run with `--update_db True` to index your PDFs:

```bash
uv run python research-assistant.py \
  --library_directory /path/to/your/pdfs \
  --chroma_db_path /path/to/your/chroma_db \
  --update_db True
```

Indexing can take several minutes for large collections. Subsequent runs without `--update_db True` start instantly.

## Integrating with Claude

### Claude Desktop

Add the server to your Claude Desktop configuration file.

**Config file location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": [
        "/absolute/path/to/research-assistant-mcp/research-assistant.py",
        "--library_directory", "/path/to/your/pdfs",
        "--chroma_db_path", "/path/to/your/chroma_db"
      ]
    }
  }
}
```

On Windows, use the full path to the venv Python executable:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\research-assistant-mcp\\research-assistant.py",
        "--library_directory", "C:\\path\\to\\your\\pdfs",
        "--chroma_db_path", "C:\\path\\to\\your\\chroma_db"
      ]
    }
  }
}
```

Restart Claude Desktop after editing the config.

### Claude Code (CLI)

Add the server to your Claude Code MCP settings:

```bash
claude mcp add research-assistant \
  /absolute/path/to/.venv/bin/python \
  -- \
  /absolute/path/to/research-assistant-mcp/research-assistant.py \
  --library_directory /path/to/your/pdfs \
  --chroma_db_path /path/to/your/chroma_db
```

Or edit `~/.claude/settings.json` (user-level) or `.claude/settings.json` (project-level) directly:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": [
        "/absolute/path/to/research-assistant-mcp/research-assistant.py",
        "--library_directory", "/path/to/your/pdfs",
        "--chroma_db_path", "/path/to/your/chroma_db"
      ]
    }
  }
}
```

## Available Tools

Once connected, Claude has access to these tools:

| Tool | Description |
|---|---|
| `search_content` | Semantic vector search across all indexed PDF content |
| `search_title` | Token-overlap search against PDF filenames (useful for author lookups) |
| `read_pdf_text` | Extract text from a PDF, optionally limited to a page range |
| `read_pdf_with_ocr` | Extract text using Tesseract OCR for scanned documents |
| `extract_pdf_images` | Save embedded images from a PDF to disk |
| `get_pdf_info` | File metadata, page count, and document statistics |
| `analyze_pdf_structure` | Page-level content categorization (text/images/mixed) |

All PDFs in `--library_directory` are also registered as `library://` resources that Claude can read directly.

## Debugging

Use the MCP Inspector to test the server interactively (requires Node.js):

```bash
./debug_mcp_inspector.sh
```

This starts the server against the `../test_data/pdfs/` and `../test_data/pdfs_db/` test fixtures and opens a browser-based inspector UI.

## Usage Tips for Claude

When the server is connected, Claude will:

- Use `search_content` for topic-based queries and cite results in APA format
- Use `search_title` when you name an author or a specific paper
- Automatically narrow page reads to relevant sections returned by search results
- Cite sources as `(Author Year)` based on PDF filenames and embedded metadata

Example prompts:

```
What does the literature say about transformer attention mechanisms?
Find papers by Smith on reinforcement learning.
Summarize the methodology section of Jones2022.pdf.
```
