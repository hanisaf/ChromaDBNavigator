# ChromaDB PDF Navigator

A desktop application for syncing PDF files with a ChromaDB vector database.

## Features

- **PDF Processing**: Automatically extracts text and chunks PDFs using semantic analysis
- **ChromaDB Integration**: Stores chunks with rich metadata including APA-style references
- **Smart Sync**: Detects new/removed PDFs and maintains database consistency
- **Search Interface**: Query the database for specific content
- **Native Windows UI**: WinForms desktop app on Windows
- **PyQt UI (legacy)**: Existing Python interface remains available

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage (Windows Native)

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Build and run the Windows UI:
   ```bash
   cd windows-native
   dotnet run
   ```
3. In the app, select database path/folder and run sync/search/browse/stats.

Notes:
- The WinForms app starts `native_backend_server.py` and reuses `chroma_manager.py`.
- If `python` is not on PATH, set `CHROMA_PYTHON_EXE` to the full Python executable path.

## Usage (PyQt Legacy)

1. Run the application:
   ```bash
   python main.py
   ```

2. Select your PDF folder
3. Click "Sync" to index all PDFs
4. Use the search interface to query your database
5. Navigate and manage your vector database

## Requirements

- Python 3.8+
- PyQt5 (legacy UI)
- ChromaDB 0.3.25
- PyPDF for PDF processing
- Sentence Transformers for embeddings
- \.NET 7 SDK or newer (Windows native UI)

## Using uv to install requirments

uv pip compile requirements.txt --universal --output-file requirements.uv

uv venv

source .venv/bin/activate

uv pip sync requirements.uv
