# ChromaDB PDF Navigator

A desktop application for syncing PDF files with a ChromaDB vector database, built with Python and PyQt5.

## Features

- **PDF Processing**: Automatically extracts text and chunks PDFs using semantic analysis
- **ChromaDB Integration**: Stores chunks with rich metadata including APA-style references
- **Smart Sync**: Detects new/removed PDFs and maintains database consistency
- **Search Interface**: Query the database for specific content
- **Modern UI**: Clean, intuitive interface with file browser and database viewer

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

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
- PyQt5
- ChromaDB 0.3.25
- PyPDF for PDF processing
- Sentence Transformers for embeddings
