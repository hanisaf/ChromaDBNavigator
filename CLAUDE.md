# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChromaDB PDF Navigator is a desktop application built with PyQt5 that synchronizes PDF files with a ChromaDB vector database. The application enables semantic search across PDF document collections with rich metadata extraction and APA reference generation.

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **main.py**: PyQt5 GUI application with multi-threaded sync operations
- **chroma_manager.py**: ChromaDB operations, embedding management, and device optimization
- **pdf_processor.py**: PDF text extraction, chunking strategies, and metadata generation  
- **config.py**: Application configuration and constants

## Key Dependencies

- **ChromaDB**: Currently pinned to version 1.0.20 (uses PersistentClient)
- **PyQt5**: GUI framework (version 5.15.9)
- **PyPDF**: PDF text extraction (version 3.17.4) 
- **Sentence Transformers**: Text embeddings with device optimization (2.2.2)
- **Pydantic**: Data validation (< 2.0.0 for ChromaDB compatibility)
- **NumPy**: Constrained to < 2.0.0 for compatibility

## Development Commands

### Running the Application
```bash
python main.py                    # Direct execution
python dev/run.py                 # Using launcher script
```

### Command Line Options
```bash
python main.py --db-path /path/to/db --pdf-folder /path/to/pdfs --collection my_collection
```

### Testing
```bash
python tests/test_installation.py    # Verify dependencies
python tests/test_metal.py          # Test Metal/MPS acceleration
python tests/test_stats_fix.py      # Test database statistics
```

### Development Tools
```bash
python dev/check_chromadb_version.py    # Check ChromaDB compatibility
python dev/install_dependencies.py      # Install with compatibility fixes
```

### Compatibility Fixes
```bash
python fixes/fix_pydantic_compatibility.py      # Fix Pydantic v2 conflicts
python fixes/fix_huggingface_compatibility.py   # Fix HuggingFace Hub issues
python fixes/fix_numpy_compatibility.py         # Fix NumPy 2.x conflicts
```

## Code Architecture Notes

### Threading and Concurrency
- Main GUI runs on primary thread
- PDF sync operations use `SyncWorker(QThread)` to prevent UI blocking
- Progress callbacks update UI thread safely via Qt signals

### Device Optimization
- `ChromaManager._get_optimal_device()` detects best embedding device:
  - CUDA GPU (highest priority)
  - Apple Metal/MPS (macOS optimization) 
  - CPU fallback
- Sentence Transformers model automatically moved to optimal device

### PDF Processing Strategies
- Page-based chunking (default): One chunk per PDF page
- Semantic chunking: Academic document structure detection
- Recursive chunking: Paragraph/sentence-based fallback
- Configurable chunk sizes and overlap in config.py

### Database Schema
Each PDF chunk includes rich metadata:
- File information (filename, path, size, page count)
- Chunk details (index, type, size, page number)  
- APA reference extraction from filename
- Processing timestamps

### Sync Operations
- Preview changes before execution with confirmation dialog
- Detect new, modified, and removed files via MD5 hashing
- Atomic operations: remove old chunks before adding updated ones
- Batch processing with progress reporting

## Important Implementation Details

### ChromaDB Version Compatibility
- Uses `chromadb.PersistentClient()` for 1.0.x compatibility
- Collection operations use newer API syntax (.add(), .query(), .get())
- Metadata filtering uses {"$in": values} syntax for bulk operations

### Error Handling
- Graceful degradation for corrupted PDFs
- Device fallback chain (GPU → Metal → CPU)
- Comprehensive exception handling with user-friendly error messages

### File Organization
- `dev/`: Development utilities and launchers
- `fixes/`: Compatibility scripts for dependency conflicts
- `tests/`: Installation and functionality verification
- `research-assistant-mcp/`: MCP server implementation (separate feature)

## Development Tips

### Adding New PDF Processing Features
1. Extend `PDFProcessor` class methods
2. Update chunk metadata schema in `chroma_manager.py:_add_file_to_db()`
3. Modify UI display in `main.py` result/document detail methods

### Database Schema Changes
1. Update metadata preparation in `chroma_manager.py:_add_file_to_db()`
2. Modify statistics collection in `get_database_stats()`
3. Update UI display methods for new fields

### UI Extensions  
1. Follow existing PyQt5 patterns in `main.py`
2. Use threaded workers for long-running operations
3. Update progress callbacks and status messages
4. Apply consistent styling via `apply_styling()` method

### Testing New Features
- Use existing test files in `tests/` as templates
- Test device optimization across different platforms
- Verify compatibility with existing ChromaDB collections