#!/bin/bash
npx @modelcontextprotocol/inspector uv run python research-assistant.py --library_directory "../test_data/pdfs" --chroma_db_path "../test_data/pdfs_db" --limit_text -1 --update_db True
