"""
Configuration file for ChromaDB PDF Navigator
"""

# Database settings
COLLECTION_NAME = "references"  # Default collection name

# PDF processing settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50

# Search settings
DEFAULT_SEARCH_RESULTS = 10
MAX_SEARCH_RESULTS = 50

# UI settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
LEFT_PANEL_WIDTH = 400
RIGHT_PANEL_WIDTH = 1000

# File settings
SUPPORTED_EXTENSIONS = ['.pdf', '.PDF']
MAX_FILE_SIZE_MB = 100  # Maximum file size to process

# Sync settings
SYNC_BATCH_SIZE = 10  # Process files in batches for better progress reporting
