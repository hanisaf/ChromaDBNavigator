#!/usr/bin/env python3
"""
Launcher script for ChromaDB PDF Navigator
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
