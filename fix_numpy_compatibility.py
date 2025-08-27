#!/usr/bin/env python3
"""
Script to fix ChromaDB compatibility issues
"""

import subprocess
import sys
import importlib

def check_numpy_version():
    """Check the current NumPy version."""
    try:
        import numpy as np
        version = np.__version__
        print(f"Current NumPy version: {version}")
        return version
    except ImportError:
        print("NumPy not installed")
        return None

def check_chromadb_version():
    """Check the current ChromaDB version."""
    try:
        import chromadb
        version = chromadb.__version__
        print(f"Current ChromaDB version: {version}")
        return version
    except ImportError:
        print("ChromaDB not installed")
        return None

def fix_compatibility():
    """Fix ChromaDB compatibility issues."""
    print("Fixing ChromaDB compatibility issues...")
    
    numpy_version = check_numpy_version()
    chromadb_version = check_chromadb_version()
    
    if chromadb_version and chromadb_version != "0.3.25":
        print(f"\n⚠️  ChromaDB version {chromadb_version} detected - this may cause compatibility issues")
        print("Attempting to fix...")
        
        try:
            # Install the correct ChromaDB version
            print("\n1. Installing ChromaDB 0.3.25...")
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb==0.3.25"], 
                         check=True, capture_output=True, text=True)
            print("✓ ChromaDB 0.3.25 installed successfully")
            
            # Check if the issue is resolved
            try:
                import chromadb
                print("✓ ChromaDB import test successful")
                return True
            except Exception as e:
                print(f"✗ ChromaDB import still failing: {e}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install ChromaDB 0.3.25: {e}")
        
        print("\n❌ Automatic fix failed. Please try manual solutions:")
        print("1. pip install chromadb==0.3.25")
        print("2. pip install -r requirements.txt")
        return False
        
    else:
        print("✓ ChromaDB version is compatible")
        return True

def test_chromadb_import():
    """Test if ChromaDB can be imported successfully."""
    try:
        import chromadb
        print("✓ ChromaDB import successful!")
        return True
    except Exception as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False

def main():
    """Main function."""
    print("ChromaDB Compatibility Fixer")
    print("=" * 40)
    
    # Check current state
    print("\nChecking current installation...")
    numpy_version = check_numpy_version()
    chromadb_version = check_chromadb_version()
    
    # Test ChromaDB import
    print("\nTesting ChromaDB import...")
    if test_chromadb_import():
        print("\n🎉 No compatibility issues detected!")
        return True
    
    # Try to fix the issue
    print("\nCompatibility issues detected. Attempting to fix...")
    if fix_compatibility():
        print("\n🎉 Compatibility issues resolved!")
        return True
    else:
        print("\n❌ Could not automatically resolve compatibility issues.")
        print("Please try the manual solutions mentioned above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
