#!/usr/bin/env python3
"""
Script to fix Pydantic compatibility issues with ChromaDB 0.3.25
"""

import subprocess
import sys
import importlib

def check_pydantic_version():
    """Check the current Pydantic version."""
    try:
        import pydantic
        version = pydantic.__version__
        print(f"Current Pydantic version: {version}")
        return version
    except ImportError:
        print("Pydantic not installed")
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

def fix_pydantic_compatibility():
    """Fix Pydantic compatibility issues."""
    print("Fixing Pydantic compatibility issues...")
    
    pydantic_version = check_pydantic_version()
    chromadb_version = check_chromadb_version()
    
    if pydantic_version and pydantic_version.startswith('2.'):
        print("\n⚠️  Pydantic 2.x detected - this causes compatibility issues with ChromaDB 0.3.25")
        print("Attempting to fix...")
        
        try:
            # Install compatible Pydantic version
            print("\n1. Installing Pydantic 1.x...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pydantic<2.0.0"], 
                         check=True, capture_output=True, text=True)
            print("✓ Pydantic 1.x installed successfully")
            
            # Install pydantic-settings
            print("\n2. Installing pydantic-settings...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pydantic-settings"], 
                         check=True, capture_output=True, text=True)
            print("✓ pydantic-settings installed successfully")
            
            # Check if the issue is resolved
            try:
                import chromadb
                print("✓ ChromaDB import test successful")
                return True
            except Exception as e:
                print(f"✗ ChromaDB import still failing: {e}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install compatible Pydantic: {e}")
        
        print("\n❌ Automatic fix failed. Please try manual solutions:")
        print("1. pip install 'pydantic<2.0.0'")
        print("2. pip install pydantic-settings")
        print("3. pip install -r requirements.txt")
        return False
        
    else:
        print("✓ Pydantic version is compatible")
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
    print("ChromaDB Pydantic Compatibility Fixer")
    print("=" * 50)
    
    # Check current state
    print("\nChecking current installation...")
    pydantic_version = check_pydantic_version()
    chromadb_version = check_chromadb_version()
    
    # Test ChromaDB import
    print("\nTesting ChromaDB import...")
    if test_chromadb_import():
        print("\n🎉 No compatibility issues detected!")
        return True
    
    # Try to fix the issue
    print("\nCompatibility issues detected. Attempting to fix...")
    if fix_pydantic_compatibility():
        print("\n🎉 Compatibility issues resolved!")
        return True
    else:
        print("\n❌ Could not automatically resolve compatibility issues.")
        print("Please try the manual solutions mentioned above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
