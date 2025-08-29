#!/usr/bin/env python3
"""
Script to fix huggingface-hub compatibility issues with sentence-transformers
"""

import subprocess
import sys
import importlib

def check_huggingface_version():
    """Check the current huggingface-hub version."""
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"Current huggingface-hub version: {version}")
        return version
    except ImportError:
        print("huggingface-hub not installed")
        return None

def check_sentence_transformers_version():
    """Check the current sentence-transformers version."""
    try:
        import sentence_transformers
        version = sentence_transformers.__version__
        print(f"Current sentence-transformers version: {version}")
        return version
    except ImportError:
        print("sentence-transformers not installed")
        return None

def fix_huggingface_compatibility():
    """Fix huggingface-hub compatibility issues."""
    print("Fixing huggingface-hub compatibility issues...")
    
    huggingface_version = check_huggingface_version()
    sentence_transformers_version = check_sentence_transformers_version()
    
    if huggingface_version and huggingface_version.startswith('0.17'):
        print("\n⚠️  huggingface-hub 0.17+ detected - this causes compatibility issues with sentence-transformers 2.2.2")
        print("Attempting to fix...")
        
        try:
            # Install compatible huggingface-hub version
            print("\n1. Installing huggingface-hub 0.16.4...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub==0.16.4"], 
                         check=True, capture_output=True, text=True)
            print("✓ huggingface-hub 0.16.4 installed successfully")
            
            # Check if the issue is resolved
            try:
                from sentence_transformers import SentenceTransformer
                print("✓ sentence-transformers import test successful")
                return True
            except Exception as e:
                print(f"✗ sentence-transformers import still failing: {e}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install compatible huggingface-hub: {e}")
        
        print("\n❌ Automatic fix failed. Please try manual solutions:")
        print("1. pip install huggingface-hub==0.16.4")
        print("2. pip install -r requirements.txt")
        return False
        
    else:
        print("✓ huggingface-hub version is compatible")
        return True

def test_sentence_transformers_import():
    """Test if sentence-transformers can be imported successfully."""
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers import successful!")
        return True
    except Exception as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False

def main():
    """Main function."""
    print("Sentence Transformers HuggingFace Compatibility Fixer")
    print("=" * 60)
    
    # Check current state
    print("\nChecking current installation...")
    huggingface_version = check_huggingface_version()
    sentence_transformers_version = check_sentence_transformers_version()
    
    # Test sentence-transformers import
    print("\nTesting sentence-transformers import...")
    if test_sentence_transformers_import():
        print("\n🎉 No compatibility issues detected!")
        return True
    
    # Try to fix the issue
    print("\nCompatibility issues detected. Attempting to fix...")
    if fix_huggingface_compatibility():
        print("\n🎉 Compatibility issues resolved!")
        return True
    else:
        print("\n❌ Could not automatically resolve compatibility issues.")
        print("Please try the manual solutions mentioned above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
