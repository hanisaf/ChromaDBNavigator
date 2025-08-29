#!/usr/bin/env python3
"""
Script to check ChromaDB version and resolve compatibility issues
"""

import subprocess
import sys
import importlib

def check_chromadb_version():
    """Check the current ChromaDB version."""
    try:
        import chromadb
        version = chromadb.__version__
        print(f"Current ChromaDB version: {version}")
        return version
    except ImportError as e:
        print(f"ChromaDB not installed: {e}")
        return None

def check_chromadb_attributes():
    """Check what attributes are available in the installed ChromaDB."""
    try:
        import chromadb
        print("\nChecking ChromaDB attributes...")
        
        # Check for different client types
        if hasattr(chromadb, 'PersistentClient'):
            print("✅ PersistentClient available")
        else:
            print("❌ PersistentClient NOT available")
        
        if hasattr(chromadb, 'Client'):
            print("✅ Client available")
        else:
            print("❌ Client NOT available")
        
        if hasattr(chromadb, 'config'):
            print("✅ config module available")
        else:
            print("❌ config module NOT available")
        
        # Try to import specific components
        try:
            from chromadb.config import Settings
            print("✅ Settings class available")
        except ImportError:
            print("❌ Settings class NOT available")
        
        return True
        
    except Exception as e:
        print(f"Error checking ChromaDB attributes: {e}")
        return False

def fix_chromadb_version():
    """Fix ChromaDB version compatibility issues."""
    print("\nFixing ChromaDB version compatibility...")
    
    current_version = check_chromadb_version()
    if not current_version:
        print("ChromaDB not installed, installing compatible version...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb==0.3.25"], 
                         check=True, capture_output=True, text=True)
            print("✅ ChromaDB 0.3.25 installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install ChromaDB: {e}")
            return False
    
    # Check if we need to change versions
    if current_version.startswith('0.3.'):
        print("✅ ChromaDB version is compatible")
        return True
    elif current_version.startswith('0.4.'):
        print("⚠️  ChromaDB 0.4.x detected - this should work with our updated code")
        return True
    else:
        print(f"⚠️  Unexpected ChromaDB version: {current_version}")
        print("Attempting to install compatible version...")
        
        try:
            # Uninstall current version
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "chromadb", "-y"], 
                         check=True, capture_output=True, text=True)
            print("✅ Current ChromaDB uninstalled")
            
            # Install compatible version
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb==0.3.25"], 
                         check=True, capture_output=True, text=True)
            print("✅ ChromaDB 0.3.25 installed successfully")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to fix ChromaDB version: {e}")
            return False

def test_chromadb_import():
    """Test if ChromaDB can be imported and used successfully."""
    try:
        import chromadb
        print(f"\n🧪 Testing ChromaDB {chromadb.__version__}...")
        
        # Test basic functionality
        if hasattr(chromadb, 'PersistentClient'):
            print("✅ PersistentClient available")
            return True
        elif hasattr(chromadb, 'Client'):
            print("✅ Client available (newer version)")
            return True
        else:
            print("❌ No compatible client found")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False

def main():
    """Main function."""
    print("ChromaDB Version Checker and Fixer")
    print("=" * 50)
    
    # Check current state
    print("\nChecking current installation...")
    if not check_chromadb_attributes():
        print("❌ Could not check ChromaDB attributes")
        return False
    
    # Test basic import
    if not test_chromadb_import():
        print("\nCompatibility issues detected. Attempting to fix...")
        if not fix_chromadb_version():
            print("\n❌ Could not automatically resolve compatibility issues.")
            print("Please try manual solutions:")
            print("1. pip uninstall chromadb -y")
            print("2. pip install chromadb==0.3.25")
            return False
    
    # Final test
    print("\nFinal compatibility test...")
    if test_chromadb_import():
        print("\n🎉 ChromaDB compatibility issues resolved!")
        print("\nYou can now run the application with:")
        print("  python main.py")
        return True
    else:
        print("\n❌ Compatibility issues persist.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
