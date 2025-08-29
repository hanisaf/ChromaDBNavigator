#!/usr/bin/env python3
"""
Step-by-step dependency installation script for ChromaDB PDF Navigator
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a pip command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ is required")
        return False

def install_dependencies():
    """Install dependencies step by step."""
    print("🚀 ChromaDB PDF Navigator - Dependency Installer")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Step 1: Upgrade pip and setuptools
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Upgrading pip, setuptools, and wheel"
    ):
        return False
    
    # Step 2: Install Pydantic first (specific version to avoid conflicts)
    if not run_command(
        f"{sys.executable} -m pip install pydantic==1.10.13",
        "Installing Pydantic 1.10.13 (compatible version)"
    ):
        return False
    
    # Step 3: Install ChromaDB
    if not run_command(
        f"{sys.executable} -m pip install chromadb==0.3.25",
        "Installing ChromaDB 0.3.25"
    ):
        return False
    
    # Step 4: Install PyPDF
    if not run_command(
        f"{sys.executable} -m pip install pypdf==3.17.4",
        "Installing PyPDF for PDF processing"
    ):
        return False
    
    # Step 5: Install Sentence Transformers
    if not run_command(
        f"{sys.executable} -m pip install sentence-transformers==2.2.2",
        "Installing Sentence Transformers for embeddings"
    ):
        return False
    
    # Step 6: Install Python DateUtil
    if not run_command(
        f"{sys.executable} -m pip install python-dateutil==2.8.2",
        "Installing Python DateUtil"
    ):
        return False
    
    # Step 7: Install PyQt5 (optional, user can install manually if needed)
    print("\n🔧 Installing PyQt5 (GUI framework)")
    print("Note: PyQt5 installation may take a while...")
    
    if not run_command(
        f"{sys.executable} -m pip install PyQt5==5.15.9",
        "Installing PyQt5 for the GUI"
    ):
        print("⚠️  PyQt5 installation failed. You may need to install it manually:")
        print("   - macOS: brew install pyqt5")
        print("   - Ubuntu: sudo apt-get install python3-pyqt5")
        print("   - Windows: pip install PyQt5-Qt5")
    
    return True

def test_imports():
    """Test if all packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    packages = [
        ("chromadb", "ChromaDB"),
        ("pypdf", "PyPDF"),
        ("sentence_transformers", "Sentence Transformers"),
        ("python_dateutil", "Python DateUtil"),
        ("pydantic", "Pydantic")
    ]
    
    all_success = True
    
    for import_name, display_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            all_success = False
    
    return all_success

def main():
    """Main installation process."""
    print("Starting dependency installation...")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some packages failed to import. The application may not work correctly.")
        return False
    
    print("\n🎉 All dependencies installed successfully!")
    print("\nYou can now run the application with:")
    print("  python main.py")
    print("  or")
    print("  python run.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
