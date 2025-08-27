#!/usr/bin/env python3
"""
Test script to verify ChromaDB PDF Navigator installation
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import PyQt5
        print("✓ PyQt5 imported successfully")
    except ImportError as e:
        print(f"✗ PyQt5 import failed: {e}")
        return False
    
    try:
        import chromadb
        print(f"✓ ChromaDB {chromadb.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False
    
    try:
        import pypdf
        print("✓ PyPDF imported successfully")
    except ImportError as e:
        print(f"✗ PyPDF import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Sentence Transformers import failed: {e}")
        return False
    
    return True

def test_local_modules():
    """Test if local application modules can be imported."""
    print("\nTesting local module imports...")
    
    try:
        from pdf_processor import PDFProcessor
        print("✓ PDFProcessor imported successfully")
    except ImportError as e:
        print(f"✗ PDFProcessor import failed: {e}")
        return False
    
    try:
        from chroma_manager import ChromaManager
        print("✓ ChromaManager imported successfully")
    except ImportError as e:
        print(f"✗ ChromaManager import failed: {e}")
        return False
    
    return True

def test_pdf_processor():
    """Test PDF processor functionality."""
    print("\nTesting PDF processor...")
    
    try:
        processor = PDFProcessor()
        print("✓ PDFProcessor initialized successfully")
        
        # Test APA reference extraction
        test_filename = "Smith Johnson 2023.pdf"
        apa_ref = processor.extract_apa_reference(test_filename)
        expected = "Smith Johnson (2023)"
        
        if apa_ref == expected:
            print("✓ APA reference extraction working correctly")
        else:
            print(f"⚠ APA reference extraction: expected '{expected}', got '{apa_ref}'")
        
        return True
        
    except Exception as e:
        print(f"✗ PDFProcessor test failed: {e}")
        return False

def test_chroma_manager():
    """Test ChromaDB manager functionality."""
    print("\nTesting ChromaDB manager...")
    
    try:
        # Test with temporary database path
        test_db_path = "./test_chroma_db"
        manager = ChromaManager(test_db_path)
        print("✓ ChromaManager initialized successfully")
        
        # Test basic operations
        stats = manager.get_database_stats()
        print(f"✓ Database stats retrieved: {stats['total_chunks']} chunks")
        
        # Clean up test database
        import shutil
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
            print("✓ Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ ChromaManager test failed: {e}")
        return False

def test_qt_components():
    """Test PyQt5 components."""
    print("\nTesting PyQt5 components...")
    
    try:
        from PyQt5.QtWidgets import QApplication, QLabel
        from PyQt5.QtCore import Qt
        
        # Create minimal app for testing
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        label = QLabel("Test Label")
        label.setAlignment(Qt.AlignCenter)
        
        print("✓ PyQt5 components working correctly")
        return True
        
    except Exception as e:
        print(f"✗ PyQt5 test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ChromaDB PDF Navigator - Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test local modules
    if not test_local_modules():
        all_tests_passed = False
    
    # Test PDF processor
    if not test_pdf_processor():
        all_tests_passed = False
    
    # Test ChromaDB manager
    if not test_chroma_manager():
        all_tests_passed = False
    
    # Test PyQt5 components
    if not test_qt_components():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Installation is successful.")
        print("\nYou can now run the application with:")
        print("  python main.py")
        print("  or")
        print("  python run.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify file permissions")
        print("4. Check available disk space")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
