#!/usr/bin/env python3
"""
Test script to verify that database statistics work correctly with large numbers of files.
This script tests the fix for the unique files count plateauing at 53.
"""

import os
import tempfile
import shutil
from chroma_manager import ChromaManager

def create_mock_pdf_files(folder_path, num_files):
    """Create mock PDF files for testing."""
    print(f"Creating {num_files} mock PDF files...")
    
    # Simple PDF content (minimal valid PDF)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
  /Font <<
    /F1 <<
      /Type /Font
      /Subtype /Type1
      /BaseFont /Helvetica
    >>
  >>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000015 00000 n 
0000000074 00000 n 
0000000120 00000 n 
0000000428 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
522
%%EOF"""
    
    for i in range(num_files):
        filename = f"test_document_{i+1:04d}.pdf"
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'wb') as f:
            f.write(pdf_content)
        
        if (i + 1) % 100 == 0:
            print(f"Created {i + 1} files...")
    
    print(f"Created {num_files} mock PDF files successfully")

def test_large_file_count():
    """Test database statistics with a large number of files."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    db_dir = tempfile.mkdtemp()
    
    try:
        print("=" * 60)
        print("TESTING DATABASE STATISTICS WITH LARGE FILE COUNT")
        print("=" * 60)
        
        # Test with different file counts
        test_counts = [50, 100, 200]  # Start with smaller numbers for quick testing
        
        for file_count in test_counts:
            print(f"\nTesting with {file_count} files...")
            
            # Clean up previous files
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            
            # Create mock PDF files
            create_mock_pdf_files(temp_dir, file_count)
            
            # Initialize ChromaDB manager with unique collection for each test
            collection_name = f"test_collection_{file_count}"
            chroma_manager = ChromaManager(db_path=db_dir, collection_name=collection_name)
            
            # Sync files to database
            print(f"Syncing {file_count} files to database...")
            added, removed, corrupted = chroma_manager.sync_database(temp_dir)
            print(f"Added: {added}, Removed: {removed}, Corrupted: {len(corrupted)}")
            
            # Get statistics
            print("Retrieving database statistics...")
            stats = chroma_manager.get_database_stats()
            
            print("\nSTATISTICS RESULTS:")
            print(f"Total Chunks: {stats['total_chunks']:,}")
            print(f"Unique Files: {stats['unique_files']:,}")
            print(f"Expected Files: {file_count}")
            
            # Verify results
            if stats['unique_files'] == file_count:
                print("✅ SUCCESS: Unique files count matches expected count")
            else:
                print(f"❌ FAILURE: Expected {file_count} unique files, got {stats['unique_files']}")
            
            if stats['total_chunks'] >= file_count:
                print(f"✅ SUCCESS: Total chunks ({stats['total_chunks']}) >= file count ({file_count})")
            else:
                print(f"❌ FAILURE: Total chunks ({stats['total_chunks']}) < file count ({file_count})")
            
            print("-" * 40)
        
        print("\n" + "=" * 60)
        print("LARGE FILE COUNT TEST COMPLETED")
        print("=" * 60)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        shutil.rmtree(db_dir)
        print("Cleaned up temporary directories")

if __name__ == "__main__":
    test_large_file_count()