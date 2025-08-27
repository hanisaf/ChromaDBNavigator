import os
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class PDFProcessor:
    """Handles PDF text extraction, chunking, and metadata generation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_apa_reference(self, filename: str) -> str:
        """Extract APA-style reference from filename (Authors year format)."""
        # Remove .pdf extension
        name = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Try to match "Authors year" pattern
        # This is a simple pattern - you might want to customize this
        match = re.match(r'^(.+?)\s+(\d{4})$', name)
        if match:
            authors = match.group(1).strip()
            year = match.group(2)
            return f"{authors} ({year})"
        
        # Fallback: just return the filename without extension
        return name
    
    def extract_text_from_pdf(self, filepath: str) -> Tuple[str, Dict]:
        """Extract text from PDF and return text content and metadata."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            # Basic metadata
            metadata = {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'total_pages': len(reader.pages),
                'extraction_date': datetime.now().isoformat(),
                'file_size': os.path.getsize(filepath)
            }
            
            return text, metadata
            
        except Exception as e:
            raise Exception(f"Error processing PDF {filepath}: {str(e)}")
    
    def chunk_text_semantically(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk text using semantic splitting with fallback to character-based splitting."""
        chunks = []
        
        try:
            # First try semantic splitting based on headers/structure
            if text.strip():
                # Try to identify sections by looking for common academic patterns
                semantic_chunks = self._split_by_academic_structure(text)
                
                if len(semantic_chunks) > 1:
                    # Use semantic chunks
                    for i, chunk in enumerate(semantic_chunks):
                        if len(chunk.strip()) > 50:  # Only keep substantial chunks
                            chunk_data = {
                                'text': chunk.strip(),
                                'chunk_index': i,
                                'chunk_type': 'semantic',
                                'chunk_size': len(chunk.strip()),
                                **metadata
                            }
                            chunks.append(chunk_data)
                else:
                    # Fallback to recursive character splitting
                    chunks = self._split_recursively(text, metadata)
            else:
                # Empty text - create single chunk
                chunks = [{
                    'text': text,
                    'chunk_index': 0,
                    'chunk_type': 'empty',
                    'chunk_size': len(text),
                    **metadata
                }]
                
        except Exception as e:
            # Final fallback to simple character-based splitting
            chunks = self._split_recursively(text, metadata)
        
        return chunks
    
    def _split_by_academic_structure(self, text: str) -> List[str]:
        """Split text by academic document structure."""
        # Common academic section headers
        section_patterns = [
            r'\n\s*(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References|Bibliography)\s*\n',
            r'\n\s*\d+\.\s+[A-Z][^.\n]*\n',  # Numbered sections
            r'\n\s*[A-Z][^.\n]*\n',  # Capitalized section headers
        ]
        
        # Try to find natural breaks
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            if matches:
                # Split at these points
                split_points = [0] + [m.start() for m in matches] + [len(text)]
                chunks = []
                for i in range(len(split_points) - 1):
                    chunk = text[split_points[i]:split_points[i+1]]
                    if chunk.strip():
                        chunks.append(chunk)
                if len(chunks) > 1:
                    return chunks
        
        # If no clear structure, return the whole text
        return [text]
    
    def _split_recursively(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text using recursive character splitting."""
        text_chunks = self._split_text_recursive(text)
        chunks = []
        
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():
                chunk_data = {
                    'text': chunk.strip(),
                    'chunk_index': i,
                    'chunk_type': 'recursive',
                    'chunk_size': len(chunk.strip()),
                    **metadata
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _split_text_recursive(self, text: str) -> List[str]:
        """Recursive text splitting implementation."""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to find good split points
        split_points = []
        
        # Look for paragraph breaks first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            current_chunk = ""
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        split_points.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
            if current_chunk:
                split_points.append(current_chunk.strip())
            
            if len(split_points) > 1:
                return split_points
        
        # Fallback to sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    split_points.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            split_points.append(current_chunk.strip())
        
        if len(split_points) > 1:
            return split_points
        
        # Final fallback: split at character limit
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def process_pdf(self, filepath: str) -> List[Dict]:
        """Main method to process a PDF file and return chunked data."""
        # Extract text and basic metadata
        text, metadata = self.extract_text_from_pdf(filepath)
        
        # Extract APA reference from filename
        apa_ref = self.extract_apa_reference(metadata['filename'])
        metadata['apa_reference'] = apa_ref
        
        # Chunk the text
        chunks = self.chunk_text_semantically(text, metadata)
        
        # Add page information to chunks
        for chunk in chunks:
            # Try to extract page number from chunk text
            page_match = re.search(r'--- Page (\d+) ---', chunk['text'])
            if page_match:
                chunk['page_number'] = int(page_match.group(1))
                # Clean up the page marker from the text
                chunk['text'] = re.sub(r'--- Page \d+ ---\s*', '', chunk['text'])
            else:
                chunk['page_number'] = 1  # Default to page 1 if can't determine
        
        return chunks
