import os
import hashlib
from typing import List, Dict, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import torch


class ChromaManager:
    """Manages ChromaDB operations for PDF chunks."""
    
    @staticmethod
    def _get_optimal_device():
        """Detect and return the optimal device for embeddings (Metal/MPS, CUDA, or CPU)."""
        try:
            # Check for CUDA first (generally fastest for large workloads)
            if torch.cuda.is_available():
                print(f"CUDA available - using GPU: {torch.cuda.get_device_name()}")
                return "cuda"
            # Check for Metal Performance Shaders (macOS)
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("Metal Performance Shaders (MPS) available - using Metal acceleration")
                print("Note: Metal acceleration works best with larger document collections")
                return "mps"
            else:
                print("Using CPU for embeddings")
                return "cpu"
        except Exception as e:
            print(f"Error detecting optimal device, falling back to CPU: {e}")
            return "cpu"
    
    def __init__(self, db_path: str = None, collection_name: str = "references"):
        if db_path is None:
            db_path = os.path.expanduser("~/.chromadb/db")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Check ChromaDB version and use appropriate client initialization
            chroma_version = chromadb.__version__
            print(f"ChromaDB version detected: {chroma_version}")
            
            # Initialize ChromaDB client - use PersistentClient for ChromaDB 1.0.x
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"PDF document chunks with metadata - {self.collection_name}"}
            )
            
            # Initialize embedding model with optimal device
            self.device = self._get_optimal_device()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            # For Metal/MPS, we may need to set additional optimizations
            if self.device == "mps":
                # Enable Metal optimizations if available
                try:
                    # Set model to use Metal backend
                    self.embedding_model = self.embedding_model.to(self.device)
                    print("Successfully moved embedding model to Metal device")
                except Exception as e:
                    print(f"Warning: Could not move model to Metal device: {e}")
                    # Fallback to CPU
                    self.device = "cpu"
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def get_device_info(self) -> Dict[str, str]:
        """Get information about the device being used for embeddings."""
        device_info = {
            'device': getattr(self, 'device', 'unknown'),
            'device_name': 'Unknown'
        }
        
        try:
            if hasattr(self, 'device'):
                device_info['device'] = self.device
                if self.device == 'mps':
                    device_info['device_name'] = 'Apple Metal Performance Shaders (GPU acceleration)'
                    device_info['note'] = 'Best performance with large document collections'
                elif self.device == 'cuda':
                    device_info['device_name'] = f'NVIDIA GPU: {torch.cuda.get_device_name()}'
                    device_info['note'] = 'Optimized for large-scale processing'
                else:
                    device_info['device_name'] = 'CPU'
                    device_info['note'] = 'Reliable performance for all workloads'
        except Exception as e:
            device_info['error'] = str(e)
            
        return device_info
    
    def _generate_file_hash(self, filepath: str) -> str:
        """Generate a hash for a file to detect changes."""
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
                return file_hash.hexdigest()
        except Exception:
            return ""
    
    def _get_existing_files(self) -> Dict[str, str]:
        """Get existing files in the database with their hashes."""
        try:
            # Get all documents and extract filename -> hash mapping
            # For ChromaDB 0.3.25, we need to use get() with proper parameters
            results = self.collection.get(
                include=['metadatas']
                # No limit - get all documents
            )
            
            existing_files = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'filename' in metadata:
                        filename = metadata['filename']
                        filepath = metadata.get('filepath', '')
                        if filepath and os.path.exists(filepath):
                            existing_files[filename] = self._generate_file_hash(filepath)
            
            return existing_files
            
        except Exception as e:
            print(f"Warning: Could not retrieve existing files: {e}")
            return {}
    
    def preview_sync_changes(self, folder_path: str) -> Dict:
        """
        Preview what changes would be made during sync without actually making them.
        Returns: dict with new_files, removed_files, modified_files, and corrupted_files
        """
        if not os.path.exists(folder_path):
            raise Exception(f"Folder does not exist: {folder_path}")
        
        # Get existing files in database
        existing_files = self._get_existing_files()
        
        # Get current files in folder
        current_files = {}
        corrupted_files = []
        
        for filename in sorted(os.listdir(folder_path)):  # Sort alphabetically
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(folder_path, filename)
                try:
                    current_files[filename] = self._generate_file_hash(filepath)
                except Exception:
                    corrupted_files.append(filename)
        
        # Find new, removed, and modified files
        new_files = [f for f in current_files if f not in existing_files]
        removed_files = [f for f in existing_files if f not in current_files]
        modified_files = [f for f in current_files if f in existing_files and current_files[f] != existing_files[f]]
        
        return {
            'new_files': new_files,
            'removed_files': removed_files,
            'modified_files': modified_files,
            'corrupted_files': corrupted_files
        }

    def sync_database(self, folder_path: str, progress_callback=None) -> Tuple[int, int, List[str]]:
        """
        Sync the database with the PDF folder.
        Returns: (added_count, removed_count, corrupted_files)
        """
        if not os.path.exists(folder_path):
            raise Exception(f"Folder does not exist: {folder_path}")
        
        # Initial progress
        if progress_callback:
            progress_callback(0, "Analyzing folder changes...")
        
        # Get sync changes preview
        changes = self.preview_sync_changes(folder_path)
        new_files = changes['new_files']
        removed_files = changes['removed_files']
        modified_files = changes['modified_files']
        corrupted_files = changes['corrupted_files'].copy()
        
        # Calculate total operations for progress tracking
        total_operations = len(removed_files) + len(modified_files) + len(new_files) + len(modified_files)
        current_operation = 0
        
        def update_progress(message, increment=True):
            nonlocal current_operation
            if increment:
                current_operation += 1
            if progress_callback and total_operations > 0:
                progress = (current_operation / total_operations) * 100
                progress_callback(min(progress, 100), message)
        
        # Remove deleted files from database
        if removed_files:
            update_progress("Removing deleted files from database...", False)
            for filename in sorted(removed_files):  # Sort alphabetically
                update_progress(f"Removing: {filename}")
        
        removed_count = self._remove_files_from_db(removed_files)
        
        # Remove modified files (they'll be re-added)
        if modified_files:
            update_progress("Removing modified files for re-indexing...", False)
            for filename in sorted(modified_files):  # Sort alphabetically
                update_progress(f"Preparing to re-index: {filename}")
        
        if modified_files:
            self._remove_files_from_db(modified_files)
        
        # Add new files and re-add modified files to database
        added_count = 0
        files_to_process = sorted(new_files + modified_files)  # Sort alphabetically
        
        if files_to_process:
            for filename in files_to_process:
                try:
                    filepath = os.path.join(folder_path, filename)
                    
                    if filename in modified_files:
                        update_progress(f"Re-indexing modified file: {filename}")
                    else:
                        update_progress(f"Processing new file: {filename}")
                    
                    self._add_file_to_db(filepath)
                    added_count += 1
                        
                except Exception as e:
                    corrupted_files.append(f"{filename} (Error: {str(e)})")
        
        # Final progress update
        if progress_callback:
            progress_callback(100, "Synchronization complete")
        
        return added_count, removed_count, corrupted_files
    
    def _remove_files_from_db(self, filenames: List[str]) -> int:
        """Remove files from the database."""
        if not filenames:
            return 0
        
        try:
            # Get all documents with matching filenames
            # For ChromaDB 0.3.25, we need to use where clause properly
            results = self.collection.get(
                where={"filename": {"$in": filenames}},
                include=['ids']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            print(f"Warning: Could not remove files from database: {e}")
            return 0
    
    def _add_file_to_db(self, filepath: str):
        """Add a single file to the database."""
        from pdf_processor import PDFProcessor
        
        # Process the PDF
        processor = PDFProcessor()
        chunks = processor.process_pdf(filepath)
        
        if not chunks:
            raise Exception("No chunks generated from PDF")
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID for this chunk
            chunk_id = f"{chunk['filename']}_{chunk['chunk_index']}_{chunk['page_number']}"
            ids.append(chunk_id)
            texts.append(chunk['text'])
            
            # Prepare metadata
            metadata = {
                'filename': chunk['filename'],
                'filepath': chunk['filepath'],
                'page_number': chunk['page_number'],
                'chunk_index': chunk['chunk_index'],
                'chunk_type': chunk['chunk_type'],
                'chunk_size': chunk['chunk_size'],
                'total_pages': chunk['total_pages'],
                'extraction_date': chunk['extraction_date'],
                'file_size': chunk['file_size'],
                'apa_reference': chunk['apa_reference']
            }
            metadatas.append(metadata)
        
        # Add to database - ChromaDB 0.3.25 syntax
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def search_database(self, query: str, n_results: int = 10, 
                       filters: Optional[Dict] = None) -> List[Dict]:
        """Search the database for relevant chunks."""
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if value:
                        where_clause[key] = value
            
            # Perform search - ChromaDB 0.3.25 syntax
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['metadatas']:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        try:
            count = self.collection.count()
            
            # Get all documents for metadata analysis (no limit)
            results = self.collection.get(include=['metadatas'])
            
            stats = {
                'total_chunks': count,
                'unique_files': 0,
                'total_pages': 0,
                'chunk_types': {},
                'file_extensions': {},
                'device_info': self.get_device_info()
            }
            
            if results['metadatas']:
                unique_files = set()
                chunk_types = {}
                file_extensions = {}
                
                for metadata in results['metadatas']:
                    if metadata:
                        if 'filename' in metadata:
                            unique_files.add(metadata['filename'])
                            
                            # Count chunk types
                            chunk_type = metadata.get('chunk_type', 'unknown')
                            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                            
                            # Count file extensions
                            ext = os.path.splitext(metadata['filename'])[1].lower()
                            file_extensions[ext] = file_extensions.get(ext, 0) + 1
                
                stats['unique_files'] = len(unique_files)
                stats['chunk_types'] = chunk_types
                stats['file_extensions'] = file_extensions
            
            return stats
            
        except Exception as e:
            print(f"Could not get database stats: {e}")
            return {'total_chunks': 0, 'error': str(e)}
    
    def get_all_documents(self, limit: int = None) -> List[Dict]:
        """Get all documents from the database."""
        try:
            if limit is not None:
                results = self.collection.get(
                    limit=limit,
                    include=['documents', 'metadatas']
                )
            else:
                results = self.collection.get(
                    include=['documents', 'metadatas']
                )
            
            documents = []
            if results['documents'] and results['metadatas']:
                for i in range(len(results['documents'])):
                    doc = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Could not retrieve documents: {e}")
            return []
