#!/usr/bin/env python3
"""
ChromaDB PDF Navigator - Main Application
A desktop app for syncing PDF files with a ChromaDB vector database.
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
                             QTextEdit, QPushButton, QLabel, QProgressBar, 
                             QFileDialog, QMessageBox, QTabWidget, QLineEdit,
                             QComboBox, QSpinBox, QGroupBox, QScrollArea,
                             QFrame, QGridLayout, QTextBrowser, QListWidget,
                             QListWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap

from chroma_manager import ChromaManager
from pdf_processor import PDFProcessor
from config import COLLECTION_NAME


class SyncWorker(QThread):
    """Worker thread for database synchronization."""
    progress_updated = pyqtSignal(int, str)
    sync_completed = pyqtSignal(int, int, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, chroma_manager, folder_path):
        super().__init__()
        self.chroma_manager = chroma_manager
        self.folder_path = folder_path
    
    def run(self):
        try:
            def progress_callback(progress, message):
                self.progress_updated.emit(progress, message)
            
            added, removed, corrupted = self.chroma_manager.sync_database(
                self.folder_path, progress_callback
            )
            self.sync_completed.emit(added, removed, corrupted)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, db_path=None, pdf_folder=None, collection_name=None):
        super().__init__()
        self.chroma_manager = None
        self.current_folder = pdf_folder or ""
        self.current_db_path = db_path or ""
        self.current_collection = collection_name or COLLECTION_NAME
        self.init_ui()
        self.init_chroma()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("ChromaDB PDF Navigator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        self.create_top_controls(main_layout)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - File browser
        self.create_file_browser(main_splitter)
        
        # Right panel - Database content
        self.create_database_panel(main_splitter)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Apply styling
        self.apply_styling()
        
        # Ensure text visibility
        self.ensure_text_visibility()
    
    def ensure_text_visibility(self):
        """Ensure all text widgets have visible text colors."""
        # Set text colors for all text widgets
        text_widgets = [
            self.file_tree,
            self.file_info_text,
            self.search_input,
            self.search_results,
            self.result_details,
            self.document_list,
            self.document_details,
            self.stats_text
        ]
        
        for widget in text_widgets:
            if hasattr(widget, 'setStyleSheet'):
                current_style = widget.styleSheet()
                if 'color:' not in current_style:
                    widget.setStyleSheet(current_style + " color: #333;")
    
    def create_top_controls(self, parent_layout):
        """Create the top control panel."""
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        
        # First row - Database settings
        db_row = QHBoxLayout()
        
        db_label = QLabel("Database Path:")
        self.db_path_label = QLabel("No database path selected")
        self.db_path_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; background: #f9f9f9;")
        self.db_path_label.setMinimumWidth(300)
        
        db_browse_btn = QPushButton("Browse DB...")
        db_browse_btn.clicked.connect(self.browse_database_path)
        
        collection_label = QLabel("Collection:")
        self.collection_combo = QComboBox()
        self.collection_combo.setEditable(True)
        self.collection_combo.addItems(["references", "papers", "documents", "research"])
        self.collection_combo.setCurrentText(self.current_collection)
        self.collection_combo.currentTextChanged.connect(self.on_collection_changed)
        
        db_row.addWidget(db_label)
        db_row.addWidget(self.db_path_label)
        db_row.addWidget(db_browse_btn)
        db_row.addWidget(collection_label)
        db_row.addWidget(self.collection_combo)
        db_row.addStretch()
        
        controls_layout.addLayout(db_row)
        
        # Second row - PDF folder and sync
        pdf_row = QHBoxLayout()
        
        folder_label = QLabel("PDF Folder:")
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; background: #f9f9f9;")
        self.folder_path_label.setMinimumWidth(300)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        
        # Sync controls
        self.sync_btn = QPushButton("Sync Database")
        self.sync_btn.clicked.connect(self.start_sync)
        self.sync_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        pdf_row.addWidget(folder_label)
        pdf_row.addWidget(self.folder_path_label)
        pdf_row.addWidget(browse_btn)
        pdf_row.addStretch()
        pdf_row.addWidget(self.sync_btn)
        pdf_row.addWidget(self.progress_bar)
        
        controls_layout.addLayout(pdf_row)
        
        parent_layout.addWidget(controls_frame)
        
        # Set initial values if provided
        if self.current_db_path:
            self.db_path_label.setText(self.current_db_path)
        if self.current_folder:
            self.folder_path_label.setText(self.current_folder)
            self.sync_btn.setEnabled(True)
            self.load_file_tree()
    
    def create_file_browser(self, parent):
        """Create the left panel file browser."""
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.StyledPanel)
        file_layout = QVBoxLayout(file_frame)
        
        # File browser header
        file_header = QLabel("PDF Files")
        file_header.setFont(QFont("Arial", 12, QFont.Bold))
        file_layout.addWidget(file_header)
        
        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("Files")
        self.file_tree.itemClicked.connect(self.on_file_selected)
        file_layout.addWidget(self.file_tree)
        
        # File info
        file_info_group = QGroupBox("File Information")
        file_info_layout = QVBoxLayout(file_info_group)
        
        self.file_info_text = QTextBrowser()
        self.file_info_text.setMaximumHeight(150)
        file_info_layout.addWidget(self.file_info_text)
        
        file_layout.addWidget(file_info_group)
        
        parent.addWidget(file_frame)
    
    def create_database_panel(self, parent):
        """Create the right panel for database content."""
        db_frame = QFrame()
        db_frame.setFrameStyle(QFrame.StyledPanel)
        db_layout = QVBoxLayout(db_frame)
        
        # Database header
        db_header = QLabel("Database Content")
        db_header.setFont(QFont("Arial", 12, QFont.Bold))
        db_layout.addWidget(db_header)
        
        # Tab widget for different views
        self.db_tabs = QTabWidget()
        
        # Search tab
        self.create_search_tab()
        
        # Browse tab
        self.create_browse_tab()
        
        # Stats tab
        self.create_stats_tab()
        
        db_layout.addWidget(self.db_tabs)
        parent.addWidget(db_frame)
    
    def create_search_tab(self):
        """Create the search tab."""
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)
        
        # Search controls
        search_controls = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_input.returnPressed.connect(self.perform_search)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.perform_search)
        
        self.results_count_spin = QSpinBox()
        self.results_count_spin.setRange(5, 50)
        self.results_count_spin.setValue(10)
        self.results_count_spin.setSuffix(" results")
        
        search_controls.addWidget(QLabel("Query:"))
        search_controls.addWidget(self.search_input)
        search_controls.addWidget(self.search_btn)
        search_controls.addWidget(self.results_count_spin)
        search_controls.addStretch()
        
        search_layout.addLayout(search_controls)
        
        # Search results
        self.search_results = QListWidget()
        self.search_results.itemClicked.connect(self.on_search_result_selected)
        search_layout.addWidget(self.search_results)
        
        # Result details
        result_details_group = QGroupBox("Result Details")
        result_details_layout = QVBoxLayout(result_details_group)
        
        self.result_details = QTextBrowser()
        result_details_layout.addWidget(self.result_details)
        
        search_layout.addWidget(result_details_group)
        
        self.db_tabs.addTab(search_widget, "Search")
    
    def create_browse_tab(self):
        """Create the browse tab."""
        browse_widget = QWidget()
        browse_layout = QVBoxLayout(browse_widget)
        
        # Browse controls
        browse_controls = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_database_view)
        
        browse_controls.addWidget(self.refresh_btn)
        browse_controls.addStretch()
        
        browse_layout.addLayout(browse_controls)
        
        # Document list
        self.document_list = QListWidget()
        self.document_list.itemClicked.connect(self.on_document_selected)
        browse_layout.addWidget(self.document_list)
        
        # Document details
        doc_details_group = QGroupBox("Document Details")
        doc_details_layout = QVBoxLayout(doc_details_group)
        
        self.document_details = QTextBrowser()
        doc_details_layout.addWidget(self.document_details)
        
        browse_layout.addWidget(doc_details_group)
        
        self.db_tabs.addTab(browse_widget, "Browse")
    
    def create_stats_tab(self):
        """Create the statistics tab."""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # Stats display
        self.stats_text = QTextBrowser()
        stats_layout.addWidget(self.stats_text)
        
        # Refresh button
        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self.update_statistics)
        stats_layout.addWidget(refresh_stats_btn)
        
        self.db_tabs.addTab(stats_widget, "Statistics")
    
    def apply_styling(self):
        """Apply custom styling to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QTreeWidget, QListWidget {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTextBrowser {
                border: 1px solid #ddd;
                background-color: white;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def init_chroma(self):
        """Initialize ChromaDB manager."""
        try:
            self.chroma_manager = ChromaManager(
                db_path=self.current_db_path,
                collection_name=self.current_collection
            )
            self.statusBar().showMessage("ChromaDB initialized successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize ChromaDB: {str(e)}")
            self.statusBar().showMessage("ChromaDB initialization failed")
    
    def browse_folder(self):
        """Browse for PDF folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select PDF Folder", "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            self.current_folder = folder
            self.folder_path_label.setText(folder)
            self.sync_btn.setEnabled(True)
            self.load_file_tree()
            self.statusBar().showMessage(f"Selected folder: {folder}")
    
    def browse_database_path(self):
        """Browse for the ChromaDB database path."""
        db_path = QFileDialog.getExistingDirectory(
            self, "Select ChromaDB Database Path", "",
            QFileDialog.ShowDirsOnly
        )
        if db_path:
            self.current_db_path = db_path
            self.db_path_label.setText(db_path)
            self.init_chroma() # Re-initialize chroma_manager with new path
            self.statusBar().showMessage(f"Selected database path: {db_path}")
    
    def on_collection_changed(self, text):
        """Handle collection name change."""
        self.current_collection = text
        self.init_chroma() # Re-initialize chroma_manager with new collection
        self.statusBar().showMessage(f"Collection changed to: {text}")
    
    def load_file_tree(self):
        """Load PDF files into the file tree."""
        if not self.current_folder:
            return
        
        self.file_tree.clear()
        
        try:
            pdf_files = [f for f in os.listdir(self.current_folder) 
                        if f.lower().endswith('.pdf')]
            
            for pdf_file in sorted(pdf_files):
                item = QTreeWidgetItem([pdf_file])
                self.file_tree.addTopLevelItem(item)
            
            self.statusBar().showMessage(f"Loaded {len(pdf_files)} PDF files")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load files: {str(e)}")
    
    def on_file_selected(self, item, column):
        """Handle file selection in the tree."""
        filename = item.text(0)
        filepath = os.path.join(self.current_folder, filename)
        
        # Display file information
        try:
            file_size = os.path.getsize(filepath)
            file_info = f"Filename: {filename}\n"
            file_info += f"Path: {filepath}\n"
            file_info += f"Size: {file_size:,} bytes\n"
            
            # Try to get basic PDF info
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                file_info += f"Pages: {len(reader.pages)}\n"
            except:
                file_info += "Pages: Could not read\n"
            
            self.file_info_text.setText(file_info)
            
        except Exception as e:
            self.file_info_text.setText(f"Error reading file: {str(e)}")
    
    def start_sync(self):
        """Start database synchronization."""
        if not self.current_folder or not self.chroma_manager:
            return
        
        # Disable sync button and show progress
        self.sync_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.sync_worker = SyncWorker(self.chroma_manager, self.current_folder)
        self.sync_worker.progress_updated.connect(self.update_sync_progress)
        self.sync_worker.sync_completed.connect(self.sync_completed)
        self.sync_worker.error_occurred.connect(self.sync_error)
        self.sync_worker.start()
    
    def update_sync_progress(self, progress, message):
        """Update sync progress bar."""
        self.progress_bar.setValue(int(progress))
        self.statusBar().showMessage(message)
    
    def sync_completed(self, added, removed, corrupted):
        """Handle sync completion."""
        self.sync_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show results
        message = f"Sync completed!\n"
        message += f"Added: {added} files\n"
        message += f"Removed: {removed} files\n"
        message += f"Corrupted: {len(corrupted)} files"
        
        if corrupted:
            message += f"\n\nCorrupted files:\n" + "\n".join(corrupted)
        
        QMessageBox.information(self, "Sync Complete", message)
        self.statusBar().showMessage("Sync completed successfully")
        
        # Refresh views
        self.refresh_database_view()
        self.update_statistics()
    
    def sync_error(self, error_message):
        """Handle sync error."""
        self.sync_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Sync Error", f"Sync failed: {error_message}")
        self.statusBar().showMessage("Sync failed")
    
    def perform_search(self):
        """Perform database search."""
        if not self.chroma_manager:
            return
        
        query = self.search_input.text().strip()
        if not query:
            return
        
        n_results = self.results_count_spin.value()
        
        try:
            results = self.chroma_manager.search_database(query, n_results)
            self.display_search_results(results)
            
        except Exception as e:
            QMessageBox.warning(self, "Search Error", f"Search failed: {str(e)}")
    
    def display_search_results(self, results):
        """Display search results."""
        self.search_results.clear()
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page_number', '?')
            chunk_type = metadata.get('chunk_type', 'unknown')
            
            item_text = f"{filename} (Page {page}, {chunk_type})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, result)
            self.search_results.addItem(item)
        
        self.statusBar().showMessage(f"Found {len(results)} results")
    
    def on_search_result_selected(self, item):
        """Handle search result selection."""
        result = item.data(Qt.UserRole)
        self.display_result_details(result)
    
    def display_result_details(self, result):
        """Display detailed information about a search result."""
        metadata = result['metadata']
        text = result['text']
        
        details = f"Filename: {metadata.get('filename', 'Unknown')}\n"
        details += f"Page: {metadata.get('page_number', '?')}\n"
        details += f"Chunk Type: {metadata.get('chunk_type', 'unknown')}\n"
        details += f"APA Reference: {metadata.get('apa_reference', 'N/A')}\n"
        details += f"Chunk Size: {metadata.get('chunk_size', '?')} characters\n"
        details += f"Extraction Date: {metadata.get('extraction_date', 'N/A')}\n"
        details += f"\n--- Content ---\n{text}"
        
        self.result_details.setText(details)
    
    def refresh_database_view(self):
        """Refresh the database browse view."""
        if not self.chroma_manager:
            return
        
        try:
            documents = self.chroma_manager.get_all_documents(1000)
            self.display_documents(documents)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not refresh database view: {str(e)}")
    
    def display_documents(self, documents):
        """Display documents in the browse view."""
        self.document_list.clear()
        
        for doc in documents:
            metadata = doc['metadata']
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page_number', '?')
            chunk_type = metadata.get('chunk_type', 'unknown')
            
            item_text = f"{filename} (Page {page}, {chunk_type})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, doc)
            self.document_list.addItem(item)
    
    def on_document_selected(self, item):
        """Handle document selection in browse view."""
        doc = item.data(Qt.UserRole)
        self.display_document_details(doc)
    
    def display_document_details(self, doc):
        """Display detailed information about a document."""
        metadata = doc['metadata']
        text = doc['text']
        
        details = f"Filename: {metadata.get('filename', 'Unknown')}\n"
        details += f"Page: {metadata.get('page_number', '?')}\n"
        details += f"Chunk Type: {metadata.get('chunk_type', 'unknown')}\n"
        details += f"APA Reference: {metadata.get('apa_reference', 'N/A')}\n"
        details += f"Chunk Size: {metadata.get('chunk_size', '?')} characters\n"
        details += f"Total Pages: {metadata.get('total_pages', '?')}\n"
        details += f"File Size: {metadata.get('file_size', '?')} bytes\n"
        details += f"Extraction Date: {metadata.get('extraction_date', 'N/A')}\n"
        details += f"\n--- Content ---\n{text}"
        
        self.document_details.setText(details)
    
    def update_statistics(self):
        """Update and display database statistics."""
        if not self.chroma_manager:
            return
        
        try:
            stats = self.chroma_manager.get_database_stats()
            self.display_statistics(stats)
            
        except Exception as e:
            self.stats_text.setText(f"Error loading statistics: {str(e)}")
    
    def display_statistics(self, stats):
        """Display database statistics."""
        if 'error' in stats:
            self.stats_text.setText(f"Error: {stats['error']}")
            return
        
        stats_text = f"Database Statistics\n"
        stats_text += f"==================\n\n"
        stats_text += f"Total Chunks: {stats['total_chunks']:,}\n"
        stats_text += f"Unique Files: {stats['unique_files']:,}\n\n"
        
        if stats['chunk_types']:
            stats_text += f"Chunk Types:\n"
            for chunk_type, count in stats['chunk_types'].items():
                stats_text += f"  {chunk_type}: {count:,}\n"
            stats_text += "\n"
        
        if stats['file_extensions']:
            stats_text += f"File Extensions:\n"
            for ext, count in stats['file_extensions'].items():
                stats_text += f"  {ext}: {count:,}\n"
        
        self.stats_text.setText(stats_text)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="ChromaDB PDF Navigator")
    parser.add_argument("--db-path", help="Path to the ChromaDB database")
    parser.add_argument("--pdf-folder", help="Path to the PDF folder to sync")
    parser.add_argument("--collection", help="Name of the collection to use", default=COLLECTION_NAME)
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ChromaDB PDF Navigator")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = MainWindow(
        db_path=args.db_path,
        pdf_folder=args.pdf_folder,
        collection_name=args.collection
    )
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
