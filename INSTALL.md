# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 2GB of free disk space for the database

## Step-by-Step Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd ChromaDBNavigator
```

Or download and extract the ZIP file to your desired location.

### 2. Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

The requirements have been simplified to avoid dependency conflicts:

```bash
pip install -r requirements.txt
```

**Note**: The first time you run this, it will download several large models:
- Sentence transformers model (~90MB)
- ChromaDB dependencies

### 4. Verify Installation

```bash
python -c "import PyQt5, chromadb, pypdf; print('All packages installed successfully!')"
```

## Common Compatibility Issues

### Pydantic Compatibility Error

If you encounter this error:
```
pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package
```

This means you have Pydantic 2.x installed, which is incompatible with ChromaDB 0.3.25.

**Solutions:**

#### Option 1: Automatic Fix (Recommended)
```bash
python fix_pydantic_compatibility.py
```

#### Option 2: Manual Fix
```bash
# Install compatible Pydantic version
pip install "pydantic<2.0.0"

# Install pydantic-settings
pip install pydantic-settings

# Then install the rest
pip install -r requirements.txt
```

#### Option 3: Clean Install
```bash
# Remove existing packages
pip uninstall pydantic chromadb -y

# Install with correct versions
pip install -r requirements.txt
```

### HuggingFace Hub Compatibility Error

If you encounter this error:
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

This means you have an incompatible version of huggingface-hub installed.

**Solutions:**

#### Option 1: Automatic Fix (Recommended)
```bash
python fix_huggingface_compatibility.py
```

#### Option 2: Manual Fix
```bash
# Install compatible huggingface-hub version
pip install huggingface-hub==0.16.4

# Then install the rest
pip install -r requirements.txt
```

#### Option 3: Clean Install
```bash
# Remove existing packages
pip uninstall huggingface-hub sentence-transformers -y

# Install with correct versions
pip install -r requirements.txt
```

## Running the Application

### Option 1: Direct Python Execution
```bash
python main.py
```

### Option 2: Using the Launcher Script
```bash
python run.py
```

### Option 3: Make Executable (macOS/Linux)
```bash
chmod +x run.py
./run.py
```

## First Run Setup

1. **Launch the application**
2. **Select your PDF folder** using the "Browse..." button
3. **Click "Sync Database"** to index all PDFs
4. **Wait for processing** - this may take several minutes depending on the number and size of PDFs

## Troubleshooting

### Common Issues

#### 1. PyQt5 Installation Problems
```bash
# On macOS with Homebrew
brew install pyqt5

# On Ubuntu/Debian
sudo apt-get install python3-pyqt5

# On Windows, try
pip install PyQt5-Qt5
```

#### 2. ChromaDB Initialization Errors
- Ensure you have write permissions in the application directory
- Check that you have at least 1GB of free disk space
- Try deleting the `chroma_db` folder and restarting
- **Note**: ChromaDB 0.3.25 is compatible with both NumPy 1.x and 2.x
- **Pydantic**: Ensure you have Pydantic < 2.0.0 installed

#### 3. PDF Processing Errors
- Verify PDF files are not corrupted
- Check file permissions
- Ensure PDFs are not password-protected

#### 4. Memory Issues
- Close other applications
- Process PDFs in smaller batches
- Increase system swap/virtual memory

### Performance Tips

1. **Batch Processing**: Process PDFs in smaller folders first
2. **Regular Syncs**: Run sync regularly to maintain database consistency
3. **Monitor Resources**: Watch CPU and memory usage during large syncs

## System Requirements

### Minimum
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: Dual-core 2.0GHz

### Recommended
- **RAM**: 8GB or more
- **Storage**: 10GB free space
- **CPU**: Quad-core 3.0GHz or better

## File Naming Convention

For optimal APA reference extraction, name your PDFs as:
```
Authors Year.pdf
```

Examples:
- `Smith Johnson 2023.pdf`
- `Brown et al 2022.pdf`
- `Wilson 2021.pdf`

## Database Location

The ChromaDB database is stored in:
- **Default**: `./chroma_db/` (relative to application directory)
- **Custom**: Can be changed in `config.py`

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Check file permissions and disk space
4. **Check Pydantic version compatibility**
5. **Check HuggingFace Hub compatibility**
6. Review the troubleshooting section above
