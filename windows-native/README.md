# Windows Native Interface

This folder contains a native Windows interface built with WinForms.

## Run

```powershell
cd windows-native
dotnet run
```

## Requirements

- \.NET 7 SDK or newer on Windows
- Python environment with project dependencies installed (`pip install -r ../requirements.txt`)

## Backend Bridge

The app launches `../native_backend_server.py` as a long-lived process and sends JSON-line commands for:
- initialize
- preview_sync
- sync
- search
- list_documents
- stats

Set `CHROMA_PYTHON_EXE` if your Python executable is not available as `python`.
