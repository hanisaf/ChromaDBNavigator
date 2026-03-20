#!/usr/bin/env python3
"""
JSON-line backend server for the native Windows UI.
Keeps ChromaManager initialized in-process to avoid repeated model loads.
"""

import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

from chroma_manager import ChromaManager
from config import COLLECTION_NAME


class BackendServer:
    def __init__(self) -> None:
        self.manager: Optional[ChromaManager] = None
        self.db_path: Optional[str] = None
        self.collection: str = COLLECTION_NAME

    def _ensure_initialized(self) -> None:
        if self.manager is None:
            self.manager = ChromaManager(db_path=self.db_path, collection_name=self.collection)

    def _initialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.db_path = payload.get("db_path") or None
        self.collection = payload.get("collection") or COLLECTION_NAME
        self.manager = ChromaManager(db_path=self.db_path, collection_name=self.collection)
        return {
            "db_path": self.manager.db_path,
            "collection": self.manager.collection_name,
            "device": getattr(self.manager, "device", "unknown"),
        }

    def _preview_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        folder = payload["folder"]
        return self.manager.preview_sync_changes(folder)

    def _sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        folder = payload["folder"]
        added, removed, corrupted = self.manager.sync_database(folder)
        return {"added": added, "removed": removed, "corrupted": corrupted}

    def _search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        query = payload["query"]
        n_results = int(payload.get("n_results", 10))
        return {"items": self.manager.search_database(query, n_results)}

    def _list_documents(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        limit = payload.get("limit")
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = None
        return {"items": self.manager.get_all_documents(limit=limit)}

    def _stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        return self.manager.get_database_stats()

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        payload = request.get("payload", {})

        if action == "initialize":
            return self._initialize(payload)
        if action == "preview_sync":
            return self._preview_sync(payload)
        if action == "sync":
            return self._sync(payload)
        if action == "search":
            return self._search(payload)
        if action == "list_documents":
            return self._list_documents(payload)
        if action == "stats":
            return self._stats(payload)

        raise ValueError(f"Unsupported action: {action}")


def write_response(response: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    server = BackendServer()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        req_id = None
        try:
            request = json.loads(line)
            req_id = request.get("id")

            # Keep protocol on stdout; send library prints/noise to stderr.
            with redirect_stdout(sys.stderr):
                result = server.handle(request)

            write_response({"id": req_id, "ok": True, "result": result})

        except Exception as exc:  # pylint: disable=broad-except
            error_payload = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_response(error_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
