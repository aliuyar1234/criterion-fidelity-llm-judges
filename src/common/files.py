from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def stable_json_dumps(payload: Any) -> str:
    """Serialize JSON deterministically for hashing and artifact writes."""

    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA256 digest for raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    """Return the hexadecimal SHA256 digest for UTF-8 text."""

    return sha256_bytes(payload.encode("utf-8"))


def sha256_json(payload: Any) -> str:
    """Return the hexadecimal SHA256 digest for a JSON-serializable payload."""

    return sha256_text(stable_json_dumps(payload))


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA256 digest for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically by replacing a temp file in the destination directory."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=file_path.parent,
        delete=False,
        newline="",
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, file_path)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write a JSON artifact atomically with stable formatting."""

    atomic_write_text(path, stable_json_dumps(payload) + "\n")
