"""Portable project locking and compatibility checks without silent reuse."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import uuid


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", run_id) or ".." in run_id:
        raise ValueError("Invalid run ID; provide the printed run folder name, not a path")
    return run_id


class ProjectLock:
    def __init__(self, path: Path):
        self.path = path
        self.token = uuid.uuid4().hex
        self.owned = False
        self._stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # OS locks are released automatically after a crash. Never unlink the lock
        # file: an unlinked inode can allow two live writers on POSIX systems.
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("Another execution holds the project lock. Wait for it to finish. "
                               "After a crash the operating system releases the lock automatically.") from error
        self._stream = stream
        self.owned = True
        metadata = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "token": self.token}).encode()
        stream.seek(0)
        stream.write(metadata)
        stream.truncate()
        os.fsync(stream.fileno())

    def release(self):
        if self._stream is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._stream.seek(0)
                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()
                self._stream = None
        self.owned = False


def verify_resume(root: Path, run_id: str, identity: dict) -> dict:
    from .artifacts import file_hash, safe_path
    run_id = validate_run_id(run_id)
    directory = safe_path(root, "outputs/runs/" + run_id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed" or status.get("operation") == "report":
        raise ValueError("Resume requires a completed check/run, not an interrupted or reporting invocation")
    for name, value in identity.items():
        if manifest.get("identity", {}).get(name) != value:
            raise ValueError("Resume identity mismatch for " + name)
    artifacts = manifest.get("artifacts", {})
    if not artifacts:
        raise ValueError("Resume manifest has no artifact checksums")
    for relative, expected in artifacts.items():
        file = safe_path(directory, relative)
        if not file.is_file() or file_hash(file) != expected["sha256"]:
            raise ValueError("Resume artifact is missing or changed: " + relative)
    return {"run_id": run_id, "validated_artifacts": len(artifacts),
            "policy": "Compatibility verified; environment and foundation checks are still refreshed"}
