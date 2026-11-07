"""One append-only UTF-8 text log, mirrored to the console."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import traceback


def now_local() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunLogger:
    def __init__(self, path: Path, run_id: str, console=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.run_id = path, run_id
        self.console = sys.stdout if console is None else console
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def _write(self, level: str, message: str, *args) -> None:
        text = message % args if args else str(message)
        with self._lock:
            for line in text.splitlines() or [""]:
                formatted = "%s | %s | %-7s | %s\n" % (now_local(), self.run_id, level, line)
                self._file.write(formatted)
                self.console.write(formatted)
            self._file.flush()
            self.console.flush()

    def info(self, message, *args):
        self._write("INFO", message, *args)

    def warning(self, message, *args):
        self._write("WARNING", message, *args)

    def error(self, message, *args):
        self._write("ERROR", message, *args)

    def exception(self, message):
        self.error(message)
        self.error(traceback.format_exc())

    def close(self):
        with self._lock:
            self._file.close()


def run_log_snapshot(path: Path, run_id: str) -> bytes:
    if not path.exists():
        return b""
    marker = " | " + run_id + " | "
    with path.open(encoding="utf-8", errors="replace") as stream:
        return "".join(line for line in stream if marker in line).encode("utf-8")
