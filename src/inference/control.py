from __future__ import annotations

import signal
from pathlib import Path


class RunControl:
    """Graceful pause / hard-abort controller for long-running inference."""

    def __init__(self, *, stop_file_path: str | Path) -> None:
        self.stop_file_path = Path(stop_file_path)
        self.stop_requested = False
        self.hard_abort_requested = False

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_signal)

    def poll_stop_file(self) -> bool:
        if self.stop_file_path.exists():
            self.stop_requested = True
            return True
        return False

    def request_stop(self) -> None:
        self.stop_requested = True

    def consume_stop_file(self) -> None:
        self.stop_file_path.unlink(missing_ok=True)

    def _handle_signal(self, signum: int, _frame) -> None:
        if self.stop_requested:
            self.hard_abort_requested = True
            return
        self.stop_requested = True
