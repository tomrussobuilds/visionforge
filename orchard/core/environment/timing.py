"""
Pipeline Timing Utilities.

Provides time tracking for pipeline execution duration measurement.
Used by RootOrchestrator to report total execution time.
"""

import time
from typing import Optional, Protocol


class TimeTrackerProtocol(Protocol):
    """Protocol for pipeline duration tracking."""

    def start(self) -> None:
        """Record pipeline start time."""
        ...  # pragma: no cover

    def stop(self) -> float:
        """Record stop time and return elapsed seconds."""
        ...  # pragma: no cover

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        ...  # pragma: no cover

    @property
    def elapsed_formatted(self) -> str:
        """Human-readable elapsed time string."""
        ...  # pragma: no cover


class TimeTracker:
    """
    Default implementation of TimeTrackerProtocol.

    Tracks elapsed time between start() and stop() calls,
    providing both raw seconds and formatted output.
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> None:
        """Record pipeline start time."""
        self._start_time = time.time()
        self._end_time = None

    def stop(self) -> float:
        """Record stop time and return elapsed seconds."""
        self._end_time = time.time()
        return self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    @property
    def elapsed_formatted(self) -> str:
        """Human-readable elapsed time string (e.g., '1h 23m 45s')."""
        total_seconds = self.elapsed_seconds
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{total_seconds:.1f}s"
