"""
Logging style constants for consistent visual hierarchy.

Provides unified formatting symbols and separators used across all logging modules.
"""


class LogStyle:
    """Unified logging style constants for consistent visual hierarchy."""

    # Level 1: Session headers (80 chars)
    HEAVY = "━" * 80

    # Level 2: Major sections (80 chars)
    DOUBLE = "═" * 80

    # Level 3: Subsections / Separators (80 chars)
    LIGHT = "─" * 80

    # Symbols
    ARROW = "»"
    BULLET = "•"
    WARNING = "⚠"
    SUCCESS = "✓"

    # Indentation
    INDENT = "  "
    DOUBLE_INDENT = "    "
