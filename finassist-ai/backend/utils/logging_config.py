"""Centralized logging setup for FinAssist AI.

Every module fetches its logger via ``get_logger(__name__)`` so that
queries, retrieved chunks, LLM responses, latency, tool calls, and errors
are all captured in a consistent, structured format.
"""

import logging
import sys


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once for the whole application.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard library Logger instance.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
