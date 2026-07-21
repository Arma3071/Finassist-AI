"""Base class for all MCP tools.

Each tool defines a Pydantic input schema for validation, implements
``_run``, and gets consistent logging, timing, and error handling for
free via :meth:`BaseTool.execute`.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.models.schemas import ToolCall
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class ToolError(Exception):
    """Raised when a tool fails after input validation succeeds."""


class BaseTool(ABC):
    """Abstract base class for MCP tools with validation and logging built in."""

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Execute the tool's core logic. Implemented by subclasses."""
        raise NotImplementedError

    def execute(self, **kwargs: Any) -> ToolCall:
        """Validate input, run the tool, and return a structured ToolCall record.

        Args:
            **kwargs: Raw arguments to validate against ``args_schema``.

        Returns:
            A ToolCall record capturing arguments, result, success, latency, and errors.
        """
        start = time.perf_counter()
        try:
            validated = self.args_schema(**kwargs)
        except ValidationError as exc:
            logger.warning("Validation failed for tool '%s': %s", self.name, exc)
            return ToolCall(
                tool_name=self.name,
                arguments=kwargs,
                result=None,
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"Invalid arguments: {exc}",
            )

        try:
            result = self._run(**validated.model_dump())
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info("Tool '%s' succeeded in %.1fms", self.name, latency_ms)
            return ToolCall(
                tool_name=self.name,
                arguments=validated.model_dump(),
                result=result,
                success=True,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception("Tool '%s' failed", self.name)
            return ToolCall(
                tool_name=self.name,
                arguments=validated.model_dump(),
                result=None,
                success=False,
                latency_ms=latency_ms,
                error=str(exc),
            )
