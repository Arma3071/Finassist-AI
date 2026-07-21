"""Unit tests for MCP tools (backend.mcp)."""

from backend.mcp.tools.calculator import CalculatorTool
from backend.mcp.tools.datetime_tool import DateTimeTool


def test_calculator_basic_arithmetic():
    tool = CalculatorTool()
    result = tool.execute(expression="(120.5 + 30) * 2")
    assert result.success is True
    assert result.result == 301.0


def test_calculator_rejects_unsafe_expression():
    tool = CalculatorTool()
    result = tool.execute(expression="__import__('os').system('echo hi')")
    assert result.success is False
    assert result.error is not None


def test_calculator_invalid_args_fail_validation():
    tool = CalculatorTool()
    result = tool.execute(missing_field_instead_of="expression")
    assert result.success is False
    assert result.error is not None


def test_calculator_division_by_zero():
    tool = CalculatorTool()
    result = tool.execute(expression="1 / 0")
    assert result.success is False


def test_datetime_tool_returns_expected_keys():
    tool = DateTimeTool()
    result = tool.execute(days_offset=0, timezone="UTC")
    assert result.success is True
    assert set(result.result.keys()) == {"iso", "date", "time", "weekday", "timezone"}


def test_datetime_tool_offset():
    tool = DateTimeTool()
    today = tool.execute(days_offset=0, timezone="UTC").result["date"]
    tomorrow = tool.execute(days_offset=1, timezone="UTC").result["date"]
    assert today != tomorrow
