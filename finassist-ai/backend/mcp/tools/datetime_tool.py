"""Date/time MCP tool: current date/time and simple date arithmetic."""

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from backend.mcp.base import BaseTool


class DateTimeArgs(BaseModel):
    """Arguments for the datetime tool."""

    days_offset: int = Field(
        default=0, description="Days to add (or subtract if negative) from today. 0 = current date/time."
    )
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'UTC', 'America/New_York'.")


class DateTimeTool(BaseTool):
    """Returns the current date/time, optionally offset by N days, in a given timezone."""

    name = "datetime"
    description = "Get the current date and time, or a date offset by N days, in a given timezone."
    args_schema = DateTimeArgs

    def _run(self, days_offset: int, timezone: str) -> dict:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone)
        now = datetime.now(tz) + timedelta(days=days_offset)
        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": timezone,
        }
