"""Company financials MCP tool backed by the Alpha Vantage API."""

import requests
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.mcp.base import BaseTool, ToolError
from backend.utils.retry import retry

_BASE_URL = "https://www.alphavantage.co/query"


class FinancialsArgs(BaseModel):
    """Arguments for the company financials tool."""

    ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    statement: str = Field(
        default="overview",
        description="Which data to fetch: 'overview', 'income_statement', or 'balance_sheet'.",
    )


class CompanyFinancialsTool(BaseTool):
    """Fetches company fundamentals/financial statements from Alpha Vantage."""

    name = "company_financials"
    description = "Get company overview, income statement, or balance sheet data for a ticker."
    args_schema = FinancialsArgs

    _FUNCTION_MAP = {
        "overview": "OVERVIEW",
        "income_statement": "INCOME_STATEMENT",
        "balance_sheet": "BALANCE_SHEET",
    }

    def _run(self, ticker: str, statement: str) -> dict:
        settings = get_settings()
        if not settings.alpha_vantage_api_key:
            raise ToolError("ALPHA_VANTAGE_API_KEY is not configured.")

        function = self._FUNCTION_MAP.get(statement)
        if function is None:
            raise ToolError(f"Unknown statement type: {statement}")

        data = self._fetch(ticker.upper(), function, settings.alpha_vantage_api_key)
        if not data or "Note" in data or "Information" in data or "Error Message" in data:
            raise ToolError(f"Alpha Vantage returned no usable data for {ticker}: {data}")

        if statement == "overview":
            return data

        # income_statement / balance_sheet: return most recent annual report only
        reports = data.get("annualReports", [])
        return {"ticker": ticker.upper(), "latest_annual_report": reports[0] if reports else {}}

    @retry(max_attempts=3, base_delay=1.0)
    def _fetch(self, ticker: str, function: str, api_key: str) -> dict:
        response = requests.get(
            _BASE_URL,
            params={"function": function, "symbol": ticker, "apikey": api_key},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
