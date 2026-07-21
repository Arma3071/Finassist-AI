"""Currency converter MCP tool backed by the ExchangeRate-API."""

import requests
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.mcp.base import BaseTool, ToolError
from backend.utils.retry import retry


class CurrencyConverterArgs(BaseModel):
    """Arguments for the currency converter tool."""

    amount: float = Field(gt=0, description="Amount to convert.")
    from_currency: str = Field(min_length=3, max_length=3, description="3-letter ISO currency code, e.g. 'USD'.")
    to_currency: str = Field(min_length=3, max_length=3, description="3-letter ISO currency code, e.g. 'PKR'.")


class CurrencyConverterTool(BaseTool):
    """Converts an amount between two currencies using live exchange rates."""

    name = "currency_converter"
    description = "Convert an amount from one currency to another using live exchange rates."
    args_schema = CurrencyConverterArgs

    def _run(self, amount: float, from_currency: str, to_currency: str) -> dict:
        settings = get_settings()
        from_currency, to_currency = from_currency.upper(), to_currency.upper()
        rate = self._fetch_rate(from_currency, to_currency, settings.exchangerate_api_key)
        converted = round(amount * rate, 4)
        return {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rate,
            "converted_amount": converted,
        }

    @retry(max_attempts=3, base_delay=0.5)
    def _fetch_rate(self, from_currency: str, to_currency: str, api_key: str) -> float:
        if api_key:
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"
        else:
            # Free, no-key fallback endpoint (rate-limited).
            url = f"https://open.er-api.com/v6/latest/{from_currency}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if api_key:
            if data.get("result") != "success":
                raise ToolError(f"Exchange rate API error: {data.get('error-type')}")
            return float(data["conversion_rate"])

        rates = data.get("rates", {})
        if to_currency not in rates:
            raise ToolError(f"Unknown target currency: {to_currency}")
        return float(rates[to_currency])
