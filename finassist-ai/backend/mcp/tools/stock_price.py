"""Stock price MCP tool backed by Yahoo Finance (yfinance)."""

from pydantic import BaseModel, Field

from backend.mcp.base import BaseTool, ToolError
from backend.utils.retry import retry


class StockPriceArgs(BaseModel):
    """Arguments for the stock price tool."""

    ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL', 'TSLA'.")


class StockPriceTool(BaseTool):
    """Fetches the latest stock price and daily change for a ticker."""

    name = "stock_price"
    description = "Get the current/latest price and daily change for a stock ticker."
    args_schema = StockPriceArgs

    def _run(self, ticker: str) -> dict:
        ticker = ticker.upper().strip()
        return self._fetch(ticker)

    @retry(max_attempts=3, base_delay=0.5)
    def _fetch(self, ticker: str) -> dict:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        history = stock.history(period="2d")
        if history.empty:
            raise ToolError(f"No price data found for ticker '{ticker}'")

        last_close = float(history["Close"].iloc[-1])
        prev_close = float(history["Close"].iloc[-2]) if len(history) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        try:
            info = stock.fast_info
            if isinstance(info, dict):
                currency = info.get("currency", "USD")
                day_high = info.get("day_high", last_close)
                day_low = info.get("day_low", last_close)
            else:
                currency = getattr(info, "currency", "USD")
                day_high = getattr(info, "day_high", last_close)
                day_low = getattr(info, "day_low", last_close)
        except Exception:
            currency = "USD"
            day_high = last_close
            day_low = last_close

        return {
            "ticker": ticker,
            "price": round(last_close, 2),
            "currency": currency,
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "day_high": round(float(day_high), 2),
            "day_low": round(float(day_low), 2),
        }
