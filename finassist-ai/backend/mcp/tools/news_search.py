"""News search MCP tool backed by NewsAPI.org."""

import requests
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.mcp.base import BaseTool, ToolError
from backend.utils.retry import retry

_BASE_URL = "https://newsapi.org/v2/everything"


class NewsSearchArgs(BaseModel):
    """Arguments for the news search tool."""

    query: str = Field(description="Search keywords, e.g. 'Tesla earnings' or 'Pakistan stock market'.")
    max_results: int = Field(default=5, ge=1, le=20)


class NewsSearchTool(BaseTool):
    """Searches recent news articles relevant to a financial query."""

    name = "news_search"
    description = "Search recent news articles for a company, market, or financial topic."
    args_schema = NewsSearchArgs

    def _run(self, query: str, max_results: int) -> list[dict]:
        settings = get_settings()
        if not settings.newsapi_api_key:
            raise ToolError("NEWSAPI_API_KEY is not configured.")

        data = self._fetch(query, max_results, settings.newsapi_api_key)
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title"),
                "source": (a.get("source") or {}).get("name"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
                "description": a.get("description"),
            }
            for a in articles[:max_results]
        ]

    @retry(max_attempts=3, base_delay=1.0)
    def _fetch(self, query: str, max_results: int, api_key: str) -> dict:
        response = requests.get(
            _BASE_URL,
            params={
                "q": query,
                "pageSize": max_results,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": api_key,
            },
            timeout=15,
        )
        if response.status_code == 429:
            raise ToolError("NewsAPI rate limit exceeded. Please wait before retrying.")
        response.raise_for_status()
        return response.json()
