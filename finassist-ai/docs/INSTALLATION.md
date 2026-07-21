# Installation Guide

## Prerequisites

- Python 3.12+
- pip
- (Optional) Docker + Docker Compose
- API keys: at least one LLM provider (Anthropic or OpenAI). Financial
  tool keys (Alpha Vantage, ExchangeRate-API, NewsAPI) are optional —
  `stock_price`, `calculator`, and `datetime` work with no keys at all.

## 1. Clone and set up a virtual environment

```bash
git clone <your-fork-url> finassist-ai
cd finassist-ai
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
LLM_PROVIDER=anthropic          # or "openai"
ANTHROPIC_API_KEY=sk-ant-...    # if using anthropic
OPENAI_API_KEY=sk-...           # if using openai
```

Optional, for the full MCP tool set:

```
ALPHA_VANTAGE_API_KEY=...       # https://www.alphavantage.co/support/#api-key
EXCHANGERATE_API_KEY=...        # https://www.exchangerate-api.com (falls back to a free endpoint if blank)
NEWSAPI_API_KEY=...             # https://newsapi.org
```

## 4. Initialize and run the backend

```bash
uvicorn backend.main:app --reload
```

The first run will download the embedding model
(`BAAI/bge-base-en-v1.5`, ~440MB) — this requires internet access once;
it's then cached locally. The API is now live at
http://localhost:8000, with interactive docs at http://localhost:8000/docs.

## 5. Run the frontend

In a second terminal (with the same virtual environment activated):

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501, register an account, log in, and try:

1. Upload `data/sample_docs/sample_report.md` from the sidebar.
2. Ask: *"What was Q3 revenue growth?"*
3. Ask: *"What's the current price of AAPL?"*
4. Open the **Admin Dashboard** tab and click **Run evaluation now**.

## 6. Run the test suite

```bash
pytest --cov=backend --cov-report=term-missing tests/
```

Some integration tests require the embedding model to be reachable/cached
(see `tests/test_api_integration.py`); they skip gracefully if not.

## 7. Run the standalone MCP server (optional)

To expose the financial tools to any MCP-compatible client (e.g. Claude
Desktop) independent of the FastAPI backend:

```bash
python -m backend.mcp.server
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker-based setup.
