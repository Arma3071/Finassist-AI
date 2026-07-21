# Deployment Guide

## Docker Compose (recommended for local/demo deployment)

```bash
cp .env.example .env   # fill in your API keys
docker compose -f docker/docker-compose.yml up --build
```

This starts:
- **backend** — FastAPI app on `:8000`, with `./data` mounted for
  persistent Chroma/SQLite storage across restarts.
- **frontend** — Streamlit app on `:8501`, pointed at the backend via the
  `BACKEND_URL` environment variable.

Visit http://localhost:8501 for the UI and http://localhost:8000/docs for
the interactive API reference.

## One-off backend image

```bash
docker build -f docker/Dockerfile -t finassist-backend .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data finassist-backend
```

## Production notes

This project is built as a demonstration/portfolio-grade system. Before
running it against real user traffic, consider:

- **Auth**: swap the PBKDF2/SQLite session store for a proper auth
  provider (OAuth, hashed-password service with rate limiting, etc.) and
  move session tokens to signed JWTs with expiry.
- **Database**: move from SQLite to Postgres for concurrent write
  throughput; the `backend/database/db.py` functions are the only place
  that would need to change.
- **Vector store**: ChromaDB's persistent client is fine for a single
  instance; for horizontal scaling, consider a hosted vector DB (Pinecone,
  Weaviate, Qdrant Cloud) or Chroma's client/server mode.
- **Secrets**: never bake API keys into the image; use your platform's
  secret manager and inject them at runtime (this is already how
  `docker-compose.yml` is set up via `env_file`).
- **Rate limiting & cost control**: financial APIs (Alpha Vantage,
  NewsAPI) have strict free-tier rate limits; add caching (e.g. Redis) in
  front of `backend/mcp/tools/` for frequently-requested tickers.
- **Observability**: the structured logging in
  `backend/utils/logging_config.py` is a good foundation — ship logs to a
  centralized system (CloudWatch, Datadog, ELK) and add request tracing.
- **CI/CD**: `.github/workflows/ci.yml` runs lint + tests on every push;
  extend it with a build-and-push-image job for your container registry
  before deploying to production infrastructure (ECS, Cloud Run, etc.).
