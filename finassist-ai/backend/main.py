"""FastAPI application entrypoint for FinAssist AI (core: RAG + MCP + Agent)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import get_settings
from backend.utils.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="FinAssist AI",
    description="RAG + MCP + Agent core for financial research assistance.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins else ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    logger.info("FinAssist AI starting up with LLM provider=%s", settings.llm_provider)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
