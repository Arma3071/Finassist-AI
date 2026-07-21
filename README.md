# Finassist-AI
A production-style financial research assistant demonstrating RAG, MCP tools, LangGraph agent orchestration, and an evaluation pipeline — built end-to-end with FastAPI, LangChain, ChromaDB, and Streamlit.

How to Run
Option 1: Local (no Docker)
Prerequisites: Python 3.10-3.12, pip.

# 1. Create venv and install deps
cd C:\Users\Downloads\finassist-ai\finassist-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure .env
cp .env.example .env
# Edit .env — add at least ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Start backend (terminal 1)
uvicorn backend.main:app --reload

# 4. Start frontend (terminal 2)
streamlit run frontend/app.py

# 5. Open http://localhost:8501
#    Register an account, log in, upload data\sample_docs\sample_report.md
Option 2: Docker
# 1. Configure
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start
docker compose -f docker/docker-compose.yml up --build

# 3. Open http://localhost:8501
Option 3: Run tests
# Activate venv, then:
pytest --cov=backend --cov-report=term-missing tests/
Note: The SentenceTransformer model (~500MB) downloads on first run. Tests that need it will be slow the first time or skipped if offline.
