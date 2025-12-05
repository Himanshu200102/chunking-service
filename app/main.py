from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from pymongo import MongoClient
from opensearchpy import OpenSearch
import lancedb

from app.db.mongo import ensure_indexes
from app.api.routes.projects import router as projects_router
from app.api.routes.files import router as files_router
from app.api.routes.query_logs import router as query_logs_router
from app.api.routes.user import router as user_router
from app.api.routes.retriever import router as retriever_router
# from app.api.routes.users import router as users_router
# from app.auth.routes import router as auth_router

# Ensure custom ADK LLMs are registered (local llama service).


APP_ENV = os.getenv("APP_ENV", "dev")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "dataroom")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = os.getenv("OPENSEARCH_PORT", "9200")
OS_URL = os.getenv("OPENSEARCH_URL", f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
LANCE_URI = os.getenv("LANCEDB_URI", "/data/lancedb")


app = FastAPI(
    title="DataRoom API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy singletons -----------------------------------------------------------
_mongo_client = None
_opensearch_client = None
_lance_conn = None


def mongo() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)[MONGO_DB]
    return _mongo_client


def opensearch() -> OpenSearch:
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = OpenSearch(OS_URL)
    return _opensearch_client


def lance() -> lancedb.DBConnection:
    global _lance_conn
    if _lance_conn is None:
        _lance_conn = lancedb.connect(LANCE_URI)
        if "health" not in _lance_conn.table_names():
            _lance_conn.create_table("health", data=[{"id": 1, "msg": "ok"}])
    return _lance_conn


@app.on_event("startup")
def _startup() -> None:
    ensure_indexes()


@app.get("/hello")
def hello():
    return {"message": "hello from dataroom api", "env": APP_ENV}


@app.get("/health")
def health():
    mongo().command("ping")
    os_ok = opensearch().ping()
    lance_ok = "health" in lance().table_names()
    return {"mongo": "ok", "opensearch": os_ok, "lancedb": lance_ok}


# Serve static files (demo HTML)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/demo")
async def demo():
    """Serve the complete pipeline demo HTML page."""
    demo_path = os.path.join(os.path.dirname(__file__), "static", "demo.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path)
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Demo file not found")


# Router registration -------------------------------------------------------
# app.include_router(auth_router)
# app.include_router(users_router)
app.include_router(projects_router)
app.include_router(files_router)
app.include_router(query_logs_router)
app.include_router(user_router)
app.include_router(retriever_router)
