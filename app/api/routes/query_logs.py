# app/api/routes/query_logs.py
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
import httpx
import os
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["query-logs"])

# Import retriever for internal health check
from app.retriever.hybrid_client import HybridClient

# Old in-memory storage removed - no longer needed


@router.get("/health")
async def health_check():
    """Health check endpoint for query logs service."""
    try:
        hybrid_client = HybridClient()
        health = hybrid_client.health_check()
        retriever_status = "connected" if health.get("status") != "error" else "error"
    except Exception as e:
        retriever_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "retriever_service_url": "http://localhost:8002/chunks",  # Now integrated
        "retriever_status": retriever_status
    }


# Old endpoints removed - use /USER/query-response instead
