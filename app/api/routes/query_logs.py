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

# Retriever service URL (internal communication)
RETRIEVER_SERVICE_URL = os.getenv("RETRIEVER_SERVICE_URL", "http://localhost:8001")

# In-memory storage for responses (in production, use Redis or database)
_query_responses: Dict[str, Dict[str, Any]] = {}


@router.get("/health")
async def health_check():
    """Health check endpoint for query logs service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{RETRIEVER_SERVICE_URL}/health")
            retriever_status = "connected" if response.status_code == 200 else "error"
    except Exception as e:
        retriever_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "retriever_service_url": RETRIEVER_SERVICE_URL,
        "retriever_status": retriever_status
    }


@router.post("/user-query")
async def log_user_query_post(
    query: str = Query(..., description="User's search query"),
    projectid: Optional[str] = Query(None, description="Project ID"),
    fileid: Optional[str] = Query(None, description="File ID (optional, for file-specific queries)"),
    user_id: Optional[str] = Query(None, description="User ID"),
    max_results: int = Query(10, description="Maximum number of results"),
    use_agent: bool = Query(True, description="Use intelligent agent for query routing"),
):
    """
    POST endpoint to submit user query and trigger processing.
    
    This endpoint:
    1. Logs the user query to retriever service
    2. Triggers query processing via retriever service (retrieval + LLM inference)
    3. Returns immediately with success status
    
    The actual response/inference will be available via GET /logs/user-response
    """
    try:
        logger.info(f"Received query request: query={query[:50]}..., projectid={projectid}, use_agent={use_agent}")
        
        # Create a unique query ID for tracking
        query_id = f"{projectid}_{datetime.now().isoformat()}_{hash(query)}"
        
        # Step 1: Log the user query to retriever service
        log_payload = {
            "query": query,
            "projectid": projectid,
            "fileid": fileid,
            "user_id": user_id,
        }
        log_payload = {k: v for k, v in log_payload.items() if v is not None}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as log_client:
                await log_client.post(
                    f"{RETRIEVER_SERVICE_URL}/logs/user-query",
                    json=log_payload
                )
        except Exception as e:
            logger.warning(f"Failed to log user query (non-critical): {e}")
        
        # Step 2: Trigger query processing via retriever service (in background)
        # We'll process it and store the result
        query_params = {
            "query": query,
            "projectid": projectid,
            "max_results_per_file": max_results,
            "max_results_global": max_results,
            "stream": False,  # Get complete response, not streaming
        }
        if fileid:
            query_params["available_fileids"] = [fileid]
        
        query_params = {k: v for k, v in query_params.items() if v is not None}
        
        # Initialize response storage
        _query_responses[query_id] = {
            "status": "processing",
            "query": query,
            "projectid": projectid,
            "fileid": fileid,
            "created_at": datetime.now().isoformat(),
            "response": None,
            "error": None
        }
        
        # Process query asynchronously
        import asyncio
        asyncio.create_task(_process_query_async(query_id, query_params, use_agent))
        
        return {
            "success": True,
            "message": "Query submitted successfully",
            "query_id": query_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error submitting query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit query: {str(e)}"
        )


async def _process_query_async(query_id: str, query_params: Dict[str, Any], use_agent: bool):
    """Process query asynchronously and store result."""
    try:
        if use_agent:
            # Use agent-based query
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{RETRIEVER_SERVICE_URL}/chunks/query/agent",
                    params=query_params
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response text
                    summaries = result.get("summaries", {})
                    if isinstance(summaries, dict):
                        response_text = "\n\n".join([f"{fid}: {s}" for fid, s in summaries.items() if s])
                    elif isinstance(summaries, str):
                        response_text = summaries
                    else:
                        response_text = "No summary generated"
                    
                    # Update stored response
                    _query_responses[query_id].update({
                        "status": "completed",
                        "response": response_text,
                        "result": result,
                        "completed_at": datetime.now().isoformat()
                    })
                    
                    # Log the response
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as log_client:
                            await log_client.post(
                                f"{RETRIEVER_SERVICE_URL}/logs/user-response",
                                json={
                                    "query": query_params.get("query"),
                                    "response": response_text,
                                    "projectid": query_params.get("projectid"),
                                    "fileid": query_params.get("available_fileids", [None])[0] if query_params.get("available_fileids") else None,
                                    "metadata": {"query_id": query_id}
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to log user response: {e}")
                else:
                    error_detail = response.text
                    _query_responses[query_id].update({
                        "status": "error",
                        "error": error_detail,
                        "completed_at": datetime.now().isoformat()
                    })
        else:
            # Use standard query
            async with httpx.AsyncClient(timeout=180.0) as client:
                query_payload = {
                    "query": query_params.get("query"),
                    "projectid": query_params.get("projectid"),
                    "fileid": query_params.get("available_fileids", [None])[0] if query_params.get("available_fileids") else None,
                    "max_results": query_params.get("max_results_global", 10),
                }
                query_payload = {k: v for k, v in query_payload.items() if v is not None}
                
                response = await client.post(
                    f"{RETRIEVER_SERVICE_URL}/chunks/query",
                    json=query_payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("summary", "No summary generated")
                    
                    _query_responses[query_id].update({
                        "status": "completed",
                        "response": response_text,
                        "result": result,
                        "completed_at": datetime.now().isoformat()
                    })
                else:
                    error_detail = response.text
                    _query_responses[query_id].update({
                        "status": "error",
                        "error": error_detail,
                        "completed_at": datetime.now().isoformat()
                    })
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}", exc_info=True)
        _query_responses[query_id].update({
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@router.get("/user-response")
async def get_user_response(
    query: Optional[str] = Query(None, description="Original query to find response"),
    query_id: Optional[str] = Query(None, description="Query ID from POST /logs/user-query"),
    projectid: Optional[str] = Query(None, description="Project ID"),
):
    """
    GET endpoint to retrieve the LLM-generated response/inference.
    
    Returns the summary/inference that was generated for the query.
    You can search by query_id (from POST response) or by query text + projectid.
    """
    try:
        # Find the response
        response_data = None
        
        if query_id:
            # Look up by query_id
            response_data = _query_responses.get(query_id)
        elif query and projectid:
            # Find most recent matching query
            matching = [
                r for r in _query_responses.values()
                if r.get("query") == query and r.get("projectid") == projectid
            ]
            if matching:
                # Get most recent
                response_data = max(matching, key=lambda x: x.get("created_at", ""))
        
        if not response_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Response not found. Please ensure the query has been processed via POST /logs/user-query"
            )
        
        if response_data.get("status") == "processing":
            return {
                "status": "processing",
                "message": "Query is still being processed. Please try again in a moment.",
                "query": response_data.get("query"),
                "query_id": query_id or "unknown"
            }
        elif response_data.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {response_data.get('error', 'Unknown error')}"
            )
        elif response_data.get("status") == "completed":
            return {
                "success": True,
                "query": response_data.get("query"),
                "response": response_data.get("response"),  # The LLM inference/summary
                "projectid": response_data.get("projectid"),
                "fileid": response_data.get("fileid"),
                "result": response_data.get("result"),  # Full result with chunks, etc.
                "created_at": response_data.get("created_at"),
                "completed_at": response_data.get("completed_at")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unknown response status"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve response: {str(e)}"
        )
