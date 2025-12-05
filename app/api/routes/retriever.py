"""Retriever service endpoints integrated into chunking service."""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import os
import json

from app.retriever.config import settings
from app.retriever.models import (
    ChunkIngestRequest,
    QueryRequest,
    QueryResponse,
    ChunkResult,
    UserQueryLog,
    UserResponseLog,
)
from app.retriever.opensearch_client import OpenSearchClient
from app.retriever.hybrid_client import HybridClient
from app.retriever.agent_retriever import AgentRetriever
from app.retriever.agent import initialize_agent
from app.retriever.chunking_integration import ChunkingIntegration, ChunkingWebhookRequest, get_integration
from app.retriever.summarizer import get_summarizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunks", tags=["retriever"])

# Initialize Hybrid client (OpenSearch + LanceDB) - singleton
_hybrid_client = None
_opensearch_client = None
_agent_retriever = None
_chunking_integration = None

def get_hybrid_client():
    """Get or create hybrid client singleton."""
    global _hybrid_client
    if _hybrid_client is None:
        _hybrid_client = HybridClient()
    return _hybrid_client

def get_opensearch_client():
    """Get or create opensearch client singleton."""
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = get_hybrid_client().opensearch_client
    return _opensearch_client

def get_agent_retriever():
    """Get or create agent retriever singleton."""
    global _agent_retriever
    if _agent_retriever is None:
        _agent_retriever = AgentRetriever(get_hybrid_client())
    return _agent_retriever

def get_chunking_integration():
    """Get or create chunking integration singleton."""
    global _chunking_integration
    if _chunking_integration is None:
        # For internal calls, we don't need auth token
        _chunking_integration = get_integration(client=get_hybrid_client(), auth_token=None)
    return _chunking_integration

# Initialize agent on module load
if settings.agent_enabled:
    try:
        initialize_agent(settings.agent_model_path)
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.warning(f"Agent initialization failed: {e}. Using rule-based fallback.")


@router.get("/health")
async def health_check():
    """Health check endpoint for both OpenSearch and LanceDB."""
    health = get_hybrid_client().health_check()
    return health


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_chunks(request: ChunkIngestRequest):
    """
    Ingest chunks from the chunking service.
    
    This endpoint receives chunks from the chunking container and stores them in OpenSearch.
    Old chunks for the same file_version_id are automatically deactivated.
    """
    try:
        # Convert chunks to dict format
        chunks_data = [chunk.model_dump() for chunk in request.chunks]
        
        result = get_hybrid_client().ingest_chunks(
            projectid=request.projectid,
            fileid=request.fileid,
            file_version_id=request.file_version_id,
            chunks=chunks_data
        )
        
        return {
            "success": True,
            "message": f"Successfully ingested {result['indexed']} chunks",
            "indexed": result["indexed"],
            "errors": result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error ingesting chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest chunks: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_chunks(request: QueryRequest):
    """
    Query chunks using semantic search.
    
    Supports two modes:
    1. File-specific query: Set search_all_files=False and provide fileid
    2. Global query: Set search_all_files=True to query across all files in a project (ignores fileid)
    
    The projectid is required for both modes.
    """
    try:
        if not request.projectid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="projectid is required for querying"
            )
        
        # If search_all_files is True, ignore fileid and search across all files
        fileid_to_use = None if request.search_all_files else request.fileid
        
        results = get_hybrid_client().search_chunks(
            query=request.query,
            projectid=request.projectid,
            fileid=fileid_to_use,
            max_results=min(request.max_results, settings.max_chunks_per_query),
            filters=request.filters
        )
        
        # Fetch file names for chunks - use internal function call
        if results:
            unique_file_ids = list(set(chunk.fileid for chunk in results if chunk.fileid))
            if unique_file_ids:
                try:
                    file_name_map = await get_chunking_integration().get_file_names(
                        project_id=request.projectid,
                        file_ids=unique_file_ids
                    )
                    # Enrich chunks with file names
                    for chunk in results:
                        if chunk.fileid in file_name_map:
                            chunk.filename = file_name_map[chunk.fileid]
                except Exception as e:
                    logger.warning(f"Failed to fetch file names: {e}")
        
        # Generate summary for top 15 chunks (with compression and citations)
        # Always generate summary if we have results (summarizer should be available)
        summary = None
        compression_stats = None
        if results:
            try:
                summarizer = get_summarizer()
                summary_result = summarizer.summarize_chunks(
                    chunks=results,
                    query=request.query,
                    projectid=request.projectid,
                    max_chunks_to_summarize=15  # Retrieve 15 chunks as requested
                )
                if summary_result:
                    summary = summary_result.get("summary")
                    # Import CompressionStats here to avoid circular imports
                    from app.retriever.models import CompressionStats
                    compression_stats = CompressionStats(**summary_result.get("compression_stats", {}))
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}")
                # Continue without summary if summarization fails
        
        return QueryResponse(
            query=request.query,
            total_results=len(results),
            chunks=results,
            projectid=request.projectid,
            fileid=request.fileid,
            summary=summary,
            compression_stats=compression_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query chunks: {str(e)}"
        )


@router.post("/query/file/{fileid}", response_model=QueryResponse)
async def query_file_chunks(
    fileid: str,
    query: str,
    projectid: str,
    max_results: int = 10,
    filters: Optional[dict] = None
):
    """
    Query chunks for a specific file (file-specific chat).
    
    This endpoint is optimized for file-specific queries where the user
    wants to chat about a single file.
    """
    try:
        request = QueryRequest(
            query=query,
            projectid=projectid,
            fileid=fileid,
            max_results=max_results,
            filters=filters
        )
        return await query_chunks(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying file chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query file chunks: {str(e)}"
        )


@router.post("/query/global", response_model=QueryResponse)
async def query_global_chunks(
    query: str,
    projectid: str,
    max_results: int = 10,
    filters: Optional[dict] = None
):
    """
    Query chunks across all files in a project (global chat).
    
    This endpoint searches across all files in a project, allowing users
    to have a unified chat experience across multiple files.
    """
    try:
        request = QueryRequest(
            query=query,
            projectid=projectid,
            fileid=None,  # No file filter for global search
            max_results=max_results,
            filters=filters
        )
        return await query_chunks(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying global chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query global chunks: {str(e)}"
        )


@router.delete("/file/{fileid}")
async def delete_file_chunks(fileid: str, projectid: str):
    """
    Delete all chunks for a specific file.
    
    Useful when a file is removed or needs to be re-processed.
    """
    try:
        result = get_hybrid_client().delete_file_chunks(
            projectid=projectid,
            fileid=fileid
        )
        return {
            "success": True,
            "message": f"Deleted {result['deleted']} chunks for file {fileid}",
            "deleted": result["deleted"]
        }
    except Exception as e:
        logger.error(f"Error deleting file chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file chunks: {str(e)}"
        )


@router.get("/stats")
async def get_stats(projectid: Optional[str] = None, fileid: Optional[str] = None):
    """
    Get statistics about stored chunks.
    
    Can be filtered by projectid and/or fileid.
    """
    try:
        # This would require additional OpenSearch query implementation
        # For now, return a placeholder
        return {
            "message": "Stats endpoint - to be implemented",
            "projectid": projectid,
            "fileid": fileid
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/query/agent")
async def agent_query(
    query: str,
    projectid: str,
    available_fileids: Optional[List[str]] = None,
    max_results_per_file: int = 10,
    max_results_global: int = 10,
    stream: bool = False
):
    """
    Intelligent agent-based query endpoint.
    
    The agent analyzes the query and automatically decides:
    - File-specific retrieval: Query each file separately and return chunks per file
    - Global retrieval: Query across all files and return top chunks overall
    
    Args:
        query: The search query
        projectid: Project identifier
        available_fileids: Optional list of file IDs to consider (if not provided, all files in project are used)
        max_results_per_file: Maximum chunks per file for file-specific retrieval
        max_results_global: Maximum chunks for global retrieval
        stream: If True and strategy is file_specific, stream results sequentially (one file at a time)
    """
    try:
        if not settings.agent_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent is disabled. Use /chunks/query endpoints instead."
            )
        
        agent_retriever = get_agent_retriever()
        chunking_integration = get_chunking_integration()
        
        # First, get agent decision
        try:
            decision = agent_retriever.agent.decide_retrieval_strategy(
                query=query,
                projectid=projectid,
                available_fileids=available_fileids or agent_retriever._get_available_fileids(projectid)
            )
            
            strategy = decision["strategy"]
            reasoning = decision["reasoning"]
        except Exception as e:
            logger.error(f"Error getting agent decision: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get agent decision: {str(e)}"
            )
        
        # If stream=True, handle both file_specific and global strategies with streaming
        if stream:
            async def generate_stream():
                try:
                    # Send initial decision
                    decision_event = f"data: {json.dumps({'type': 'decision', 'strategy': strategy, 'reasoning': reasoning})}\n\n"
                    yield decision_event.encode('utf-8')
                    
                    if strategy == "file_specific":
                        # File-specific: Process files sequentially
                        fileids_to_process = decision.get("fileids") or available_fileids or agent_retriever._get_available_fileids(projectid)
                        file_name_map = {}
                        if fileids_to_process:
                            try:
                                file_name_map = await chunking_integration.get_file_names(
                                    project_id=projectid,
                                    file_ids=fileids_to_process
                                )
                            except Exception as e:
                                logger.warning(f"Failed to fetch file names: {e}")
                        
                        try:
                            async for result in agent_retriever.retrieve_sequential(
                                query=query,
                                projectid=projectid,
                                available_fileids=fileids_to_process,
                                max_results_per_file=max_results_per_file
                            ):
                                try:
                                    # Handle different result types
                                    if result.get("type") == "complete":
                                        # Completion message - send as-is
                                        event_data = json.dumps(result)
                                        event = f"data: {event_data}\n\n"
                                        yield event.encode('utf-8')
                                        logger.info("Sent completion message")
                                        break  # Exit loop after completion
                                    elif result.get("type") == "file_result":
                                        # File result - need to convert ChunkResult objects to dicts
                                        if result.get("chunks"):
                                            enriched_chunks = []
                                            for chunk in result["chunks"]:
                                                if hasattr(chunk, '__dict__'):
                                                    # It's a ChunkResult object
                                                    chunk_dict = {
                                                        "chunk_ref": chunk.chunk_ref,
                                                        "text": chunk.text,
                                                        "fileid": chunk.fileid,
                                                        "filename": file_name_map.get(chunk.fileid, getattr(chunk, 'filename', None)),
                                                        "file_version_id": chunk.file_version_id,
                                                        "score": chunk.score,
                                                        "section_path": chunk.section_path,
                                                        "object_type": chunk.object_type,
                                                        "page_range": chunk.page_range,
                                                        "caption": chunk.caption,
                                                        "metadata": chunk.metadata
                                                    }
                                                    enriched_chunks.append(chunk_dict)
                                                else:
                                                    # Already a dict
                                                    enriched_chunks.append(chunk)
                                            result["chunks"] = enriched_chunks
                                        # Now serialize
                                        event_data = json.dumps(result)
                                        event = f"data: {event_data}\n\n"
                                        yield event.encode('utf-8')
                                    elif result.get("type") in ["status", "file_error"]:
                                        # Status or file error - send as-is
                                        event_data = json.dumps(result)
                                        event = f"data: {event_data}\n\n"
                                        yield event.encode('utf-8')
                                    else:
                                        # Other result types - send as-is
                                        event_data = json.dumps(result)
                                        event = f"data: {event_data}\n\n"
                                        yield event.encode('utf-8')
                                except Exception as e:
                                    logger.error(f"Error serializing result: {e}", exc_info=True)
                                    error_event = f"data: {json.dumps({'type': 'error', 'message': f'Error processing result: {str(e)}'})}\n\n"
                                    yield error_event.encode('utf-8')
                        except Exception as e:
                            logger.error(f"Error in retrieve_sequential: {e}", exc_info=True)
                            error_event = f"data: {json.dumps({'type': 'error', 'message': f'Error processing files: {str(e)}'})}\n\n"
                            yield error_event.encode('utf-8')
                    
                        # Completion message should already be sent by retrieve_sequential
                        logger.info("Finished processing all files in retrieve_sequential")
                    else:
                        # Global strategy: Retrieve all chunks and generate summary
                        logger.info(f"Global strategy: Retrieving chunks and generating summary")
                        result = agent_retriever.retrieve(
                            query=query,
                            projectid=projectid,
                            available_fileids=available_fileids,
                            max_results_per_file=max_results_per_file,
                            max_results_global=max_results_global
                        )
                        
                        # Fetch file names for chunks
                        all_file_ids = set()
                        chunks_list = []
                        if isinstance(result.get("results"), list):
                            chunks_list = result["results"]
                            for chunk in chunks_list:
                                if chunk.fileid:
                                    all_file_ids.add(chunk.fileid)
                        elif isinstance(result.get("results"), dict):
                            # Shouldn't happen for global, but handle it
                            for fileid, file_chunks in result["results"].items():
                                all_file_ids.add(fileid)
                                chunks_list.extend(file_chunks)
                        
                        file_name_map = {}
                        if all_file_ids:
                            try:
                                file_name_map = await chunking_integration.get_file_names(
                                    project_id=projectid,
                                    file_ids=list(all_file_ids)
                                )
                                for chunk in chunks_list:
                                    if chunk.fileid in file_name_map:
                                        chunk.filename = file_name_map[chunk.fileid]
                            except Exception as e:
                                logger.warning(f"Failed to fetch file names: {e}")
                        
                        # Convert chunks to dicts for JSON serialization
                        enriched_chunks = []
                        for chunk in chunks_list:
                            chunk_dict = {
                                "chunk_ref": chunk.chunk_ref,
                                "text": chunk.text,
                                "fileid": chunk.fileid,
                                "filename": getattr(chunk, 'filename', file_name_map.get(chunk.fileid, None)),
                                "file_version_id": chunk.file_version_id,
                                "score": chunk.score,
                                "section_path": chunk.section_path,
                                "object_type": chunk.object_type,
                                "page_range": chunk.page_range,
                                "caption": chunk.caption,
                                "metadata": chunk.metadata
                            }
                            enriched_chunks.append(chunk_dict)
                        
                        # Get summary
                        summaries = result.get("summaries", "")
                        summary_text = summaries if isinstance(summaries, str) else ""
                        
                        logger.info(f"Global strategy result: {len(enriched_chunks)} chunks, summary length: {len(summary_text) if summary_text else 0}")
                        
                        # Send summary first (this is the inference/answer)
                        if summary_text:
                            summary_event = {
                                "type": "summary",
                                "summary": summary_text,
                                "message": "AI inference generated"
                            }
                            yield f"data: {json.dumps(summary_event)}\n\n".encode('utf-8')
                            logger.info("Sent summary event for global strategy")
                        else:
                            # If no summary, send a status message
                            no_summary_event = {
                                "type": "status",
                                "message": f"Retrieved {len(enriched_chunks)} chunks, but no summary was generated"
                            }
                            yield f"data: {json.dumps(no_summary_event)}\n\n".encode('utf-8')
                            logger.warning(f"No summary generated for global strategy with {len(enriched_chunks)} chunks")
                        
                        # Send chunks (for reference)
                        if enriched_chunks:
                            chunks_event = {
                                "type": "chunks",
                                "chunks": enriched_chunks,
                                "total": len(enriched_chunks)
                            }
                            yield f"data: {json.dumps(chunks_event)}\n\n".encode('utf-8')
                            logger.info(f"Sent {len(enriched_chunks)} chunks for global strategy")
                        
                        # Send completion
                        completion_event = {
                            "type": "complete",
                            "message": "Global search completed"
                        }
                        yield f"data: {json.dumps(completion_event)}\n\n".encode('utf-8')
                        logger.info("Sent global search completion event")
                except Exception as e:
                    logger.error(f"Error in generate_stream: {e}", exc_info=True)
                    error_event = f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                    yield error_event.encode('utf-8')
                finally:
                    # Ensure stream is properly closed
                    logger.info("Stream generator finished")
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Content-Type": "text/event-stream; charset=utf-8"
                }
            )
        
        # Otherwise, use standard retrieval
        result = agent_retriever.retrieve(
            query=query,
            projectid=projectid,
            available_fileids=available_fileids,
            max_results_per_file=max_results_per_file,
            max_results_global=max_results_global
        )
        
        # Fetch file names for chunks in the result
        if result.get("results"):
            all_file_ids = set()
            if isinstance(result["results"], dict):
                # File-specific: results is a dict of fileid -> chunks
                for fileid, chunks in result["results"].items():
                    all_file_ids.add(fileid)
                    for chunk in chunks:
                        if chunk.fileid:
                            all_file_ids.add(chunk.fileid)
            elif isinstance(result["results"], list):
                # Global: results is a list of chunks
                for chunk in result["results"]:
                    if chunk.fileid:
                        all_file_ids.add(chunk.fileid)
            
            if all_file_ids:
                try:
                    file_name_map = await chunking_integration.get_file_names(
                        project_id=projectid,
                        file_ids=list(all_file_ids)
                    )
                    # Enrich chunks with file names
                    if isinstance(result["results"], dict):
                        for fileid, chunks in result["results"].items():
                            for chunk in chunks:
                                if chunk.fileid in file_name_map:
                                    chunk.filename = file_name_map[chunk.fileid]
                    elif isinstance(result["results"], list):
                        for chunk in result["results"]:
                            if chunk.fileid in file_name_map:
                                chunk.filename = file_name_map[chunk.fileid]
                except Exception as e:
                    logger.warning(f"Failed to fetch file names for agent query: {e}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agent query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process agent query: {str(e)}"
        )


@router.post("/webhook/chunking", status_code=status.HTTP_200_OK)
async def chunking_webhook(
    request: ChunkingWebhookRequest,
    background_tasks: BackgroundTasks
):
    """
    Webhook endpoint to receive notifications from chunking service.
    
    When the chunking service finishes processing a file, it calls this endpoint
    with the chunk file location. This endpoint then ingests the chunks into OpenSearch.
    """
    try:
        if request.status != "parsed":
            return {
                "success": False,
                "message": f"Status is '{request.status}', not 'parsed'. Chunks not ready yet."
            }
        
        # Process in background to avoid blocking
        background_tasks.add_task(
            get_chunking_integration().ingest_chunks_from_chunking_service,
            project_id=request.project_id,
            file_id=request.file_id,
            file_version_id=request.file_version_id,
            chunk_uri=request.chunk_uri
        )
        
        return {
            "success": True,
            "message": "Chunk ingestion queued",
            "project_id": request.project_id,
            "file_id": request.file_id,
            "file_version_id": request.file_version_id
        }
        
    except Exception as e:
        logger.error(f"Error in chunking webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process webhook: {str(e)}"
        )


@router.post("/sync/{project_id}/{file_id}", status_code=status.HTTP_200_OK)
async def sync_chunks_from_chunking(
    project_id: str,
    file_id: str,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger sync of chunks from chunking service for a single file.
    
    This endpoint calls parse_latest on the chunking service and then ingests
    the resulting chunks into OpenSearch.
    """
    try:
        # First, try to fetch chunk info from chunking service (this triggers parsing if needed)
        chunk_info = await get_chunking_integration().fetch_chunks_from_chunking_api(
            project_id=project_id,
            file_id=file_id
        )
        
        chunk_uri = None
        file_version_id = None
        version = 1
        
        if chunk_info:
            # Check if file is parsed
            status_value = chunk_info.get("status")
            if status_value != "parsed":
                logger.warning(f"File status is '{status_value}', not 'parsed'. Will try to find chunk file directly.")
            else:
                # Get chunk_uri from response
                chunk_uri = chunk_info.get("chunk_uri")
                version = chunk_info.get("version", 1)
                file_version_id = chunk_info.get("version_id") or f"v_{file_id}_{version}"
        
        # If chunk_uri not found, try to construct it and find the file directly
        if not chunk_uri:
            logger.info(f"chunk_uri not in response, trying to find chunk file directly for {file_id}")
            # Standard path: /app/uploads/projects/{project_id}/derived/{file_id}/1/all_chunks.jsonl
            constructed_paths = [
                f"/app/uploads/projects/{project_id}/derived/{file_id}/{version}/all_chunks.jsonl",
                f"/app/uploads/projects/{project_id}/derived/{file_id}/{version}/chunks.jsonl",
                f"/app/uploads/projects/{project_id}/derived/{file_id}/1/all_chunks.jsonl",
                f"/app/uploads/projects/{project_id}/derived/{file_id}/1/chunks.jsonl"
            ]
            
            # Try to resolve one of these paths using the integration's path resolution
            for path in constructed_paths:
                try:
                    # Use the private method to resolve paths (it handles container vs host paths)
                    resolved = get_chunking_integration()._resolve_chunk_uri(path)
                    if resolved and os.path.exists(resolved):
                        chunk_uri = path
                        logger.info(f"Found chunk file directly: {chunk_uri} (resolved to: {resolved})")
                        if not file_version_id:
                            file_version_id = f"v_{file_id}_{version}"
                        break
                except Exception as e:
                    logger.debug(f"Could not resolve path {path}: {e}")
                    continue
        
        # If still no chunk_uri found, raise error
        if not chunk_uri:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not find chunk file. Tried: {', '.join(constructed_paths)}. "
                       f"File may not be parsed yet or chunk file doesn't exist. "
                       f"Try uploading and parsing the file first in Step 1 and Step 2."
            )
        
        # Ensure file_version_id is set
        if not file_version_id:
            file_version_id = f"v_{file_id}_{version}"
        
        # Process in background
        background_tasks.add_task(
            get_chunking_integration().ingest_chunks_from_chunking_service,
            project_id=project_id,
            file_id=file_id,
            file_version_id=file_version_id,
            chunk_uri=chunk_uri
        )
        
        return {
            "success": True,
            "message": "Chunk sync queued",
            "project_id": project_id,
            "file_id": file_id,
            "file_version_id": file_version_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync chunks: {str(e)}"
        )


@router.post("/sync-all/{project_id}", status_code=status.HTTP_200_OK)
async def sync_all_chunks_from_chunking(
    project_id: str,
    do_ocr: bool = False,
    force: bool = False,
    only_status: Optional[str] = None,
    background: bool = False,
    force_parse_for_skipped: bool = True,  # If True, force parse skipped files to get chunk_uri
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Parse and sync all files in a project from chunking service.
    
    This endpoint:
    1. Calls parse_all_latest on the chunking service to parse all files
    2. Automatically ingests all resulting chunks into OpenSearch
    
    Args:
        project_id: Project identifier
        do_ocr: Whether to perform OCR during parsing
        force: Re-parse even if already parsed
        only_status: Only process files with this status (e.g., "queued")
        background: If True, process in background (default: False for immediate processing)
    """
    try:
        logger.info(f"Starting sync-all for project {project_id}, force={force}")
        
        # Process directly by default (for demo/testing), or in background if requested
        if background and background_tasks:
            background_tasks.add_task(
                get_chunking_integration().parse_and_sync_all_files,
                project_id=project_id,
                do_ocr=do_ocr,
                force=force,
                only_status=only_status
            )
            return {
                "success": True,
                "message": "Parse and sync queued for all files",
                "project_id": project_id,
                "ingested_chunks": 0
            }
        else:
            # Process directly (synchronous)
            try:
                logger.info(f"Calling parse_and_sync_all_files for project {project_id}")
                result = await get_chunking_integration().parse_and_sync_all_files(
                    project_id=project_id,
                    do_ocr=do_ocr,
                    force=force,
                    only_status=only_status,
                    force_parse_for_skipped=force_parse_for_skipped
                )
                logger.info(f"parse_and_sync_all_files returned: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                # Ensure result has the expected format
                if not isinstance(result, dict):
                    result = {"success": True, "message": "Sync completed", "result": result, "ingested_chunks": 0}
                
                # Ensure ingested_chunks is in the response (check multiple possible keys)
                if "ingested_chunks" not in result:
                    result["ingested_chunks"] = result.get("total_ingested", result.get("ingested", 0))
                
                logger.info(f"Returning sync result with ingested_chunks: {result.get('ingested_chunks', 0)}")
                return result
            except Exception as e:
                logger.error(f"Error in parse_and_sync_all_files: {e}", exc_info=True)
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing all chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync all chunks: {str(e)}"
        )


@router.post("/logs/user-query")
async def log_user_query(payload: UserQueryLog):
    """
    API 1: Receive and log the raw user query.

    This endpoint is meant to be called right when the user asks a question.
    It does not perform retrieval itself; it just records the query context.
    """
    try:
        logger.info(f"[USER_QUERY] {payload.model_dump()}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error logging user query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log user query",
        )


@router.post("/logs/user-response")
async def log_user_response(payload: UserResponseLog):
    """
    API 2: Receive and log the final response returned to the user.

    Typical flow:
    1) Frontend/backend calls /logs/user-query when user asks a question.
    2) Your system does retrieval + generation.
    3) After you have the final answer string, call /logs/user-response
       with the text plus any metadata you want to track.
    """
    try:
        logger.info(f"[USER_RESPONSE] {payload.model_dump()}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error logging user response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log user response",
    )


