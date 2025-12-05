# app/api/routes/user.py
from fastapi import APIRouter, HTTPException, Query, status, Body
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import httpx
import os
import logging
import json
import asyncio
import re
from datetime import datetime

# Import retriever functions for internal calls
from app.retriever.hybrid_client import HybridClient
from app.retriever.agent_retriever import AgentRetriever
from app.retriever.models import QueryRequest, QueryResponse
from app.retriever.chunking_integration import get_integration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/USER", tags=["USER"])

# Internal retriever clients (singletons)
_hybrid_client = None
_agent_retriever = None
_chunking_integration = None

def get_hybrid_client():
    """Get or create hybrid client singleton."""
    global _hybrid_client
    if _hybrid_client is None:
        _hybrid_client = HybridClient()
    return _hybrid_client

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
        _chunking_integration = get_integration(client=get_hybrid_client(), auth_token=None)
    return _chunking_integration


@router.post("/query-response")
async def query_response(
    query: str = Body(..., description="User's search query"),
    projectid: str = Body(..., description="Project ID"),
    fileid: Optional[str] = Body(None, description="Single file ID to search (if not provided, searches all files in project)"),
    user_id: Optional[str] = Body(None, description="User ID"),
    max_results: int = Body(10, description="Maximum number of results"),
    use_agent: bool = Body(True, description="Use intelligent agent for query routing"),
):
    """
    POST streaming endpoint for query and response.
    
    This endpoint:
    1. Accepts user query with optional file selection (single file OR all files)
    2. Streams responses as Server-Sent Events (SSE)
    3. Sends heartbeat every 5 seconds to keep connection alive
    4. Returns LLM inference/summary with retrieved chunks
    
    Parameters:
    - query: The search query
    - projectid: Project ID (required)
    - fileid: Optional single file ID. If not provided, searches all files in project
    - user_id: Optional user ID for logging
    - max_results: Maximum number of results to return
    - use_agent: Whether to use intelligent agent for query routing
    """
    try:
        logger.info(f"Received query: query={query[:50]}..., projectid={projectid}, fileid={fileid}, use_agent={use_agent}")
        
        # Step 1: Log the user query to retriever service (non-blocking)
        log_payload = {
            "query": query,
            "projectid": projectid,
            "user_id": user_id,
        }
        if fileid:
            log_payload["fileid"] = fileid
        
        log_payload = {k: v for k, v in log_payload.items() if v is not None}
        
        # Log query in background (don't wait)
        asyncio.create_task(_log_user_query(log_payload))
        
        # Step 2: Create streaming response generator with processing messages
        async def generate_stream():
            # Data collection for final response
            collected_chunks = []
            collected_summary = []  # List to store summary
            collected_fileids = set()
            stream_done = asyncio.Event()
            error_occurred = asyncio.Event()
            error_message = [None]
            inference_done = asyncio.Event()
            
            # Queue for streaming events from retriever
            data_queue = asyncio.Queue()
            
            # Track current file being processed (for file-specific strategy)
            current_file_index = [0]  # Use list to allow modification in nested functions
            current_fileid = [None]
            is_file_specific_mode = [False]
            
            # Task to read from retriever service (using internal calls)
            # Capture fileid in closure to avoid scope issues
            captured_fileid = fileid
            async def read_from_retriever():
                try:
                    # Use internal retriever functions instead of HTTP
                    if captured_fileid:
                        # Single file: Direct retrieval + inference (no agent)
                        logger.info(f"Single file query - using direct retrieval for fileid: {captured_fileid}")
                        
                        # Call internal retriever function
                        hybrid_client = get_hybrid_client()
                        chunking_integration = get_chunking_integration()
                        
                        # Search chunks
                        results = hybrid_client.search_chunks(
                            query=query,
                            projectid=projectid,
                            fileid=captured_fileid,
                            max_results=max_results
                        )
                        
                        # Fetch file names
                        if results:
                            unique_file_ids = list(set(chunk.fileid for chunk in results if chunk.fileid))
                            if unique_file_ids:
                                try:
                                    file_name_map = await chunking_integration.get_file_names(
                                        project_id=projectid,
                                        file_ids=unique_file_ids
                                    )
                                    for chunk in results:
                                        if chunk.fileid in file_name_map:
                                            chunk.filename = file_name_map[chunk.fileid]
                                except Exception as e:
                                    logger.warning(f"Failed to fetch file names: {e}")
                        
                        # Generate summary
                        from app.retriever.summarizer import get_summarizer
                        summary = None
                        if results:
                            try:
                                summarizer = get_summarizer()
                                summary_result = summarizer.summarize_chunks(
                                    chunks=results,
                                    query=query,
                                    projectid=projectid,
                                    max_chunks_to_summarize=15
                                )
                                if summary_result:
                                    summary = summary_result.get("summary")
                            except Exception as e:
                                logger.warning(f"Failed to generate summary: {e}")
                        
                        # Build result dict
                        result = {
                            "query": query,
                            "total_results": len(results),
                            "chunks": [{
                                "chunk_ref": ch.chunk_ref,
                                "text": ch.text,
                                "fileid": ch.fileid,
                                "filename": getattr(ch, 'filename', None),
                                "file_version_id": ch.file_version_id,
                                "score": ch.score,
                                "section_path": ch.section_path,
                                "object_type": ch.object_type,
                                "page_range": ch.page_range,
                                "caption": ch.caption,
                                "metadata": ch.metadata
                            } for ch in results],
                            "projectid": projectid,
                            "fileid": captured_fileid,
                            "summary": summary
                        }
                        
                        # Collect data from result
                        await _collect_result_data(result, collected_chunks, collected_summary, collected_fileids)
                    elif use_agent:
                        # All files with intelligent agent:
                        # Call internal agent retriever
                        logger.info(f"Calling internal agent retriever for query: {query[:50]}...")
                        
                        agent_retriever = get_agent_retriever()
                        chunking_integration = get_chunking_integration()
                        
                        # Fetch file names BEFORE calling retrieve (so we can pass them to summarizer)
                        file_name_map = {}
                        try:
                            # Use MongoDB directly to get file names (no HTTP calls needed)
                            from app.db.mongo import get_db
                            mongo_db = get_db()
                            files_collection = mongo_db["files"]
                            files_cursor = files_collection.find({"project_id": projectid})
                            for file_doc in files_cursor:
                                file_id = file_doc.get("_id") or str(file_doc.get("_id"))
                                filename = file_doc.get("filename") or file_doc.get("name")
                                if file_id and filename:
                                    file_name_map[file_id] = filename
                            logger.info(f"✅ Fetched {len(file_name_map)} file names from MongoDB for project {projectid}: {file_name_map}")
                        except Exception as e:
                            logger.warning(f"❌ Failed to fetch file names from MongoDB: {e}")
                            # Fallback: try chunking_integration
                            try:
                                hybrid_client = get_hybrid_client()
                                sample_chunks = hybrid_client.opensearch_client.search_chunks(
                                    query="*",
                                    projectid=projectid,
                                    fileid=None,
                                    max_results=100
                                )
                                all_file_ids = list(set(chunk.fileid for chunk in sample_chunks if chunk.fileid))
                                if all_file_ids:
                                    file_name_map = await chunking_integration.get_file_names(
                                        project_id=projectid,
                                        file_ids=all_file_ids
                                    )
                                    logger.info(f"Fetched {len(file_name_map)} file names via chunking_integration")
                            except Exception as e2:
                                logger.warning(f"Failed to fetch file names via chunking_integration: {e2}")
                        
                        # Get agent decision first
                        agent = agent_retriever.agent
                        available_fileids = agent_retriever._get_available_fileids(projectid)
                        decision = agent.decide_retrieval_strategy(
                            query=query,
                            projectid=projectid,
                            available_fileids=available_fileids
                        )
                        
                        strategy = decision.get("strategy")
                        reasoning = decision.get("reasoning")
                        logger.info(f"✅ Agent decision: {strategy} - {reasoning}")
                        
                        # Send decision event to client
                        decision_event = {
                            "type": "decision",
                            "strategy": strategy,
                            "reasoning": reasoning,
                        }
                        await data_queue.put((
                            "data",
                            f"data: {json.dumps(decision_event)}\n\n".encode('utf-8')
                        ))
                        
                        # File-specific strategy: use sequential processing (TRUE sequential!)
                        if strategy == "file_specific":
                            is_file_specific_mode[0] = True
                            current_file_index[0] = 0
                            
                            # Use retrieve_sequential - processes files one at a time
                            async for event in agent_retriever.retrieve_sequential(
                                query=query,
                                projectid=projectid,
                                available_fileids=available_fileids,
                                max_results_per_file=max_results,
                                file_id_to_name=file_name_map
                            ):
                                event_type = event.get("type")
                                
                                if event_type == "status":
                                    # Status update for current file
                                    await data_queue.put((
                                        "data",
                                        f"data: {json.dumps(event)}\n\n".encode('utf-8')
                                    ))
                                
                                elif event_type == "file_result":
                                    # File processing complete - send final_response
                                    current_file_index[0] += 1
                                    fid = event.get("fileid")
                                    filename = event.get("filename", fid)
                                    chunks = event.get("chunks", [])
                                    summary = event.get("summary")
                                    
                                    logger.info(f"📄 File {current_file_index[0]}: {filename} - {len(chunks)} chunks, summary: {len(summary) if summary else 0} chars")
                                    
                                    file_final_response = _build_file_final_response(
                                        query=query,
                                        projectid=projectid,
                                        fileid=fid,
                                        file_result=event
                                    )
                                    
                                    # Log truncated final response JSON
                                    try:
                                        logged = json.dumps(file_final_response)
                                        if len(logged) > 800:
                                            logged = logged[:800] + "...[truncated]"
                                        logger.info(f"[FINAL_RESPONSE] {logged}")
                                    except Exception as log_e:
                                        logger.warning(f"Failed to log final_response JSON: {log_e}")
                                    
                                    await data_queue.put((
                                        "data",
                                        f"data: {json.dumps(file_final_response)}\n\n".encode('utf-8')
                                    ))
                                
                                elif event_type == "file_error":
                                    # Error processing file
                                    logger.error(f"❌ Error processing file {event.get('fileid')}: {event.get('error')}")
                                    await data_queue.put((
                                        "data",
                                        f"data: {json.dumps(event)}\n\n".encode('utf-8')
                                    ))
                                
                                elif event_type == "complete":
                                    # All files processed
                                    logger.info("✅ All files processed sequentially")
                                    await data_queue.put((
                                        "data",
                                        f"data: {json.dumps(event)}\n\n".encode('utf-8')
                                    ))
                                    inference_done.set()
                                    stream_done.set()
                                    break
                        
                        else:
                            # Global strategy: treat result as a single combined result
                            is_file_specific_mode[0] = False
                            
                            # Call retrieve to get the result for global strategy
                            result = agent_retriever.retrieve(
                                query=query,
                                projectid=projectid,
                                available_fileids=available_fileids,
                                max_results_per_file=max_results,
                                max_results_global=max_results,
                                file_id_to_name=file_name_map
                            )
                            
                            # Flatten results into chunks + single summary
                            enriched_chunks: List[Dict[str, Any]] = []
                            results_field = result.get("results")
                            
                            if isinstance(results_field, list):
                                enriched_chunks = results_field
                            elif isinstance(results_field, dict):
                                for fid, file_chunks in results_field.items():
                                    for ch in file_chunks:
                                        enriched_chunks.append(ch)
                            
                            # Summaries may be a string or dict; handle both
                            summaries_field = result.get("summaries")
                            if isinstance(summaries_field, str) and summaries_field.strip():
                                summary_text = summaries_field
                            elif isinstance(summaries_field, dict):
                                # If dict, join all summaries with line breaks
                                summary_text = "\n\n".join(
                                    v for v in summaries_field.values() if isinstance(v, str) and v.strip()
                                )
                                if not summary_text.strip():
                                    summary_text = None
                            else:
                                summary_text = None
                            
                            # Log if summary is missing
                            if not summary_text:
                                logger.warning(f"No summary generated for global strategy. Summaries field type: {type(summaries_field)}, value: {summaries_field}")
                            else:
                                logger.info(f"Summary text extracted: {len(summary_text)} chars, preview: {summary_text[:100]}...")
                            
                            # Use existing collector helper
                            aggregate_result = {
                                "chunks": enriched_chunks,
                                "summary": summary_text,
                            }
                            await _collect_result_data(
                                aggregate_result,
                                collected_chunks,
                                collected_summary,
                                collected_fileids,
                            )
                    else:
                        # All files without agent: Use standard query
                        logger.info(f"All files query - using direct retrieval")
                        
                        hybrid_client = get_hybrid_client()
                        chunking_integration = get_chunking_integration()
                        
                        # Search chunks across all files
                        results = hybrid_client.search_chunks(
                            query=query,
                            projectid=projectid,
                            fileid=None,  # No file filter for global search
                            max_results=max_results
                        )
                        
                        # Fetch file names
                        if results:
                            unique_file_ids = list(set(chunk.fileid for chunk in results if chunk.fileid))
                            if unique_file_ids:
                                try:
                                    file_name_map = await chunking_integration.get_file_names(
                                        project_id=projectid,
                                        file_ids=unique_file_ids
                                    )
                                    for chunk in results:
                                        if chunk.fileid in file_name_map:
                                            chunk.filename = file_name_map[chunk.fileid]
                                except Exception as e:
                                    logger.warning(f"Failed to fetch file names: {e}")
                        
                        # Generate summary
                        from app.retriever.summarizer import get_summarizer
                        summary = None
                        if results:
                            try:
                                summarizer = get_summarizer()
                                summary_result = summarizer.summarize_chunks(
                                    chunks=results,
                                    query=query,
                                    projectid=projectid,
                                    max_chunks_to_summarize=15,
                                    file_id_to_name=file_name_map if 'file_name_map' in locals() else None
                                )
                                if summary_result:
                                    summary = summary_result.get("summary")
                                    if summary:
                                        logger.info(f"Generated summary: {len(summary)} chars, preview: {summary[:100]}...")
                                    else:
                                        logger.warning("Summary result exists but summary field is None or empty")
                                else:
                                    logger.warning("Summary result is None")
                            except Exception as e:
                                logger.error(f"Failed to generate summary: {e}", exc_info=True)
                        
                        # Build result dict
                        result = {
                            "query": query,
                            "total_results": len(results),
                            "chunks": [{
                                "chunk_ref": ch.chunk_ref,
                                "text": ch.text,
                                "fileid": ch.fileid,
                                "filename": getattr(ch, 'filename', None),
                                "file_version_id": ch.file_version_id,
                                "score": ch.score,
                                "section_path": ch.section_path,
                                "object_type": ch.object_type,
                                "page_range": ch.page_range,
                                "caption": ch.caption,
                                "metadata": ch.metadata
                            } for ch in results],
                            "projectid": projectid,
                            "fileid": None,
                            "summary": summary if summary else "No inference generated."
                        }
                        
                        # Log summary status
                        if summary:
                            logger.info(f"Summary included in result: {len(summary)} chars")
                        else:
                            logger.warning("No summary generated, using fallback message")
                        
                        await _collect_result_data(result, collected_chunks, collected_summary, collected_fileids)
                    
                    inference_done.set()
                    stream_done.set()
                    
                    # Log the response to retriever service (non-blocking, after completion)
                    # Collect summary for logging (if available)
                    response_summary = None
                    if collected_summary and len(collected_summary) > 0:
                        # Get the first summary (or combine if multiple)
                        response_summary = collected_summary[0]
                        # Truncate if too long for logging
                        if len(response_summary) > 500:
                            response_summary = response_summary[:500] + "..."
                    elif is_file_specific_mode[0]:
                        # For file-specific, we sent per-file responses
                        files_processed = current_file_index[0] if isinstance(current_file_index, list) and len(current_file_index) > 0 else (current_file_index if isinstance(current_file_index, int) else 0)
                        response_summary = f"Processed {files_processed} file(s) with individual responses"
                    else:
                        response_summary = "Response streamed to client"
                    
                    # Log the response (non-blocking, after completion)
                    asyncio.create_task(_log_user_response({
                        "query": query,
                        "response": response_summary,
                        "projectid": projectid,
                        "fileid": captured_fileid,
                        "user_id": user_id,
                        "metadata": {
                            "streaming": True,
                            "chunks_count": len(collected_chunks),
                            "files_processed": current_file_index[0] if is_file_specific_mode[0] else None
                        }
                    }))
                    
                except Exception as e:
                    logger.error(f"Error in retriever query: {e}", exc_info=True)
                    error_message[0] = f"Query processing failed: {str(e)}"
                    error_occurred.set()
            
            # Start the reader task
            reader_task = asyncio.create_task(read_from_retriever())
            
            # Processing message sender (every 5 seconds)
            processing_interval = 5.0  # 5 seconds
            processing_stop = asyncio.Event()
            processing_queue = asyncio.Queue()
            current_filename = [None]  # Track current filename for processing messages
            
            async def processing_sender():
                """Send 'processing' message every 5 seconds while stream is active"""
                while not stream_done.is_set() and not error_occurred.is_set() and not processing_stop.is_set():
                    await asyncio.sleep(processing_interval)
                    if not stream_done.is_set() and not error_occurred.is_set() and not processing_stop.is_set():
                        # For file-specific mode, include file number and filename
                        if is_file_specific_mode[0] and current_file_index[0] > 0:
                            file_num = current_file_index[0]
                            fileid = current_fileid[0]
                            filename = current_filename[0] or fileid or "file"
                            file_num_text = f"{file_num}{'st' if file_num == 1 else 'nd' if file_num == 2 else 'rd' if file_num == 3 else 'th'}"
                            processing_event = {
                                "type": "processing",
                                "message": f"Processing {file_num_text} file: {filename}...",
                                "fileid": fileid,
                                "filename": filename,
                                "file_number": file_num
                            }
                        else:
                            processing_event = {
                                "type": "processing",
                                "message": "Processing the document..."
                            }
                        await processing_queue.put(f"data: {json.dumps(processing_event)}\n\n".encode('utf-8'))
            
            # Start processing message task
            processing_task = asyncio.create_task(processing_sender())
            
            # Generator that yields processing messages and stream events
            try:
                # Main loop: send processing messages and stream events
                while True:
                    try:
                        # Check processing queue FIRST (with short timeout) to ensure messages are sent every 5 seconds
                        try:
                            processing_msg = await asyncio.wait_for(processing_queue.get(), timeout=0.1)
                            yield processing_msg
                            # After yielding processing message, continue to check data_queue
                        except asyncio.TimeoutError:
                            pass
                        
                        # Then check data_queue (with longer timeout)
                        try:
                            item_type, item_data = await asyncio.wait_for(data_queue.get(), timeout=0.4)
                            if item_type == "data":
                                yield item_data
                        except asyncio.TimeoutError:
                            # If both queues are empty, continue loop to check again
                            pass
                        
                        # Exit when reader_task is done and queues are drained, or on error
                        if error_occurred.is_set():
                            break
                        
                        if reader_task.done() and data_queue.empty() and processing_queue.empty():
                            # Small delay to allow any pending processing messages
                            await asyncio.sleep(0.1)
                            if processing_queue.empty():
                                break
                    except Exception as e:
                        logger.error(f"Error in stream generator: {e}")
                        break
                
                # For global strategy or single file, build final response
                if not is_file_specific_mode[0]:
                    # Wait a bit more to ensure all data is collected
                    await asyncio.sleep(0.5)
                    
                    if error_occurred.is_set():
                        error_event = {
                            "type": "error",
                            "message": error_message[0] or "Unknown error occurred"
                        }
                        yield f"data: {json.dumps(error_event)}\n\n".encode('utf-8')
                    else:
                        # Build final response with inference, citations, etc.
                        final_response = _build_final_response(
                            query=query,
                            projectid=projectid,
                            fileid=captured_fileid,
                            collected_fileids=collected_fileids,
                            inference=collected_summary,
                            chunks=collected_chunks
                        )
                        yield f"data: {json.dumps(final_response)}\n\n".encode('utf-8')
                        
            finally:
                # Clean up
                processing_stop.set()
                if not processing_task.done():
                    processing_task.cancel()
                    try:
                        await processing_task
                    except asyncio.CancelledError:
                        pass
                
                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        pass
        
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
        
    except Exception as e:
        logger.error(f"Error in query-response endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


async def _collect_result_data(
    result: Dict[str, Any],
    collected_chunks: List[Dict[str, Any]],
    collected_summary: List[str],
    collected_fileids: set
):
    """Collect chunks and summary from a JSON result."""
    chunks = result.get("chunks", [])
    summary = result.get("summary")
    
    # Collect summary (only if not already collected)
    if summary:
        # Only add if not already present (avoid duplicates)
        if len(collected_summary) == 0:
            collected_summary.append(summary)
        elif collected_summary[0] != summary:
            # If different summary, replace it
            collected_summary[0] = summary
    
    # Collect chunks
    for chunk in chunks:
        if hasattr(chunk, '__dict__'):
            # It's an object, convert to dict
            chunk_dict = {
                "chunk_ref": getattr(chunk, 'chunk_ref', None),
                "text": getattr(chunk, 'text', ''),
                "fileid": getattr(chunk, 'fileid', None),
                "filename": getattr(chunk, 'filename', None),
                "file_version_id": getattr(chunk, 'file_version_id', None),
                "section_path": getattr(chunk, 'section_path', []),
                "object_type": getattr(chunk, 'object_type', None),
                "page_range": getattr(chunk, 'page_range', None),
                "caption": getattr(chunk, 'caption', None),
                "metadata": getattr(chunk, 'metadata', {})
            }
        else:
            # It's already a dict
            chunk_dict = chunk.copy()
            # Remove score if present
            chunk_dict.pop("score", None)
        
        collected_chunks.append(chunk_dict)
        if chunk_dict.get("fileid"):
            collected_fileids.add(chunk_dict["fileid"])


async def _collect_stream_event(
    event: Dict[str, Any],
    collected_chunks: List[Dict[str, Any]],
    collected_summary: List[str],
    collected_fileids: set
):
    """Collect data from SSE stream events."""
    event_type = event.get("type")
    
    if event_type == "summary":
        summary = event.get("summary")
        if summary and len(collected_summary) == 0:
            collected_summary.append(summary)
    elif event_type == "chunks":
        chunks = event.get("chunks", [])
        for chunk in chunks:
            chunk_dict = chunk.copy() if isinstance(chunk, dict) else chunk
            # Remove score if present
            chunk_dict.pop("score", None)
            collected_chunks.append(chunk_dict)
            if chunk_dict.get("fileid"):
                collected_fileids.add(chunk_dict["fileid"])
    elif event_type == "file_result":
        # Handle file_result events from agent streaming
        chunks = event.get("chunks", [])
        for chunk in chunks:
            chunk_dict = chunk.copy() if isinstance(chunk, dict) else chunk
            chunk_dict.pop("score", None)
            collected_chunks.append(chunk_dict)
            if chunk_dict.get("fileid"):
                collected_fileids.add(chunk_dict["fileid"])
        # Check for summary in file_result
        summary = event.get("summary")
        if summary and len(collected_summary) == 0:
            collected_summary.append(summary)


def _format_citations_html(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format citations from chunks as HTML-friendly structure."""
    citations = []
    seen_citations = set()  # To avoid duplicates
    
    for chunk in chunks:
        fileid = chunk.get("fileid")
        filename = chunk.get("filename") or fileid
        section_path = chunk.get("section_path", [])
        page_range = chunk.get("page_range", [])
        object_type = chunk.get("object_type", "narrative")
        
        # Create citation key to avoid duplicates
        citation_key = (fileid, tuple(section_path), tuple(page_range) if page_range else None)
        if citation_key in seen_citations:
            continue
        seen_citations.add(citation_key)
        
        # Build citation text
        citation_parts = []
        
        # File name
        if filename:
            citation_parts.append(f"<strong>{filename}</strong>")
        
        # Section path
        if section_path and isinstance(section_path, list) and len(section_path) > 0:
            section_text = " > ".join(str(s) for s in section_path)
            citation_parts.append(f"Section: {section_text}")
        
        # Page range
        if page_range and isinstance(page_range, list) and len(page_range) >= 2:
            if page_range[0] == page_range[1]:
                citation_parts.append(f"Page {page_range[0]}")
            else:
                citation_parts.append(f"Pages {page_range[0]}-{page_range[1]}")
        
        # Object type
        if object_type and object_type != "narrative":
            citation_parts.append(f"Type: {object_type}")
        
        # Build HTML-friendly citation
        citation_html = " | ".join(citation_parts)
        
        citations.append({
            "fileid": fileid,
            "filename": filename,
            "section_path": section_path,
            "page_range": page_range,
            "object_type": object_type,
            "citation_html": citation_html
        })
    
    return citations


def _format_inference_html(inference_text: str, file_id_to_name: Optional[Dict[str, str]] = None) -> str:
    """
    Convert inference text to HTML-friendly format, converting citations to HTML.
    
    Also ensures that file IDs in citations are replaced with filenames if file_id_to_name mapping is provided.
    """
    if file_id_to_name is None:
        file_id_to_name = {}
    
    # First, replace any file IDs in citations with filenames
    citation_pattern = r"\(Citation:\s*Doc=([^,]+),\s*Page=(\d+)\)"
    
    def replace_fileid_in_citation(match):
        doc_ref = match.group(1).strip()
        page = match.group(2).strip()
        # If doc_ref is a file_id, replace with filename
        if doc_ref in file_id_to_name:
            doc_ref = file_id_to_name[doc_ref]
        elif doc_ref.startswith('f_') and len(doc_ref) > 2:
            # Try to find the file_id in our mapping (case-insensitive)
            doc_ref_lower = doc_ref.lower()
            for fid, fname in file_id_to_name.items():
                if fid.lower() == doc_ref_lower:
                    doc_ref = fname
                    break
        return f"(Citation: Doc={doc_ref}, Page={page})"
    
    # Replace file IDs with filenames in citations
    inference_text = re.sub(citation_pattern, replace_fileid_in_citation, inference_text, flags=re.IGNORECASE)
    
    # Escape HTML special characters
    html_text = inference_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Convert citations from (Citation: Doc=..., Page=...) to HTML format
    def replace_citation_html(match):
        doc_name = match.group(1).strip()
        page = match.group(2).strip()
        # Create HTML citation with styling
        return f'<span class="citation" data-doc="{doc_name}" data-page="{page}">(Citation: Doc={doc_name}, Page={page})</span>'
    
    html_text = re.sub(citation_pattern, replace_citation_html, html_text, flags=re.IGNORECASE)
    
    # Convert newlines to <br> tags
    html_text = html_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    
    return html_text


def _build_file_final_response(
    query: str,
    projectid: str,
    fileid: str,
    file_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the final response for a single file in file-specific mode."""
    # Get inference/summary from file_result
    inference_text = file_result.get("summary") or "No inference generated."
    
    # Get chunks from file_result
    chunks = file_result.get("chunks", [])
    
    # Convert chunks to dicts if needed and build file_id_to_name mapping
    chunks_list = []
    file_id_to_name = {}
    for chunk in chunks:
        if hasattr(chunk, '__dict__'):
            chunk_dict = {
                "chunk_ref": getattr(chunk, 'chunk_ref', None),
                "text": getattr(chunk, 'text', ''),
                "fileid": getattr(chunk, 'fileid', fileid),
                "filename": getattr(chunk, 'filename', None),
                "section_path": getattr(chunk, 'section_path', []),
                "object_type": getattr(chunk, 'object_type', None),
                "page_range": getattr(chunk, 'page_range', None),
                "caption": getattr(chunk, 'caption', None),
                "metadata": getattr(chunk, 'metadata', {})
            }
        else:
            chunk_dict = chunk.copy() if isinstance(chunk, dict) else chunk
            chunk_dict.pop("score", None)  # Remove score
        
        # Build file_id_to_name mapping for citation replacement
        chunk_fileid = chunk_dict.get("fileid")
        chunk_filename = chunk_dict.get("filename")
        if chunk_fileid and chunk_filename:
            file_id_to_name[chunk_fileid] = chunk_filename
        
        chunks_list.append(chunk_dict)
    
    # Also add the fileid from file_result if available
    if fileid and file_result.get("filename"):
        file_id_to_name[fileid] = file_result.get("filename")
    
    # Convert inference to HTML-friendly format with file_id_to_name mapping
    inference_html = _format_inference_html(inference_text, file_id_to_name)
    
    # Format citations
    citations = _format_citations_html(chunks_list)
    
    return {
        "type": "final_response",
        "query": query,
        "projectid": projectid,
        "fileid": fileid,
        "filename": file_result.get("filename"),
        "inference": inference_html,
        "citations": citations
    }


def _build_final_response(
    query: str,
    projectid: str,
    fileid: Optional[str],
    collected_fileids: set,
    inference: List[str],
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the final response with inference, citations, and metadata."""
    # Get inference text
    inference_text = inference[0] if inference else "No inference generated."
    
    # Build file_id_to_name mapping from chunks for citation replacement
    file_id_to_name = {}
    for chunk in chunks:
        chunk_fileid = chunk.get("fileid")
        chunk_filename = chunk.get("filename")
        if chunk_fileid and chunk_filename:
            file_id_to_name[chunk_fileid] = chunk_filename
    
    # Convert inference to HTML-friendly format with file_id_to_name mapping
    inference_html = _format_inference_html(inference_text, file_id_to_name)
    
    # Determine fileid(s)
    if fileid:
        response_fileid = fileid
    elif collected_fileids:
        if len(collected_fileids) == 1:
            response_fileid = list(collected_fileids)[0]
        else:
            response_fileid = list(collected_fileids)  # Multiple files
    else:
        response_fileid = None
    
    # Format citations
    citations = _format_citations_html(chunks)
    
    return {
        "type": "final_response",
        "query": query,
        "projectid": projectid,
        "fileid": response_fileid,
        "inference": inference_html,  # HTML-friendly inference
        "citations": citations
    }


async def _log_user_query(log_payload: Dict[str, Any]) -> None:
    """Log user query (non-blocking)."""
    try:
        from app.retriever.models import UserQueryLog
        log_entry = UserQueryLog(**log_payload)
        logger.info(f"[USER_QUERY] {log_entry.model_dump()}")
    except Exception as e:
        logger.warning(f"Failed to log user query: {e}")


async def _log_user_response(log_payload: Dict[str, Any]) -> None:
    """Log user response (non-blocking)."""
    try:
        from app.retriever.models import UserResponseLog
        log_entry = UserResponseLog(**log_payload)
        logger.info(f"[USER_RESPONSE] {log_entry.model_dump()}")
    except Exception as e:
        logger.warning(f"Failed to log user response: {e}")

