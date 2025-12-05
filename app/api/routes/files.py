# app/routes/files.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, Query, Body
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator
import os, uuid
from typing import Optional, Set
from pydantic import BaseModel
import json
import time
import threading

from app.db.mongo import db
from app.utils.storage import save_upload_file, sha256_of_file, ensure_dir
from app.utils.stream import stream_error
from app.lancedb_client import get_lancedb
from app.deps import get_opensearch
from app.utils.create_chunks import create_chunk_page_by_page
from app.utils.docling_converter import convert_document
from app.utils.structure_aware_chunks import create_chunks
from app.utils.lance_client import delete_chunks_by_doc, delete_chunks_by_dataroom

import logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/files", tags=["files"])

# ---------- Request Models ----------
class ParseLatestRequest(BaseModel):
    user_id: str = "temp_user_001"
    do_ocr: bool = False
    force: bool = False

class ParseAllLatestRequest(BaseModel):
    user_id: str = "temp_user_001"
    do_ocr: bool = False
    force: bool = False
    only_status: Optional[str] = None
    limit: int = 1000

class DeleteFileRequest(BaseModel):
    user_id: str = "temp_user_001"

# ---------- helpers ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _load_chunks_from_uri(chunk_uri: str) -> List[Dict[str, Any]]:
    """Load chunks from a chunk_uri file (supports both .json and .jsonl formats)."""
    if not chunk_uri:
        return []
    
    # Resolve path (handle container paths)
    possible_paths = [chunk_uri]
    if chunk_uri.startswith("/app/uploads"):
        # Try host path
        host_path = chunk_uri.replace("/app/uploads", "/home/himanshu-gcp/DataRoom-ai-sheetal/uploads")
        possible_paths.insert(0, host_path)
        # Also try alternative filenames
        if "chunks.json" in host_path and not host_path.endswith(".jsonl"):
            possible_paths.insert(0, host_path.replace("chunks.json", "all_chunks.jsonl"))
            possible_paths.insert(0, host_path.replace("chunks.json", "chunks.jsonl"))
        elif "chunks.jsonl" in host_path:
            possible_paths.insert(0, host_path.replace("chunks.jsonl", "all_chunks.jsonl"))
        elif "all_chunks.jsonl" in host_path:
            possible_paths.insert(0, host_path.replace("all_chunks.jsonl", "chunks.jsonl"))
    
    chunk_file = None
    for path in possible_paths:
        if os.path.exists(path):
            chunk_file = path
            logger.info(f"Found chunk file at: {path}")
            break
    
    if not chunk_file:
        logger.warning(f"Chunk file not found at {chunk_uri} or alternatives. Tried: {possible_paths}")
        return []
    
    chunks = []
    try:
        # Check if file is .json (single JSON array) or .jsonl (line-delimited JSON)
        is_json_array = chunk_file.endswith('.json') and not chunk_file.endswith('.jsonl')
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            if is_json_array:
                # Single JSON array format
                data = json.load(f)
                if isinstance(data, list):
                    # Normalize chunk format to match expected structure
                    for chunk in data:
                        if isinstance(chunk, dict) and "text" in chunk:
                            # Normalize field names and structure
                            normalized_chunk = {
                                "chunk_ref": chunk.get("chunk_ref") or chunk.get("chunk_id") or chunk.get("_id", ""),
                                "text": chunk.get("text", ""),
                                "fileid": chunk.get("fileid") or chunk.get("doc_id", ""),
                                "section_path": chunk.get("section_path", []),
                                "object_type": chunk.get("object_type", "narrative"),
                                "page_range": chunk.get("page_range", []),
                                "caption": chunk.get("caption"),
                                "metadata": chunk.get("metadata", {})
                            }
                            
                            # Handle section_path - convert string to list if needed
                            if isinstance(normalized_chunk["section_path"], str):
                                normalized_chunk["section_path"] = [normalized_chunk["section_path"]] if normalized_chunk["section_path"] else []
                            
                            # Handle page_range - convert page_number to page_range if needed
                            if not normalized_chunk["page_range"] and "page_number" in chunk:
                                page_num = chunk.get("page_number", 1)
                                normalized_chunk["page_range"] = [page_num, page_num]
                            elif not normalized_chunk["page_range"]:
                                normalized_chunk["page_range"] = [1, 1]
                            
                            # Preserve other fields that might be useful
                            for key in ["chunk_id", "doc_id", "file_version_id", "dataroom_id"]:
                                if key in chunk and key not in normalized_chunk:
                                    normalized_chunk[key] = chunk[key]
                            
                            chunks.append(normalized_chunk)
                else:
                    logger.warning(f"Expected JSON array in {chunk_file}, got {type(data)}")
            else:
                # JSONL format (line-delimited JSON)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if isinstance(chunk, dict) and "text" in chunk:
                            # Normalize chunk format
                            normalized_chunk = {
                                "chunk_ref": chunk.get("chunk_ref") or chunk.get("chunk_id") or chunk.get("_id", ""),
                                "text": chunk.get("text", ""),
                                "fileid": chunk.get("fileid") or chunk.get("doc_id", ""),
                                "section_path": chunk.get("section_path", []),
                                "object_type": chunk.get("object_type", "narrative"),
                                "page_range": chunk.get("page_range", []),
                                "caption": chunk.get("caption"),
                                "metadata": chunk.get("metadata", {})
                            }
                            
                            # Handle section_path
                            if isinstance(normalized_chunk["section_path"], str):
                                normalized_chunk["section_path"] = [normalized_chunk["section_path"]] if normalized_chunk["section_path"] else []
                            
                            # Handle page_range
                            if not normalized_chunk["page_range"] and "page_number" in chunk:
                                page_num = chunk.get("page_number", 1)
                                normalized_chunk["page_range"] = [page_num, page_num]
                            elif not normalized_chunk["page_range"]:
                                normalized_chunk["page_range"] = [1, 1]
                            
                            chunks.append(normalized_chunk)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chunk line: {e}")
                        continue
        
        logger.info(f"Successfully loaded {len(chunks)} chunks from {chunk_file}")
    except Exception as e:
        logger.error(f"Error loading chunks from {chunk_file}: {e}", exc_info=True)
    
    return chunks


def _get_latest_version_doc(project_id: str, file_id: str) -> Optional[Dict[str, Any]]:
    return db.file_versions.find_one(
        {"project_id": project_id, "file_id": file_id},
        sort=[("version", -1)]
    )


def _trigger_retriever_sync_all(project_id: str) -> None:
    """
    After parsing completes for a project, trigger background sync of chunks
    to the retriever service so any backend client automatically benefits.
    
    Since we're now integrated, we directly sync chunks from the parsed files
    instead of making HTTP calls.
    """
    try:
        from app.retriever.chunking_integration import get_integration
        from app.retriever.hybrid_client import HybridClient
        
        logger.info(f"Auto-syncing chunks to retriever for project {project_id}")
        
        # Use internal integration to sync
        hybrid_client = HybridClient()
        integration = get_integration(client=hybrid_client, auth_token=None)
        
        # Get all files in the project that have been parsed
        files_cur = db.files.find({"project_id": project_id, "deleted_at": None}, {"_id": 1})
        file_ids = [f["_id"] for f in files_cur]
        
        if not file_ids:
            logger.info(f"No files found for project {project_id}")
            return
        
        ingested_count = 0
        errors = []
        
        # Process each file
        for file_id in file_ids:
            try:
                # Get latest version doc
                vdoc = _get_latest_version_doc(project_id, file_id)
                if not vdoc:
                    logger.debug(f"No version doc found for file {file_id}")
                    continue
                
                file_status = vdoc.get("status", "unknown")
                if file_status != "parsed":
                    logger.debug(f"File {file_id} status is {file_status}, not parsed. Skipping.")
                    continue
                
                chunk_uri = (vdoc.get("storage") or {}).get("chunk_uri")
                if not chunk_uri:
                    logger.warning(f"No chunk_uri for file {file_id}, skipping sync")
                    continue
                
                version = vdoc.get("version", 1)
                file_version_id = f"v_{file_id}_{version}"
                
                # Load and ingest chunks directly (no HTTP call needed)
                logger.info(f"Syncing chunks for file {file_id} from {chunk_uri}")
                ingest_result = integration.ingest_chunks_from_chunking_service(
                    project_id=project_id,
                    file_id=file_id,
                    file_version_id=file_version_id,
                    chunk_uri=chunk_uri
                )
                
                if ingest_result.get("success"):
                    indexed = ingest_result.get("indexed", 0)
                    ingested_count += indexed
                    logger.info(f"Successfully synced {indexed} chunks for file {file_id}")
                else:
                    error_msg = ingest_result.get("message", "Unknown error")
                    errors.append(f"File {file_id}: {error_msg}")
                    logger.warning(f"Failed to sync chunks for file {file_id}: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                errors.append(f"File {file_id}: {error_msg}")
                logger.error(f"Error syncing file {file_id}: {e}", exc_info=True)
        
        logger.info(
            f"Auto-sync to retriever completed for project {project_id}, "
            f"ingested_chunks={ingested_count}, errors={len(errors)}"
        )
        if errors:
            logger.warning(f"Sync errors: {errors[:5]}")  # Log first 5 errors
        
    except Exception as e:
        logger.warning(
            f"Failed to auto-sync chunks to retriever for project {project_id}: {e}",
            exc_info=True
        )

def _run_parse_for_version(project_id: str, file_id: str, do_ocr: bool, version_doc: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Run Docling for a specific version_doc. Updates DB and returns a small result dict with actual chunks."""
    vid = version_doc["_id"]
    status = version_doc.get("status")
    raw_uri = (version_doc.get("storage") or {}).get("raw_uri")
    logger.info(f"Starting parse for file_id={file_id}, version={version_doc['version']}, vid={vid}, status={status}, raw_uri={raw_uri}")

    if not raw_uri or not os.path.isfile(raw_uri):
        raise HTTPException(status_code=400, detail=f"Missing raw file on disk for version {vid}")

    if status == "parsed" and not force:
        # File already parsed - get chunk_uri from version document
        chunk_uri = (version_doc.get("storage") or {}).get("chunk_uri")
        
        # Load existing chunks from chunk_uri
        chunks = []
        if chunk_uri:
            try:
                chunks = _load_chunks_from_uri(chunk_uri)
                logger.info(f"Loaded {len(chunks)} chunks from existing chunk_uri: {chunk_uri}")
            except Exception as e:
                logger.warning(f"Failed to load chunks from {chunk_uri}: {e}")
                # Continue with empty chunks if loading fails
        
        return {
            "file_id": file_id,
            "version": version_doc["version"],
            "status": "parsed",
            "skipped": True,
            "reason": "already parsed",
            "chunk_uri": chunk_uri,  # Include chunk_uri even for skipped files
            "chunks": chunks,  # Load existing chunks
            "chunks_count": len(chunks)
        }

    # mark 'parsing'
    now = _now_iso()
    db.file_versions.update_one(
        {"_id": vid},
        {"$set": {"status": "parsing", "updated_at": now, "error": None}}
    )

    # derived dir
    base_dir = "/app/uploads"
    out_dir = os.path.join(base_dir, "projects", project_id, "derived", file_id, str(version_doc["version"]))
    ensure_dir(out_dir)
    
    # run docling
    conv_result = convert_document(raw_uri, do_ocr)
    chunks, chunk_uri = create_chunks(
        conv_result,
        doc_id=file_id,
        dataroom_id=project_id,
        file_version_id=vid,
        max_tokens=512,
        output_dir=out_dir
    )

    # commit 'parsed'
    now2 = _now_iso()
    db.file_versions.update_one(
        {"_id": vid},
        {"$set": {
            "status": "parsed",
            "storage.chunk_uri": chunk_uri,
            "chunks_count": len(chunks),
            "updated_at": now2
        }}
    )

    return {
        "ok": True,
        "file_id": file_id,
        "version": int(version_doc["version"]),
        "status": "parsed",
        "chunk_uri": chunk_uri,
        "chunks_count": len(chunks),
        "chunks": chunks,  # Include actual chunk data
        "updated_at": now2
    }

def _stream_parse_progress(
    project_id: str,
    file_ids: List[str],
    do_ocr: bool,
    force: bool,
    only_status: Optional[str]
) -> Generator[str, None, None]:
    """
    Generator that yields Server-Sent Events for each file processing result.
    SSE format: data: {json}\n\n
    
    Includes:
    - Current status messages every 5 seconds (GUARANTEED, even during long parsing)
    - Progress updates for each file
    - Final complete message with all chunks
    """
    
    processed = 0
    skipped = 0
    failed = 0
    total = len(file_ids)
    all_chunks = []  # Collect all chunks from all files
    
    # Track time for current status
    start_time = time.time()
    last_current_status = start_time
    CURRENT_STATUS_INTERVAL = 5  # seconds
    current_status_count = 0

    # Send initial status
    yield f"data: {json.dumps({'type': 'init', 'project_id': project_id, 'total_files': total, 'force': force, 'filter_status': only_status, 'started_at': _now_iso()})}\n\n"
    
    # Helper function to send current status if needed
    def send_current_status_if_needed(idx, fid=None, force_send=False):
        nonlocal last_current_status, current_status_count
        current_time = time.time()
        if force_send or (current_time - last_current_status >= CURRENT_STATUS_INTERVAL):
            current_status_count += 1
            elapsed_total = int(current_time - start_time)
            msg = {
                'type': 'current_status',
                'current_status_number': current_status_count,
                'timestamp': _now_iso(),
                'current_file': idx,
                'total_files': total,
                'elapsed_total_seconds': elapsed_total,
                'message': 'PROCESSING...'
            }
            if fid:
                msg['processing_file_id'] = fid
            
            last_current_status = current_time
            return f"data: {json.dumps(msg)}\n\n"
        return None

    for idx, fid in enumerate(file_ids, 1):
        # Send current status before each file
        cs = send_current_status_if_needed(idx, fid)
        if cs:
            yield cs

        # Send progress update
        yield f"data: {json.dumps({'type': 'progress', 'current': idx, 'total': total, 'file_id': fid, 'status': 'processing'})}\n\n"

        vdoc = _get_latest_version_doc(project_id, fid)
        if not vdoc:
            result = {"file_id": fid, "skipped": True, "reason": "no versions", "chunks": []}
            skipped += 1
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"
            continue

        # Only apply status filter if force is False
        if not force and only_status and vdoc.get("status") != only_status:
            result = {"file_id": fid, "skipped": True, "reason": f"status != {only_status}", "chunks": []}
            skipped += 1
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"
            continue

        try:
            # IMPORTANT: For long-running parsing, we need to send current status DURING processing
            # We'll use a thread to run parsing and monitor it
            
            parse_result = {}
            parse_error = {}
            parse_complete = threading.Event()
            
            def run_parse():
                try:
                    parse_result['data'] = _run_parse_for_version(project_id, fid, do_ocr, vdoc, force=force)
                except Exception as e:
                    parse_error['error'] = e
                finally:
                    parse_complete.set()
            
            # Start parsing in background thread
            parse_thread = threading.Thread(target=run_parse, daemon=True)
            parse_thread.start()
            
            # Monitor progress and send current status while parsing runs
            while not parse_complete.is_set():
                # Check if current status needed
                cs = send_current_status_if_needed(idx, fid)
                if cs:
                    yield cs
                
                # Wait a bit before checking again (don't spin CPU)
                parse_complete.wait(timeout=1.0)
            
            # Wait for thread to finish
            parse_thread.join(timeout=1.0)
            
            # Check if there was an error
            if 'error' in parse_error:
                raise parse_error['error']
            
            # Get the result
            res = parse_result.get('data', {})
            
            # Send one more current status after parsing (if needed)
            cs = send_current_status_if_needed(idx, fid)
            if cs:
                yield cs
            
            # Collect chunks from this file
            file_chunks = res.get("chunks", [])
            all_chunks.extend(file_chunks)
            
            if res.get("ok"):
                processed += 1
            elif res.get("skipped"):
                skipped += 1
            
            # Send result with chunks included
            yield f"data: {json.dumps({'type': 'result', **res})}\n\n"
            
        except HTTPException as he:
            failed += 1
            result = {"file_id": fid, "error": he.detail, "failed": True, "chunks": []}
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"
        except Exception as e:
            failed += 1
            result = {"file_id": fid, "error": str(e), "failed": True, "chunks": []}
            logger.exception(f"Error parsing file {fid}")
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"

    # Send final current status before complete
    cs = send_current_status_if_needed(total)
    if cs:
        yield cs

    # Send final summary with all chunks
    total_elapsed = int(time.time() - start_time)
    summary = {
        'type': 'complete',
        'summary': {
            'processed': processed,
            'skipped': skipped,
            'failed': failed,
            'total': total,
            'total_chunks': len(all_chunks),
            'total_elapsed_seconds': total_elapsed,
            'current_status_sent': current_status_count
        },
        'chunks': all_chunks  # All chunks from all processed files
    }
    yield f"data: {json.dumps(summary)}\n\n"

    # NOTE: Auto-sync has been disabled.
    # After parsing completes, call POST /chunks/sync-all/{project_id} manually to sync chunks.


@router.get("/parse-all-latest")
def parse_all_latest(
    project_id: str,
    request_body: ParseAllLatestRequest = Query(...),
):
    """
    For each active file in the project:
      - pick its LATEST version
      - parse it with Docling (unless already parsed and force=False)
    Optional filter: only process versions whose current status == only_status.
    
    Returns a streaming Server-Sent Events (SSE) response with real-time progress updates.
    Each event is in the format: data: {json}\n\n
    
    Event types:
    - init: Initial message with total file count
    - current_status: Sent every 5 seconds during processing
    - progress: Progress update for each file being processed
    - result: Result after processing each file (ok, skipped, or failed) - includes chunks
    - complete: Final summary with totals and ALL chunks from all files
    """
    # list active files in project
    files_cur = db.files.find({"project_id": project_id, "deleted_at": None}, {"_id": 1})
    file_ids = [f["_id"] for f in files_cur][:max(1, request_body.limit)]

    return StreamingResponse(
        _stream_parse_progress(
            project_id=project_id,
            file_ids=file_ids,
            do_ocr=request_body.do_ocr,
            force=request_body.force,
            only_status=request_body.only_status
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


#---------- list (handy for checking) ----------
@router.get("", status_code=200)
def list_files(
    project_id: str,
    user_id: str = Query(default="temp_user_001", description="User ID")
):
    cur = db.files.find({"project_id": project_id, "deleted_at": None})
    return [{**f, "_id": str(f["_id"])} for f in cur]

# ---------- SINGLE-FILE upload (Swagger-friendly) ----------
@router.post("/upload-single", status_code=201, summary="Upload Single File (Swagger-friendly)")
async def upload_single_file(
    project_id: str,
    file: UploadFile = File(..., description="File to upload"),
    replace: bool = Form(False, description="Replace existing file with same name"),
    user_id: str = Form(default="temp_user_001", description="User ID"),
):
    """
    Upload a single file. **Use this endpoint in Swagger UI**.
    
    For multiple files, call this endpoint multiple times.
    
    - If replace=False (default): create NEW file_id (counts toward 20-file cap)
    - If replace=True: reuse existing file_id by filename (bump version), DOES NOT count toward cap
    
    Returns file information including file_id and raw_uri.
    """
    # Project must exist
    prj = db.projects.find_one({"_id": project_id}, {"_id": 1})
    if not prj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file limit
    active_count = db.files.count_documents({"project_id": project_id, "deleted_at": None})
    
    original_name = file.filename
    existing = db.files.find_one(
        {"project_id": project_id, "filename": original_name, "deleted_at": None},
        {"_id": 1}
    )
    
    # If not replacing and would exceed limit
    if not existing and not replace and active_count >= 20:
        raise HTTPException(
            status_code=409,
            detail=f"Project has {active_count} files. Limit is 20. Delete some files or use replace=true for existing files."
        )
    
    try:
        result = await _process_single_file(project_id, file, replace)
        return result
    except Exception as e:
        logger.exception(f"Error uploading file {original_name}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- MULTI-FILE upload / replace ----------
from fastapi import File as FastAPIFile

@router.post("", status_code=201, summary="Upload Multiple Files")
async def upload_or_replace_files(
    project_id: str,
    files: List[UploadFile] = FastAPIFile(..., description="Select multiple files to upload"),
    replace: bool = Form(False, description="Replace existing files with same names"),
    user_id: str = Form(default="temp_user_001", description="User ID"),
):
    """
    Upload multiple files at once.
    
    **Swagger UI Instructions**:
    1. Click "Try it out"
    2. In the 'files' field, click "Choose Files" 
    3. Hold Ctrl/Cmd and select multiple files
    4. Or select files one by one if prompted
    5. Click "Execute"
    
    - If replace=False (default): create NEW file_ids (each counts toward the 20-file cap)
    - If replace=True: reuse existing file_ids by filename (bump versions), DOES NOT count toward cap
    
    Returns a list of results, one per file.
    
    ### Example with cURL:
    ```bash
    curl -X POST "http://localhost:8000/projects/{project_id}/files" \\
      -F "files=@document1.pdf" \\
      -F "files=@document2.pdf" \\
      -F "user_id=temp_user_001" \\
      -F "replace=false"
    ```
    """
    # 0) Project must exist
    prj = db.projects.find_one({"_id": project_id}, {"_id": 1})
    if not prj:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if no files were uploaded
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # 1) Check current file count ONCE at the start
    active_count = db.files.count_documents({"project_id": project_id, "deleted_at": None})
    
    # 2) Count how many NEW files will be created (not replacements)
    new_files_count = 0
    for file in files:
        original_name = file.filename or "unnamed"
        existing = db.files.find_one(
            {"project_id": project_id, "filename": original_name, "deleted_at": None},
            {"_id": 1}
        )
        if not existing:
            new_files_count += 1

    # 3) Enforce 20-file limit
    if active_count + new_files_count > 20:
        raise HTTPException(
            status_code=409,
            detail=f"Project has {active_count} files. Adding {new_files_count} new files would exceed the limit of 20. Delete some files or use replace=true for existing files."
        )

    # 4) Process each file
    results = []
    for file in files:
        try:
            result = await _process_single_file(project_id, file, replace)
            results.append(result)
        except Exception as e:
            # Include error for this file but continue processing others
            results.append({
                "filename": file.filename or "unnamed",
                "success": False,
                "error": str(e)
            })

    return {
        "uploaded": len([r for r in results if r.get("success", True)]),
        "failed": len([r for r in results if not r.get("success", True)]),
        "results": results
    }


async def _process_single_file(project_id: str, file: UploadFile, replace: bool) -> dict:
    """
    Process a single file upload. Returns a dict with file info.
    """
    original_name = file.filename or "unnamed"
    mime = file.content_type or "application/octet-stream"

    # 1) If replace=True, try to reuse existing file by filename
    existing = None
    if replace:
        existing = db.files.find_one(
            {"project_id": project_id, "filename": original_name, "deleted_at": None},
            {"_id": 1}
        )

    # 2) Decide file_id + next version
    if existing:
        file_id = existing["_id"]
        latest = db.file_versions.find_one(
            {"file_id": file_id},
            sort=[("version", -1)],
            projection={"version": 1}
        )
        next_version = (latest["version"] + 1) if latest else 1
    else:
        file_id = _new_id("f")
        next_version = 1
        now = _now_iso()
        db.files.insert_one({
            "_id": file_id,
            "project_id": project_id,
            "filename": original_name,
            "mime": mime,
            "size": 0,
            "checksum": None,
            "deleted_at": None,
            "created_at": now,
            "updated_at": now,
        })

    version_id = f"v_{file_id}_{next_version}"

    # 3) Persist raw file → /app/uploads/projects/<pid>/raw/<file_id>/<version>/<filename>
    base_dir = "/app/uploads"
    raw_dir = os.path.join(base_dir, "projects", project_id, "raw", file_id, str(next_version))
    ensure_dir(raw_dir)
    dest_path = os.path.join(raw_dir, original_name)

    size_bytes, abs_path = save_upload_file(file, dest_path)
    checksum = sha256_of_file(file.file)

    # 4) Update file doc (size/mime/checksum)
    now = _now_iso()
    db.files.update_one(
        {"_id": file_id},
        {"$set": {"mime": mime, "size": size_bytes, "checksum": checksum, "updated_at": now}},
        upsert=True,
    )

    # 5) Create file_version(status='queued')
    db.file_versions.insert_one({
        "_id": version_id,
        "file_id": file_id,
        "project_id": project_id,
        "version": next_version,
        "status": "queued",
        "storage": {
            "raw_uri": abs_path,
            "docling_json_uri": None,
            "md_uri": None,
            "chunks_uri": None
        },
        "error": None,
        "created_at": now,
        "updated_at": now
    })

    return {
        "success": True,
        "file_id": file_id,
        "version_id": version_id,
        "version": next_version,
        "filename": original_name,
        "size": size_bytes,
        "checksum": checksum,
        "status": "queued",
        "raw_uri": abs_path,
    }


@router.post("/{file_id}/parse_latest", status_code=200)
def parse_latest_version(
    project_id: str,
    file_id: str,
    request_body: ParseLatestRequest = Body(...),
):
    """
    Parse the latest version of a file and return the full result including chunk_uri.
    
    Returns:
        Full result dict with file_id, version, status, chunk_uri, chunks, etc.
    """
    vdoc = _get_latest_version_doc(project_id, file_id)
    if not vdoc:
        raise HTTPException(status_code=404, detail="No versions found for this file")
    
    result = _run_parse_for_version(project_id, file_id, request_body.do_ocr, vdoc, force=request_body.force)
    
    # Return the full result (not just chunks) so chunk_uri is available
    return result

@router.delete("/{file_id}", status_code=200)
async def hard_delete_file(
    request: Request,
    project_id: str,
    file_id: str,
    user_id: str = Query(..., description="User ID performing the deletion"),
):
    """
    Hard delete a file and all its versions/chunks from MongoDB, OpenSearch, and LanceDB.
    """
    f = db.files.find_one({"_id": file_id, "project_id": project_id})
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    # Gather versions
    versions = list(db.file_versions.find({"file_id": file_id}, {"_id": 1, "version": 1}))
    version_ids = [v["_id"] for v in versions]

    # Delete Mongo chunks
    chunks_res = db.chunks.delete_many({"file_version_id": {"$in": version_ids}})
    chunks_deleted = int(chunks_res.deleted_count or 0)

    # Delete from OpenSearch (best-effort)
    os_deleted = 0
    try:
        os_client = get_opensearch()
        if version_ids:
            resp = os_client.delete_by_query(
                index="chunks",
                body={"query": {"terms": {"file_version_id": version_ids}}},
                refresh=True,
                conflicts="proceed",
            )
            os_deleted = int(resp.get("deleted", 0) or 0)
    except Exception:
        os_deleted = -1

    # Delete from LanceDB (best-effort)
    lance_deleted = 0
    try:
        lance_deleted = delete_chunks_by_doc(file_id)
    except Exception as e:
        logger.error(f"Error deleting from LanceDB: {e}")
        lance_deleted = -1

    # Delete versions + file
    fv_res = db.file_versions.delete_many({"file_id": file_id})
    file_res = db.files.delete_one({"_id": file_id})
    versions_deleted = int(fv_res.deleted_count or 0)
    file_deleted = int(file_res.deleted_count or 0)

    # Delete blobs from disk
    base_dir = "/app/uploads"
    file_root = os.path.join(base_dir, "projects", project_id, "raw", file_id)

    try:
        import shutil
        shutil.rmtree(file_root, ignore_errors=True)
    except Exception:
        pass

    # Write audit
    client_ip = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None

    audit_doc = {
        "_id": _new_id("a"),
        "project_id": project_id,
        "action": "file_deleted",
        "actor_user_id": user_id,
        "file_id": file_id,
        "version_ids": version_ids,
        "counts": {
            "chunks": chunks_deleted,
            "versions": versions_deleted,
            "file_docs": file_deleted,
            "os_docs": os_deleted,
            "lance_rows": lance_deleted
        },
        "blobs_root_deleted": file_root,
        "client": {"ip": client_ip, "user_agent": user_agent},
        "ts": _now_iso(),
    }
    db.audit.insert_one(audit_doc)

    return {
        "ok": True,
        "file_id": file_id,
        "versions_deleted": versions_deleted,
        "chunks_deleted": chunks_deleted,
        "os_docs_deleted": os_deleted,
        "lance_rows_deleted": lance_deleted,
        "uploads_deleted": file_root,
        "audit_id": audit_doc["_id"],
    }