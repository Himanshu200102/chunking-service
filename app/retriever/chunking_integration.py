"""Integration service to receive chunks from the chunking service."""
import logging
import json
import os
import glob
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
import httpx

from app.retriever.opensearch_client import OpenSearchClient
from app.retriever.hybrid_client import HybridClient

logger = logging.getLogger(__name__)


class ChunkingWebhookRequest(BaseModel):
    """Webhook request from chunking service when chunks are ready."""
    project_id: str = Field(..., description="Project identifier")
    file_id: str = Field(..., description="File identifier")
    file_version_id: str = Field(..., description="File version identifier")
    chunk_uri: str = Field(..., description="Path to JSONL file containing chunks")
    status: str = Field(..., description="Status: 'parsed' means chunks are ready")
    
    class Config:
        populate_by_name = True


class ChunkingIntegration:
    """Service to integrate with the chunking container."""
    
    def __init__(self, client, auth_token: Optional[str] = None):
        """
        Initialize with either HybridClient or OpenSearchClient for backward compatibility.
        
        Args:
            client: HybridClient or OpenSearchClient instance
            auth_token: Optional auth token for chunking service
        """
        self.client = client
        # For backward compatibility, support both HybridClient and OpenSearchClient
        if isinstance(client, HybridClient):
            self.opensearch_client = client.opensearch_client
        else:
            self.opensearch_client = client
        # Use port 8002 for both internal and external access (unified port configuration)
        self.chunking_service_url = os.getenv("CHUNKING_SERVICE_URL", "http://localhost:8002")
        self.auth_token = auth_token or os.getenv("CHUNKING_SERVICE_AUTH_TOKEN", None)
    
    def load_chunks_from_jsonl(self, chunk_uri: str) -> List[Dict[str, Any]]:
        """Load chunks from a JSONL file."""
        try:
            # Handle paths that might be in chunking container
            # If path starts with /app/uploads, try to find it in shared volume or construct alternative path
            original_path = chunk_uri
            
            # Try multiple possible paths
            possible_paths = [chunk_uri]
            
            # If path is from chunking container (/app/uploads), try local alternatives
            if chunk_uri.startswith("/app/uploads"):
                # Priority: Host path first (most likely to work)
                host_path = chunk_uri.replace("/app/uploads", "/home/himanshu-gcp/DataRoom-ai-sheetal/uploads")
                possible_paths.insert(0, host_path)  # Insert at beginning for priority
                # Also try with all_chunks.jsonl if it's chunks.jsonl (or vice versa)
                # Also try .jsonl variants if it's .json (or vice versa)
                if "chunks.jsonl" in host_path:
                    possible_paths.insert(0, host_path.replace("chunks.jsonl", "all_chunks.jsonl"))
                elif "all_chunks.jsonl" in host_path:
                    possible_paths.insert(0, host_path.replace("all_chunks.jsonl", "chunks.jsonl"))
                elif "chunks.json" in host_path and not host_path.endswith(".jsonl"):
                    # Try .jsonl variants
                    possible_paths.insert(0, host_path.replace("chunks.json", "chunks.jsonl"))
                    possible_paths.insert(0, host_path.replace("chunks.json", "all_chunks.jsonl"))
                # Try the same path (if volumes are shared)
                possible_paths.append(chunk_uri)
                # Try relative to current directory
                relative_path = chunk_uri.replace("/app/uploads", "./uploads")
                possible_paths.append(relative_path)
            
            chunk_uri_to_use = None
            for path in possible_paths:
                if os.path.exists(path):
                    chunk_uri_to_use = path
                    logger.info(f"Found chunk file at: {path} (original: {original_path})")
                    break
            
            if not chunk_uri_to_use:
                # Last resort: try to find the file by searching
                logger.warning(f"Chunk file not found at {original_path}, searching...")
                # Extract project_id and file_id from path if possible
                # Format: /app/uploads/projects/{project_id}/derived/{file_id}/{version}/chunks.jsonl
                parts = original_path.split("/")
                if "projects" in parts and "derived" in parts:
                    proj_idx = parts.index("projects")
                    if proj_idx + 1 < len(parts):
                        project_id = parts[proj_idx + 1]
                        if "derived" in parts:
                            derived_idx = parts.index("derived")
                            if derived_idx + 1 < len(parts):
                                file_id = parts[derived_idx + 1]
                                # Search in possible locations
                                search_paths = [
                                    f"/home/himanshu-gcp/DataRoom-ai-sheetal/uploads/projects/{project_id}/derived/{file_id}",
                                    f"./uploads/projects/{project_id}/derived/{file_id}",
                                    f"/app/uploads/projects/{project_id}/derived/{file_id}"
                                ]
                                for search_base in search_paths:
                                    if os.path.exists(search_base):
                                        # Find chunks.jsonl or all_chunks.jsonl files
                                        found_files = glob.glob(f"{search_base}/*/chunks.jsonl")
                                        found_files.extend(glob.glob(f"{search_base}/*/all_chunks.jsonl"))
                                        if found_files:
                                            chunk_uri_to_use = sorted(found_files)[-1]  # Use latest version
                                            logger.info(f"Found chunk file by searching: {chunk_uri_to_use}")
                                            break
            
            if not chunk_uri_to_use:
                raise FileNotFoundError(f"Chunk file not found at {original_path} or any alternative paths")
            
            chunks = []
            # Check if file is .json (single JSON array) or .jsonl (line-delimited JSON)
            is_json_array = chunk_uri_to_use.endswith('.json') and not chunk_uri_to_use.endswith('.jsonl')
            
            with open(chunk_uri_to_use, 'r', encoding='utf-8') as f:
                if is_json_array:
                    # Single JSON array format
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            # It's an array of chunks
                            for chunk in data:
                                if isinstance(chunk, dict) and "text" in chunk:
                                    # Handle both chunk_ref and chunk_id formats
                                    chunk_ref = chunk.get("chunk_ref") or chunk.get("chunk_id") or chunk.get("_id")
                                    if not chunk_ref:
                                        logger.warning(f"Skipping chunk: no chunk_ref/chunk_id/_id found")
                                        continue
                                    # Normalize section_path - ensure it's a list
                                    section_path = chunk.get("section_path", [])
                                    if isinstance(section_path, str):
                                        # If it's a string, convert to list
                                        if section_path.strip():
                                            section_path = [section_path]  # Wrap single string in list
                                        else:
                                            section_path = []  # Empty string becomes empty list
                                    elif not isinstance(section_path, list):
                                        section_path = []  # Ensure it's a list
                                    
                                    # Normalize page_range - ensure it's a list of 2 integers
                                    page_range = chunk.get("page_range", [1, 1])
                                    if not isinstance(page_range, list) or len(page_range) != 2:
                                        page_range = [1, 1]  # Default to [1, 1]
                                    
                                    normalized_chunk = {
                                        "chunk_ref": chunk_ref,
                                        "text": chunk["text"],
                                        "section_path": section_path,
                                        "object_type": chunk.get("object_type", "narrative"),
                                        "page_range": page_range,
                                        "caption": chunk.get("caption"),
                                        "metadata": chunk.get("metadata", {})
                                    }
                                    # Preserve other fields that might be useful
                                    if "chunk_id" in chunk:
                                        normalized_chunk["chunk_id"] = chunk["chunk_id"]
                                    if "doc_id" in chunk:
                                        normalized_chunk["doc_id"] = chunk["doc_id"]
                                    if "file_version_id" in chunk:
                                        normalized_chunk["file_version_id"] = chunk["file_version_id"]
                                    # IMPORTANT: Preserve embedding if it exists (from parsing)
                                    if "embedding" in chunk and chunk["embedding"] is not None:
                                        normalized_chunk["embedding"] = chunk["embedding"]
                                    chunks.append(normalized_chunk)
                                else:
                                    logger.warning(f"Skipping invalid chunk: missing text field")
                        else:
                            logger.error(f"Expected JSON array, got {type(data)}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON file: {e}")
                        raise
                else:
                    # JSONL format (line-delimited JSON)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            # Ensure all required fields are present
                            if "text" in chunk:
                                # Handle both chunk_ref and chunk_id formats
                                chunk_ref = chunk.get("chunk_ref") or chunk.get("chunk_id") or chunk.get("_id")
                                if not chunk_ref:
                                    logger.warning(f"Skipping chunk: no chunk_ref/chunk_id/_id found")
                                    continue
                                # Normalize section_path - ensure it's a list
                                section_path = chunk.get("section_path", [])
                                if isinstance(section_path, str):
                                    # If it's a string, convert to list (split by common delimiters or wrap in list)
                                    if section_path.strip():
                                        section_path = [section_path]  # Wrap single string in list
                                    else:
                                        section_path = []  # Empty string becomes empty list
                                elif not isinstance(section_path, list):
                                    section_path = []  # Ensure it's a list
                                
                                # Normalize page_range - ensure it's a list of 2 integers
                                page_range = chunk.get("page_range", [1, 1])
                                if not isinstance(page_range, list) or len(page_range) != 2:
                                    page_range = [1, 1]  # Default to [1, 1]
                                
                                # Normalize chunk format
                                normalized_chunk = {
                                    "chunk_ref": chunk_ref,
                                    "text": chunk["text"],
                                    "section_path": section_path,
                                    "object_type": chunk.get("object_type", "narrative"),
                                    "page_range": page_range,
                                    "caption": chunk.get("caption"),
                                    "metadata": chunk.get("metadata", {})
                                }
                                # Preserve other fields that might be useful
                                if "chunk_id" in chunk:
                                    normalized_chunk["chunk_id"] = chunk["chunk_id"]
                                if "doc_id" in chunk:
                                    normalized_chunk["doc_id"] = chunk["doc_id"]
                                if "file_version_id" in chunk:
                                    normalized_chunk["file_version_id"] = chunk["file_version_id"]
                                # IMPORTANT: Preserve embedding if it exists (from parsing)
                                if "embedding" in chunk and chunk["embedding"] is not None:
                                    normalized_chunk["embedding"] = chunk["embedding"]
                                chunks.append(normalized_chunk)
                            else:
                                logger.warning(f"Skipping invalid chunk: missing text field")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse chunk line: {e}")
                            continue
            
            logger.info(f"Loaded {len(chunks)} chunks from {chunk_uri}")
            return chunks
        except FileNotFoundError:
            logger.error(f"Chunk file not found: {chunk_uri}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk file not found: {chunk_uri}"
            )
        except Exception as e:
            logger.error(f"Error loading chunks from {chunk_uri}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load chunks: {str(e)}"
            )
    
    def _resolve_chunk_uri(self, chunk_uri: str) -> Optional[str]:
        """Resolve chunk_uri from container path to host path, handling both chunks.jsonl, all_chunks.jsonl, and chunks.json."""
        original_path = chunk_uri
        
        # Try multiple possible paths
        possible_paths = [chunk_uri]
        
        # If path is from chunking container (/app/uploads), try local alternatives
        if chunk_uri.startswith("/app/uploads"):
            # Priority: Host path first (most likely to work)
            host_path = chunk_uri.replace("/app/uploads", "/home/himanshu-gcp/DataRoom-ai-sheetal/uploads")
            possible_paths.insert(0, host_path)
            # Also try with all_chunks.jsonl if it's chunks.jsonl (or vice versa)
            # Also try .jsonl variants if it's .json (or vice versa)
            if "chunks.jsonl" in host_path:
                possible_paths.insert(0, host_path.replace("chunks.jsonl", "all_chunks.jsonl"))
            elif "all_chunks.jsonl" in host_path:
                possible_paths.insert(0, host_path.replace("all_chunks.jsonl", "chunks.jsonl"))
            elif "chunks.json" in host_path and not host_path.endswith(".jsonl"):
                # Try .jsonl variants
                possible_paths.insert(0, host_path.replace("chunks.json", "chunks.jsonl"))
                possible_paths.insert(0, host_path.replace("chunks.json", "all_chunks.jsonl"))
            # Try the same path (if volumes are shared)
            possible_paths.append(chunk_uri)
            # Try relative to current directory
            relative_path = chunk_uri.replace("/app/uploads", "./uploads")
            possible_paths.append(relative_path)
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Resolved chunk_uri: {original_path} -> {path}")
                return path
        
        # If not found, try searching by extracting project_id and file_id
        parts = original_path.split("/")
        if "projects" in parts and "derived" in parts:
            proj_idx = parts.index("projects")
            if proj_idx + 1 < len(parts):
                project_id = parts[proj_idx + 1]
                if "derived" in parts:
                    derived_idx = parts.index("derived")
                    if derived_idx + 1 < len(parts):
                        file_id = parts[derived_idx + 1]
                        # Search in possible locations
                        search_paths = [
                            f"/home/himanshu-gcp/DataRoom-ai-sheetal/uploads/projects/{project_id}/derived/{file_id}",
                            f"./uploads/projects/{project_id}/derived/{file_id}",
                            f"/app/uploads/projects/{project_id}/derived/{file_id}"
                        ]
                        for search_base in search_paths:
                            if os.path.exists(search_base):
                                # Find chunks.jsonl or all_chunks.jsonl files
                                found_files = glob.glob(f"{search_base}/*/chunks.jsonl")
                                found_files.extend(glob.glob(f"{search_base}/*/all_chunks.jsonl"))
                                if found_files:
                                    resolved_path = sorted(found_files)[-1]  # Use latest version
                                    logger.info(f"Resolved chunk_uri by searching: {original_path} -> {resolved_path}")
                                    return resolved_path
        
        return None
    
    def ingest_chunks_from_chunking_service(
        self,
        project_id: str,
        file_id: str,
        file_version_id: str,
        chunk_uri: str
    ) -> Dict[str, Any]:
        """Ingest chunks from chunking service into OpenSearch."""
        try:
            # Resolve chunk_uri to actual accessible path
            resolved_chunk_uri = self._resolve_chunk_uri(chunk_uri)
            if not resolved_chunk_uri:
                return {
                    "success": False,
                    "message": f"Chunk file not found at: {chunk_uri} (tried multiple path variations)",
                    "indexed": 0
                }
            
            # Check file size
            file_size = os.path.getsize(resolved_chunk_uri)
            if file_size == 0:
                return {
                    "success": False,
                    "message": f"Chunk file is empty (0 bytes). The file may not have been parsed yet. Try 'Force re-parse' to regenerate chunks: {resolved_chunk_uri}",
                    "indexed": 0
                }
            
            # Load chunks from JSONL file (use resolved path)
            chunks = self.load_chunks_from_jsonl(resolved_chunk_uri)
            
            if not chunks:
                # Check if file has any content at all
                try:
                    with open(resolved_chunk_uri, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if not first_line:
                            return {
                                "success": False,
                                "message": f"Chunk file exists but is empty or contains only whitespace. Try 'Force re-parse' to regenerate chunks: {resolved_chunk_uri}",
                                "indexed": 0
                            }
                except Exception as e:
                    logger.warning(f"Could not read file to check content: {e}")
                
                return {
                    "success": False,
                    "message": f"No valid chunks found in file (file exists, {file_size} bytes, but no parseable chunks). The file may be corrupted or in an unexpected format. Try 'Force re-parse' to regenerate chunks: {resolved_chunk_uri}",
                    "indexed": 0
                }
            
            # Ingest into both OpenSearch and LanceDB (hybrid)
            # Use hybrid client if available, otherwise fall back to opensearch_client
            ingest_client = self.client if isinstance(self.client, HybridClient) else self.opensearch_client
            logger.info(f"Ingesting {len(chunks)} chunks for file {file_id}, project {project_id}, version {file_version_id}")
            
            # Ensure chunks are properly normalized for LanceDB
            normalized_chunks = []
            for chunk in chunks:
                normalized_chunk = chunk.copy()
                # Ensure section_path is a list (LanceDB client will convert to JSON string)
                if "section_path" in normalized_chunk:
                    if isinstance(normalized_chunk["section_path"], str):
                        try:
                            import json
                            normalized_chunk["section_path"] = json.loads(normalized_chunk["section_path"])
                        except:
                            normalized_chunk["section_path"] = [normalized_chunk["section_path"]] if normalized_chunk["section_path"] else []
                    elif not isinstance(normalized_chunk["section_path"], list):
                        normalized_chunk["section_path"] = []
                else:
                    normalized_chunk["section_path"] = []
                
                # Ensure page_range is a list
                if "page_range" in normalized_chunk:
                    if not isinstance(normalized_chunk["page_range"], list):
                        try:
                            import json
                            if isinstance(normalized_chunk["page_range"], str):
                                normalized_chunk["page_range"] = json.loads(normalized_chunk["page_range"])
                            else:
                                normalized_chunk["page_range"] = [normalized_chunk["page_range"]]
                        except:
                            normalized_chunk["page_range"] = [1, 1]
                else:
                    normalized_chunk["page_range"] = [1, 1]
                
                # Ensure metadata is a dict
                if "metadata" in normalized_chunk:
                    if isinstance(normalized_chunk["metadata"], str):
                        try:
                            import json
                            normalized_chunk["metadata"] = json.loads(normalized_chunk["metadata"])
                        except:
                            normalized_chunk["metadata"] = {}
                    elif not isinstance(normalized_chunk["metadata"], dict):
                        normalized_chunk["metadata"] = {}
                else:
                    normalized_chunk["metadata"] = {}
                
                normalized_chunks.append(normalized_chunk)
            
            try:
                result = ingest_client.ingest_chunks(
                    projectid=project_id,
                    fileid=file_id,
                    file_version_id=file_version_id,
                    chunks=normalized_chunks
                )
            except Exception as e:
                logger.error(f"Error ingesting chunks with hybrid client: {e}", exc_info=True)
                # Fallback to OpenSearch only if hybrid fails
                logger.warning("Falling back to OpenSearch-only ingestion")
                result = self.opensearch_client.ingest_chunks(
                    projectid=project_id,
                    fileid=file_id,
                    file_version_id=file_version_id,
                    chunks=normalized_chunks
                )
            
            logger.info(f"Ingest result: {result}")
            
            # Check if ingestion was successful
            if result.get("indexed", 0) > 0:
                return {
                    "success": True,
                    "message": f"Successfully ingested {result['indexed']} chunks",
                    "indexed": result["indexed"],
                    "errors": result.get("errors", [])
                }
            else:
                # Ingestion returned 0 indexed chunks - this is an error
                error_msg = result.get("message")
                if not error_msg or (isinstance(error_msg, str) and error_msg.strip() == ""):
                    error_msg = result.get("error")
                if not error_msg or (isinstance(error_msg, str) and error_msg.strip() == ""):
                    error_msg = f"No chunks were indexed (expected {len(chunks)} chunks). Result: {result}"
                logger.error(f"Ingestion failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg,
                    "indexed": 0,
                    "errors": result.get("errors", [])
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ingest chunks: {str(e)}"
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication if token is available."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    async def fetch_chunks_from_chunking_api(
        self,
        project_id: str,
        file_id: str,
        force: bool = False,
        user_id: str = "temp_user_001"
    ) -> Optional[Dict[str, Any]]:
        """Fetch chunk information from chunking service API for a single file."""
        try:
            async with httpx.AsyncClient(timeout=None) as client:  # No timeout for parsing
                # Call parse_latest endpoint (POST) to trigger parsing and get chunk info
                # If force=True, it will re-parse even if already parsed
                # New API uses Body with ParseLatestRequest model
                request_body = {
                    "user_id": user_id,
                    "do_ocr": False,
                    "force": force
                }
                response = await client.post(
                    f"{self.chunking_service_url}/projects/{project_id}/files/{file_id}/parse_latest",
                    json=request_body,
                    headers=self._get_headers()
                )
                if response.status_code == 200:
                    data = response.json()
                    # If file was skipped but already parsed, try to get version info
                    # Check if chunk_uri is present (should be now that we fixed the endpoint)
                    if data.get("skipped") and data.get("status") == "parsed":
                        if data.get("chunk_uri"):
                            logger.info(f"File {file_id} already parsed, chunk_uri found: {data.get('chunk_uri')}")
                        else:
                            logger.warning(f"File {file_id} already parsed but chunk_uri not in response")
                    return data
                elif response.status_code == 403:
                    logger.error(f"Authentication failed: 403 Forbidden. Check CHUNKING_SERVICE_AUTH_TOKEN")
                return None
        except Exception as e:
            logger.error(f"Error fetching from chunking service: {e}")
            return None
    
    async def get_file_names(self, project_id: str, file_ids: List[str]) -> Dict[str, str]:
        """Fetch file names from MongoDB directly (no HTTP calls needed since we're integrated)."""
        file_name_map = {}
        if not project_id or not file_ids:
            return file_name_map
        
        try:
            # Use MongoDB directly instead of HTTP call
            from app.db.mongo import db
            files_collection = db["files"]
            # Query for files in this project with matching IDs
            query = {"project_id": project_id, "_id": {"$in": file_ids}}
            files_cursor = files_collection.find(query, {"_id": 1, "filename": 1, "name": 1})
            
            for file_doc in files_cursor:
                file_id = file_doc.get("_id") or str(file_doc.get("_id"))
                filename = file_doc.get("filename") or file_doc.get("name")
                if file_id and filename and file_id in file_ids:
                    file_name_map[file_id] = filename
            
            logger.info(f"Fetched {len(file_name_map)} file names from MongoDB for {len(file_ids)} file IDs")
        except Exception as e:
            logger.warning(f"Failed to fetch file names from MongoDB: {e}")
            # Fallback to HTTP (might work in some cases)
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Get list of files from chunking service
                    response = await client.get(
                        f"{self.chunking_service_url}/projects/{project_id}/files",
                        params={"user_id": "temp_user_001"},
                        headers=self._get_headers()
                    )
                    
                    if response.status_code == 200:
                        files = response.json()
                        for file_info in files:
                            file_id = file_info.get("_id") or file_info.get("id")
                            filename = file_info.get("filename") or file_info.get("name")
                            if file_id and filename and file_id in file_ids:
                                file_name_map[file_id] = filename
            except Exception as e2:
                logger.warning(f"Failed to fetch file names via HTTP fallback: {e2}")
        
        return file_name_map
    
    async def get_file_version_info(
        self,
        project_id: str,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get file version information including chunk_uri from chunking service."""
        try:
            async with httpx.AsyncClient(timeout=None) as client:  # No timeout for parsing
                # Try to get file info - check if there's a file details endpoint
                # For now, we'll use parse_latest with force=False to get status
                # But we need chunk_uri which might be in the file version document
                # Let's try calling parse_latest - if it's already parsed, we need another way
                
                # Alternative: List files and find the one we need
                # New API requires user_id query parameter
                response = await client.get(
                    f"{self.chunking_service_url}/projects/{project_id}/files",
                    params={"user_id": "temp_user_001"},
                    headers=self._get_headers()
                )
                if response.status_code == 200:
                    files = response.json()
                    for file_info in files:
                        if file_info.get("_id") == file_id or file_info.get("id") == file_id:
                            # Found the file, but we still need version info with chunk_uri
                            # The file list might not have chunk_uri
                            pass
                
                # Try parse_latest - if already parsed, it might not return chunk_uri
                # So we'll need to force parse or find another endpoint
                return await self.fetch_chunks_from_chunking_api(project_id, file_id, force=False)
        except Exception as e:
            logger.error(f"Error getting file version info: {e}")
            return None
    
    async def parse_and_sync_all_files(
        self,
        project_id: str,
        do_ocr: bool = False,
        force: bool = False,
        only_status: Optional[str] = None,
        force_parse_for_skipped: bool = True,
        user_id: str = "temp_user_001",
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Trigger parsing for all files in a project and sync chunks.
        
        Calls the parse_all_latest endpoint which:
        1. Parses all latest file versions
        2. Returns results with chunk_uri for each file
        3. We then ingest all chunks into OpenSearch
        """
        try:
            async with httpx.AsyncClient(timeout=None) as client:  # No timeout for batch parsing
                # Call parse-all-latest endpoint (GET with query params, returns SSE stream)
                params = {
                    "user_id": user_id,
                    "do_ocr": str(do_ocr).lower(),
                    "force": str(force).lower(),
                    "limit": str(limit)
                }
                if only_status:
                    params["only_status"] = only_status
                
                response = await client.get(
                    f"{self.chunking_service_url}/projects/{project_id}/files/parse-all-latest",
                    params=params,
                    headers=self._get_headers()
                )
                
                if response.status_code == 403:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Authentication failed: 403 Forbidden. The retriever service needs an auth token to call the chunking service. Set CHUNKING_SERVICE_AUTH_TOKEN environment variable."
                    )
                elif response.status_code != 200:
                    error_detail = f"Chunking service returned status {response.status_code}"
                    try:
                        error_text = await response.aread()
                        error_detail = error_text.decode('utf-8')[:500]  # Limit error message length
                    except:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=error_detail
                    )
                
                # Parse SSE stream
                results = []
                summary = {}
                buffer = ""
                
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8', errors='ignore')
                    lines = buffer.split('\n')
                    buffer = lines.pop() if lines else ""
                    
                    for line in lines:
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                
                                if data.get("type") == "result":
                                    # Extract file result from SSE event
                                    file_result = {
                                        "file_id": data.get("file_id"),
                                        "status": data.get("status"),
                                        "skipped": data.get("skipped", False),
                                        "version": data.get("version", 1),
                                        "chunk_uri": data.get("chunk_uri"),
                                        "ok": not data.get("skipped", False) and data.get("status") == "parsed"
                                    }
                                    results.append(file_result)
                                elif data.get("type") == "complete":
                                    # Final summary
                                    summary = data.get("summary", {})
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE data line: {line[:100]}")
                                continue
                
                # Ingest chunks for each successfully parsed file
                ingested_count = 0
                errors = []
                
                logger.info(f"Processing {len(results)} file results from parsing")
                for file_result in results:
                    # Initialize all variables at the start to avoid scope issues
                    # Use file_status to avoid shadowing FastAPI status module
                    file_id = file_result.get("file_id")
                    file_status = file_result.get("status", "unknown")
                    skipped = file_result.get("skipped", False)
                    chunk_uri = file_result.get("chunk_uri")  # Try to get from result first
                    version = file_result.get("version", 1)
                    
                    if not file_id:
                        logger.warning(f"Skipping file result with no file_id: {file_result}")
                        continue
                    logger.info(f"File result for {file_id}: status={file_status}, skipped={skipped}, chunk_uri={chunk_uri is not None}, version={version}, all_keys={list(file_result.keys())}")
                    
                    # Handle already-parsed files: fetch chunk_uri from chunking service if not already in result
                    if skipped and file_status == "parsed":
                        # File was already parsed - check if chunk_uri is already in the result
                        if not chunk_uri:
                            logger.info(f"File {file_id} already parsed (skipped) but chunk_uri not in SSE result, fetching chunk info...")
                        version = file_result.get("version", 1)
                        
                        # First, try to get chunk_uri from the parse_latest endpoint
                        chunk_info = await self.fetch_chunks_from_chunking_api(project_id, file_id, force=False)
                        if chunk_info:
                            chunk_uri = chunk_info.get("chunk_uri")
                            version = chunk_info.get("version", version)
                            logger.info(f"Got chunk_uri from parse_latest (force=False): {chunk_uri is not None}")
                        
                        # If still no chunk_uri and force_parse_for_skipped is True, force parse
                        if not chunk_uri and force_parse_for_skipped:
                            logger.info(f"File {file_id} skipped without chunk_uri, trying force parse...")
                            chunk_info = await self.fetch_chunks_from_chunking_api(project_id, file_id, force=True)
                            if chunk_info:
                                chunk_uri = chunk_info.get("chunk_uri")
                                version = chunk_info.get("version", version)
                                logger.info(f"Force parse result for {file_id}: chunk_uri={chunk_uri is not None}, version={version}")
                        
                        # If we still don't have chunk_uri, try to find it by searching filesystem
                        if not chunk_uri:
                            logger.info(f"Still no chunk_uri for {file_id}, searching filesystem...")
                            # Try multiple possible base paths (in order of likelihood)
                            base_paths = [
                                "/home/himanshu-gcp/DataRoom-ai-sheetal/uploads",  # Host path (most likely)
                                "/app/uploads",  # Chunking container path (if mounted)
                                "./uploads",  # Relative path
                                "../DataRoom-ai-sheetal/uploads",  # Relative from retriever container
                            ]
                            
                            logger.info(f"Searching for chunk file for {file_id} in project {project_id}, version {version}")
                            
                            for base_path in base_paths:
                                # Try constructed path with version - check both chunks.jsonl and all_chunks.jsonl, and .json
                                for filename in ["all_chunks.jsonl", "chunks.jsonl", "chunks.json", f"{file_id}_chunks.json"]:
                                    potential_path = f"{base_path}/projects/{project_id}/derived/{file_id}/{version}/{filename}"
                                    logger.info(f"Trying path: {potential_path}")
                                    if os.path.exists(potential_path):
                                        chunk_uri = potential_path
                                        logger.info(f"✅ Found chunk file at: {chunk_uri}")
                                        break
                                
                                if chunk_uri:
                                    break
                                
                                # Try to find the chunk file in the derived directory (any version)
                                derived_dir = f"{base_path}/projects/{project_id}/derived/{file_id}"
                                logger.info(f"Checking derived directory: {derived_dir}")
                                if os.path.exists(derived_dir):
                                    logger.info(f"✅ Derived directory exists: {derived_dir}, searching for chunk files...")
                                    # List what's in the directory
                                    try:
                                        dir_contents = os.listdir(derived_dir)
                                        logger.info(f"Contents of {derived_dir}: {dir_contents}")
                                    except Exception as e:
                                        logger.warning(f"Could not list directory {derived_dir}: {e}")
                                    
                                    # Look for version directories - check both filenames and .json
                                    version_files = glob.glob(f"{derived_dir}/*/chunks.jsonl")
                                    version_files.extend(glob.glob(f"{derived_dir}/*/all_chunks.jsonl"))
                                    version_files.extend(glob.glob(f"{derived_dir}/*/chunks.json"))
                                    version_files.extend(glob.glob(f"{derived_dir}/*/*_chunks.json"))
                                    logger.info(f"Found {len(version_files)} chunk file(s) in {derived_dir}")
                                    if version_files:
                                        # Use the most recent one (by modification time)
                                        chunk_uri = sorted(version_files, key=lambda x: os.path.getmtime(x))[-1]
                                        logger.info(f"✅ Found chunk file by searching in {derived_dir}: {chunk_uri}")
                                        break
                                else:
                                    logger.warning(f"❌ Derived directory does not exist: {derived_dir}")
                            
                            if not chunk_uri:
                                logger.warning(f"❌ Could not find chunk file for {file_id} in any expected location. Tried base paths: {base_paths}")
                                # Log what directories do exist for debugging
                                for base_path in base_paths[:2]:  # Check first 2 most likely paths
                                    projects_dir = f"{base_path}/projects"
                                    if os.path.exists(projects_dir):
                                        logger.info(f"Projects directory exists: {projects_dir}")
                                        proj_dir = f"{projects_dir}/{project_id}"
                                        if os.path.exists(proj_dir):
                                            logger.info(f"Project directory exists: {proj_dir}")
                                            derived_base = f"{proj_dir}/derived"
                                            if os.path.exists(derived_base):
                                                logger.info(f"Derived directory exists: {derived_base}")
                                                # List available file_ids
                                                available_files = [d for d in os.listdir(derived_base) if os.path.isdir(os.path.join(derived_base, d))]
                                                logger.info(f"Available file_ids in derived: {available_files[:10]}")
                        else:
                            # chunk_uri is already in the SSE result, use it
                            logger.info(f"File {file_id} already parsed (skipped), using chunk_uri from SSE result: {chunk_uri}")
                    elif file_result.get("ok") and file_status == "parsed":
                        # Newly parsed file - should have chunk_uri in result
                        chunk_uri = file_result.get("chunk_uri")
                        version = file_result.get("version", 1)
                        logger.info(f"Newly parsed file {file_id}: chunk_uri={chunk_uri}, version={version}")
                        if not chunk_uri:
                            logger.warning(f"Newly parsed file {file_id} has no chunk_uri in result! Result keys: {file_result.keys()}")
                    else:
                        # File not parsed or failed - skip this file
                        logger.warning(f"Skipping file {file_id}: status={file_status}, skipped={skipped}, ok={file_result.get('ok')}")
                        continue
                    
                    file_version_id = f"v_{file_id}_{version}"
                    
                    logger.info(f"Processing file {file_id}: chunk_uri={chunk_uri}, version={version}, file_version_id={file_version_id}")
                    
                    if chunk_uri:
                        logger.info(f"Attempting to ingest chunks from {chunk_uri} for file {file_id}")
                        try:
                            # Note: ingest_chunks_from_chunking_service is synchronous
                            # In production, consider making it async or using asyncio.to_thread
                            ingest_result = self.ingest_chunks_from_chunking_service(
                                project_id=project_id,
                                file_id=file_id,
                                file_version_id=file_version_id,
                                chunk_uri=chunk_uri
                            )
                            logger.info(f"Ingestion result for {file_id}: {ingest_result}")
                            if ingest_result.get("success"):
                                indexed = ingest_result.get("indexed", 0)
                                ingested_count += indexed
                                logger.info(f"Successfully ingested {indexed} chunks for file {file_id}")
                            else:
                                # Ingestion failed - get error message
                                error_msg = ingest_result.get('message')
                                if not error_msg or (isinstance(error_msg, str) and error_msg.strip() == ""):
                                    # Try to get error from other fields
                                    error_msg = ingest_result.get('error') or ingest_result.get('detail')
                                if not error_msg or (isinstance(error_msg, str) and error_msg.strip() == ""):
                                    # Construct error message from result
                                    indexed = ingest_result.get('indexed', 0)
                                    errors_list = ingest_result.get('errors', [])
                                    if errors_list:
                                        error_msg = f"Ingestion failed: {errors_list[0] if isinstance(errors_list, list) and len(errors_list) > 0 else str(errors_list)}"
                                    else:
                                        error_msg = f"Ingestion failed: No chunks were indexed (result: {ingest_result})"
                                logger.error(f"Ingestion failed for {file_id}: {error_msg}. Full result: {ingest_result}")
                                
                                # If file is empty and force_parse_for_skipped is True, try to re-parse
                                if "empty" in error_msg.lower() and force_parse_for_skipped:
                                    logger.info(f"File {file_id} has empty chunk file, attempting force re-parse...")
                                    try:
                                        chunk_info = await self.fetch_chunks_from_chunking_api(project_id, file_id, force=True)
                                        if chunk_info and chunk_info.get("chunk_uri"):
                                            # Try ingesting again with the new chunk_uri
                                            new_chunk_uri = chunk_info.get("chunk_uri")
                                            resolved_new_uri = self._resolve_chunk_uri(new_chunk_uri)
                                            if resolved_new_uri and os.path.exists(resolved_new_uri) and os.path.getsize(resolved_new_uri) > 0:
                                                retry_result = self.ingest_chunks_from_chunking_service(
                                                    project_id=project_id,
                                                    file_id=file_id,
                                                    file_version_id=f"v_{file_id}_{chunk_info.get('version', version)}",
                                                    chunk_uri=new_chunk_uri
                                                )
                                                if retry_result.get("success"):
                                                    ingested_count += retry_result.get("indexed", 0)
                                                    logger.info(f"Successfully ingested {retry_result.get('indexed', 0)} chunks for file {file_id} after force re-parse")
                                                    continue  # Skip adding error, ingestion succeeded
                                    except Exception as e:
                                        logger.warning(f"Failed to force re-parse file {file_id}: {e}")
                                
                                errors.append(f"File {file_id}: {error_msg}")
                                logger.error(f"Failed to ingest chunks for {file_id}: {error_msg}")
                        except FileNotFoundError as e:
                            error_msg = f"Chunk file not found: {chunk_uri}. The file might be in the chunking container and not accessible from the retriever."
                            logger.error(f"File {file_id}: {error_msg}")
                            errors.append(f"File {file_id}: {error_msg}")
                        except HTTPException as he:
                            error_msg = he.detail if hasattr(he, 'detail') and he.detail else str(he)
                            if not error_msg or error_msg.strip() == "":
                                error_msg = f"HTTPException {he.status_code}: {type(he).__name__}"
                            logger.error(f"HTTPException during ingestion for {file_id}: {error_msg}", exc_info=True)
                            errors.append(f"File {file_id}: {error_msg}")
                        except Exception as e:
                            error_msg = str(e) if e else "Unknown exception occurred"
                            if not error_msg or error_msg.strip() == "":
                                error_msg = f"Exception of type {type(e).__name__} occurred during ingestion: {repr(e)}"
                            logger.error(f"Error ingesting chunks for file {file_id}: {error_msg}", exc_info=True)
                            errors.append(f"File {file_id}: {error_msg}")
                    else:
                        # No chunk_uri available - try to fetch it from API
                        logger.warning(f"File {file_id}: No chunk_uri available. Status: {status}, Skipped: {skipped}, OK: {file_result.get('ok')}, chunk_uri from result: {file_result.get('chunk_uri')}")
                        error_msg = f"No chunk_uri available. File may not be parsed yet or chunk_uri path is not accessible. Status: {status}, Skipped: {skipped}, OK: {file_result.get('ok')}"
                        logger.warning(f"File {file_id}: {error_msg}")
                        # Try one more time to get chunk_uri from the database/API
                        logger.info(f"Attempting to fetch chunk_uri from API for file {file_id}...")
                        chunk_uri_fetched = False
                        try:
                            chunk_info = await self.fetch_chunks_from_chunking_api(project_id, file_id, force=False)
                            if chunk_info and chunk_info.get("chunk_uri"):
                                chunk_uri = chunk_info.get("chunk_uri")
                                version = chunk_info.get("version", version)
                                logger.info(f"Successfully fetched chunk_uri from API: {chunk_uri}")
                                # Retry ingestion with the fetched chunk_uri
                                file_version_id = f"v_{file_id}_{version}"
                                try:
                                    ingest_result = self.ingest_chunks_from_chunking_service(
                                        project_id=project_id,
                                        file_id=file_id,
                                        file_version_id=file_version_id,
                                        chunk_uri=chunk_uri
                                    )
                                    if ingest_result.get("success"):
                                        ingested_count += ingest_result.get("indexed", 0)
                                        logger.info(f"Successfully ingested {ingest_result.get('indexed', 0)} chunks for file {file_id} after fetching chunk_uri")
                                        chunk_uri_fetched = True  # Mark as successful
                                    else:
                                        error_msg = f"Failed to ingest after fetching chunk_uri: {ingest_result.get('message', 'Unknown error')}"
                                        logger.error(f"File {file_id}: {error_msg}")
                                except Exception as e:
                                    error_msg = f"Error ingesting after fetching chunk_uri: {str(e)}"
                                    logger.error(f"File {file_id}: {error_msg}", exc_info=True)
                            else:
                                error_msg = f"Could not fetch chunk_uri from API. Response: {chunk_info}"
                                logger.warning(f"File {file_id}: {error_msg}")
                        except Exception as e:
                            error_msg = f"Failed to fetch chunk_uri from API: {str(e)}"
                            logger.warning(f"File {file_id}: {error_msg}")
                        
                        # Only add error if we didn't successfully fetch and ingest
                        if not chunk_uri_fetched:
                            errors.append(f"File {file_id}: {error_msg}")
                
                return {
                    "success": True,
                    "message": f"Processed {len(results)} files, ingested {ingested_count} chunks",
                    "parsing_summary": summary,
                    "ingested_chunks": ingested_count,
                    "errors": errors
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error parsing and syncing all files: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse and sync files: {str(e)}"
            )


# Global integration instance
_integration_instance: Optional[ChunkingIntegration] = None


def get_integration(client=None, opensearch_client: Optional[OpenSearchClient] = None, auth_token: Optional[str] = None) -> ChunkingIntegration:
    """Get or create the global integration instance."""
    global _integration_instance
    if _integration_instance is None:
        # Use client if provided (HybridClient), otherwise use opensearch_client
        if client is None:
            if opensearch_client is None:
                from app.retriever.opensearch_client import OpenSearchClient
                opensearch_client = OpenSearchClient()
            client = opensearch_client
        # Get auth token from environment if not provided
        if auth_token is None:
            auth_token = os.getenv("CHUNKING_SERVICE_AUTH_TOKEN", None)
        _integration_instance = ChunkingIntegration(client, auth_token=auth_token)
    return _integration_instance

