"""Data models for the retriever service."""
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for tables and figures."""
    num_rows: Optional[int] = Field(None, description="Number of rows (for tables)")
    num_cols: Optional[int] = Field(None, description="Number of columns (for tables)")
    has_header: Optional[bool] = Field(None, description="Whether table has header row")
    figure_type: Optional[str] = Field(None, description="Figure type: image, chart, diagram")


class ChunkInput(BaseModel):
    """Chunk structure received from chunking service."""
    chunk_ref: str = Field(..., description="Stable per-version unique id")
    text: str = Field(..., description="Chunk text content (400-800 tokens)")
    section_path: List[str] = Field(default_factory=list, description="Best-effort heading lineage")
    object_type: Literal["narrative", "table", "figure", "code", "other"] = Field(
        default="narrative", description="Type of content object"
    )
    page_range: List[int] = Field(..., min_items=2, max_items=2, description="Inclusive start/end page numbers")
    caption: Optional[str] = Field(None, description="Caption for tables/figures, null for narrative")
    metadata: Optional[ChunkMetadata] = Field(None, description="Optional metadata for tables/figures")


class ChunkDocument(BaseModel):
    """Chunk document structure for OpenSearch."""
    id: str = Field(..., alias="_id", description="Same as chunk_ref")
    projectid: str = Field(..., description="Project identifier")
    fileid: str = Field(..., description="File identifier")
    file_version_id: str = Field(..., description="Links to specific version")
    chunk_ref: str = Field(..., description="Stable per-version unique id")
    text: str = Field(..., description="Chunk text content")
    section_path: List[str] = Field(default_factory=list, description="Best-effort heading lineage")
    object_type: Literal["narrative", "table", "figure", "code", "other"] = Field(
        default="narrative", description="Type of content object"
    )
    page_range: List[int] = Field(..., min_items=2, max_items=2, description="Inclusive start/end page numbers")
    caption: Optional[str] = Field(None, description="Caption for tables/figures, null for narrative")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for tables/figures")
    is_active: bool = Field(default=True, description="Flip old chunks to false on re-chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "c_00017",
                "projectid": "p_123",
                "fileid": "f_456",
                "file_version_id": "vf_789_3",
                "chunk_ref": "c_00017",
                "text": "This is a sample chunk text...",
                "section_path": ["1. Intro", "Scope"],
                "object_type": "narrative",
                "page_range": [3, 4],
                "caption": None,
                "metadata": None,
                "is_active": True,
                "created_at": "2025-11-10T21:13:00Z",
                "updated_at": "2025-11-10T21:13:00Z"
            }
        }


class ChunkIngestRequest(BaseModel):
    """Request model for ingesting chunks."""
    projectid: str = Field(..., description="Project identifier")
    fileid: str = Field(..., description="File identifier")
    file_version_id: str = Field(..., description="File version identifier")
    chunks: List[ChunkInput] = Field(..., description="List of chunks to ingest")


class QueryRequest(BaseModel):
    """Request model for querying chunks."""
    query: str = Field(..., description="Search query text")
    projectid: Optional[str] = Field(None, description="Project identifier (optional for global search)")
    fileid: Optional[str] = Field(None, description="File identifier (optional for file-specific search)")
    search_all_files: bool = Field(default=False, description="If True, search across all files in project (ignores fileid)")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    filters: Optional[dict] = Field(None, description="Additional filters (object_type, page_range, etc.)")


class ChunkResult(BaseModel):
    """Result model for retrieved chunks."""
    chunk_ref: str
    text: str
    section_path: List[str]
    object_type: str
    page_range: List[int]
    caption: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    fileid: str
    file_version_id: str
    filename: Optional[str] = Field(None, description="File name (if available)")
    score: float = Field(..., description="Relevance score from OpenSearch")


class CompressionStats(BaseModel):
    """Compression statistics for debugging."""
    chunks_processed: int
    original: Dict[str, Any]
    after_extractive_compression: Dict[str, Any]
    final_context: Dict[str, Any]
    summary: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    total_results: int
    chunks: List[ChunkResult]
    projectid: Optional[str] = None
    fileid: Optional[str] = None
    summary: Optional[str] = Field(None, description="AI-generated summary of top chunks")
    compression_stats: Optional[CompressionStats] = Field(None, description="Compression statistics for debugging")


class UserQueryLog(BaseModel):
    """Model for logging or tracking a raw user query."""
    query: str = Field(..., description="Original user query text")
    projectid: Optional[str] = Field(None, description="Project identifier, if applicable")
    fileid: Optional[str] = Field(None, description="File identifier, if applicable")
    user_id: Optional[str] = Field(None, description="End-user identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary client metadata (session, UI state, etc.)"
    )


class UserResponseLog(BaseModel):
    """Model for logging or tracking the final response shown to the user."""
    query: str = Field(..., description="Original user query text (for correlation)")
    response: str = Field(..., description="Final answer text returned to the user")
    projectid: Optional[str] = Field(None, description="Project identifier, if applicable")
    fileid: Optional[str] = Field(None, description="File identifier, if applicable")
    user_id: Optional[str] = Field(None, description="End-user identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary client metadata (ranking info, model name, etc.)"
    )


