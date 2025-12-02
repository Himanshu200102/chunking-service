import os
import logging
import lancedb
from typing import List, Dict, Optional
import pyarrow as pa

logger = logging.getLogger(__name__)

_LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")
_db = None

def get_lancedb() -> lancedb.DBConnection:
    """Return a singleton LanceDB connection."""
    global _db
    if _db is None:
        _db = lancedb.connect(_LANCEDB_URI)
        # Create a tiny health table if missing (useful for /health checks)
        if "health" not in _db.table_names():
            _db.create_table("health", data=[{"id": 1, "msg": "ok"}])
    return _db


def get_or_create_chunks_table(embedding_dim: int = 384):
    """
    Get or create the chunks table with vector embeddings.
    
    Schema:
    - chunk_ref (str): Unique chunk identifier
    - file_version_id (str): Version this chunk belongs to
    - project_id (str): Project identifier
    - text (str): Chunk text content
    - section_path (list[str]): Section hierarchy
    - object_type (str): Type of content (narrative, table, figure, code)
    - page_range (list[int]): [start_page, end_page]
    - embedding (vector): Dense embedding vector
    - created_at (str): ISO timestamp
    
    Args:
        embedding_dim: Dimension of embedding vectors (default: 384 for all-MiniLM-L6-v2)
    
    Returns:
        LanceDB table instance
    """
    db = get_lancedb()
    
    if "chunks" in db.table_names():
        logger.info("Opening existing chunks table")
        return db.open_table("chunks")
    
    # Define schema
    schema = pa.schema([
        pa.field("chunk_ref", pa.string()),
        pa.field("file_version_id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("section_path", pa.list_(pa.string())),
        pa.field("object_type", pa.string()),
        pa.field("page_range", pa.list_(pa.int32())),
        pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),  # Fixed-size vector
        pa.field("created_at", pa.string()),
    ])
    
    logger.info(f"Creating chunks table with schema (embedding_dim={embedding_dim})")
    
    # Create table with empty data (schema only)
    table = db.create_table("chunks", schema=schema, mode="create")
    
    logger.info("Chunks table created successfully")
    return table


def insert_chunks_with_embeddings(
    chunks: List[Dict],
    batch_size: int = 100
) -> int:
    """
    Insert chunks with embeddings into LanceDB.
    
    Args:
        chunks: List of chunk dicts with 'embedding' field
        batch_size: Batch size for insertion
    
    Returns:
        Number of chunks inserted
    """
    if not chunks:
        return 0
    
    # Get embedding dimension from first chunk
    embedding_dim = len(chunks[0].get("embedding", []))
    if embedding_dim == 0:
        raise ValueError("Chunks must have 'embedding' field")
    
    table = get_or_create_chunks_table(embedding_dim)
    
    # Insert in batches
    total_inserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        try:
            table.add(batch)
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
            raise
    
    logger.info(f"Total inserted: {total_inserted} chunks")
    return total_inserted


def search_similar_chunks(
    query_embedding: List[float],
    limit: int = 10,
    project_id: Optional[str] = None,
    file_version_id: Optional[str] = None,
    object_types: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Search for similar chunks using vector similarity.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results
        project_id: Optional project filter
        file_version_id: Optional version filter
        object_types: Optional object type filter (e.g., ["narrative", "table"])
    
    Returns:
        List of similar chunks with scores
    """
    db = get_lancedb()
    
    if "chunks" not in db.table_names():
        logger.warning("Chunks table doesn't exist")
        return []
    
    table = db.open_table("chunks")
    
    # Build filter expression
    filters = []
    if project_id:
        filters.append(f"project_id = '{project_id}'")
    if file_version_id:
        filters.append(f"file_version_id = '{file_version_id}'")
    if object_types:
        types_str = "', '".join(object_types)
        filters.append(f"object_type IN ('{types_str}')")
    
    where_clause = " AND ".join(filters) if filters else None
    
    # Vector search
    try:
        results = (
            table.search(query_embedding)
            .limit(limit)
            .where(where_clause, prefilter=True if where_clause else False)
            .to_list()
        )
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def delete_chunks_by_version(file_version_id: str) -> int:
    """
    Delete all chunks for a specific file version.
    
    Args:
        file_version_id: Version identifier
    
    Returns:
        Number of chunks deleted
    """
    db = get_lancedb()
    
    if "chunks" not in db.table_names():
        return 0
    
    table = db.open_table("chunks")
    
    try:
        table.delete(f"file_version_id = '{file_version_id}'")
        logger.info(f"Deleted chunks for version: {file_version_id}")
        return 1  # LanceDB doesn't return count
    except Exception as e:
        logger.error(f"Failed to delete chunks: {e}")
        return 0
