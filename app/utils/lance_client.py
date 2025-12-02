# app/lancedb_client.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import lancedb
from lancedb.table import Table
import pyarrow as pa
import numpy as np

logger = logging.getLogger(__name__)

# LanceDB configuration
LANCEDB_URI = os.getenv("LANCEDB_URI", "/app/lancedb_data")
CHUNKS_TABLE_NAME = "chunk_embeddings"

# Singleton connection
_db = None


def get_lancedb() -> lancedb.DBConnection:
    """
    Get or create LanceDB connection (singleton pattern).
    
    Returns:
        LanceDB connection instance
    """
    global _db
    if _db is None:
        logger.info(f"Connecting to LanceDB at: {LANCEDB_URI}")
        _db = lancedb.connect(LANCEDB_URI)
        logger.info("LanceDB connection established")
    return _db


def get_chunks_table(create_if_not_exists: bool = True) -> Optional[Table]:
    """
    Get the chunks embeddings table.
    
    Args:
        create_if_not_exists: If True, create table if it doesn't exist
        
    Returns:
        LanceDB table or None if not exists and create_if_not_exists=False
    """
    db = get_lancedb()
    
    # Check if table exists
    if CHUNKS_TABLE_NAME in db.table_names():
        return db.open_table(CHUNKS_TABLE_NAME)
    
    if create_if_not_exists:
        logger.info(f"Creating chunks table: {CHUNKS_TABLE_NAME}")
        # Create table with schema
        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("dataroom_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_version_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("page_number", pa.int32()),
            pa.field("block_type", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 384)),  # bge-small-en-v1.5 dimension
            pa.field("created_at", pa.string()),
        ])
        
        # Create empty table with schema
        table = db.create_table(CHUNKS_TABLE_NAME, schema=schema, mode="overwrite")
        logger.info(f"Created table {CHUNKS_TABLE_NAME}")
        return table
    
    return None


def insert_chunk_embeddings(chunks: List[Dict[str, Any]]) -> int:
    """
    Insert chunk embeddings into LanceDB.
    
    Args:
        chunks: List of chunk dictionaries with embeddings
        
    Returns:
        Number of chunks inserted
    """
    if not chunks:
        logger.warning("No chunks to insert")
        return 0
    
    # Filter chunks that have embeddings
    chunks_with_embeddings = [
        chunk for chunk in chunks 
        if chunk.get("embedding") is not None
    ]
    
    if not chunks_with_embeddings:
        logger.warning("No chunks have embeddings")
        return 0
    
    # Prepare data for LanceDB
    data = []
    for chunk in chunks_with_embeddings:
        embedding = chunk.get("embedding")
        
        # Ensure embedding is the right format
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.astype(np.float32)
        else:
            logger.warning(f"Invalid embedding format for chunk {chunk.get('chunk_id')}")
            continue
        
        data.append({
            "chunk_id": chunk.get("chunk_id") or chunk.get("_id"),
            "dataroom_id": chunk.get("dataroom_id"),
            "doc_id": chunk.get("doc_id"),
            "file_version_id": chunk.get("file_version_id"),
            "chunk_index": chunk.get("chunk_index", 0),
            "page_number": chunk.get("page_number") or 0,
            "block_type": chunk.get("block_type", "text"),
            "embedding": embedding.tolist(),
            "created_at": chunk.get("created_at", ""),
        })
    
    if not data:
        logger.warning("No valid data to insert")
        return 0
    
    try:
        table = get_chunks_table(create_if_not_exists=True)
        table.add(data)
        logger.info(f"Inserted {len(data)} chunk embeddings into LanceDB")
        return len(data)
    except Exception as e:
        logger.error(f"Error inserting embeddings into LanceDB: {e}")
        raise


def delete_chunks_by_version(file_version_id: str) -> int:
    """
    Delete all chunks for a specific file version.
    
    Args:
        file_version_id: File version ID
        
    Returns:
        Number of chunks deleted
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return 0
        
        # LanceDB delete syntax
        result = table.delete(f"file_version_id = '{file_version_id}'")
        logger.info(f"Deleted chunks for version {file_version_id} from LanceDB")
        return result
    except Exception as e:
        logger.error(f"Error deleting chunks from LanceDB: {e}")
        return 0


def delete_chunks_by_doc(doc_id: str) -> int:
    """
    Delete all chunks for a specific document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Number of chunks deleted
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return 0
        
        result = table.delete(f"doc_id = '{doc_id}'")
        logger.info(f"Deleted chunks for doc {doc_id} from LanceDB")
        return result
    except Exception as e:
        logger.error(f"Error deleting chunks from LanceDB: {e}")
        return 0


def delete_chunks_by_dataroom(dataroom_id: str) -> int:
    """
    Delete all chunks for a specific dataroom.
    
    Args:
        dataroom_id: Dataroom ID
        
    Returns:
        Number of chunks deleted
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return 0
        
        result = table.delete(f"dataroom_id = '{dataroom_id}'")
        logger.info(f"Deleted chunks for dataroom {dataroom_id} from LanceDB")
        return result
    except Exception as e:
        logger.error(f"Error deleting chunks from LanceDB: {e}")
        return 0


def search_similar_chunks(
    query_embedding: List[float],
    dataroom_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    limit: int = 10,
    metric: str = "cosine"
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks using vector similarity.
    
    Args:
        query_embedding: Query embedding vector
        dataroom_id: Optional filter by dataroom
        doc_id: Optional filter by document
        limit: Maximum number of results
        metric: Distance metric (cosine, l2, dot)
        
    Returns:
        List of similar chunks with scores
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return []
        
        # Convert embedding to numpy array
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # Build search query
        search_query = table.search(query_vector, vector_column_name="embedding")
        
        # Apply filters
        if dataroom_id:
            search_query = search_query.where(f"dataroom_id = '{dataroom_id}'")
        
        if doc_id:
            search_query = search_query.where(f"doc_id = '{doc_id}'")
        
        # Set metric and limit
        search_query = search_query.metric(metric).limit(limit)
        
        # Execute search
        results = search_query.to_list()
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
    except Exception as e:
        logger.error(f"Error searching LanceDB: {e}")
        return []


def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific chunk by ID.
    
    Args:
        chunk_id: Chunk ID
        
    Returns:
        Chunk data or None
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            return None
        
        results = table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()
        
        if results:
            return results[0]
        return None
    except Exception as e:
        logger.error(f"Error getting chunk from LanceDB: {e}")
        return None


def count_chunks(
    dataroom_id: Optional[str] = None,
    doc_id: Optional[str] = None
) -> int:
    """
    Count chunks in LanceDB.
    
    Args:
        dataroom_id: Optional filter by dataroom
        doc_id: Optional filter by document
        
    Returns:
        Number of chunks
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            return 0
        
        # Build query
        query = table.search()
        
        if dataroom_id:
            query = query.where(f"dataroom_id = '{dataroom_id}'")
        
        if doc_id:
            query = query.where(f"doc_id = '{doc_id}'")
        
        # Count results
        count = len(query.limit(1000000).to_list())  # LanceDB doesn't have direct count
        
        return count
    except Exception as e:
        logger.error(f"Error counting chunks in LanceDB: {e}")
        return 0


def update_chunk_embedding(chunk_id: str, embedding: List[float]) -> bool:
    """
    Update embedding for a specific chunk.
    Note: LanceDB doesn't support in-place updates, so we delete and reinsert.
    
    Args:
        chunk_id: Chunk ID
        embedding: New embedding vector
        
    Returns:
        True if successful, False otherwise
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            return False
        
        # Get existing chunk
        existing = get_chunk_by_id(chunk_id)
        if not existing:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
        
        # Delete old entry
        table.delete(f"chunk_id = '{chunk_id}'")
        
        # Update embedding
        existing["embedding"] = np.array(embedding, dtype=np.float32).tolist()
        
        # Reinsert
        table.add([existing])
        
        logger.info(f"Updated embedding for chunk {chunk_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating chunk embedding: {e}")
        return False


def create_index(
    metric: str = "cosine",
    num_partitions: int = 256,
    num_sub_vectors: int = 96
):
    """
    Create IVF-PQ index for faster similarity search.
    
    Args:
        metric: Distance metric (cosine, l2, dot)
        num_partitions: Number of IVF partitions
        num_sub_vectors: Number of sub-vectors for PQ
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return
        
        logger.info("Creating IVF-PQ index...")
        table.create_index(
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors
        )
        logger.info("Index created successfully")
    except Exception as e:
        logger.error(f"Error creating index: {e}")


def optimize_table():
    """
    Optimize the chunks table by compacting small files.
    """
    try:
        table = get_chunks_table(create_if_not_exists=False)
        if not table:
            logger.warning("Chunks table does not exist")
            return
        
        logger.info("Optimizing table...")
        table.optimize()
        logger.info("Table optimized successfully")
    except Exception as e:
        logger.error(f"Error optimizing table: {e}")