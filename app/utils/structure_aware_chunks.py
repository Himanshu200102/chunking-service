# app/utils/structure_aware_chunks.py
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone

from docling.chunking import HierarchicalChunker
from app.db.mongo import db
from app.utils.embedding import embed_chunks_batch
from app.utils.lance_client import insert_chunk_embeddings, delete_chunks_by_version

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for better matching and search.
    - Lowercase
    - Remove extra whitespace
    - Remove special characters (optional)
    """
    # Lowercase
    normalized = text.lower()
    # Replace multiple spaces/newlines with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    return normalized


def extract_table_schema(chunk, block_type: str) -> Optional[Dict[str, Any]]:
    """
    Extract table schema if the chunk contains a table.
    
    Returns:
        Table schema dict or None
    """
    if block_type != "table":
        return None
    
    # Try to extract table structure from chunk metadata
    table_schema = {
        "columns": [],
        "row_count": 0,
        "has_header": False
    }
    
    # If chunk has table data in doc_items, extract it
    if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
        for item in chunk.meta.doc_items:
            if hasattr(item, 'label') and 'table' in str(item.label).lower():
                # Extract table metadata if available
                # This is a placeholder - adjust based on actual Docling table structure
                if hasattr(item, 'data') and item.data:
                    table_schema["columns"] = getattr(item.data, 'columns', [])
                    table_schema["row_count"] = getattr(item.data, 'row_count', 0)
                    table_schema["has_header"] = getattr(item.data, 'has_header', False)
                break
    
    return table_schema if table_schema["columns"] or table_schema["row_count"] > 0 else None


def extract_figure_id(chunk, block_type: str) -> Optional[str]:
    """
    Extract figure ID if the chunk contains a figure/image.
    
    Returns:
        Figure ID string or None
    """
    if block_type not in ["picture", "figure", "image"]:
        return None
    
    # Try to extract figure reference from chunk metadata
    if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
        for item in chunk.meta.doc_items:
            if hasattr(item, 'label') and any(fig_type in str(item.label).lower() 
                                              for fig_type in ["picture", "figure", "image"]):
                # Extract figure ID if available
                if hasattr(item, 'self_ref'):
                    return item.self_ref
                break
    
    return None


def create_chunks(
    conv_result,
    doc_id: str,
    dataroom_id: str,
    file_version_id: str,
    max_tokens: int = 512,
    output_dir: str = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Create chunks from an existing Docling conversion result and store them in MongoDB.
    
    Args:
        conv_result: Docling DocumentConverter result
        doc_id: Document ID (file_id)
        dataroom_id: Data room ID (project_id)
        file_version_id: File version ID (for internal tracking)
        max_tokens: Maximum tokens per chunk
        output_dir: Directory to save JSON file
        
    Returns:
        Tuple of (chunks_with_metadata, output_path)
    """
    document = conv_result.document
    
    # Initialize hierarchical chunker
    chunker = HierarchicalChunker(
        max_tokens=max_tokens,
        include_metadata=True
    )
    
    # Create chunks
    chunk_iter = chunker.chunk(document)
    
    chunks_with_metadata = []
    chunk_index = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    
    for chunk in chunk_iter:
        # Extract section headers directly from meta.headings
        heading_context = []
        if chunk.meta and chunk.meta.headings:
            heading_context = chunk.meta.headings
        
        # Create section_path from heading hierarchy
        section_path = " > ".join(heading_context) if heading_context else ""
        
        # Get page numbers from doc_items provenance
        page_numbers = []
        block_types = []
        
        if chunk.meta and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                # Extract block type from label
                if hasattr(item, 'label') and item.label:
                    block_type_str = str(item.label).replace('DocItemLabel.', '').lower()
                    block_types.append(block_type_str)
                
                # Extract page numbers from provenance
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no') and prov.page_no:
                            page_numbers.append(prov.page_no)
        
        # Remove duplicates and sort page numbers
        page_numbers = sorted(list(set(page_numbers))) if page_numbers else []
        
        # Determine primary block type (most common or first)
        if block_types:
            from collections import Counter
            block_type_counts = Counter(block_types)
            block_type = block_type_counts.most_common(1)[0][0]
        else:
            block_type = "text"
        
        # Generate unique chunk ID
        chunk_id = f"c_{file_version_id}_{chunk_index}"
        
        # Extract text and create normalized version
        text = chunk.text
        normalized_text = normalize_text(text)
        
        # Extract table schema or figure ID based on block type
        table_schema = extract_table_schema(chunk, block_type)
        figure_id = extract_figure_id(chunk, block_type)
        
        # Build chunk metadata for MongoDB
        chunk_doc = {
            "_id": chunk_id,
            "chunk_id": chunk_id,  # Explicit chunk_id field
            "doc_id": doc_id,  # Document/File ID
            "dataroom_id": dataroom_id,  # Project/Data room ID
            "file_version_id": file_version_id,  # Internal tracking only
            "page_number": page_numbers[0] if page_numbers else None,  # Primary page
            "section_path": section_path,  # Hierarchical path as string (e.g., "Chapter 1 > Section 1.1")
            "heading_context": heading_context,  # Full list of hierarchical headers
            "block_type": block_type,  # Type: text, table, list, title, section_header, etc.
            "chunk_index": chunk_index,  # Sequential index within document
            "text": text,  # Original chunk text
            "normalized_text": normalized_text,  # Normalized text for matching/search
            "embedding": None,  # Will be populated by embedding service later
            "char_count": len(text),  # Character count
            "token_count": max_tokens,  # Approximate token count
            "is_active": True,  # Active/inactive flag for versioning
            "created_at": now_iso,
            "updated_at": now_iso
        }
        
        # Add table_schema if it's a table
        if table_schema:
            chunk_doc["table_schema"] = table_schema
        
        # Add figure_id if it's a figure/image
        if figure_id:
            chunk_doc["figure_id"] = figure_id
        
        chunks_with_metadata.append(chunk_doc)
        chunk_index += 1
    
    logger.info(f"Created {len(chunks_with_metadata)} hierarchical chunks for file_version_id={file_version_id}")
    
    # Generate embeddings (disable progress bar to avoid performance issues)
    # Check if embeddings should be generated (can be disabled for CPU-only systems)
    import os
    enable_embeddings = os.getenv("ENABLE_EMBEDDINGS_DURING_PARSING", "false").lower() == "true"
    
    if enable_embeddings:
        try:
            logger.info(f"Starting embedding generation for {len(chunks_with_metadata)} chunks...")
            embeddings = embed_chunks_batch(
                chunks_with_metadata,
                show_progress=False,  # Disable progress bar for better performance
                batch_size=16  # Smaller batch size for faster processing
            )
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks_with_metadata, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"✅ Successfully generated embeddings for {len(embeddings)} chunks")
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}", exc_info=True)
            # Continue without embeddings rather than failing
    else:
        logger.warning(f"⚠️  Embedding generation DISABLED during parsing (ENABLE_EMBEDDINGS_DURING_PARSING=false)")
        logger.warning(f"⚠️  Chunks will be saved without embeddings. LanceDB will generate embeddings during sync.")
    
    # Store chunks in MongoDB
    if chunks_with_metadata:
        store_chunks_in_db(chunks_with_metadata, file_version_id)
    
    # Save to JSON file
    output_path = str(Path(output_dir) / f"{doc_id}_chunks.json") if output_dir else f"{doc_id}_chunks.json"
    save_chunks_to_json(chunks_with_metadata, output_path)
    
    return chunks_with_metadata, output_path



def store_chunks_in_db(chunks: List[Dict[str, Any]], file_version_id: str) -> None:
    """
    Store chunks in MongoDB. Deletes old chunks for this version first.
    
    Args:
        chunks: List of chunk documents
        file_version_id: File version ID
    """
    try:
        # DELETE (not deactivate) any existing chunks for this version
        # This allows re-parsing with force=true to work properly
        delete_result = db.chunks.delete_many({"file_version_id": file_version_id})
        if delete_result.deleted_count > 0:
            logger.info(f"Deleted {delete_result.deleted_count} old chunks for version {file_version_id}")
        delete_chunks_by_version(file_version_id)

        # Insert new chunks
        if chunks:
            db.chunks.insert_many(chunks, ordered=False)
            logger.info(f"Stored {len(chunks)} chunks in MongoDB for version {file_version_id}")
        # Insert embeddings into LanceDB (only chunks with embeddings)
        chunks_with_embeddings = [c for c in chunks if c.get("embedding") is not None]
        if chunks_with_embeddings:
            lance_count = insert_chunk_embeddings(chunks_with_embeddings)
            logger.info(f"Stored {lance_count} chunk embeddings in LanceDB")
    except Exception as e:
        logger.error(f"Error storing chunks in MongoDB: {e}")
        raise


def save_chunks_to_json(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save chunks with metadata to a JSON file.
    
    Args:
        chunks: List of chunks with metadata
        output_path: Path to save the JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def get_chunks_for_version(file_version_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for a specific file version from MongoDB.
    
    Args:
        file_version_id: File version ID
        active_only: If True, only return active chunks
        
    Returns:
        List of chunk documents
    """
    query = {"file_version_id": file_version_id}
    if active_only:
        query["is_active"] = True
    
    chunks = list(db.chunks.find(query).sort("chunk_index", 1))
    return chunks


def get_chunks_for_document(doc_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a document across all versions.
    
    Args:
        doc_id: Document ID
        active_only: If True, only return active chunks
        
    Returns:
        List of chunk documents
    """
    query = {"doc_id": doc_id}
    if active_only:
        query["is_active"] = True
    
    chunks = list(db.chunks.find(query).sort([("file_version_id", 1), ("chunk_index", 1)]))
    return chunks


def get_chunks_for_dataroom(dataroom_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a data room.
    
    Args:
        dataroom_id: Data room ID (project_id)
        active_only: If True, only return active chunks
        
    Returns:
        List of chunk documents
    """
    query = {"dataroom_id": dataroom_id}
    if active_only:
        query["is_active"] = True
    
    chunks = list(db.chunks.find(query).sort([("doc_id", 1), ("chunk_index", 1)]))
    return chunks


def search_chunks_by_text(
    dataroom_id: str,
    search_text: str,
    limit: int = 10,
    active_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Search chunks by text content (searches both text and normalized_text).
    
    Args:
        dataroom_id: Data room ID to search within
        search_text: Text to search for
        limit: Maximum number of results
        active_only: If True, only search active chunks
        
    Returns:
        List of matching chunk documents
    """
    query = {
        "dataroom_id": dataroom_id,
        "$or": [
            {"text": {"$regex": search_text, "$options": "i"}},
            {"normalized_text": {"$regex": normalize_text(search_text), "$options": "i"}}
        ]
    }
    if active_only:
        query["is_active"] = True
    
    chunks = list(db.chunks.find(query).limit(limit))
    return chunks


def search_chunks_by_section(
    dataroom_id: str,
    section_path: str,
    active_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Search chunks by section path.
    
    Args:
        dataroom_id: Data room ID to search within
        section_path: Section path to search for (e.g., "Chapter 1 > Section 1.1")
        active_only: If True, only search active chunks
        
    Returns:
        List of matching chunk documents
    """
    query = {
        "dataroom_id": dataroom_id,
        "section_path": {"$regex": section_path, "$options": "i"}
    }
    if active_only:
        query["is_active"] = True
    
    chunks = list(db.chunks.find(query).sort("chunk_index", 1))
    return chunks


def update_chunk_embedding(chunk_id: str, embedding: List[float]) -> bool:
    """
    Update the embedding for a specific chunk.
    
    Args:
        chunk_id: Chunk ID
        embedding: Embedding vector
        
    Returns:
        True if successful, False otherwise
    """
    try:
        result = db.chunks.update_one(
            {"_id": chunk_id},
            {
                "$set": {
                    "embedding": embedding,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error updating chunk embedding: {e}")
        return False


def batch_update_embeddings(chunk_embeddings: Dict[str, List[float]]) -> int:
    """
    Batch update embeddings for multiple chunks.
    
    Args:
        chunk_embeddings: Dict mapping chunk_id to embedding vector
        
    Returns:
        Number of chunks updated
    """
    updated_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    
    try:
        for chunk_id, embedding in chunk_embeddings.items():
            result = db.chunks.update_one(
                {"_id": chunk_id},
                {
                    "$set": {
                        "embedding": embedding,
                        "updated_at": now_iso
                    }
                }
            )
            if result.modified_count > 0:
                updated_count += 1
        
        logger.info(f"Updated embeddings for {updated_count} chunks")
    except Exception as e:
        logger.error(f"Error batch updating embeddings: {e}")
    
    return updated_count