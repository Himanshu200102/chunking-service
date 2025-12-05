"""LanceDB client wrapper for vector-based chunk storage and retrieval."""
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

from app.retriever.config import settings
from app.retriever.models import ChunkDocument, ChunkResult

logger = logging.getLogger(__name__)


class LanceDBClient:
    """LanceDB client for managing chunks with vector embeddings."""
    
    def __init__(self, db_path: Optional[str] = None, embedding_model_name: Optional[str] = None):
        """
        Initialize LanceDB client.
        
        Args:
            db_path: Path to LanceDB database directory (default: ./lancedb_data)
            embedding_model_name: Name of the sentence transformer model for embeddings
                                  (default: from settings or BAAI/bge-small-en-v1.5)
        """
        self.db_path = db_path or getattr(settings, 'lancedb_path', './lancedb_data')
        # Use same model as rest of system for consistency
        self.embedding_model_name = embedding_model_name or getattr(settings, 'lancedb_embedding_model', None) or "BAAI/bge-small-en-v1.5"
        # Fixed embedding dimension (384 for all-MiniLM-L6-v2, 384 for bge-small-en-v1.5)
        self.embedding_dimension = 384
        
        # Create database directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(self.db_path)
        
        # Lazy-load embedding model (only when needed for generating new embeddings)
        self._embedding_model = None
        logger.info(f"LanceDB client initialized (embedding model will be loaded on-demand if needed)")
        
        # Table name
        self.table_name = getattr(settings, 'lancedb_table_name', 'chunks')
        
        # Ensure table exists
        self._ensure_table_exists()
    
    def _get_embedding_model(self):
        """Lazy-load embedding model only when needed."""
        if self._embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                # Set environment variables for better CPU performance
                os.environ["OMP_NUM_THREADS"] = "4"
                os.environ["MKL_NUM_THREADS"] = "4"
                os.environ["OPENBLAS_NUM_THREADS"] = "4"
                self._embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                logger.info(f"✅ Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def _ensure_table_exists(self):
        """Create table if it doesn't exist."""
        try:
            if self.table_name not in self.db.table_names():
                # Create empty table with schema
                schema = {
                    "vector": "float32",
                    "chunk_ref": "string",
                    "projectid": "string",
                    "fileid": "string",
                    "file_version_id": "string",
                    "text": "string",
                    "section_path": "string",  # JSON string
                    "object_type": "string",
                    "page_range": "string",  # JSON string "[start, end]"
                    "caption": "string",
                    "metadata": "string",  # JSON string
                    "is_active": "bool",
                    "created_at": "string",
                    "updated_at": "string"
                }
                # Create with sample data (all strings, no None values to avoid schema conflicts)
                sample_data = [{
                    "vector": np.zeros(self.embedding_dimension, dtype=np.float32),
                    "chunk_ref": "",
                    "projectid": "",
                    "fileid": "",
                    "file_version_id": "",
                    "text": "",
                    "section_path": "[]",
                    "object_type": "narrative",
                    "page_range": "[0, 0]",
                    "caption": "",
                    "metadata": "{}",
                    "is_active": True,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }]
                self.db.create_table(self.table_name, sample_data, mode="overwrite")
                logger.info(f"Created LanceDB table: {self.table_name}")
        except Exception as e:
            logger.error(f"Error ensuring table exists: {e}")
            # Table might already exist, try to open it
            try:
                self.table = self.db.open_table(self.table_name)
                logger.info(f"Opened existing LanceDB table: {self.table_name}")
            except Exception as e2:
                logger.error(f"Failed to open table: {e2}")
                raise
    
    def _get_table(self):
        """Get or create the table."""
        try:
            if self.table_name in self.db.table_names():
                return self.db.open_table(self.table_name)
            else:
                self._ensure_table_exists()
                return self.db.open_table(self.table_name)
        except Exception as e:
            logger.error(f"Error getting table: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (lazy-loads model if needed)."""
        try:
            model = self._get_embedding_model()  # Lazy load
            embedding = model.encode(text, show_progress_bar=False, batch_size=16)
            return np.array(embedding).astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch (more efficient)."""
        try:
            if not texts:
                return []
            
            model = self._get_embedding_model()  # Lazy load
            logger.info(f"🔄 Encoding {len(texts)} texts in batch (batch_size=4)...")
            
            # Use smaller batch size for CPU stability (4 instead of 16)
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                batch_size=4  # Reduced from 16 for better CPU stability
            )
            logger.info(f"✅ Successfully encoded {len(texts)} texts")
            return [np.array(emb).astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"❌ Error generating batch embeddings: {e}", exc_info=True)
            raise
    
    def _chunk_to_dict_with_embedding(self, chunk_data: Dict[str, Any], projectid: str, fileid: str, file_version_id: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Convert chunk data to LanceDB format with pre-generated embedding."""
        import json
        
        # Use provided embedding (already generated in batch)
        text = chunk_data.get("text", "")
        if not text:
            raise ValueError(f"Chunk {chunk_data.get('chunk_ref', 'unknown')} has no text field")
        
        # Normalize section_path (ensure it's a list)
        section_path = chunk_data.get("section_path", [])
        if isinstance(section_path, str):
            try:
                section_path = json.loads(section_path)
            except:
                section_path = [section_path] if section_path else []
        if not isinstance(section_path, list):
            section_path = []
        
        # Normalize page_range (ensure it's a list)
        page_range = chunk_data.get("page_range", [0, 0])
        if isinstance(page_range, str):
            try:
                page_range = json.loads(page_range)
            except:
                page_range = [1, 1]
        if not isinstance(page_range, list):
            page_range = [page_range] if page_range else [1, 1]
        if len(page_range) < 2:
            page_range = [page_range[0] if page_range else 1, page_range[0] if page_range else 1]
        
        # Normalize metadata (ensure it's a dict)
        metadata = chunk_data.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Ensure caption is a string (never None)
        caption = chunk_data.get("caption")
        if caption is None or caption == "None":
            caption = ""
        elif not isinstance(caption, str):
            caption = str(caption)
        
        return {
            "vector": embedding,
            "chunk_ref": chunk_data.get("chunk_ref", ""),
            "projectid": projectid or "",
            "fileid": fileid or "",
            "file_version_id": file_version_id or "",
            "text": text,
            "section_path": json.dumps(section_path),
            "object_type": chunk_data.get("object_type", "narrative"),
            "page_range": json.dumps(page_range),
            "caption": caption,
            "metadata": json.dumps(metadata),
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    def ingest_chunks(
        self,
        projectid: str,
        fileid: str,
        file_version_id: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest chunks into LanceDB.
        
        First, deactivates old chunks for the same file_version_id,
        then indexes new chunks with embeddings.
        """
        try:
            table = self._get_table()
            
            # Deactivate old chunks for this file version
            self._deactivate_old_chunks(fileid, file_version_id)
            
            # Prepare chunks with embeddings (batch processing for efficiency)
            chunk_records = []
            indexed_count = 0
            errors = []
            
            # Check if chunks already have embeddings (from parsing)
            chunks_with_embeddings = [c for c in chunks if c.get("embedding") is not None]
            chunks_without_embeddings = [c for c in chunks if c.get("embedding") is None]
            
            logger.info(f"📊 Chunks status: {len(chunks_with_embeddings)} with embeddings, {len(chunks_without_embeddings)} without")
            
            embeddings = []
            
            if chunks_without_embeddings:
                # Generate embeddings only for chunks that don't have them
                texts_to_encode = [chunk.get("text", "") for chunk in chunks_without_embeddings]
                logger.info(f"🔄 Generating {len(texts_to_encode)} missing embeddings in batch...")
                
                try:
                    new_embeddings = self._generate_embeddings_batch(texts_to_encode)
                    logger.info(f"✅ Generated {len(new_embeddings)} new embeddings successfully")
                    
                    # Assign new embeddings to chunks
                    for chunk, emb in zip(chunks_without_embeddings, new_embeddings):
                        chunk["embedding"] = emb
                except Exception as e:
                    logger.error(f"❌ Failed to generate batch embeddings: {e}")
                    raise
            else:
                logger.info(f"✅ All chunks already have embeddings from parsing - skipping generation!")
            
            # Now create chunk records (all chunks should have embeddings now)
            for chunk_data in chunks:
                try:
                    # Use embedding from chunk (either pre-existing or just generated)
                    embedding = chunk_data.get("embedding")
                    if embedding is None:
                        logger.warning(f"Chunk {chunk_data.get('chunk_ref')} still missing embedding, using zero vector")
                        embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                    elif isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    chunk_dict = self._chunk_to_dict_with_embedding(
                        chunk_data, projectid, fileid, file_version_id, embedding
                    )
                    chunk_records.append(chunk_dict)
                    indexed_count += 1
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_data.get('chunk_ref', 'unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Insert chunks into table
            if chunk_records:
                try:
                    table.add(chunk_records)
                    logger.info(f"Inserted {indexed_count} chunks into LanceDB")
                except Exception as e:
                    logger.error(f"Error inserting chunks: {e}")
                    raise
            
            return {
                "success": True,
                "indexed": indexed_count,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error ingesting chunks into LanceDB: {e}")
            raise
    
    def _deactivate_old_chunks(self, fileid: str, file_version_id: str):
        """
        Deactivate old chunks for a file when re-chunking.
        """
        try:
            table = self._get_table()
            
            # Query for active chunks with this fileid
            # Use a dummy vector for the search (we're filtering by metadata, not similarity)
            dummy_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
            results = table.search(dummy_vector).where(f"fileid = '{fileid}' AND is_active = true").limit(10000).to_pandas()
            
            if len(results) > 0:
                # Update is_active to False (batch embedding generation)
                texts = [row["text"] for _, row in results.iterrows()]
                logger.info(f"Deactivating {len(texts)} old chunks, generating embeddings in batch...")
                embeddings = self._generate_embeddings_batch(texts)
                
                update_data = []
                for i, (_, row) in enumerate(results.iterrows()):
                    row_dict = row.to_dict()
                    row_dict["is_active"] = False
                    row_dict["updated_at"] = datetime.utcnow().isoformat()
                    # Use pre-generated embedding
                    row_dict["vector"] = embeddings[i]
                    update_data.append(row_dict)
                
                # Delete old records and insert updated ones
                # Note: LanceDB doesn't have direct update, so we delete and re-insert
                chunk_refs_to_delete = results["chunk_ref"].tolist()
                for chunk_ref in chunk_refs_to_delete:
                    table.delete(f"chunk_ref = '{chunk_ref}'")
                
                if update_data:
                    table.add(update_data)
                
                logger.info(f"Deactivated {len(update_data)} old chunks for fileid: {fileid}")
        except Exception as e:
            logger.warning(f"Error deactivating old chunks in LanceDB: {e}")
    
    def search_chunks(
        self,
        query: str,
        projectid: Optional[str] = None,
        fileid: Optional[str] = None,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Search for chunks using vector similarity search.
        
        Supports:
        - Global search across all files in a project
        - File-specific search
        - Additional filters
        """
        try:
            table = self._get_table()
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build where clause
            where_clauses = ["is_active = true"]
            
            if projectid:
                where_clauses.append(f"projectid = '{projectid}'")
            
            if fileid:
                where_clauses.append(f"fileid = '{fileid}'")
            
            if filters:
                if "object_type" in filters:
                    where_clauses.append(f"object_type = '{filters['object_type']}'")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else None
            
            # Perform vector search
            search_builder = table.search(query_embedding).limit(max_results)
            
            if where_clause:
                search_builder = search_builder.where(where_clause)
            
            try:
                results_df = search_builder.to_pandas()
            except Exception as e:
                logger.warning(f"Error converting to pandas, trying to_list: {e}")
                # Fallback: convert to list and then to pandas manually
                results_list = search_builder.to_list()
                if not results_list:
                    return []
                import pandas as pd
                results_df = pd.DataFrame(results_list)
            
            # Parse results
            results = []
            import json
            
            for _, row in results_df.iterrows():
                try:
                    # Parse JSON fields
                    section_path = json.loads(row.get("section_path", "[]"))
                    page_range = json.loads(row.get("page_range", "[0, 0]"))
                    metadata = json.loads(row.get("metadata", "{}")) if row.get("metadata") else None
                    
                    # Get distance (LanceDB returns distance, lower is better)
                    # Convert to score (higher is better) by inverting
                    distance = row.get("_distance", 0.0)
                    score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    results.append(ChunkResult(
                        chunk_ref=row["chunk_ref"],
                        text=row["text"],
                        section_path=section_path,
                        object_type=row.get("object_type", "narrative"),
                        page_range=page_range,
                        caption=row.get("caption") or None,
                        metadata=metadata,
                        fileid=row["fileid"],
                        file_version_id=row["file_version_id"],
                        score=float(score)
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing chunk result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks in LanceDB: {e}")
            raise
    
    def delete_file_chunks(self, projectid: str, fileid: str) -> Dict[str, Any]:
        """Delete all chunks for a specific file."""
        try:
            table = self._get_table()
            
            # Find chunks to delete
            # Use a dummy vector for the search (we're filtering by metadata, not similarity)
            dummy_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
            results = table.search(dummy_vector).where(f"projectid = '{projectid}' AND fileid = '{fileid}'").limit(10000).to_pandas()
            
            deleted_count = 0
            for chunk_ref in results["chunk_ref"].tolist():
                table.delete(f"chunk_ref = '{chunk_ref}'")
                deleted_count += 1
            
            return {
                "success": True,
                "deleted": deleted_count
            }
        except Exception as e:
            logger.error(f"Error deleting file chunks from LanceDB: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check LanceDB health."""
        try:
            table = self._get_table()
            count = table.count_rows()
            return {
                "status": "healthy",
                "table_name": self.table_name,
                "total_chunks": count
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

