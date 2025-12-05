"""OpenSearch client wrapper for chunk storage and retrieval."""
import logging
from typing import List, Optional, Dict, Any
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError, RequestError
from datetime import datetime

from app.retriever.config import settings
from app.retriever.models import ChunkDocument, ChunkResult

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client for managing chunks."""
    
    def __init__(self):
        """Initialize OpenSearch client."""
        self.client = OpenSearch(
            hosts=[{
                'host': settings.opensearch_host,
                'port': settings.opensearch_port
            }],
            http_compress=True,
            http_auth=(
                settings.opensearch_username,
                settings.opensearch_password
            ) if settings.opensearch_username and settings.opensearch_password else None,
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        self.index_name = settings.opensearch_index_name
        # Try to ensure index exists, but don't fail if OpenSearch isn't ready yet
        try:
            self._ensure_index_exists()
        except Exception as e:
            logger.warning(f"Could not ensure index exists during initialization (OpenSearch may not be ready): {e}")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist with proper mapping."""
        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "projectid": {"type": "keyword"},
                    "fileid": {"type": "keyword"},
                    "file_version_id": {"type": "keyword"},
                    "chunk_ref": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "section_path": {"type": "keyword"},
                    "object_type": {"type": "keyword"},
                    "page_range": {"type": "integer"},
                    "caption": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "num_rows": {"type": "integer"},
                            "num_cols": {"type": "integer"},
                            "has_header": {"type": "boolean"},
                            "figure_type": {"type": "keyword"}
                        }
                    },
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist, creating it...")
                self.client.indices.create(index=self.index_name, body=index_body)
                logger.info(f"✅ Created OpenSearch index: {self.index_name}")
            else:
                logger.debug(f"Index {self.index_name} already exists")
        except NotFoundError:
            # Index doesn't exist (exists() may raise NotFoundError in some cases)
            logger.info(f"Index {self.index_name} not found, attempting to create...")
            try:
                self.client.indices.create(index=self.index_name, body=index_body)
                logger.info(f"✅ Created OpenSearch index: {self.index_name}")
            except Exception as create_error:
                logger.error(f"Failed to create index {self.index_name}: {create_error}")
                raise
        except Exception as e:
            logger.error(f"Error checking/creating index {self.index_name}: {e}")
            raise
    
    def ingest_chunks(
        self,
        projectid: str,
        fileid: str,
        file_version_id: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest chunks into OpenSearch.
        
        First, deactivates old chunks for the same file_version_id,
        then indexes new chunks.
        """
        try:
            # Ensure index exists before ingesting
            self._ensure_index_exists()
            # Deactivate old chunks for this file version
            self._deactivate_old_chunks(fileid, file_version_id)
            
            # Prepare bulk operations
            bulk_operations = []
            indexed_count = 0
            errors = []
            
            for chunk_data in chunks:
                # Extract metadata (already a dict from model_dump())
                metadata = chunk_data.get("metadata")
                
                chunk_doc = ChunkDocument(
                    id=chunk_data["chunk_ref"],
                    projectid=projectid,
                    fileid=fileid,
                    file_version_id=file_version_id,
                    chunk_ref=chunk_data["chunk_ref"],
                    text=chunk_data["text"],
                    section_path=chunk_data.get("section_path", []),
                    object_type=chunk_data.get("object_type", "narrative"),
                    page_range=chunk_data["page_range"],
                    caption=chunk_data.get("caption"),
                    metadata=metadata,
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                # Add index operation
                bulk_operations.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": chunk_doc.id
                    }
                })
                # Exclude _id from document body (it's a metadata field, not a document field)
                doc_dict = chunk_doc.model_dump(by_alias=True, exclude_none=True)
                # Remove _id if it exists in the dict (it shouldn't with by_alias=True, but just in case)
                doc_dict.pop("_id", None)
                bulk_operations.append(doc_dict)
                indexed_count += 1
            
            # Execute bulk operation
            if bulk_operations:
                response = self.client.bulk(body=bulk_operations, refresh=True)
                
                # Check for errors
                if response.get("errors"):
                    for item in response.get("items", []):
                        if "error" in item.get("index", {}):
                            errors.append(item["index"]["error"])
                    logger.warning(f"Some chunks failed to index: {errors}")
            
            return {
                "success": True,
                "indexed": indexed_count,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")
            raise
    
    def _deactivate_old_chunks(self, fileid: str, file_version_id: str):
        """
        Deactivate old chunks for a file when re-chunking.
        
        When new chunks are ingested, deactivates all existing active chunks
        for the same fileid to ensure only the latest chunks are active.
        This handles both re-chunking the same version and creating new versions.
        """
        try:
            # Deactivate all active chunks for this fileid
            # This ensures that when re-chunking, all old chunks are deactivated
            update_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"fileid": fileid}},
                            {"term": {"is_active": True}}
                        ]
                    }
                },
                "script": {
                    "source": "ctx._source.is_active = false; ctx._source.updated_at = params.now",
                    "params": {
                        "now": datetime.utcnow().isoformat()
                    }
                }
            }
            
            result = self.client.update_by_query(
                index=self.index_name,
                body=update_query,
                refresh=True
            )
            updated_count = result.get("updated", 0)
            if updated_count > 0:
                logger.info(f"Deactivated {updated_count} old chunks for fileid: {fileid} (new version: {file_version_id})")
        except Exception as e:
            logger.warning(f"Error deactivating old chunks: {e}")
    
    def search_chunks(
        self,
        query: str,
        projectid: Optional[str] = None,
        fileid: Optional[str] = None,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Search for chunks using OpenSearch.
        
        Supports:
        - Global search across all files in a project
        - File-specific search
        - Additional filters
        """
        try:
            # Ensure index exists before searching
            self._ensure_index_exists()
            # Build query
            must_clauses = [
                {"term": {"is_active": True}},
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "section_path^1.5", "caption^1.2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                }
            ]
            
            # Add project filter if provided
            if projectid:
                must_clauses.append({"term": {"projectid": projectid}})
            
            # Add file filter if provided (for file-specific chat)
            if fileid:
                must_clauses.append({"term": {"fileid": fileid}})
            
            # Add additional filters
            if filters:
                if "object_type" in filters:
                    must_clauses.append({"term": {"object_type": filters["object_type"]}})
                if "page_range" in filters:
                    page_range = filters["page_range"]
                    if isinstance(page_range, list) and len(page_range) == 2:
                        must_clauses.append({
                            "range": {
                                "page_range": {
                                    "gte": page_range[0],
                                    "lte": page_range[1]
                                }
                            }
                        })
            
            search_body = {
                "query": {
                    "bool": {
                        "must": must_clauses
                    }
                },
                "size": max_results,
                "_source": {
                    "excludes": []
                }
            }
            
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Parse results
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit["_source"]
                results.append(ChunkResult(
                    chunk_ref=source["chunk_ref"],
                    text=source["text"],
                    section_path=source.get("section_path", []),
                    object_type=source.get("object_type", "narrative"),
                    page_range=source.get("page_range", []),
                    caption=source.get("caption"),
                    metadata=source.get("metadata"),
                    fileid=source["fileid"],
                    file_version_id=source["file_version_id"],
                    score=hit["_score"]
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
    
    def delete_file_chunks(self, projectid: str, fileid: str) -> Dict[str, Any]:
        """Delete all chunks for a specific file."""
        try:
            delete_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"projectid": projectid}},
                            {"term": {"fileid": fileid}}
                        ]
                    }
                }
            }
            
            response = self.client.delete_by_query(
                index=self.index_name,
                body=delete_query,
                refresh=True
            )
            
            return {
                "success": True,
                "deleted": response.get("deleted", 0)
            }
        except Exception as e:
            logger.error(f"Error deleting file chunks: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch cluster health."""
        try:
            health = self.client.cluster.health()
            return {
                "status": health.get("status"),
                "cluster_name": health.get("cluster_name"),
                "number_of_nodes": health.get("number_of_nodes")
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

