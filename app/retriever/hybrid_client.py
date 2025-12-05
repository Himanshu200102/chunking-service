"""Hybrid client that combines OpenSearch and LanceDB for dual-storage retrieval."""
import logging
from typing import List, Optional, Dict, Any
from collections import defaultdict

from app.retriever.opensearch_client import OpenSearchClient
from app.retriever.lancedb_client import LanceDBClient
from app.retriever.models import ChunkResult

logger = logging.getLogger(__name__)


class HybridClient:
    """
    Hybrid client that uses both OpenSearch (keyword/BM25) and LanceDB (vector similarity).
    
    - Ingests chunks into both databases
    - Retrieves from both and merges results using reciprocal rank fusion (RRF)
    """
    
    def __init__(self):
        """Initialize both OpenSearch and LanceDB clients."""
        self.opensearch_client = OpenSearchClient()
        self.lancedb_client = LanceDBClient()
        logger.info("Initialized HybridClient with OpenSearch and LanceDB")
    
    def ingest_chunks(
        self,
        projectid: str,
        fileid: str,
        file_version_id: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest chunks into both OpenSearch and LanceDB.
        
        Returns combined results from both operations.
        """
        try:
            # Ingest into OpenSearch
            opensearch_result = self.opensearch_client.ingest_chunks(
                projectid=projectid,
                fileid=fileid,
                file_version_id=file_version_id,
                chunks=chunks
            )
            
            # Ingest into LanceDB
            lancedb_result = self.lancedb_client.ingest_chunks(
                projectid=projectid,
                fileid=fileid,
                file_version_id=file_version_id,
                chunks=chunks
            )
            
            # Combine results
            total_indexed = opensearch_result.get("indexed", 0)
            total_errors = opensearch_result.get("errors", []) + lancedb_result.get("errors", [])
            
            return {
                "success": True,
                "indexed": total_indexed,
                "opensearch": opensearch_result,
                "lancedb": lancedb_result,
                "errors": total_errors
            }
            
        except Exception as e:
            logger.error(f"Error ingesting chunks into hybrid storage: {e}")
            raise
    
    def search_chunks(
        self,
        query: str,
        projectid: Optional[str] = None,
        fileid: Optional[str] = None,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fusion_k: int = 60
    ) -> List[ChunkResult]:
        """
        Search chunks using both OpenSearch and LanceDB, then merge results using RRF.
        
        Args:
            query: Search query text
            projectid: Optional project ID filter
            fileid: Optional file ID filter
            max_results: Maximum number of results to return
            filters: Additional filters
            fusion_k: RRF constant (higher = more weight to top results)
        
        Returns:
            Merged and deduplicated results sorted by combined score
        """
        try:
            # Search OpenSearch (keyword/BM25)
            opensearch_results = self.opensearch_client.search_chunks(
                query=query,
                projectid=projectid,
                fileid=fileid,
                max_results=max_results * 2,  # Get more results for better fusion
                filters=filters
            )
            
            # Search LanceDB (vector similarity)
            lancedb_results = self.lancedb_client.search_chunks(
                query=query,
                projectid=projectid,
                fileid=fileid,
                max_results=max_results * 2,  # Get more results for better fusion
                filters=filters
            )
            
            # Merge results using Reciprocal Rank Fusion (RRF)
            merged_results = self._merge_results_rrf(
                opensearch_results,
                lancedb_results,
                max_results=max_results,
                fusion_k=fusion_k
            )
            
            logger.info(
                f"Hybrid search: OpenSearch={len(opensearch_results)}, "
                f"LanceDB={len(lancedb_results)}, Merged={len(merged_results)}"
            )
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to OpenSearch only if LanceDB fails
            logger.warning("Falling back to OpenSearch only")
            return self.opensearch_client.search_chunks(
                query=query,
                projectid=projectid,
                fileid=fileid,
                max_results=max_results,
                filters=filters
            )
    
    def _merge_results_rrf(
        self,
        opensearch_results: List[ChunkResult],
        lancedb_results: List[ChunkResult],
        max_results: int = 10,
        fusion_k: int = 60
    ) -> List[ChunkResult]:
        """
        Merge results from OpenSearch and LanceDB using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each result set
        Higher RRF score = better overall ranking
        """
        # Build maps: chunk_ref -> (rank, result)
        opensearch_map = {}
        lancedb_map = {}
        
        for rank, result in enumerate(opensearch_results, start=1):
            opensearch_map[result.chunk_ref] = (rank, result)
        
        for rank, result in enumerate(lancedb_results, start=1):
            lancedb_map[result.chunk_ref] = (rank, result)
        
        # Get all unique chunk_refs
        all_chunk_refs = set(opensearch_map.keys()) | set(lancedb_map.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        merged_results_map = {}
        
        for chunk_ref in all_chunk_refs:
            rrf_score = 0.0
            
            # Add OpenSearch contribution
            if chunk_ref in opensearch_map:
                rank, result = opensearch_map[chunk_ref]
                rrf_score += 1.0 / (fusion_k + rank)
                merged_results_map[chunk_ref] = result
            
            # Add LanceDB contribution
            if chunk_ref in lancedb_map:
                rank, result = lancedb_map[chunk_ref]
                rrf_score += 1.0 / (fusion_k + rank)
                # Use LanceDB result if OpenSearch doesn't have it, or prefer the one with better individual score
                if chunk_ref not in merged_results_map:
                    merged_results_map[chunk_ref] = result
                else:
                    # Prefer result with higher individual score
                    existing_score = merged_results_map[chunk_ref].score
                    if result.score > existing_score:
                        merged_results_map[chunk_ref] = result
            
            rrf_scores[chunk_ref] = rrf_score
        
        # Sort by RRF score (descending)
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final results with updated scores
        final_results = []
        for chunk_ref, rrf_score in sorted_chunks[:max_results]:
            result = merged_results_map[chunk_ref]
            # Update score to RRF score (normalized to 0-1 range)
            # RRF scores are typically small, so we normalize them
            normalized_score = min(1.0, rrf_score * 10)  # Scale up for readability
            result.score = normalized_score
            final_results.append(result)
        
        return final_results
    
    def delete_file_chunks(self, projectid: str, fileid: str) -> Dict[str, Any]:
        """Delete all chunks for a specific file from both databases."""
        try:
            opensearch_result = self.opensearch_client.delete_file_chunks(projectid, fileid)
            lancedb_result = self.lancedb_client.delete_file_chunks(projectid, fileid)
            
            return {
                "success": True,
                "opensearch": opensearch_result,
                "lancedb": lancedb_result,
                "total_deleted": opensearch_result.get("deleted", 0) + lancedb_result.get("deleted", 0)
            }
        except Exception as e:
            logger.error(f"Error deleting file chunks: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of both databases."""
        opensearch_health = self.opensearch_client.health_check()
        lancedb_health = self.lancedb_client.health_check()
        
        return {
            "opensearch": opensearch_health,
            "lancedb": lancedb_health,
            "status": "healthy" if (
                opensearch_health.get("status") != "error" and
                lancedb_health.get("status") != "error"
            ) else "degraded"
        }

