"""Agent-based retrieval service that intelligently routes queries."""
import logging
from typing import List, Optional, Dict, Any
from app.retriever.opensearch_client import OpenSearchClient
from app.retriever.hybrid_client import HybridClient
from app.retriever.agent import get_agent
from app.retriever.models import ChunkResult
from app.retriever.summarizer import get_summarizer

logger = logging.getLogger(__name__)


class AgentRetriever:
    """
    Intelligent retrieval service that uses an agent to decide
    between file-specific and global retrieval strategies.
    """
    
    def __init__(self, client):
        """
        Initialize the agent retriever.
        
        Args:
            client: HybridClient or OpenSearchClient instance
        """
        self.client = client
        # For backward compatibility, support both HybridClient and OpenSearchClient
        if isinstance(client, HybridClient):
            self.opensearch_client = client.opensearch_client
        else:
            self.opensearch_client = client
        self.agent = get_agent()
    
    def retrieve(
        self,
        query: str,
        projectid: str,
        available_fileids: Optional[List[str]] = None,
        max_results_per_file: int = 10,
        max_results_global: int = 10,
        file_id_to_name: Optional[Dict[str, str]] = None,  # Pre-fetched file name mapping
    ) -> Dict[str, Any]:
        """
        Intelligently retrieve chunks based on agent's decision.
        
        Returns:
            {
                "strategy": "file_specific" | "global",
                "reasoning": "explanation",
                "results": {
                    "fileid1": [chunks],
                    "fileid2": [chunks],
                    ...
                } or [chunks] for global
            }
        """
        # Get available fileids if not provided
        if available_fileids is None:
            available_fileids = self._get_available_fileids(projectid)
        
        # Agent decides the strategy
        decision = self.agent.decide_retrieval_strategy(
            query=query,
            projectid=projectid,
            available_fileids=available_fileids
        )
        
        strategy = decision["strategy"]
        reasoning = decision["reasoning"]
        
        logger.info(f"Agent decision: {strategy} - {reasoning}")
        
        # Execute retrieval based on strategy
        if strategy == "file_specific":
            results = self._file_specific_retrieval(
                query=query,
                projectid=projectid,
                fileids=decision.get("fileids") or available_fileids,
                max_results=max_results_per_file
            )
            # Generate summaries for each file's top chunks
            summaries = self._generate_file_summaries(results, query, projectid, file_id_to_name)
        else:
            results = self._global_retrieval(
                query=query,
                projectid=projectid,
                max_results=max_results_global
            )
            # Generate summary for top chunks
            summaries = self._generate_global_summary(results, query, projectid, file_id_to_name)
        
        return {
            "strategy": strategy,
            "reasoning": reasoning,
            "query": query,
            "projectid": projectid,
            "results": results,
            "summaries": summaries
        }
    
    def _file_specific_retrieval(
        self,
        query: str,
        projectid: str,
        fileids: List[str],
        max_results: int
    ) -> Dict[str, List[ChunkResult]]:
        """Retrieve chunks for each file separately."""
        file_results = {}
        
        for fileid in fileids:
            try:
                # Use hybrid client if available, otherwise fall back to opensearch_client
                search_client = self.client if isinstance(self.client, HybridClient) else self.opensearch_client
                chunks = search_client.search_chunks(
                    query=query,
                    projectid=projectid,
                    fileid=fileid,
                    max_results=max_results
                )
                file_results[fileid] = chunks
                logger.info(f"Retrieved {len(chunks)} chunks for file {fileid}")
            except Exception as e:
                logger.error(f"Error retrieving chunks for file {fileid}: {e}")
                file_results[fileid] = []
        
        return file_results
    
    def _global_retrieval(
        self,
        query: str,
        projectid: str,
        max_results: int
    ) -> List[ChunkResult]:
        """Retrieve top chunks across all files."""
        try:
            chunks = self.opensearch_client.search_chunks(
                query=query,
                projectid=projectid,
                fileid=None,  # No file filter for global search
                max_results=max_results
            )
            logger.info(f"Retrieved {len(chunks)} chunks globally")
            return chunks
        except Exception as e:
            logger.error(f"Error in global retrieval: {e}")
            return []
    
    def _generate_file_summaries(
        self,
        file_results: Dict[str, List[ChunkResult]],
        query: str,
        projectid: str,
        file_id_to_name: Optional[Dict[str, str]] = None
    ) -> Dict[str, Optional[str]]:
        """Generate summaries for each file's chunks."""
        summaries = {}
        try:
            summarizer = get_summarizer()
            for fileid, chunks in file_results.items():
                if chunks:
                    summary_result = summarizer.summarize_chunks(
                        chunks=chunks,
                        query=query,
                        projectid=projectid,
                        max_chunks_to_summarize=4,
                        file_id_to_name=file_id_to_name
                    )
                    summaries[fileid] = summary_result.get("summary") if summary_result else None
                else:
                    summaries[fileid] = None
        except Exception as e:
            logger.warning(f"Failed to generate file summaries: {e}")
            # Return None for all files if summarization fails
            summaries = {fileid: None for fileid in file_results.keys()}
        return summaries
    
    async def retrieve_sequential(
        self,
        query: str,
        projectid: str,
        available_fileids: Optional[List[str]] = None,
        max_results_per_file: int = 10,
        file_id_to_name: Optional[Dict[str, str]] = None
    ):
        """
        Sequential retrieval and inference for file-specific queries.
        
        For each file:
        1. Retrieve chunks
        2. Generate inference/summary
        3. Yield result
        
        This is a generator that yields results as they're processed.
        
        Args:
            file_id_to_name: Pre-fetched file names mapping (fileid -> filename)
        """
        # Get available fileids if not provided
        if available_fileids is None:
            available_fileids = self._get_available_fileids(projectid)
        
        if not available_fileids:
            yield {
                "type": "error",
                "message": "No files available in project"
            }
            return
        
        # Process each file sequentially
        for fileid in available_fileids:
            try:
                # Step 1: Retrieve chunks for this file
                yield {
                    "type": "status",
                    "fileid": fileid,
                    "message": f"Retrieving chunks for file {fileid}..."
                }
                
                # Use hybrid client if available, otherwise fall back to opensearch_client
                search_client = self.client if isinstance(self.client, HybridClient) else self.opensearch_client
                chunks = search_client.search_chunks(
                    query=query,
                    projectid=projectid,
                    fileid=fileid,
                    max_results=max_results_per_file
                )
                
                if not chunks:
                    yield {
                        "type": "file_result",
                        "fileid": fileid,
                        "chunks": [],
                        "summary": None,
                        "message": f"No relevant chunks found for file {fileid}"
                    }
                    continue
                
                # Step 2: Generate inference/summary for this file
                filename = file_id_to_name.get(fileid) if file_id_to_name else fileid
                yield {
                    "type": "status",
                    "fileid": fileid,
                    "filename": filename,
                    "message": f"Generating inference for file {filename}..."
                }
                
                summarizer = get_summarizer()
                try:
                    summary_result = summarizer.summarize_chunks(
                        chunks=chunks,
                        query=query,
                        projectid=projectid,
                        max_chunks_to_summarize=max_results_per_file,
                        file_id_to_name=file_id_to_name
                    )
                    
                    summary = summary_result.get("summary") if summary_result else None
                    compression_stats = summary_result.get("compression_stats") if summary_result else None
                    
                    if not summary:
                        logger.warning(f"❌ Summarizer returned empty summary for file {fileid}. Summary result: {summary_result}")
                        # Fallback: use first few chunks as summary
                        if chunks:
                            fallback_texts = [ch.text[:200] for ch in chunks[:3]]
                            summary = f"[Inference generation failed. Here are the relevant excerpts:]<br><br>" + "<br><br>".join(fallback_texts)
                            logger.info(f"Using fallback summary for file {fileid}")
                    else:
                        logger.info(f"✅ Generated summary for file {fileid}: {len(summary)} chars")
                except Exception as e:
                    logger.error(f"❌ Error generating summary for file {fileid}: {e}", exc_info=True)
                    summary = None
                    compression_stats = None
                    # Fallback: use first few chunks
                    if chunks:
                        fallback_texts = [ch.text[:200] for ch in chunks[:3]]
                        summary = f"[Error generating inference: {str(e)}]<br><br>Relevant excerpts:<br><br>" + "<br><br>".join(fallback_texts)
                        logger.info(f"Using error fallback summary for file {fileid}")
                
                # Step 3: Yield result for this file
                yield {
                    "type": "file_result",
                    "fileid": fileid,
                    "filename": filename,
                    "chunks": chunks,
                    "summary": summary,
                    "compression_stats": compression_stats,
                    "chunks_count": len(chunks)
                }
                
            except Exception as e:
                logger.error(f"Error processing file {fileid}: {e}")
                yield {
                    "type": "file_error",
                    "fileid": fileid,
                    "error": str(e)
                }
        
        # Final completion message
        yield {
            "type": "complete",
            "message": "All files processed"
        }
    
    def _generate_global_summary(
        self,
        chunks: List[ChunkResult],
        query: str,
        projectid: str,
        file_id_to_name: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Generate summary for global retrieval results."""
        if not chunks:
            logger.warning("No chunks available for global summary generation")
            return None
        try:
            logger.info(f"Generating global summary for {len(chunks)} chunks with query: {query[:50]}...")
            summarizer = get_summarizer()
            summary_result = summarizer.summarize_chunks(
                chunks=chunks,
                query=query,
                projectid=projectid,
                max_chunks_to_summarize=15,  # Retrieve 15 chunks for global summary
                file_id_to_name=file_id_to_name
            )
            if summary_result:
                summary = summary_result.get("summary")
                if summary:
                    logger.info(f"Successfully generated global summary ({len(summary)} chars)")
                    return summary
                else:
                    logger.warning("Summary result exists but 'summary' field is empty")
            else:
                logger.warning("summarize_chunks returned None")
            return None
        except Exception as e:
            logger.error(f"Failed to generate global summary: {e}", exc_info=True)
            return None
    
    def _get_available_fileids(self, projectid: str) -> List[str]:
        """Get list of available fileids for a project."""
        try:
            # Query OpenSearch to get distinct fileids
            search_body = {
                "size": 0,
                "aggs": {
                    "unique_fileids": {
                        "terms": {
                            "field": "fileid",
                            "size": 100
                        }
                    }
                },
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"projectid": projectid}},
                            {"term": {"is_active": True}}
                        ]
                    }
                }
            }
            
            response = self.opensearch_client.client.search(
                index=self.opensearch_client.index_name,
                body=search_body
            )
            
            fileids = [
                bucket["key"]
                for bucket in response.get("aggregations", {})
                .get("unique_fileids", {})
                .get("buckets", [])
            ]
            
            return fileids
        except Exception as e:
            logger.error(f"Error getting available fileids: {e}")
            return []

