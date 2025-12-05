"""Summarization and context-compression module using Llama GGUF with provenance headers and citations."""

import logging
import os
import re
from typing import List, Optional, Dict, Tuple, Any, Set

import numpy as np
import httpx
from app.retriever.agent import get_agent
from app.retriever.config import settings
from app.retriever.models import ChunkResult

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore

logger = logging.getLogger(__name__)


_summarizer_llama_model = None  # Global cache for summarizer Llama model


class ChunkSummarizer:
    """
    Summarizes chunks using the Llama GGUF model, with:
    - Query-aware context compression (extractive + LLM)
    - Provenance headers (Doc, Page, Section)
    - Token-aware context merging (1,000-2,500 tokens)
    - Citation extraction and enforcement
    """

    def __init__(self):
        """
        Initialize the summarizer model.
        
        Priority:
        1. If SUMMARIZER_MODEL_PATH is configured and llama_cpp is available, load a dedicated
           Llama model from that GGUF file.
        2. Otherwise, fall back to the agent's model (so existing behavior keeps working).
        """
        global _summarizer_llama_model

        self.agent = get_agent()
        self.model = None

        # Prefer dedicated summarizer model if configured
        model_path = settings.summarizer_model_path
        if model_path and Llama is not None:
            try:
                if _summarizer_llama_model is None:
                    logger.info(
                        f"Loading summarizer Llama model from SUMMARIZER_MODEL_PATH={model_path}"
                    )
                    _summarizer_llama_model = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_threads=os.cpu_count() or 4,
                    )
                self.model = _summarizer_llama_model
            except Exception as e:  # pragma: no cover - runtime failure
                logger.error(
                    f"Failed to load summarizer model at {model_path}: {e}. "
                    "Falling back to agent model."
                )
                self.model = self.agent.model if hasattr(self.agent, "model") else None
        else:
            # Fallback: reuse agent model (previous behavior)
            if model_path and Llama is None:
                logger.warning(
                    "SUMMARIZER_MODEL_PATH is set but llama_cpp is not installed; "
                    "falling back to agent model."
                )
            self.model = self.agent.model if hasattr(self.agent, "model") else None
        self._embedding_model = None  # Lazy-loaded sentence embedding model
        self._tokenizer = None  # Lazy-loaded tokenizer for token counting
        self._file_name_cache: Dict[str, str] = {}  # Cache for file_id -> filename mapping

    # ---------------------------
    # Public API
    # ---------------------------

    def summarize_chunks(
        self,
        chunks: List[ChunkResult],
        query: str,
        projectid: Optional[str] = None,
        max_chunks_to_summarize: int = 15,  # Retrieve 15 chunks as requested
        sentences_per_chunk: int = 5,
        min_context_tokens: int = 1000,
        max_context_tokens: int = 2500,
        file_id_to_name: Optional[Dict[str, str]] = None,  # Pre-fetched file name mapping
    ) -> Optional[Dict[str, Any]]:
        """
        Summarize the top chunks related to a query, with context compression and citations.

        Pipeline:
        1. Take top-N chunks by score (default 15)
        2. Extractive compression per chunk (keep top-K sentences)
        3. Add provenance headers: [Doc: X | Page: Y | Section: Z]
        4. Merge context blocks with token limits (1,000-2,500 tokens)
        5. LLM generates answer with enforced citations
        6. Extract and format citations

        Returns:
            Dict with 'summary' (str) and 'compression_stats' (dict), or None if summarization fails
        """
        if not chunks:
            logger.warning("summarize_chunks called with empty chunks list")
            return None

        logger.info(f"Summarizing {len(chunks)} chunks for query: {query[:100]}...")

        # Fetch file names for all unique file IDs in chunks
        if file_id_to_name:
            # Use pre-fetched file names (from API layer)
            for file_id, filename in file_id_to_name.items():
                if file_id and filename:
                    self._file_name_cache[file_id] = filename
            # Store for use in citation extraction
            self._prefetched_file_names = file_id_to_name.copy()
            logger.info(f"✅ Loaded {len(file_id_to_name)} file names from pre-fetched mapping: {file_id_to_name}")
        else:
            self._prefetched_file_names = {}
            logger.warning("⚠ No file_id_to_name mapping provided to summarizer!")
        
        # Log what file IDs are in the chunks
        unique_file_ids = list(set(chunk.fileid for chunk in chunks if chunk.fileid))
        logger.info(f"Chunks contain {len(unique_file_ids)} unique file IDs: {unique_file_ids}")
        
        if projectid and not file_id_to_name:
            self._fetch_file_names(projectid, chunks)
            # Log file name cache status
            cached_count = sum(1 for fid in unique_file_ids if fid in self._file_name_cache and self._file_name_cache[fid] != fid)
            logger.info(f"File name cache: {cached_count}/{len(unique_file_ids)} file IDs have cached filenames")
        
        # Final check: log the mapping for chunks
        for fid in unique_file_ids:
            cached_name = self._file_name_cache.get(fid, "NOT_FOUND")
            logger.info(f"File ID {fid} -> {cached_name}")

        # Even if LLM model is not available, we can still do extractive compression and show stats
        # The LLM step will be skipped, but compression stats will still be generated
        llm_available = self.model is not None
        if not llm_available:
            logger.warning("Llama model not available. Will perform extractive compression only (no LLM summarization).")

        # Take top N chunks (already sorted by relevance score)
        chunks_to_summarize = chunks[:max_chunks_to_summarize]
        if not chunks_to_summarize:
            logger.warning(f"No chunks to summarize after taking top {max_chunks_to_summarize}")
            return None
        
        logger.info(f"Processing {len(chunks_to_summarize)} chunks for summarization")

        try:
            # Calculate original stats
            original_total_chars = sum(len(chunk.text) for chunk in chunks_to_summarize)
            original_total_tokens = sum(self._count_tokens(chunk.text) for chunk in chunks_to_summarize)
            original_total_sentences = sum(len(self._split_into_sentences(chunk.text)) for chunk in chunks_to_summarize)

            # 3A. Extractive compression (semantic shrinking)
            compressed_texts, compression_info = self._compress_chunks_extractive(
                chunks_to_summarize,
                query=query,
                sentences_per_chunk=sentences_per_chunk,
            )

            # Calculate compressed stats
            compressed_total_chars = sum(len(text) for text in compressed_texts.values())
            compressed_total_tokens = sum(self._count_tokens(text) for text in compressed_texts.values())
            compressed_total_sentences = sum(info.get('sentences_kept', 0) for info in compression_info.values())

            # 4A. Add provenance headers and merge context blocks
            final_context, context_stats = self._build_final_context(
                chunks_to_summarize,
                compressed_texts,
                min_tokens=min_context_tokens,
                max_tokens=max_context_tokens,
            )

            # 5. LLM compression / summarization with citation enforcement (if model available)
            if llm_available:
                prompt = self._create_citation_prompt(query, final_context)

                response = self.model(
                    prompt,
                    max_tokens=1000,  # Reduced from 2500 for faster generation (4.78 tokens/sec = ~3.5 min vs 10 min)
                    temperature=0.2,  # Lower temperature for more focused, less repetitive summaries
                    stop=["</answer>", "\n\n\n", "The document includes", "The document also includes"],
                    echo=False,
                )

                answer = response["choices"][0]["text"].strip()
                logger.info(f"Raw LLM answer length: {len(answer)} chars")
                
                if not answer or len(answer.strip()) == 0:
                    logger.warning("LLM returned empty answer, using fallback summary")
                    answer_with_citations = self._create_fallback_summary(
                        chunks_to_summarize, query, compressed_texts
                    )
                else:
                    answer = self._clean_summary(answer)
                    logger.info(f"Cleaned answer length: {len(answer)} chars")
                    
                    # 6. Citation extraction (automatic)
                    answer_with_citations = self._extract_and_format_citations(
                        answer, chunks_to_summarize
                    )
                    logger.info(f"Final answer with citations length: {len(answer_with_citations)} chars")
            else:
                # If LLM not available, create a simple summary from compressed chunks
                answer_with_citations = self._create_fallback_summary(
                    chunks_to_summarize, query, compressed_texts
                )

            # Build compression statistics
            compression_stats = {
                "chunks_processed": len(chunks_to_summarize),
                "original": {
                    "total_chars": original_total_chars,
                    "total_tokens": original_total_tokens,
                    "total_sentences": original_total_sentences,
                    "avg_chars_per_chunk": original_total_chars // len(chunks_to_summarize) if chunks_to_summarize else 0,
                    "avg_tokens_per_chunk": original_total_tokens // len(chunks_to_summarize) if chunks_to_summarize else 0,
                },
                "after_extractive_compression": {
                    "total_chars": compressed_total_chars,
                    "total_tokens": compressed_total_tokens,
                    "total_sentences": compressed_total_sentences,
                    "avg_chars_per_chunk": compressed_total_chars // len(compressed_texts) if compressed_texts else 0,
                    "avg_tokens_per_chunk": compressed_total_tokens // len(compressed_texts) if compressed_texts else 0,
                    "compression_ratio": round(compressed_total_chars / original_total_chars, 3) if original_total_chars > 0 else 0,
                    "sentences_kept_per_chunk": sentences_per_chunk,
                },
                "final_context": {
                    "blocks_used": context_stats.get("blocks_used", 0),
                    "total_tokens": context_stats.get("total_tokens", 0),
                    "target_range": f"{min_context_tokens}-{max_context_tokens}",
                },
                "summary": {
                    "length_chars": len(answer_with_citations),
                    "length_tokens": self._count_tokens(answer_with_citations),
                }
            }

            logger.info(
                f"Generated compressed summary for {len(chunks_to_summarize)} chunks with citations. "
                f"Compression: {compression_stats['after_extractive_compression']['compression_ratio']:.1%}"
            )
            
            return {
                "summary": answer_with_citations,
                "compression_stats": compression_stats
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    # ---------------------------
    # Token counting
    # ---------------------------

    def _get_tokenizer(self):
        """Lazy-load tokenizer for token counting (approximate if tiktoken not available)."""
        if self._tokenizer is None:
            try:
                import tiktoken

                # Use cl100k_base (GPT-4 tokenizer) as a reasonable approximation
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("Loaded tiktoken for token counting")
            except ImportError:
                logger.warning(
                    "tiktoken not available, using approximate token counting (4 chars = 1 token)"
                )
                self._tokenizer = None
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or approximation."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Approximation: ~4 characters per token
            return len(text) // 4

    # ---------------------------
    # Extractive compression
    # ---------------------------

    def _get_embedding_model(self):
        """Lazy-load sentence embedding model, shared with LanceDB config if available."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Use same embedding model as rest of system (BAAI/bge-small-en-v1.5)
                model_name = getattr(
                    settings, "lancedb_embedding_model", "BAAI/bge-small-en-v1.5"
                ) or "BAAI/bge-small-en-v1.5"
                self._embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence embedding model for compression: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model for compression: {e}")
                raise
        return self._embedding_model

    def _split_into_sentences(self, text: str) -> List[str]:
        """Very lightweight sentence splitter."""
        if not text:
            return []
        # Split on punctuation followed by whitespace, keep punctuation
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        # Filter out empty pieces
        return [s.strip() for s in parts if s.strip()]

    def _compress_chunks_extractive(
        self,
        chunks: List[ChunkResult],
        query: str,
        sentences_per_chunk: int = 5,
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        For each chunk, keep only the top-K most query-relevant sentences.

        Returns:
            Tuple of:
            - Dict mapping chunk_ref -> compressed text (without provenance header)
            - Dict mapping chunk_ref -> compression info (sentences_kept, sentences_total, etc.)
        """
        model = self._get_embedding_model()

        # Encode query once
        query_embedding = model.encode(query)

        compressed: Dict[str, str] = {}
        compression_info: Dict[str, Dict] = {}

        for chunk in chunks:
            sentences = self._split_into_sentences(chunk.text)
            if not sentences:
                continue

            try:
                sent_embeddings = model.encode(sentences)
            except Exception as e:
                logger.warning(
                    f"Failed to embed sentences for chunk {chunk.chunk_ref}: {e}"
                )
                continue

            # Cosine similarity between query and each sentence
            q = np.array(query_embedding)
            s = np.array(sent_embeddings)
            # Normalize
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            s_norm = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
            sims = s_norm @ q_norm

            # Get indices of top-K sentences
            k = min(sentences_per_chunk, len(sentences))
            top_idx = np.argsort(-sims)[:k]
            # Preserve original order of sentences in text
            top_idx_sorted = sorted(top_idx.tolist())

            selected_sentences = [sentences[i] for i in top_idx_sorted]
            compressed_text = " ".join(selected_sentences)
            compressed[chunk.chunk_ref] = compressed_text
            
            # Store compression info
            compression_info[chunk.chunk_ref] = {
                "sentences_total": len(sentences),
                "sentences_kept": k,
                "compression_ratio": round(k / len(sentences), 3) if sentences else 0,
            }

        return compressed, compression_info

    # ---------------------------
    # Context building with provenance
    # ---------------------------

    def _build_provenance_header(self, chunk: ChunkResult) -> str:
        """
        Build provenance header in format: [Doc: <filename> | Page: <page> | Section: <section>]
        
        These headers are NOT shown to the user later, but allow the LLM to cite correctly.
        """
        parts = []
        
        # Doc: Use filename if available, otherwise fileid
        doc_name = self._get_file_name(chunk.fileid)
        parts.append(f"Doc: {doc_name}")
        
        # Page: Use page range
        page_start = chunk.page_range[0] if chunk.page_range else 1
        parts.append(f"Page: {page_start}")
        
        # Section: Use section_path if available
        if chunk.section_path:
            section = " > ".join(chunk.section_path)
            parts.append(f"Section: {section}")
        
        return f"[{' | '.join(parts)}]"

    def _build_final_context(
        self,
        chunks: List[ChunkResult],
        compressed_texts: Dict[str, str],
        min_tokens: int = 1000,
        max_tokens: int = 2500,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build final context with provenance headers, merging blocks to fit token limits.
        
        Target: 1,000-2,500 tokens to fit within small open-source LLM window.
        
        Returns:
            Tuple of (final_context_string, stats_dict)
        """
        context_blocks = []
        total_tokens = 0

        for chunk in chunks:
            compressed_text = compressed_texts.get(chunk.chunk_ref, chunk.text)
            if not compressed_text:
                continue

            # Build provenance header
            header = self._build_provenance_header(chunk)
            
            # Build block: header + compressed text
            block = f"{header}\n{compressed_text}"
            block_tokens = self._count_tokens(block)

            # Check if adding this block would exceed max_tokens
            if total_tokens + block_tokens > max_tokens:
                # If we're below min_tokens, try to add a truncated version
                if total_tokens < min_tokens:
                    remaining_tokens = max_tokens - total_tokens
                    # Truncate the compressed text to fit
                    truncated = self._truncate_to_tokens(
                        compressed_text, remaining_tokens - self._count_tokens(header) - 10
                    )
                    if truncated:
                        block = f"{header}\n{truncated}"
                        context_blocks.append(block)
                        total_tokens += self._count_tokens(block)
                break

            context_blocks.append(block)
            total_tokens += block_tokens

            # If we've reached a good size (above min_tokens), we can stop early
            if total_tokens >= min_tokens and len(context_blocks) >= 3:
                # But continue if we're still below max_tokens and have more chunks
                if total_tokens + 200 > max_tokens:  # Leave some buffer
                    break

        final_context = "\n\n".join(context_blocks)
        stats = {
            "blocks_used": len(context_blocks),
            "total_tokens": total_tokens,
        }
        logger.info(
            f"Built final context: {len(context_blocks)} blocks, ~{total_tokens} tokens"
        )
        return final_context, stats

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        if max_tokens <= 0:
            return ""
        
        tokenizer = self._get_tokenizer()
        if tokenizer:
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens)
        else:
            # Approximation
            max_chars = max_tokens * 4
            return text[:max_chars] + "..." if len(text) > max_chars else text

    # ---------------------------
    # Prompt construction with citation enforcement
    # ---------------------------

    def _create_citation_prompt(self, query: str, final_context: str) -> str:
        """
        Create prompt that enforces citation format and encourages rich summaries.
        
        Format for inline citations in the answer:
            (Citation: Doc=<DOC_ID>, Page=<PAGE>)
        
        Notes for the model:
        - Context blocks begin with headers like:
              [Doc: <DOC_NAME> | Page: <PAGE> | Section: <SECTION>]
        - Use these headers to understand which document and section each passage comes from.
        """
        prompt = f"""You are a data-room assistant.

You MUST answer based ONLY on the provided context.

General instructions:
- Read all context blocks carefully. Each block starts with a header like:
  [Doc: <DOC_NAME> | Page: <PAGE> | Section: <SECTION>]
- Use these headers to understand which document a passage belongs to and what section it is in.
- Give a clear, helpful summary or answer based on the information you DO see in the context.
- If the query asks about topics in each document separately, organize your answer BY DOCUMENT.
  For each distinct Doc name you see in the headers, describe its main topics in separate bullet points or paragraphs.
- DO NOT answer "Not found in the provided documents." if you can reasonably infer an answer
  from the context, even if the topics are not explicitly labeled as "main topics".
- Only if NONE of the context is relevant to the query, then say:
  "Not found in the provided documents."

Citation instructions:
- When you make a factual statement that comes from the documents, add citations
  immediately after the sentence using this exact format:
  (Citation: Doc=<DOC_NAME>, Page=<PAGE>)
- <DOC_NAME> MUST match the EXACT Doc name from the context headers (e.g., "Exhibit 10.1 Lease Agreement.pdf").
- DO NOT use file IDs (like f_abc123). Always use the document filename from the headers.
- <PAGE> should be the page number from the header.
- Limit citations: Only cite once per unique topic or section. Avoid repeating the same citation multiple times.

Answer style:
- Write a natural, flowing summary in paragraph form, not bullet points.
- Describe the main topics and key information in complete sentences.
- Make it read like a professional document summary, not a list.
- Be concise but comprehensive. Cover all major topics mentioned in the documents.
- DO NOT repeat the same information or sentences multiple times.
- Group related information together in coherent paragraphs.
- Only use bullet points if the query specifically asks for a list format.

# Query:
{query}

# Context:
{final_context}

# Answer:"""
        return prompt

    # ---------------------------
    # Citation extraction
    # ---------------------------

    def _extract_and_format_citations(
        self, answer: str, chunks: List[ChunkResult]
    ) -> str:
        """
        Extract citations from answer and ensure they're properly formatted.
        
        Looks for patterns like:
        - (Citation: Doc=X, Page=Y)
        - Doc: X | Page: Y (from provenance headers)
        
        Also attaches source metadata if citations are missing.
        """
        # Pattern to match citation format: (Citation: Doc=..., Page=...)
        citation_pattern = r"\(Citation:\s*Doc=([^,]+),\s*Page=(\d+)\)"
        
        # Find all citations in the answer
        citations_found = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        # Create a mapping of file_id -> filename for citation replacement
        # First, use any pre-fetched file names passed to summarize_chunks
        file_id_to_name = {}
        if hasattr(self, '_prefetched_file_names') and self._prefetched_file_names:
            file_id_to_name.update(self._prefetched_file_names)
            logger.info(f"Using {len(self._prefetched_file_names)} pre-fetched file names for citation replacement")
        
        # Create a mapping of filename -> chunks for page number lookup
        # This allows us to use the actual page_range from chunks instead of trusting LLM's page number
        filename_to_chunks = {}
        for chunk in chunks:
            filename = self._get_file_name(chunk.fileid)
            if filename and filename != chunk.fileid:
                if filename not in filename_to_chunks:
                    filename_to_chunks[filename] = []
                filename_to_chunks[filename].append(chunk)
        
        # Then add from chunks (using cache) - this is a fallback
        for chunk in chunks:
            if chunk.fileid and chunk.fileid not in file_id_to_name:
                filename = self._get_file_name(chunk.fileid)
                if filename and filename != chunk.fileid:  # Only add if we have a real filename
                    file_id_to_name[chunk.fileid] = filename
                    # Also add case-insensitive mapping
                    file_id_to_name[chunk.fileid.lower()] = filename
        
        # Log if we're missing file names
        missing_names = [fid for fid, name in file_id_to_name.items() if name == fid and fid.startswith('f_')]
        if missing_names:
            logger.warning(f"File names not found for file IDs: {missing_names[:5]}")
        else:
            logger.info(f"File name mapping ready: {len(file_id_to_name)} file IDs mapped")
        
        # Replace file IDs with file names in citations AND use actual page numbers from chunks
        def replace_citation(match):
            doc_ref = match.group(1).strip()
            llm_page = match.group(2).strip()  # Page number from LLM (may be wrong)
            original_doc_ref = doc_ref
            
            # If doc_ref is a file_id (starts with f_ or matches a known file_id), replace with filename
            if doc_ref in file_id_to_name:
                doc_ref = file_id_to_name[doc_ref]
                logger.debug(f"Replaced citation: {original_doc_ref} -> {doc_ref}")
            elif doc_ref.lower() in file_id_to_name:
                doc_ref = file_id_to_name[doc_ref.lower()]
                logger.debug(f"Replaced citation (case-insensitive): {original_doc_ref} -> {doc_ref}")
            elif doc_ref.startswith('f_') and len(doc_ref) > 2:
                # Try to find the file_id in our mapping (case-insensitive)
                doc_ref_lower = doc_ref.lower()
                for fid, fname in file_id_to_name.items():
                    if not fid.startswith('f_'):
                        continue
                    if fid.lower() == doc_ref_lower:
                        doc_ref = fname
                        logger.debug(f"Replaced citation (loop search): {original_doc_ref} -> {doc_ref}")
                        break
                # If still not found, try to fetch it now (last resort)
                if doc_ref == original_doc_ref and doc_ref.startswith('f_'):
                    # Try fetching from cache one more time
                    fetched_name = self._get_file_name(doc_ref)
                    if fetched_name != doc_ref:
                        doc_ref = fetched_name
                        file_id_to_name[doc_ref] = fetched_name
                        logger.debug(f"Replaced citation (cache fetch): {original_doc_ref} -> {doc_ref}")
                    else:
                        logger.warning(f"Could not find filename for file ID: {doc_ref} (file_id_to_name has {len(file_id_to_name)} entries)")
            
            # Now look up the actual page number from chunks instead of using LLM's page number
            # The LLM might generate incorrect page numbers, so we use the actual chunk data
            actual_page = llm_page  # Default to LLM's page if we can't find a match
            if doc_ref in filename_to_chunks:
                # Find the chunk that best matches
                matching_chunks = filename_to_chunks[doc_ref]
                if matching_chunks:
                    # Strategy: Try to find a chunk with page_range matching the LLM's page first
                    # If found, use that chunk's page (which should match)
                    # If not found, use the first chunk's page (better than defaulting to LLM's wrong page)
                    found_match = False
                    for chunk in matching_chunks:
                        if chunk.page_range and len(chunk.page_range) >= 1:
                            chunk_page = str(chunk.page_range[0])
                            # If LLM's page matches a chunk's page, use it
                            if chunk_page == llm_page:
                                actual_page = chunk_page
                                found_match = True
                                logger.debug(f"Citation page match: LLM said {llm_page}, chunk has {chunk_page} - using {actual_page}")
                                break
                    
                    # If no exact match, use the first chunk's page (better than wrong LLM page)
                    if not found_match and matching_chunks[0].page_range and len(matching_chunks[0].page_range) >= 1:
                        actual_page = str(matching_chunks[0].page_range[0])
                        logger.debug(f"Citation page fallback: LLM said {llm_page}, using first chunk's page {actual_page} from {doc_ref}")
                else:
                    logger.warning(f"No chunks found for filename: {doc_ref}")
            else:
                # Try case-insensitive lookup
                for filename, chunk_list in filename_to_chunks.items():
                    if filename.lower() == doc_ref.lower():
                        if chunk_list and chunk_list[0].page_range and len(chunk_list[0].page_range) >= 1:
                            actual_page = str(chunk_list[0].page_range[0])
                            logger.debug(f"Citation page (case-insensitive): LLM said {llm_page}, using actual page {actual_page}")
                        break
            
            return f"(Citation: Doc={doc_ref}, Page={actual_page})"
        
        # If no citations found, try to infer from context
        if not citations_found:
            # Check if answer mentions any fileids or page numbers
            fileid_pattern = r"\b(f_[a-z0-9]+|p_[a-z0-9]+)\b"
            page_pattern = r"\b(page|pages?)\s+(\d+)\b"
            
            fileids_mentioned = re.findall(fileid_pattern, answer, re.IGNORECASE)
            pages_mentioned = re.findall(page_pattern, answer, re.IGNORECASE)
            
            # If we find mentions but no citations, add them at the end
            if fileids_mentioned or pages_mentioned:
                # Try to match with chunks
                sources = []
                for chunk in chunks[:5]:  # Check top 5 chunks
                    if chunk.fileid in answer or any(
                        str(p) in answer for p in (chunk.page_range if chunk.page_range else [])
                    ):
                        file_name = self._get_file_name(chunk.fileid)
                        # Use actual page_range from chunk
                        page_num = chunk.page_range[0] if chunk.page_range and len(chunk.page_range) >= 1 else 1
                        sources.append(f"(Citation: Doc={file_name}, Page={page_num})")
                
                if sources:
                    answer += "\n\nSources: " + ", ".join(set(sources))
        
        # Ensure citations are properly formatted with file names
        answer = re.sub(citation_pattern, replace_citation, answer, flags=re.IGNORECASE)
        
        # Clean up any incomplete/broken citations at the end (aggressive patterns)
        # Remove patterns like "(C", "(Ci", "(Cit", "(Citation", "(Citation: ", "(Citation: D", etc.
        answer = re.sub(r'\(C[itation]*:?\s*D[oc]*=?[^,)]*,?\s*P[age]*=?\d*\)?$', '', answer, flags=re.IGNORECASE)  # Any incomplete citation at end
        answer = re.sub(r'\(C[itation]*:?\s*D[oc]*=?[^,)]*$', '', answer, flags=re.IGNORECASE)  # "(Citation: Doc=" or shorter at end
        answer = re.sub(r'\(C[itation]*:?\s*$', '', answer, flags=re.IGNORECASE)  # Just "(C" or "(Citation: " at end
        answer = re.sub(r'\([^)]*$', '', answer)  # Any incomplete parenthesis at the end
        answer = answer.strip()
        
        # Remove trailing punctuation artifacts (commas, periods, incomplete sentences)
        answer = re.sub(r'[\.,;:\s]+$', '', answer)
        answer = answer.strip()

        # Deduplicate excessively repeated identical citations within the text.
        # Strategy: Remove citations that appear more than 2-3 times, and collapse consecutive duplicates
        def _collapse_repeated_citations(text: str) -> str:
            # First, collapse consecutive duplicates (2+ in a row)
            pattern = r"(\(Citation:\s*Doc=[^,]+,\s*Page=\d+\)\s*){2,}"

            def repl(m: re.Match) -> str:
                # Keep a single instance of the first citation in the sequence
                seq = m.group(0)
                first = re.search(
                    r"\(Citation:\s*Doc=[^,]+,\s*Page=\d+\)", seq, re.IGNORECASE
                )
                return first.group(0) + " " if first else seq

            text = re.sub(pattern, repl, text)
            
            # Second, limit citations per unique citation to max 2-3 occurrences total
            # Find all unique citations and their positions
            all_citations = re.findall(citation_pattern, text, re.IGNORECASE)
            citation_counts = {}
            for citation in all_citations:
                citation_key = f"{citation[0].strip()}_{citation[1].strip()}"
                citation_counts[citation_key] = citation_counts.get(citation_key, 0) + 1
            
            # If a citation appears more than 3 times, remove excess occurrences
            for citation_key, count in citation_counts.items():
                if count > 3:
                    # Find all occurrences of this citation
                    doc_name, page = citation_key.split('_')
                    citation_regex = r"\(Citation:\s*Doc=" + re.escape(doc_name) + r",\s*Page=" + page + r"\)"
                    matches = list(re.finditer(citation_regex, text, re.IGNORECASE))
                    # Keep first 2, remove the rest
                    for match in reversed(matches[2:]):
                        text = text[:match.start()] + text[match.end():]
            
            return text

        answer = _collapse_repeated_citations(answer)
        
        # Additional cleanup: Remove repetitive sentence patterns
        # Remove sentences that start with "The document includes" or "The document requires" if they're too repetitive
        sentences = re.split(r'([.!?]\s+)', answer)
        seen_content = set()
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            # Check if this sentence is too similar to previous ones
            sentence_lower = sentence.lower().strip()
            # Extract key content (remove citations and common words)
            key_content = re.sub(r'\(Citation:[^)]+\)', '', sentence_lower)
            key_content = re.sub(r'\b(the|document|includes|requires|also|provisions?)\b', '', key_content)
            key_content = key_content.strip()[:50]  # First 50 chars as key
            
            if key_content and key_content in seen_content and len(key_content) > 10:
                # Skip this sentence if we've seen similar content
                continue
            if key_content:
                seen_content.add(key_content)
            cleaned_sentences.append(sentence)
        
        answer = "".join(cleaned_sentences)

        return answer

    def _create_fallback_summary(
        self,
        chunks: List[ChunkResult],
        query: str,
        compressed_texts: Dict[str, str]
    ) -> str:
        """Create a simple summary when LLM is not available."""
        summary_parts = [f"Based on the query '{query}', here are the key points from the documents:\n"]
        
        for i, chunk in enumerate(chunks[:5], 1):  # Top 5 chunks
            compressed = compressed_texts.get(chunk.chunk_ref, chunk.text[:200])
            summary_parts.append(f"{i}. {compressed[:150]}...")
            if chunk.page_range:
                # Use filename instead of fileid for user-friendly display
                file_name = self._get_file_name(chunk.fileid)
                summary_parts.append(f"   (Source: {file_name}, Page {chunk.page_range[0]})")
        
        return "\n".join(summary_parts)

    # ---------------------------
    # File name fetching
    # ---------------------------

    def _fetch_file_names(self, projectid: str, chunks: List[ChunkResult]) -> None:
        """Fetch file names from chunking service and cache them."""
        if not projectid or not chunks:
            return
        
        # Get unique file IDs from chunks
        unique_file_ids = list(set(chunk.fileid for chunk in chunks if chunk.fileid))
        if not unique_file_ids:
            return
        
        # Check cache first - only fetch if we don't have all file names
        missing_file_ids = [fid for fid in unique_file_ids if fid not in self._file_name_cache]
        if not missing_file_ids:
            return  # All file names already cached
        
        try:
            # Try to use internal chunking integration first (if available)
            try:
                from app.retriever.chunking_integration import get_chunking_integration
                chunking_integration = get_chunking_integration()
                if chunking_integration:
                    # Use internal call (synchronous wrapper)
                    # Note: We can't use asyncio.run_until_complete if there's already a running loop
                    # So we'll skip internal calls in async contexts and use HTTP instead
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        # If there's a running loop, skip internal call and use HTTP
                        logger.debug("Running event loop detected, using HTTP for file name fetch")
                        raise RuntimeError("Event loop already running")
                    except RuntimeError:
                        # No running loop, safe to create new one
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        file_name_map = loop.run_until_complete(
                            chunking_integration.get_file_names(
                                project_id=projectid,
                                file_ids=missing_file_ids
                            )
                        )
                        for file_id, filename in file_name_map.items():
                            if file_id and filename:
                                self._file_name_cache[file_id] = filename
                        logger.info(f"Fetched {len(file_name_map)} file names via internal call")
                        return
            except Exception as internal_error:
                logger.debug(f"Internal file name fetch failed, trying HTTP: {internal_error}")
            
            # Fallback to HTTP call
            chunking_service_url = os.getenv("CHUNKING_SERVICE_URL", "http://localhost:8002")
            auth_token = os.getenv("CHUNKING_SERVICE_AUTH_TOKEN")
            
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            # Use sync httpx client
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{chunking_service_url}/projects/{projectid}/files",
                    params={"user_id": "temp_user_001"},
                    headers=headers
                )
                
                if response.status_code == 200:
                    files = response.json()
                    fetched_count = 0
                    for file_info in files:
                        file_id = file_info.get("_id") or file_info.get("id")
                        filename = file_info.get("filename") or file_info.get("name")
                        if file_id and filename and file_id in missing_file_ids:
                            self._file_name_cache[file_id] = filename
                            fetched_count += 1
                    logger.info(f"Fetched {fetched_count} file names via HTTP")
                else:
                    logger.warning(f"Failed to fetch file names: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to fetch file names: {e}")
    
    def _get_file_name(self, fileid: str) -> str:
        """Get file name from cache, or return fileid if not found."""
        return self._file_name_cache.get(fileid, fileid)

    # ---------------------------
    # Utils
    # ---------------------------

    def _clean_summary(self, summary: str) -> str:
        """Clean and format the summary text."""
        summary = summary.strip()

        prefixes_to_remove = [
            "Summary:",
            "Here's a summary:",
            "Based on the chunks:",
            "The summary is:",
            "Answer:",
            "# Answer:",
        ]

        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix) :].strip()

        if summary and not summary[0].isupper():
            summary = (
                summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
            )

        return summary


# Global summarizer instance
_summarizer_instance: Optional[ChunkSummarizer] = None


def get_summarizer() -> ChunkSummarizer:
    """Get or create the global summarizer instance."""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ChunkSummarizer()
    return _summarizer_instance
