# app/utils/embeddings.py
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Model configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Dimension for bge-small-en-v1.5


@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Get or create cached embedding model.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        SentenceTransformer model instance
    """
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = True,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model
        normalize: Whether to normalize embeddings
        show_progress: Whether to show progress bar
        
    Returns:
        List of embedding vectors
    """
    embedder = get_embedder(model_name)
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        batch_size=32
    )
    return embeddings.tolist()


def create_combined_text(
    text: str,
    section_headers: List[str],
    header_weight: float = 0.3
) -> str:
    """
    Combine text and section headers into a single string for embedding.
    Section headers are given more weight by repeating them.
    
    Args:
        text: Main text content
        section_headers: List of hierarchical section headers
        header_weight: Weight given to headers (0.0-1.0)
        
    Returns:
        Combined text string
    """
    if not section_headers:
        return text
    
    # Combine section headers into a path
    section_path = " > ".join(section_headers)
    
    # Repeat headers based on weight to give them more importance
    # For 0.3 weight, we repeat headers 1 time (30% of content)
    repeat_count = max(1, int(header_weight * 3))
    header_text = " ".join([section_path] * repeat_count)
    
    # Combine: headers first, then text
    combined = f"{header_text} {text}"
    
    return combined


def embed_chunk_with_context(
    chunk: Dict[str, Any],
    model_name: str = EMBEDDING_MODEL
) -> List[float]:
    """
    Create a single embedding vector for a chunk that includes both text and section context.
    
    Args:
        chunk: Chunk dictionary with 'text' and 'section_headers' fields
        model_name: Name of the embedding model
        
    Returns:
        Embedding vector as list of floats
    """
    text = chunk.get("text", "")
    section_headers = chunk.get("section_headers", []) or chunk.get("heading_context", [])
    
    # Create combined text
    combined_text = create_combined_text(text, section_headers)
    
    # Generate embedding
    embeddings = embed_texts([combined_text], model_name=model_name, show_progress=False)
    
    return embeddings[0]


def embed_chunks_batch(
    chunks: List[Dict[str, Any]],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for multiple chunks in batch.
    More efficient than embedding one at a time.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the embedding model
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
        
    Returns:
        List of embedding vectors
    """
    if not chunks:
        return []
    
    # Prepare combined texts for all chunks
    combined_texts = []
    for chunk in chunks:
        text = chunk.get("text", "")
        section_headers = chunk.get("section_headers", []) or chunk.get("heading_context", [])
        combined = create_combined_text(text, section_headers)
        combined_texts.append(combined)
    
    # Generate embeddings in batch
    embedder = get_embedder(model_name)
    embeddings = embedder.encode(
        combined_texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=batch_size
    )
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    return embeddings.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))