import os
import logging
from typing import List, Dict, Optional, Tuple
from app.utils.docling_page_by_page import run_docling_on_page,extract_pdf_page,get_pdf_page_count
from app.utils.agent_chunker import _call_google_agent_sdk,_parse_jsonl
logger = logging.getLogger(__name__)
import json
def create_chunk_page_by_page(
    pdf_path: str,
    do_ocr: bool,
    output_dir: str,
    page_range: Optional[Tuple[int, int]] = None,
    keep_temp_pages: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process PDF page by page with Docling and chunk each page.
    
    Args:
        pdf_path: Path to source PDF
        do_ocr: Whether to perform OCR
        output_dir: Directory for outputs
        page_range: Optional (start_page, end_page) tuple (1-indexed, inclusive)
        keep_temp_pages: Whether to keep extracted single-page PDFs
    
    Returns:
        Tuple of (page_results, all_chunks)
    """
    # CRITICAL: Create ALL directories upfront
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created main output directory: {output_dir}")
    
    # Create subdirectories
    temp_pages_dir = os.path.join(output_dir, "temp_pages")
    page_outputs_dir = os.path.join(output_dir, "page_outputs")
    
    os.makedirs(temp_pages_dir, exist_ok=True)
    os.makedirs(page_outputs_dir, exist_ok=True)
    
    logger.info(f"Created temp_pages directory: {temp_pages_dir}")
    logger.info(f"Created page_outputs directory: {page_outputs_dir}")
    
    # Verify PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return [], []
    
    # Get page count
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        logger.error("Could not determine page count")
        return [], []
    
    # Determine page range
    if page_range:
        start_page, end_page = page_range
        start_page = max(1, start_page)
        end_page = min(total_pages, end_page)
    else:
        start_page, end_page = 1, total_pages
    
    logger.info(f"Processing pages {start_page}-{end_page} of {total_pages}")
    
    results = []
    all_chunks = []
    
    for page_num in range(start_page, end_page + 1):
        logger.info(f"Processing page {page_num}/{total_pages}")
        
        try:
            # Extract single page
            logger.debug(f"Extracting page {page_num} to {temp_pages_dir}")
            page_pdf_path = extract_pdf_page(pdf_path, page_num, temp_pages_dir)
            
            # Verify the extracted page exists
            if not os.path.exists(page_pdf_path):
                raise FileNotFoundError(f"Failed to create page PDF: {page_pdf_path}")
            
            logger.debug(f"Successfully extracted page to: {page_pdf_path}")
            
            # Process with Docling
            logger.debug(f"Running Docling on page {page_num}")
            conv_result = run_docling_on_page(
                page_pdf_path,
                do_ocr,
                page_outputs_dir,
                page_num
            )
            
            # Chunk this page with agent
            logger.debug(f"Chunking page {page_num}")
            response_text = _call_google_agent_sdk(conv_result.document.export_to_markdown(), conv_result.document.export_to_dict())
            
            # Parse response
            page_chunks = _parse_jsonl(response_text)
            
            # Add page number to chunks
            for chunk in page_chunks:
                if chunk.get("page_range") == [1, 1]:  # Default value
                    chunk["page_range"] = [page_num, page_num]
            
            # Add chunks to all_chunks
            all_chunks.extend(page_chunks)
            
            results.append({
                "page_num": page_num,
                
                "docling_dict": conv_result.document.export_to_dict(),
                "temp_pdf_path": page_pdf_path,
                "num_chunks": len(page_chunks)
            })
            
            logger.info(f"✓ Completed page {page_num} - {len(page_chunks)} chunks created")
            
            # Clean up temp page PDF if requested
            if not keep_temp_pages:
                try:
                    os.remove(page_pdf_path)
                    logger.debug(f"Removed temp page: {page_pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temp page {page_pdf_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
            results.append({
                "page_num": page_num,
                "error": str(e)
            })
    
    # Clean up temp directory if empty
    if not keep_temp_pages:
        try:
            if os.path.exists(temp_pages_dir) and not os.listdir(temp_pages_dir):
                os.rmdir(temp_pages_dir)
                logger.info(f"Removed empty temp_pages directory")
        except Exception as e:
            logger.debug(f"Could not remove temp directory: {e}")
    
    logger.info(f"Completed processing {len(results)} pages")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    with open(os.path.join(output_dir, "all_chunks.jsonl"), "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    return results, os.path.join(output_dir, "all_chunks.jsonl")