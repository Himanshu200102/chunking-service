# app/utils/docling_page_by_page.py

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in a PDF."""
    try:
        # Try PyMuPDF first
        import fitz
        pdf = fitz.open(pdf_path)
        count = len(pdf)
        pdf.close()
        return count
    except ImportError:
        pass
    
    try:
        # Fallback to pypdfium2
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        count = len(pdf)
        pdf.close()
        return count
    except ImportError:
        pass
    
    try:
        # Fallback to PyPDF2
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except ImportError:
        logger.error("No PDF library available. Install one of: pymupdf, pypdfium2, PyPDF2")
        return 0


def extract_pdf_page(pdf_path: str, page_num: int, output_dir: str) -> str:
    """
    Extract a single page from PDF and save as a new PDF.
    
    Args:
        pdf_path: Path to source PDF
        page_num: Page number (1-indexed)
        output_dir: Directory to save extracted page
    
    Returns:
        Path to the extracted page PDF
    """
    # CRITICAL: Ensure output directory exists FIRST
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {output_dir}")
    
    output_path = os.path.join(
        output_dir,
        f"{Path(pdf_path).stem}_page_{page_num:04d}.pdf"
    )
    
    page_index = page_num - 1  # Convert to 0-indexed
    
    # Try PyMuPDF first (fastest and most reliable)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        
        if page_index >= len(doc) or page_index < 0:
            doc.close()
            raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        new_doc.save(output_path)
        new_doc.close()
        doc.close()
        
        logger.debug(f"Extracted page {page_num} using PyMuPDF to {output_path}")
        return output_path
        
    except ImportError:
        logger.debug("PyMuPDF not available, trying PyPDF2")
    
    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader, PdfWriter
        
        reader = PdfReader(pdf_path)
        
        if page_index >= len(reader.pages) or page_index < 0:
            raise ValueError(f"Page {page_num} out of range (PDF has {len(reader.pages)} pages)")
        
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        logger.debug(f"Extracted page {page_num} using PyPDF2 to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("No PDF library available for page extraction")
        raise ImportError(
            "Please install a PDF library: pip install pymupdf OR pip install PyPDF2"
        )

def run_docling_on_page(
    page_pdf_path: str,
    do_ocr: bool,
    out_dir: str,
    page_num: int
) -> Tuple[Optional[str], str, Dict]:
    """
    Run Docling on a single-page PDF.
    
    Returns:
        Tuple of (json_path, markdown_path, docling_dict)
    """
    # CRITICAL: Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    logger.debug(f"Ensured Docling output directory exists: {out_dir}")
    
    if do_ocr:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["es"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )
        
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        conv_result = doc_converter.convert(page_pdf_path)
    else:
        doc_converter = DocumentConverter()
        conv_result = doc_converter.convert(page_pdf_path)
    
    return conv_result