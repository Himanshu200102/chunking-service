# app/utils/docling.py
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from app.utils.storage import ensure_dir

logger = logging.getLogger(__name__)


def convert_document(raw_path: str, do_ocr: bool):
    if do_ocr:
        # Full pipeline with OCR (slower but more accurate)
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
        conv_result = doc_converter.convert(raw_path)
    else:
        # Fast pipeline without OCR (optimized for speed)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Explicitly disable OCR
        pipeline_options.do_table_structure = False  # Disable slow table detection
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=os.cpu_count() or 4,  # Use all available cores
            device=AcceleratorDevice.CPU  # Explicit CPU to avoid GPU initialization delays
        )
        
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        conv_result = doc_converter.convert(raw_path)

    return conv_result
    