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


def run_docling(raw_path: str, do_ocr: bool):
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
        conv_result = doc_converter.convert(raw_path)
    else:
        doc_converter = DocumentConverter()
        conv_result = doc_converter.convert(raw_path)
    # json_output_path = os.path.join(
    #     out_dir,
    #     f"{Path(raw_path).stem}_docling_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    # )
    # markdown_output_path = os.path.join(
    #     out_dir,
    #     f"{Path(raw_path).stem}_docling_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
    # )
    # with open(json_output_path, "w", encoding="utf-8") as json_file:
    #     json.dump(conv_result.document.export_to_dict(), json_file, ensure_ascii=False, indent=4)
    # with open(markdown_output_path, "w", encoding="utf-8") as md_file:
    #     md_file.write(conv_result.document.export_to_markdown())
    return conv_result
    