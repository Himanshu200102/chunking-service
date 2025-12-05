# Parsing Speed Optimization

## Problem
Parsing was taking 11+ minutes for 2 small PDFs (79KB and 85KB).

## Root Cause
When `do_ocr=False`, Docling was still using **default heavy processing**:
- ❌ Table structure detection (very slow)
- ❌ Complex layout analysis
- ❌ ML model loading for various tasks
- ❌ Image processing

Even without OCR, these operations made small PDFs take 10+ minutes to parse!

## Solution

### Optimized `app/utils/docling_converter.py`:

**Before** (do_ocr=False):
```python
doc_converter = DocumentConverter()  # Default = SLOW
```

**After** (do_ocr=False):
```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  # Explicitly disable OCR
pipeline_options.do_table_structure = False  # Disable slow table detection
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=os.cpu_count() or 4,  # Use all CPU cores
    device=AcceleratorDevice.CPU  # No GPU initialization delays
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

## Changes Made

1. **Explicitly disable OCR**: `do_ocr = False`
2. **Disable table structure detection**: `do_table_structure = False` (major speedup!)
3. **Use all CPU cores**: `num_threads = os.cpu_count()`
4. **Explicit CPU device**: Avoids GPU initialization overhead

## Expected Speed Improvement

| Configuration | Time for 2 small PDFs | Speed |
|--------------|----------------------|-------|
| **Before** (default) | 11+ minutes | ❌ Very Slow |
| **After** (optimized) | ~30-60 seconds | ✅ Fast |

**Speedup**: ~10-20x faster! 🚀

## Trade-offs

### What You Lose (when do_ocr=False):
- ❌ No table structure detection (tables treated as text blocks)
- ❌ No advanced layout analysis
- ❌ No OCR for scanned documents

### What You Keep:
- ✅ Text extraction from PDFs
- ✅ Basic layout detection
- ✅ Section/paragraph chunking
- ✅ Page numbers and document structure

## When to Use Each Mode

### Fast Mode (`do_ocr=False`) - Recommended for most cases:
- ✅ Regular PDFs with text
- ✅ Quick processing needed
- ✅ Simple documents
- ⚡ 30-60 seconds per small PDF

### Full Mode (`do_ocr=True`) - Use when needed:
- ✅ Scanned documents (images)
- ✅ Complex tables that need structure
- ✅ Multi-language documents
- ⏱️ 5-10 minutes per PDF (slower but more accurate)

## Next Steps for You

1. **Refresh your browser** (the old parsing session timed out)
2. **Re-parse the files** with the optimized settings
3. **Expected time**: ~1-2 minutes for both PDFs (instead of 11+ minutes)

## Monitoring

Check parsing progress in the demo HTML:
- You should see faster "current_status" updates
- Parsing should complete in < 2 minutes

If it's still slow, we can:
- Further reduce Docling processing
- Use alternative parsing libraries (PyPDF2, pdfplumber)
- Pre-process PDFs to simplify them

