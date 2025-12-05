# Citations and Streaming Behavior - Fixed Issues

## ✅ Issues Fixed

### 1. Citations Now Show Filenames Instead of File IDs

**Problem**: Citations in the inference/summary were showing file IDs (e.g., `f_a7c404dc4c`) instead of user-friendly filenames.

**Solution**: 
- Updated `_extract_and_format_citations()` in `summarizer.py` to replace file IDs with filenames
- Updated `_format_inference_html()` in `user.py` to replace file IDs in citations with filenames before HTML conversion
- Both `_build_final_response()` and `_build_file_final_response()` now build `file_id_to_name` mappings and pass them to `_format_inference_html()`
- Fallback summary in `summarizer.py` now uses `_get_file_name()` to show filenames

**Result**: All citations now display as:
- `(Citation: Doc=Exhibit 10.1 Lease Agreement.pdf, Page=1)` ✅
- Instead of: `(Citation: Doc=f_a7c404dc4c, Page=1)` ❌

### 2. Processing Messages Include Filename

**Problem**: Processing messages every 5 seconds only showed file number (e.g., "Processing 1st file...") without the filename.

**Solution**:
- Added `current_filename` tracking variable
- Updated file-specific processing loop to fetch and store filenames
- Updated `processing_sender()` to include filename in processing messages

**Result**: Processing messages now show:
```json
{
  "type": "processing",
  "message": "Processing 1st file: Exhibit 10.1 Lease Agreement.pdf...",
  "fileid": "f_a7c404dc4c",
  "filename": "Exhibit 10.1 Lease Agreement.pdf",
  "file_number": 1
}
```

### 3. File-Specific Sequential Processing Confirmed

**Behavior**: When agent decides `file_specific` strategy:
1. **Sequential Processing**: Files are processed one at a time
2. **Per-File Flow**: For each file:
   - Retrieve chunks for that file
   - Generate inference/summary for that file
   - Send `status` event: "Generating inference for file: {filename}..."
   - Send `final_response` event with inference and citations
   - Move to next file
3. **Processing Messages**: Every 5 seconds, sends a processing message with the current filename

**Code Location**: `app/api/routes/user.py` lines 254-321

**Example Flow**:
```
1. Agent decides: file_specific
2. File 1 (f_a7c404dc4c):
   - Retrieve chunks → Generate inference → Send final_response
3. File 2 (f_xyz123):
   - Retrieve chunks → Generate inference → Send final_response
4. Complete event
```

### 4. Global vs File-Specific Selection

**Functionality**: ✅ Yes, it exists!

**How it works**:
- **If `fileid` is provided**: Single file query (no agent, direct retrieval)
- **If `fileid` is NOT provided and `use_agent=true`**: Agent decides between:
  - `global`: Search all files together, single inference
  - `file_specific`: Process each file separately, separate inference per file
- **If `fileid` is NOT provided and `use_agent=false`**: Global search (all files together)

**Frontend Control**: In the demo HTML, users can:
- Select "Search all files in project" → Agent decides strategy
- Select "Search a specific file" → Single file query

## Streaming Events Sent

### Every 5 Seconds (Heartbeat/Processing)
```json
{
  "type": "processing",
  "message": "Processing 1st file: Exhibit 10.1 Lease Agreement.pdf...",
  "fileid": "f_a7c404dc4c",
  "filename": "Exhibit 10.1 Lease Agreement.pdf",
  "file_number": 1
}
```

### File-Specific Mode Events
1. **Decision Event**:
```json
{
  "type": "decision",
  "strategy": "file_specific",
  "reasoning": "Query suggests file-specific information"
}
```

2. **Status Event** (per file):
```json
{
  "type": "status",
  "fileid": "f_a7c404dc4c",
  "filename": "Exhibit 10.1 Lease Agreement.pdf",
  "message": "Generating inference for file: Exhibit 10.1 Lease Agreement.pdf..."
}
```

3. **Final Response Event** (per file):
```json
{
  "type": "final_response",
  "query": "summarize the doc",
  "projectid": "p_43b3e12cfe",
  "fileid": "f_a7c404dc4c",
  "filename": "Exhibit 10.1 Lease Agreement.pdf",
  "inference": "Based on the query... (Citation: Doc=Exhibit 10.1 Lease Agreement.pdf, Page=1)...",
  "citations": [
    {
      "fileid": "f_a7c404dc4c",
      "filename": "Exhibit 10.1 Lease Agreement.pdf",
      "section_path": ["18. ASSIGNMENT AND SUBLETTING"],
      "page_range": [1, 1],
      "citation_html": "<strong>Exhibit 10.1 Lease Agreement.pdf</strong> | Section: 18. ASSIGNMENT AND SUBLETTING | Page 1"
    }
  ]
}
```

4. **Complete Event**:
```json
{
  "type": "complete",
  "message": "All files processed"
}
```

## Summary

✅ **Citations**: Now show filenames instead of file IDs  
✅ **Processing Messages**: Include filename every 5 seconds  
✅ **File-Specific Processing**: Sequential (retrieve → infer → next file)  
✅ **Global vs File-Specific**: Agent decides or user selects single file  

All functionality is working as expected!

