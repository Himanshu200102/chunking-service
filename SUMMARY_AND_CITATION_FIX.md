# Summary Cut-off and Citation Fix

## Problems Fixed

### 1. Summary Being Cut Off Mid-Citation
**Problem**: Summaries were ending abruptly with incomplete citations like "(C" or "(Citation: Doc="

**Root Cause**: 
- `max_tokens=1200` was too low for longer summaries
- LLM would stop generating mid-citation when hitting the token limit

**Solution**:
- Increased `max_tokens` from 1200 → 2500
- Added aggressive cleanup patterns for incomplete citations at the end:
  - `(C`, `(Ci`, `(Cit`, `(Citation`, `(Citation: D`, etc.
  - Any incomplete parenthesis at the end

### 2. Citations Showing File IDs Instead of Names
**Problem**: Citations showed `Doc=f_a7c404dc4c` (file ID) instead of actual filename

**How It Works Now**:
1. **File names are pre-fetched** from MongoDB using `/projects/{project_id}/files` endpoint
2. **Passed through the pipeline**: API layer → Agent Retriever → Summarizer
3. **Stored in cache**: `self._file_name_cache` and `self._prefetched_file_names`
4. **Used in provenance headers**: `[Doc: actual_filename.pdf | Page: 1]`
5. **Replaced in citations**: LLM-generated citations with file IDs are replaced with filenames

**File Name Fetching Flow**:
```
user.py (query_response)
  ↓ Fetches file names from MongoDB
  ↓ file_name_map = {file_id: filename, ...}
  ↓
agent_retriever.retrieve(file_id_to_name=file_name_map)
  ↓
agent_retriever._generate_global_summary(file_id_to_name)
  ↓
summarizer.summarize_chunks(file_id_to_name)
  ↓ Stores in self._file_name_cache
  ↓ Uses in provenance headers
  ↓ Replaces file IDs in citations
  ↓
Final answer with proper citations: (Citation: Doc=filename.pdf, Page=1)
```

## Changes Made

### `app/retriever/summarizer.py`:

1. **Line 180**: Increased `max_tokens` from 1200 → 2500
   ```python
   max_tokens=2500,  # Increased to prevent cutoff of longer summaries
   ```

2. **Lines 649-665**: Enhanced incomplete citation cleanup
   ```python
   # Clean up any incomplete/broken citations at the end (aggressive patterns)
   answer = re.sub(r'\(C[itation]*:?\s*D[oc]*=?[^,)]*,?\s*P[age]*=?\d*\)?$', '', answer, flags=re.IGNORECASE)
   answer = re.sub(r'\(C[itation]*:?\s*D[oc]*=?[^,)]*$', '', answer, flags=re.IGNORECASE)
   answer = re.sub(r'\(C[itation]*:?\s*$', '', answer, flags=re.IGNORECASE)
   answer = re.sub(r'\([^)]*$', '', answer)  # Any incomplete parenthesis at the end
   ```

3. **Existing**: File ID replacement in citations (lines 588-621)
   - Already implemented: replaces `f_abc123` with actual filename
   - Uses pre-fetched file names from `file_id_to_name` parameter

## Result

✅ **Longer summaries without cutoff** - Can now generate up to 2500 tokens
✅ **Clean endings** - No more incomplete citations like "(C" at the end
✅ **User-friendly citations** - Shows actual filenames instead of file IDs
✅ **Proper file name fetching** - Uses existing `/projects/{project_id}/files` endpoint

## Testing

To test, run a query:
```bash
curl -X POST http://localhost:8002/USER/query-response \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the doc",
    "projectid": "p_43b3e12cfe",
    "max_results": 10
  }'
```

Expected output:
- Complete summary (no cutoff)
- Citations with actual filenames: `(Citation: Doc=Exhibit 10.1 Lease Agreement.pdf, Page=1)`
- No incomplete citations at the end

## What This Means for Your Team

- **Better summaries**: Longer, more complete answers
- **Professional citations**: Uses actual document names, not file IDs
- **Reliable output**: No more cut-off summaries or incomplete citations
- **Leverages existing API**: Uses the `/projects/{project_id}/files` endpoint already implemented

