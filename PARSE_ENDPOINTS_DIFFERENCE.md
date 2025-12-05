# Parse Latest vs Parse All Latest - Differences

## Overview

Both endpoints parse files and extract chunks, but they differ in scope and how they handle multiple files.

---

## 1. `parse_latest` - Single File Parsing

**Endpoint:** `POST /projects/{project_id}/files/{file_id}/parse_latest`

### Characteristics:
- ✅ Parses **ONE specific file**
- ✅ Synchronous (returns immediately after parsing)
- ✅ Returns full result with chunks
- ✅ **Does NOT auto-sync to retriever** (you need to manually sync or use parse_all_latest)

### Request:
```json
{
  "user_id": "temp_user_001",
  "do_ocr": false,
  "force": false
}
```

### Response:
```json
{
  "ok": true,
  "file_id": "f_456",
  "status": "parsed",
  "version": 1,
  "chunk_uri": "/app/uploads/projects/p_123/derived/f_456/1/all_chunks.jsonl",
  "chunks": [...]
}
```

### Use Case:
- Parse a single file that was just uploaded
- Re-parse a specific file (with `force: true`)
- Quick parsing of one file

---

## 2. `parse_all_latest` - Batch File Parsing

**Endpoint:** `GET /projects/{project_id}/files/parse-all-latest`

### Characteristics:
- ✅ Parses **ALL files** in the project
- ✅ Streaming (Server-Sent Events - SSE)
- ✅ Real-time progress updates
- ✅ **AUTO-SYNCS chunks to retriever** after completion ✅

### Query Parameters:
- `user_id`: `temp_user_001`
- `do_ocr`: `false` (or `true` for OCR)
- `force`: `false` (or `true` to re-parse)
- `only_status`: (Optional) Filter by status
- `limit`: `1000` (max files to process)

### Response Format (SSE Stream):
```
data: {"type": "init", "total": 3}
data: {"type": "progress", "current": 1, "total": 3, "file_id": "f_1"}
data: {"type": "result", "file_id": "f_1", "status": "parsed", "chunks": [...]}
data: {"type": "progress", "current": 2, "total": 3, "file_id": "f_2"}
data: {"type": "result", "file_id": "f_2", "status": "parsed", "chunks": [...]}
data: {"type": "complete", "summary": {"processed": 2, "skipped": 1, ...}}
```

### Use Case:
- Parse all files in a project at once
- Initial setup of a project with multiple files
- Re-parse all files after updates
- When you want automatic sync to retriever

---

## Key Differences Summary

| Feature | `parse_latest` | `parse_all_latest` |
|---------|----------------|-------------------|
| **Scope** | Single file | All files in project |
| **Response Type** | JSON (synchronous) | SSE Stream (asynchronous) |
| **Progress Updates** | No | Yes (real-time) |
| **Auto-Sync to Retriever** | ❌ **NO** | ✅ **YES** |
| **Speed** | Fast (one file) | Slower (all files) |
| **Use When** | Single file needed | Batch processing needed |

---

## Auto-Sync Behavior

### `parse_latest` (Single File):
```python
# Does NOT auto-sync
# You need to manually call:
POST /chunks/sync/{project_id}/{file_id}
# OR
POST /chunks/sync-all/{project_id}  # Syncs all files
```

### `parse_all_latest` (All Files):
```python
# Automatically syncs after completion
# Code in _stream_parse_progress():
threading.Thread(
    target=_trigger_retriever_sync_all,
    args=(project_id,),
    daemon=True,
).start()
```

---

## Recommended Workflow

### Option 1: Parse All Files (Recommended)
```
1. Upload files
2. POST /projects/{id}/files/parse-all-latest
   → Parses all files
   → Auto-syncs to retriever ✅
3. Query immediately (chunks are ready!)
```

### Option 2: Parse Single File
```
1. Upload file
2. POST /projects/{id}/files/{file_id}/parse_latest
   → Parses one file
3. POST /chunks/sync/{project_id}/{file_id}  ← Manual sync needed
   → Syncs chunks to retriever
4. Query
```

---

## Why Different Behavior?

**`parse_latest`** doesn't auto-sync because:
- It's designed for single-file operations
- You might want to parse multiple files individually before syncing
- Gives you control over when to sync

**`parse_all_latest`** auto-syncs because:
- It's designed for batch operations
- After parsing all files, you typically want them all synced
- Convenience for initial project setup

---

## Example Scenarios

### Scenario 1: New Project Setup
```
✅ Use parse_all_latest
→ Uploads all files
→ Parses all at once
→ Auto-syncs all chunks
→ Ready to query!
```

### Scenario 2: Add One File to Existing Project
```
Option A: Use parse_latest + manual sync
→ Parse single file
→ Manually sync that file

Option B: Use parse_all_latest
→ Re-parse all files (with force=false, skips already parsed)
→ Auto-syncs all (including new file)
```

### Scenario 3: Re-parse One File
```
✅ Use parse_latest with force=true
→ Re-parses that file
→ Manually sync if needed
```

---

## Summary

**Both endpoints parse files, but:**
- `parse_latest` = Single file, no auto-sync
- `parse_all_latest` = All files, **auto-syncs to retriever** ✅

**For most use cases, use `parse_all_latest`** because:
1. It handles multiple files efficiently
2. Auto-syncs chunks automatically
3. Provides progress updates
4. One call does everything

