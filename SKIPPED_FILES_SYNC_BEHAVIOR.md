# Skipped Files Sync Behavior

## Question: If a file is already parsed (skipped), will it still sync to retriever?

## Answer: ✅ **YES, it will still sync!**

---

## How It Works

### Flow for `parse-all-latest`:

1. **Parsing Phase:**
   ```
   parse-all-latest → processes all files
   - If file already parsed (force=false) → skipped: true, status: "parsed"
   - chunk_uri is included in the result even for skipped files
   ```

2. **Auto-Sync Phase (After Parsing Completes):**
   ```
   _trigger_retriever_sync_all() → calls retriever's parse_and_sync_all_files()
   ```

3. **Sync Logic for Skipped Files:**
   ```python
   if skipped and status == "parsed":
       # File was already parsed
       chunk_uri = file_result.get("chunk_uri")  # Gets chunk_uri from skipped result
       
       # If chunk_uri exists, ingest chunks
       if chunk_uri:
           ingest_chunks_from_chunking_service(
               project_id, file_id, file_version_id, chunk_uri
           )
   ```

---

## Example Scenario

### Input:
```
File: f_123
Status: Already parsed (skipped: true)
chunk_uri: "/app/uploads/projects/p_1/derived/f_123/1/chunks.json"
```

### What Happens:

1. **During parse-all-latest:**
   ```
   data: {"type": "result", "file_id": "f_123", "skipped": true, 
          "status": "parsed", "chunk_uri": "/app/uploads/..."}
   ```

2. **After parsing completes:**
   ```
   Auto-sync triggered → parse_and_sync_all_files()
   ```

3. **Sync processes skipped file:**
   ```
   - Detects: skipped=true, status="parsed"
   - Gets chunk_uri from result
   - Loads chunks from chunk_uri
   - Ingests chunks into retriever (OpenSearch + LanceDB)
   ```

4. **Result:**
   ```
   ✅ Chunks synced to retriever even though file was skipped!
   ```

---

## Code Evidence

### In `chunking_integration.py` (lines 672-780):

```python
# Handle already-parsed files
if skipped and status == "parsed":
    # File was already parsed - check if chunk_uri is already in the result
    chunk_uri = file_result.get("chunk_uri")
    
    # ... tries to get chunk_uri if not present ...
    
    if chunk_uri:
        # Ingests chunks even for skipped files!
        ingest_result = self.ingest_chunks_from_chunking_service(
            project_id=project_id,
            file_id=file_id,
            file_version_id=file_version_id,
            chunk_uri=chunk_uri
        )
```

---

## Summary

| Scenario | File Status | Sync to Retriever? |
|----------|-------------|-------------------|
| New file parsed | `skipped: false`, `status: "parsed"` | ✅ YES |
| Already parsed file | `skipped: true`, `status: "parsed"` | ✅ **YES** |
| Failed to parse | `skipped: false`, `status: "error"` | ❌ NO |

---

## Key Points

1. ✅ **Skipped files ARE synced** - The sync happens after parsing completes
2. ✅ **chunk_uri is included** - Even skipped files have chunk_uri in the result
3. ✅ **Auto-sync handles both** - Newly parsed AND already-parsed files
4. ✅ **No manual sync needed** - Everything happens automatically

---

## Why This Design?

- **Convenience**: You don't need to manually sync files that were already parsed
- **Consistency**: All files in the project get synced, regardless of when they were parsed
- **Reliability**: Even if a file was parsed before but not synced, it will be synced now

---

## Verification

To verify chunks are synced for skipped files, check the logs:

```
File result for f_123: status=parsed, skipped=True, chunk_uri=/app/uploads/...
✅ Found chunk file at: /app/uploads/...
Ingesting 114 chunks for file f_123...
Successfully ingested 114 chunks for file f_123
```

---

## Conclusion

**Yes, skipped files (already parsed) WILL sync to retriever automatically!** 🎉

The auto-sync function specifically handles this case and ingests chunks from the `chunk_uri` even when the file was skipped during parsing.


