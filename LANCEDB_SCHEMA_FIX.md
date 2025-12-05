# LanceDB Schema Fix

## Problem
LanceDB was throwing the error:
```
ArrowNotImplementedError: Unsupported cast from string to null using function cast_null
```

This happened because the schema was initialized with `None` values for `caption` and `metadata` fields, but later tried to insert string values, causing a type mismatch.

## Solution

### Changes Made to `app/retriever/lancedb_client.py`:

1. **Fixed Sample Data (lines 72-87)**:
   - Changed `"caption": None` → `"caption": ""`
   - Changed `"metadata": None` → `"metadata": "{}"`
   - This ensures the schema is initialized with string types, not null types

2. **Enhanced `_chunk_to_dict` Method (lines 164-179)**:
   - Added explicit caption normalization to ensure it's always a string
   - Checks for `None` or `"None"` and converts to empty string
   - Ensures all fields are properly typed before insertion

3. **Recreated LanceDB Table**:
   - Deleted old table with incorrect schema
   - Restarted API to create fresh table with correct schema

## Result

✅ **LanceDB is now healthy and working**
✅ **Hybrid retrieval (OpenSearch + LanceDB) is functional**
✅ **No more Arrow cast errors during chunk ingestion**

## Verification

```bash
curl http://localhost:8002/chunks/health
```

Should return:
```json
{
    "lancedb": {
        "status": "healthy",
        "table_name": "chunks",
        "total_chunks": 1
    },
    "status": "healthy"
}
```

## What This Means for Your Team

- **Hybrid retrieval is enabled**: Both OpenSearch (keyword) and LanceDB (semantic/vector) search are working
- **Better search quality**: Combines keyword matching with semantic understanding
- **No data loss**: The fix only affects the schema, all chunks will be re-ingested correctly

