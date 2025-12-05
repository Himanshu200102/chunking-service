# Swagger API Endpoints Guide

## Accessing Swagger UI

1. Start the service: `./run.sh`
2. Open browser: `http://localhost:8002/docs`
3. You'll see all available endpoints organized by tags

---

## Complete Workflow Flow

### Step 1: Create/Get Project

**Endpoint:** `POST /projects`

**Purpose:** Create a new project or get existing project ID

**Request Body:**
```json
{
  "name": "My Project",
  "user_id": "temp_user_001"
}
```

**Response:**
```json
{
  "_id": "p_86484360e2",
  "name": "My Project",
  "created_at": "2024-12-03T..."
}
```

**Note:** Save the `_id` (project_id) for next steps!

---

### Step 2: Upload File

**Endpoint:** `POST /projects/{project_id}/files`

**Purpose:** Upload a PDF/document to the project

**Parameters:**
- `project_id`: The project ID from Step 1
- `user_id`: Query parameter (e.g., `?user_id=temp_user_001`)

**Request:**
- Use "Try it out" button
- Click "Choose File" and select your PDF
- Click "Execute"

**Response:**
```json
{
  "_id": "f_ef454b0540",
  "filename": "document.pdf",
  "project_id": "p_86484360e2",
  "status": "uploaded"
}
```

**Note:** Save the `_id` (file_id) for next steps!

---

### Step 3: Parse File

**Endpoint:** `POST /projects/{project_id}/files/{file_id}/parse_latest`

**Purpose:** Parse the uploaded file to extract text and create chunks

**Path Parameters:**
- `project_id`: From Step 1
- `file_id`: From Step 2

**Request Body:**
```json
{
  "user_id": "temp_user_001",
  "do_ocr": false,
  "force": false
}
```

**Parameters:**
- `do_ocr`: Set to `true` if you want OCR for scanned documents
- `force`: Set to `true` to re-parse even if already parsed

**Response:**
```json
{
  "ok": true,
  "file_id": "f_ef454b0540",
  "status": "parsed",
  "version": 1,
  "chunk_uri": "/app/uploads/projects/p_86484360e2/derived/f_ef454b0540/1/all_chunks.jsonl",
  "chunks": [...]
}
```

**Note:** After parsing, chunks are automatically synced to the retriever (no manual sync needed!)

---

### Step 4: Parse All Files (Alternative)

**Endpoint:** `GET /projects/{project_id}/files/parse-all-latest`

**Purpose:** Parse all files in a project at once (streaming endpoint)

**Path Parameters:**
- `project_id`: From Step 1

**Query Parameters:**
- `user_id`: `temp_user_001`
- `do_ocr`: `false` (or `true` for OCR)
- `force`: `false` (or `true` to re-parse)
- `limit`: `1000` (max files to process)

**Response:** Server-Sent Events (SSE) stream with progress updates

**Note:** This endpoint automatically syncs chunks to retriever after parsing completes!

---

### Step 5: Query/Inference (Get Answers)

**Endpoint:** `POST /USER/query-response`

**Purpose:** Ask questions about your documents and get AI-generated answers

**Request Body:**
```json
{
  "query": "What are the main topics mentioned in the documents?",
  "projectid": "p_86484360e2",
  "fileid": "f_ef454b0540",
  "user_id": "temp_user_001",
  "max_results": 10,
  "use_agent": true
}
```

**Parameters:**
- `query`: Your question
- `projectid`: Project ID from Step 1
- `fileid`: (Optional) Specific file ID, or omit to search all files
- `user_id`: User identifier
- `max_results`: Number of chunks to retrieve (default: 10)
- `use_agent`: `true` for intelligent routing, `false` for simple search

**Response:** Server-Sent Events (SSE) stream with:
- `decision`: Agent's strategy decision
- `status`: Processing updates
- `final_response`: Complete answer with citations
- `complete`: End of stream

**Example Response Events:**
```
data: {"type": "decision", "strategy": "global", "reasoning": "..."}
data: {"type": "status", "message": "Retrieving chunks..."}
data: {"type": "final_response", "inference": "The documents discuss...", "citations": [...]}
data: {"type": "complete", "message": "Query completed"}
```

---

## Retriever Endpoints (Under `/chunks`)

### Health Check

**Endpoint:** `GET /chunks/health`

**Purpose:** Check if retriever service is healthy

**Response:**
```json
{
  "opensearch": {"status": "green"},
  "lancedb": {"status": "ok"},
  "status": "healthy"
}
```

---

### Direct Query (Without Agent)

**Endpoint:** `POST /chunks/query`

**Purpose:** Query chunks directly without agent routing

**Request Body:**
```json
{
  "query": "What is mentioned about revenue?",
  "projectid": "p_86484360e2",
  "fileid": "f_ef454b0540",
  "search_all_files": false,
  "max_results": 10,
  "filters": null
}
```

**Response:**
```json
{
  "query": "What is mentioned about revenue?",
  "total_results": 5,
  "chunks": [...],
  "projectid": "p_86484360e2",
  "fileid": "f_ef454b0540",
  "summary": "The documents mention revenue of $X million...",
  "compression_stats": {...}
}
```

---

### Agent-Based Query

**Endpoint:** `POST /chunks/query/agent`

**Purpose:** Intelligent agent decides whether to search per-file or globally

**Query Parameters:**
- `query`: Your question
- `projectid`: Project ID
- `available_fileids`: (Optional) List of file IDs to consider
- `max_results_per_file`: 10
- `max_results_global`: 10
- `stream`: `false` (or `true` for streaming)

**Response:**
```json
{
  "strategy": "file_specific",
  "reasoning": "Query asks about each file separately",
  "query": "...",
  "projectid": "...",
  "results": {
    "f_file1": [chunks...],
    "f_file2": [chunks...]
  },
  "summaries": {
    "f_file1": "Summary for file 1...",
    "f_file2": "Summary for file 2..."
  }
}
```

---

### Sync Chunks (Manual)

**Endpoint:** `POST /chunks/sync-all/{project_id}`

**Purpose:** Manually trigger sync of all chunks to retriever

**Path Parameters:**
- `project_id`: Project ID

**Query Parameters:**
- `do_ocr`: `false`
- `force`: `false`
- `only_status`: (Optional) Filter by status
- `background`: `false`

**Response:**
```json
{
  "success": true,
  "message": "Processed 3 files, ingested 58 chunks",
  "ingested_chunks": 58,
  "errors": []
}
```

**Note:** Usually not needed - chunks auto-sync after parsing!

---

### Sync Single File

**Endpoint:** `POST /chunks/sync/{project_id}/{file_id}`

**Purpose:** Manually sync chunks for a single file

**Path Parameters:**
- `project_id`: Project ID
- `file_id`: File ID

**Response:**
```json
{
  "success": true,
  "message": "Chunk sync queued",
  "project_id": "p_86484360e2",
  "file_id": "f_ef454b0540"
}
```

---

## Complete Example Flow in Swagger

### 1. Create Project
```
POST /projects
Body: {"name": "Test Project", "user_id": "temp_user_001"}
→ Save project_id: "p_123"
```

### 2. Upload File
```
POST /projects/p_123/files?user_id=temp_user_001
File: document.pdf
→ Save file_id: "f_456"
```

### 3. Parse File
```
POST /projects/p_123/files/f_456/parse_latest
Body: {"user_id": "temp_user_001", "do_ocr": false, "force": false}
→ Wait for "status": "parsed"
```

### 4. Query (Get Answer)
```
POST /USER/query-response
Body: {
  "query": "What are the main topics?",
  "projectid": "p_123",
  "fileid": "f_456",
  "user_id": "temp_user_001",
  "use_agent": true
}
→ Stream response with final_answer
```

---

## Tips for Using Swagger

1. **Use "Try it out"**: Click the button to enable editing
2. **Check Required Fields**: Red asterisk (*) means required
3. **View Schema**: Click "Schema" to see request/response structure
4. **Copy cURL**: Use "Copy" button to get cURL command
5. **Streaming Endpoints**: For SSE endpoints, use cURL or Postman (Swagger UI may not show streams properly)

---

## Common Use Cases

### Use Case 1: Single File Q&A
1. Upload file → Parse → Query with `fileid`

### Use Case 2: Multi-File Search
1. Upload multiple files → Parse all → Query without `fileid` (searches all)

### Use Case 3: Compare Files
1. Upload files → Parse all → Query with `use_agent: true` (agent will use file_specific strategy)

### Use Case 4: Global Summary
1. Upload files → Parse all → Query without `fileid` and `use_agent: true` (agent will use global strategy)

---

## Troubleshooting

**Problem:** Endpoint returns 404
- **Solution:** Check that service is running: `curl http://localhost:8002/health`

**Problem:** Parsing takes too long
- **Solution:** Normal for large PDFs. Check logs: `docker logs -f DataRoom-api`

**Problem:** No chunks found
- **Solution:** Ensure file is parsed (`status: "parsed"`) and chunks are synced

**Problem:** Query returns empty
- **Solution:** Check that chunks exist: `GET /chunks/health` and verify project_id/file_id are correct

---

## Endpoint Summary Table

| Endpoint | Method | Purpose | Auto-Sync? |
|----------|--------|---------|------------|
| `/projects` | POST | Create project | N/A |
| `/projects/{id}/files` | POST | Upload file | No |
| `/projects/{id}/files/{fid}/parse_latest` | POST | Parse single file | ✅ Yes |
| `/projects/{id}/files/parse-all-latest` | GET | Parse all files | ✅ Yes |
| `/USER/query-response` | POST | Query & get answers | N/A |
| `/chunks/query` | POST | Direct query | N/A |
| `/chunks/query/agent` | POST | Agent-based query | N/A |
| `/chunks/sync-all/{id}` | POST | Manual sync all | N/A |
| `/chunks/health` | GET | Health check | N/A |

---

## Next Steps

1. **Test the flow** using Swagger UI at `http://localhost:8002/docs`
2. **Use the demo page** at `http://localhost:8002/demo` for a visual interface
3. **Check logs** if something doesn't work: `docker logs -f DataRoom-api`


