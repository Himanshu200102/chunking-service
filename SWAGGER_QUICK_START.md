# Swagger Quick Start Guide

## 🚀 Quick Access

1. **Start Service**: `./run.sh`
2. **Open Swagger**: `http://localhost:8002/docs`
3. **Follow the flow below**

---

## 📋 Complete Flow (Step by Step)

```
┌─────────────────────────────────────────────────────────────┐
│                    SWAGGER API FLOW                         │
└─────────────────────────────────────────────────────────────┘

Step 1: CREATE PROJECT
├─ Endpoint: POST /projects
├─ Body: {"name": "My Project", "user_id": "temp_user_001"}
└─ Response: {"_id": "p_123", ...}  ← SAVE THIS project_id

Step 2: UPLOAD FILE
├─ Endpoint: POST /projects/{project_id}/files
├─ Query: ?user_id=temp_user_001
├─ File: Choose PDF file
└─ Response: {"_id": "f_456", ...}  ← SAVE THIS file_id

Step 3: PARSE FILE
├─ Endpoint: POST /projects/{project_id}/files/{file_id}/parse_latest
├─ Body: {"user_id": "temp_user_001", "do_ocr": false, "force": false}
└─ Response: {"status": "parsed", "chunk_uri": "...", ...}
   ⚠️ WAIT for status: "parsed" (chunks auto-sync to retriever!)

Step 4: QUERY (GET ANSWERS)
├─ Endpoint: POST /USER/query-response
├─ Body: {
│     "query": "What are the main topics?",
│     "projectid": "p_123",
│     "fileid": "f_456",  ← Optional: omit to search all files
│     "user_id": "temp_user_001",
│     "use_agent": true
│   }
└─ Response: SSE stream with final_answer
```

---

## 🎯 Most Common Endpoints

### 1. **Create Project**
```
POST /projects
```
**Why:** Start a new project to organize your documents

---

### 2. **Upload File**
```
POST /projects/{project_id}/files?user_id=temp_user_001
```
**Why:** Add PDFs/documents to your project

---

### 3. **Parse File**
```
POST /projects/{project_id}/files/{file_id}/parse_latest
```
**Why:** Extract text and create searchable chunks (auto-syncs to retriever!)

---

### 4. **Query & Get Answers**
```
POST /USER/query-response
```
**Why:** Ask questions and get AI-generated answers with citations

---

## 🔍 Alternative: Parse All Files at Once

Instead of parsing files one by one, you can parse all files in a project:

```
GET /projects/{project_id}/files/parse-all-latest?user_id=temp_user_001&force=false
```

**Benefits:**
- Parses all files in one go
- Automatically syncs all chunks to retriever
- Returns streaming progress updates

---

## 💡 Key Concepts Explained

### **Project vs File**
- **Project**: Container for multiple files (like a folder)
- **File**: Individual document (PDF, etc.)

### **Parse vs Sync**
- **Parse**: Extracts text from PDF and creates chunks
- **Sync**: Ingests chunks into vector database (OpenSearch/LanceDB)
- **Auto-Sync**: Happens automatically after parsing! ✅

### **Query Types**

1. **Single File Query**
   - Include `fileid` in request
   - Searches only that file

2. **All Files Query**
   - Omit `fileid` in request
   - Searches across all files in project

3. **Agent Query** (`use_agent: true`)
   - AI decides: search per-file or globally
   - Better for complex questions

---

## 📝 Example Request Bodies

### Create Project
```json
{
  "name": "Financial Reports",
  "user_id": "temp_user_001"
}
```

### Parse File
```json
{
  "user_id": "temp_user_001",
  "do_ocr": false,
  "force": false
}
```

### Query (Single File)
```json
{
  "query": "What is the revenue mentioned?",
  "projectid": "p_86484360e2",
  "fileid": "f_ef454b0540",
  "user_id": "temp_user_001",
  "max_results": 10,
  "use_agent": true
}
```

### Query (All Files)
```json
{
  "query": "What are the main topics across all documents?",
  "projectid": "p_86484360e2",
  "user_id": "temp_user_001",
  "max_results": 10,
  "use_agent": true
}
```

---

## 🎨 Swagger UI Tips

1. **Click "Try it out"** to enable editing
2. **Required fields** are marked with red asterisk (*)
3. **View Schema** to see all available fields
4. **Copy cURL** to test from command line
5. **Check Response** to see example outputs

---

## ⚡ Quick Test Flow

1. **Create Project**
   - `POST /projects` → Get `project_id`

2. **Upload PDF**
   - `POST /projects/{project_id}/files` → Get `file_id`

3. **Parse**
   - `POST /projects/{project_id}/files/{file_id}/parse_latest`
   - Wait for `"status": "parsed"`

4. **Query**
   - `POST /USER/query-response`
   - Get answer with citations!

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| 404 Not Found | Check service is running: `curl http://localhost:8002/health` |
| Parsing slow | Normal for large PDFs. Check logs: `docker logs -f DataRoom-api` |
| No chunks found | Ensure file is parsed (`status: "parsed"`) |
| Empty query results | Verify `projectid` and `fileid` are correct |

---

## 📚 Full Documentation

See `SWAGGER_ENDPOINTS_GUIDE.md` for detailed endpoint documentation.

---

## 🎯 Next Steps

1. Open `http://localhost:8002/docs`
2. Try the flow above
3. Check the demo page: `http://localhost:8002/demo`


