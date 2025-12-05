# Streaming Behavior for File-Specific Strategy

## Overview

When querying all files with the agent, and the agent decides on **file-specific** strategy, the streaming works as follows:

## Flow for File-Specific Strategy

### 1. Agent Decision
- Stream sends: `{"type": "decision", "strategy": "file_specific", "reasoning": "..."}`

### 2. For Each File (Sequentially)

#### File 1:
1. **Status Event**: `{"type": "status", "fileid": "file1", "message": "Retrieving chunks for file file1..."}`
2. **Processing Messages** (every 5 seconds):
   - `{"type": "processing", "message": "Processing 1st file..."}`
   - `{"type": "processing", "message": "Processing 1st file..."}` (every 5 seconds)
3. **Status Event**: `{"type": "status", "fileid": "file1", "message": "Generating inference for file file1..."}`
4. **Processing Messages** (every 5 seconds, continues during inference):
   - `{"type": "processing", "message": "Processing 1st file..."}`
5. **Final Response for File 1**:
   ```json
   {
     "type": "final_response",
     "query": "user query",
     "projectid": "project_id",
     "fileid": "file1",
     "inference": "HTML-friendly inference with citations...",
     "citations": [...]
   }
   ```

#### File 2:
1. **Status Event**: `{"type": "status", "fileid": "file2", "message": "Retrieving chunks for file file2..."}`
2. **Processing Messages** (every 5 seconds):
   - `{"type": "processing", "message": "Processing 2nd file..."}`
   - `{"type": "processing", "message": "Processing 2nd file..."}` (every 5 seconds)
3. **Status Event**: `{"type": "status", "fileid": "file2", "message": "Generating inference for file file2..."}`
4. **Processing Messages** (every 5 seconds, continues during inference):
   - `{"type": "processing", "message": "Processing 2nd file..."}`
5. **Final Response for File 2**:
   ```json
   {
     "type": "final_response",
     "query": "user query",
     "projectid": "project_id",
     "fileid": "file2",
     "inference": "HTML-friendly inference with citations...",
     "citations": [...]
   }
   ```

#### File N:
- Same pattern continues for each file...

### 3. Completion
- Stream sends: `{"type": "complete", "message": "All files processed"}`

## Key Features

✅ **Per-File Processing Messages**: Shows "Processing 1st file...", "Processing 2nd file...", etc. every 5 seconds
✅ **Per-File Final Responses**: Each file gets its own final response with inference and citations
✅ **Sequential Processing**: Files are processed one at a time
✅ **Real-Time Updates**: Status events show progress for each file
✅ **HTML-Friendly Inference**: Each inference is formatted as HTML with citations

## Example Stream Sequence

```
data: {"type":"decision","strategy":"file_specific","reasoning":"..."}

data: {"type":"status","fileid":"file1","message":"Retrieving chunks for file file1..."}
data: {"type":"processing","message":"Processing 1st file..."}
data: {"type":"processing","message":"Processing 1st file..."}
data: {"type":"status","fileid":"file1","message":"Generating inference for file file1..."}
data: {"type":"processing","message":"Processing 1st file..."}
data: {"type":"final_response","query":"...","projectid":"...","fileid":"file1","inference":"...","citations":[...]}

data: {"type":"status","fileid":"file2","message":"Retrieving chunks for file file2..."}
data: {"type":"processing","message":"Processing 2nd file..."}
data: {"type":"processing","message":"Processing 2nd file..."}
data: {"type":"status","fileid":"file2","message":"Generating inference for file file2..."}
data: {"type":"processing","message":"Processing 2nd file..."}
data: {"type":"final_response","query":"...","projectid":"...","fileid":"file2","inference":"...","citations":[...]}

data: {"type":"complete","message":"All files processed"}
```

## Implementation Details

- **Processing Messages**: Sent every 5 seconds while a file is being processed
- **File Tracking**: Tracks current file index (1st, 2nd, 3rd, etc.) and updates processing messages accordingly
- **Final Response Format**: Each file gets a `final_response` event with:
  - `query`: Original user query
  - `projectid`: Project ID
  - `fileid`: Specific file ID
  - `inference`: HTML-friendly inference text with citations
  - `citations`: Array of citation objects

