# FileID Behavior Explanation

## Current Behavior

The system handles `fileid` in the following way:

### Logic Flow:

```
1. If fileid is provided (not None, not empty string):
   → Search ONLY that specific file
   
2. If fileid is None/empty AND use_agent=True:
   → Use intelligent agent (decides: file_specific or global strategy)
   
3. If fileid is None/empty AND use_agent=False:
   → Search ALL files in the project
```

### Code Logic:

```python
if captured_fileid:  # Truthy check (not None, not "", not False)
    # Single file search
    search_chunks(fileid=captured_fileid)
elif use_agent:
    # Agent decides strategy
    agent_retriever.retrieve(...)
else:
    # All files search
    search_chunks(fileid=None)  # No file filter
```

## Examples

### Example 1: Single File Query
```json
{
  "query": "What is revenue?",
  "projectid": "p_123",
  "fileid": "f_456",  ← Provided
  "use_agent": true
}
```
**Result:** Searches ONLY file `f_456`

---

### Example 2: All Files with Agent
```json
{
  "query": "What are the main topics?",
  "projectid": "p_123",
  "fileid": null,  ← Not provided (or empty string)
  "use_agent": true
}
```
**Result:** Agent decides:
- If query suggests per-file comparison → `file_specific` strategy (searches each file separately)
- If query suggests general info → `global` strategy (searches all files together)

---

### Example 3: All Files without Agent
```json
{
  "query": "What are the main topics?",
  "projectid": "p_123",
  "fileid": null,  ← Not provided
  "use_agent": false
}
```
**Result:** Searches ALL files in project (global search)

---

## Edge Cases

### Empty String `""`
```json
{
  "fileid": ""
}
```
**Current Behavior:** Treated as "all files" (because `if ""` is False)

### Null/None
```json
{
  "fileid": null
}
```
**Current Behavior:** Treated as "all files"

### Not Provided
```json
{
  // fileid field omitted
}
```
**Current Behavior:** Treated as "all files" (defaults to None)

---

## Summary Table

| fileid Value | use_agent | Behavior |
|--------------|-----------|----------|
| `"f_123"` (provided) | any | Search only file `f_123` |
| `null` or `""` | `true` | Agent decides (file_specific or global) |
| `null` or `""` | `false` | Search all files (global) |

---

## Recommendation

The current behavior is correct:
- ✅ **fileid provided** → Single file search
- ✅ **fileid null/empty + use_agent=true** → Intelligent routing
- ✅ **fileid null/empty + use_agent=false** → All files search

This gives users flexibility:
1. Want specific file? → Provide fileid
2. Want smart routing? → Omit fileid, set use_agent=true
3. Want simple all-files search? → Omit fileid, set use_agent=false


