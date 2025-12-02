# Local Llama ADK Integration

This guide explains how to run the Google Agent Development Kit (ADK) with a fully offline Llama 3 8B GGUF model.

## 1. Install Dependencies

```bash
conda activate DataRoom
pip install -r requirements.txt
```

`requirements.txt` now includes:
- `google-adk==1.18.0`
- `llama-cpp-python`

> **Note:** ADK pulls in several Google Cloud packages. They are safe to keep even if you only use local models.

## 2. Download a GGUF Model

Choose a GGUF build of Llama 3 8B Instruct, for example from the Hugging Face repo [`TheBloke/Llama-3-8B-Instruct-GGUF`](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF).

Download a quantization that fits your hardware (e.g. `*Q4_K_M.gguf`) and place it somewhere on disk, for example:

```
/models/llama-3-8b-instruct-q4_K_M.gguf
```

## 3. Configure Environment Variables

Add the following variables to your shell (or `.env.local`):

```bash
export LLAMA_MODEL_PATH=/models/llama-3-8b-instruct-q4_K_M.gguf
export LLAMA_CTX_SIZE=4096        # optional
export LLAMA_GPU_LAYERS=0         # >0 to offload to GPU if available
```

## 4. Start the API Server

```bash
cd /home/sheetalsharma/DataRoom-ai
export MONGO_URI=mongodb://localhost:27017
export OPENSEARCH_URL=http://localhost:9200
export LANCEDB_URI=/tmp/lancedb
export APP_ENV=dev
export EMBEDDING_MODEL=all-MiniLM-L6-v2
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

When `app.main` loads it imports `app.agents`, which registers the local Llama provider with ADK. Any ADK agents that use `model="local/llama-8b-gguf"` will automatically hit your offline model.

## 5. Quick Test Script

Run the sample agent to verify inference:

```bash
python scripts/run_local_agent.py "Explain retrieval augmented generation in simple terms."
```

Expected output:
```
[local/llama-8b-gguf] -> <agent reply>
```

## 6. Using ADK in Your Code

Inside your application you can now create ADK agents that call the local model:

```python
from google.adk import Agent
import app.agents  # registers the local llama service

agent = Agent(
    name="local_helper",
    model="local/llama-8b-gguf",
    instruction="You are a concise technical assistant."
)

result = await agent.run("Summarize the latest project status.")
print(result.output)
```

Because ADK is model-agnostic, the rest of the agent tooling (tools, evaluations, multi-agent flows) works unchanged.

## 7. Notes

- The integration is minimal: it focuses on text-only prompts and responses.
- Streaming is supported, but it uses a blocking llama.cpp iterator under the hood; adjust `LLAMA_CTX_SIZE` / quantization for comfort.
- If you switch to a different model name, update `SUPPORTED_MODEL_REGEX` in `app/agents/local_llama.py`.

Happy hacking!
