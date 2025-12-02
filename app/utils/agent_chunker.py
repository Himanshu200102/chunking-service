# app/utils/agent_chunker.py
import json
import logging
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

SYSTEM_RULES = """You are a document chunking agent.
Produce JSON lines, one chunk per line.
Each JSON object must contain:
- "chunk_ref": stable unique string (REQUIRED)
- "text": chunk text (400-800 tokens target) (REQUIRED)
- "section_path": array of strings - MANDATORY, never empty (REQUIRED)
- "object_type": one of ["narrative","table","figure","code","other"] (REQUIRED)
- "page_range": [start_page, end_page] (REQUIRED)
- "caption": string (REQUIRED for tables/figures, null for narrative)
- "metadata": object with additional context (optional)

CRITICAL RULES - DO NOT FABRICATE:
1. section_path is MANDATORY and must NEVER be empty []
2. ONLY use section headings/titles that ACTUALLY APPEAR in the document content provided
3. DO NOT invent, make up, or assume section names that are not explicitly in the content
4. If you see "# Introduction" in the content, use ["Introduction"]
5. If you see "## 2.1 Data Analysis", use ["2.1 Data Analysis"] or ["Chapter 2", "2.1 Data Analysis"]
6. If NO headings are present in the content, use ONLY these fallback options:
   - ["Document"] - for general content
   - ["Page N"] - where N is the actual page number
   - NEVER use invented names like "Main Content", "Body", "Analysis" unless those exact words appear as headings

Section Path Rules (ONLY from actual content):
- Extract section_path from ACTUAL headings in the markdown (lines starting with #, ##, ###, etc.)
- Use the exact wording from the heading, do not paraphrase or summarize
- For hierarchical headings, build path from parent to child: ["Chapter 1", "Section 1.2", "Subsection 1.2.1"]
- For tables/figures, use the section heading under which they appear
- If content has NO heading: use ["Document"] or ["Page N"] where N is the page number
- If between sections: use the most recent preceding heading
- Preserve numbering if present: "1.2 Methods" should be ["1.2 Methods"], not ["Methods"]

DO NOT:
- ❌ Invent section names that don't exist in the content
- ❌ Use generic names like "Introduction", "Main Content", "Body" unless they are ACTUAL headings
- ❌ Paraphrase or reword section headings
- ❌ Assume document structure that isn't explicitly present
- ❌ Use "Uncategorized" unless absolutely necessary (no headings at all in entire content)

Rules for ALL content:
- Use 50–100 token overlap ONLY for narrative text. No overlap for table/figure/code.
- Do not include Markdown fences or extra formatting in output JSON.
- Output ONLY JSON lines (no prose).

Rules for TABLES:
- Keep tables atomic (entire table in one chunk).
- Extract caption from the actual document if present (look for "Table N:", "Table:", captions above/below table)
- If NO caption exists in the document, generate a descriptive caption based on:
  * Column headers visible in the table
  * Table content/data
  * DO NOT make up what the table shows - describe only what is visible
  * Example: "Table with columns: Quarter, Revenue, Growth" (factual description)
  * NOT: "Table showing quarterly financial performance" (interpretation)
- Include the full table markdown in the "text" field.
- Set object_type to "table".
- Add to metadata: {"num_rows": N, "num_cols": M, "has_header": true/false}
- section_path MUST be from the actual heading where this table appears

Rules for FIGURES/IMAGES:
- Keep figures atomic (entire figure description in one chunk).
- Extract caption from the actual document if present (look for "Figure N:", "Fig N:", captions near image)
- If NO caption exists in the document, generate a descriptive caption based on:
  * Alt text or description that is actually in the content
  * DO NOT interpret or assume what the figure shows
  * Example: "Figure containing a bar chart" (factual)
  * NOT: "Figure showing sales trends over time" (interpretation without evidence)
- In the "text" field, include ONLY what is actually present:
  * The actual caption if it exists
  * Any alt text or description that is in the content
  * Figure reference text like "Figure 1" or "Fig. 2.3" if present
- Set object_type to "figure".
- Add to metadata: {"figure_type": "chart/diagram/photo/etc", "referenced_as": "Figure 1"} - use only if this info is in content
- section_path MUST be from the actual heading where this figure appears

Rules for NARRATIVE text:
- Split into logical chunks of 400-800 tokens.
- Use 50-100 token overlap between consecutive chunks.
- Prefer breaking at section/paragraph boundaries.
- Set object_type to "narrative".
- caption should be null for narrative text.
- section_path MUST be from actual headings in the content

Caption Generation Guidelines (when caption is missing):
- Be purely descriptive, not interpretive
- Base caption ONLY on what is visible/present in the content
- For tables: describe structure (columns, data type), not meaning
- For figures: describe type (chart, diagram, image), not interpretation
- Keep it factual: "Table with 3 columns" not "Table showing sales performance"
- Be concise (1 sentence)

Example outputs:

Narrative with actual heading:
{"chunk_ref": "p1_narrative_001", "text": "The introduction section explains the background...", "section_path": ["1. Introduction"], "object_type": "narrative", "page_range": [1, 2], "caption": null}

Narrative without heading (fallback):
{"chunk_ref": "p7_narrative_015", "text": "This paragraph discusses various considerations...", "section_path": ["Page 7"], "object_type": "narrative", "page_range": [7, 7], "caption": null}

Table with actual caption and heading:
{"chunk_ref": "p5_table_001", "text": "Table 3: Quarterly Results\\n| Quarter | Revenue | Growth |\\n|---------|---------|--------|\\n| Q1 | $1.2M | 15% |", "section_path": ["3. Financial Analysis", "3.2 Quarterly Review"], "object_type": "table", "page_range": [5, 5], "caption": "Quarterly Results", "metadata": {"num_rows": 2, "num_cols": 3, "has_header": true}}

Table without caption (descriptive only):
{"chunk_ref": "p8_table_002", "text": "| Product | Units | Price |\\n|---------|-------|-------|\\n| Widget A | 450 | $12.99 |", "section_path": ["4. Product Data"], "object_type": "table", "page_range": [8, 8], "caption": "Table with columns: Product, Units, Price", "metadata": {"num_rows": 2, "num_cols": 3, "has_header": true}}

Figure with actual caption:
{"chunk_ref": "p12_fig_001", "text": "Figure 3.2: System Architecture\\n[Diagram present in document]", "section_path": ["3. System Design", "3.2 Architecture"], "object_type": "figure", "page_range": [12, 12], "caption": "System Architecture", "metadata": {"figure_type": "diagram", "referenced_as": "Figure 3.2"}}

Figure without caption (descriptive only):
{"chunk_ref": "p15_fig_003", "text": "[Bar chart image present]", "section_path": ["5. Results"], "object_type": "figure", "page_range": [15, 15], "caption": "Figure containing a bar chart", "metadata": {"figure_type": "chart"}}

REMEMBER: 
- ONLY use information that is ACTUALLY in the provided content
- DO NOT invent, assume, or interpret beyond what is explicitly present
- section_path must come from ACTUAL headings in the document
- Captions must be based on what is visible, not assumptions about meaning
- When in doubt, be more literal and less interpretive
"""


def _call_google_agent_sdk(md_content, json_content: Optional[Dict]) -> str:
    """
    Call Google Gemini API to chunk a markdown document.
    
    Args:
        md_content: Markdown content of the document
        json_content: Optional Docling JSON content for context
    Returns:
        Raw response text from the model
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    
    
    # Build the prompt
    prompt_parts = []
    
    # Add JSON context if provided
    if json_content:
        docling_json = json_content
        
        # Extract useful metadata from JSON
        context_info = {
            "total_pages": len(docling_json.get("pages", [])),
            "num_tables": len(docling_json.get("tables", [])),
            "num_figures": len(docling_json.get("pictures", [])),
            "sections": [
                {
                    "title": t.get("text", ""),
                    "level": t.get("level", 1),
                    "page": t.get("prov", [{}])[0].get("page_no", 1) if t.get("prov") else 1
                }
                for t in docling_json.get("texts", [])
                if t.get("label") in ("section_header", "title")
            ][:10]  # Limit to first 10 sections
        }
        
        prompt_parts.append("Document Context (from Docling extraction):")
        prompt_parts.append(json.dumps(context_info, indent=2))
        prompt_parts.append("\n--- DOCUMENT CONTENT TO CHUNK ---\n")
    
    # Add markdown content
    prompt_parts.append(md_content)
    
    prompt = "\n".join(prompt_parts)
    
    # Configure and call the model
    genai.configure(api_key=api_key)
    model_name = "gemini-2.5-flash"
    
    try:
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_RULES
        )
        
        resp = model_instance.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192
            )
        )
        
        return resp.text
    
    except Exception as e:
        logger.error(f"Google AI Studio API error: {e}")
        raise


def _parse_jsonl(response_text: str) -> List[Dict]:
    """
    Parse JSONL response from the agent.
    
    Args:
        response_text: Raw text response from model
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    for line in response_text.strip().split("\n"):
        line = line.strip()
        
        # Skip empty lines and markdown fences
        if not line or line.startswith("```"):
            continue
        
        try:
            chunk = json.loads(line)
            
            # Validate required fields
            if "chunk_ref" in chunk and "text" in chunk:
                # Set defaults for optional fields
                chunk.setdefault("section_path", [])
                chunk.setdefault("object_type", "narrative")
                chunk.setdefault("page_range", [1, 1])
                
                chunks.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk (missing required fields): {line[:100]}")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
    
    return chunks
