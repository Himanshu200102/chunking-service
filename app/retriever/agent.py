"""Agent module for intelligent retrieval routing using local GGUF model."""
import logging
import json
import os
from typing import Dict, List, Optional, Literal
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from app.retriever.config import settings

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Agent that decides retrieval strategy using a local GGUF model.
    
    The agent analyzes user queries to determine:
    1. File-specific retrieval: Query each file separately and return chunks per file
    2. Global retrieval: Query across all files and return top chunks overall
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the agent with a GGUF model."""
        self.model = None
        self.model_path = model_path or getattr(settings, 'agent_model_path', None)
        
        if Llama is None:
            logger.warning("llama-cpp-python not installed. Agent will use rule-based fallback.")
            return
        
        # Resolve relative paths to absolute
        if self.model_path and not os.path.isabs(self.model_path):
            # Try relative to current directory first
            if os.path.exists(self.model_path):
                self.model_path = os.path.abspath(self.model_path)
            else:
                # Try relative to config directory
                config_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(config_dir, self.model_path)
                if os.path.exists(resolved_path):
                    self.model_path = os.path.abspath(resolved_path)
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Loading GGUF model from {self.model_path}...")
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,  # Context window
                    n_threads=4,  # CPU threads
                    verbose=False,
                    n_gpu_layers=0  # Set to >0 if GPU available
                )
                logger.info(f"✓ Successfully loaded GGUF model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}", exc_info=True)
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.model = None
        else:
            logger.warning(f"GGUF model not found at {self.model_path}. Using rule-based fallback.")
    
    def decide_retrieval_strategy(
        self,
        query: str,
        projectid: str,
        available_fileids: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Decide whether to use file-specific or global retrieval strategy.
        
        Returns:
            {
                "strategy": "file_specific" | "global",
                "reasoning": "explanation",
                "fileids": [list of fileids if file_specific, None if global]
            }
        """
        if self.model is None:
            return self._rule_based_decision(query, available_fileids)
        
        return self._model_based_decision(query, projectid, available_fileids)
    
    def _model_based_decision(
        self,
        query: str,
        projectid: str,
        available_fileids: Optional[List[str]]
    ) -> Dict[str, any]:
        """Use GGUF model to make intelligent decision."""
        try:
            # Create prompt for the agent
            prompt = self._create_decision_prompt(query, available_fileids)
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent decisions
                stop=["</response>", "\n\n"],
                echo=False
            )
            
            text = response['choices'][0]['text'].strip()
            
            # Parse the response
            return self._parse_model_response(text, available_fileids)
            
        except Exception as e:
            logger.error(f"Error in model-based decision: {e}")
            return self._rule_based_decision(query, available_fileids)
    
    def _create_decision_prompt(
        self,
        query: str,
        available_fileids: Optional[List[str]]
    ) -> str:
        """Create prompt for the agent to make decision."""
        files_info = ""
        if available_fileids:
            files_info = f"\nAvailable files: {', '.join(available_fileids)}"
        
        prompt = f"""You are a retrieval routing agent. Analyze the user query and decide the best retrieval strategy.

Query: "{query}"
{files_info}

Decision criteria:
- Use "file_specific" if the query asks about:
  * Specific files or documents
  * Comparisons between files
  * Information from particular sources
  * "each file", "per file", "for each file"
  
- Use "global" if the query asks about:
  * General information across all files
  * Aggregated or combined information
  * Overall patterns or trends
  * Questions that don't specify a file

Respond in JSON format:
{{
    "strategy": "file_specific" or "global",
    "reasoning": "brief explanation",
    "fileids": ["file1", "file2"] or null
}}

Response:"""
        return prompt
    
    def _parse_model_response(
        self,
        text: str,
        available_fileids: Optional[List[str]]
    ) -> Dict[str, any]:
        """Parse model response into structured decision."""
        try:
            # Try to extract JSON from response
            text = text.strip()
            if "{" in text and "}" in text:
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                json_str = text[json_start:json_end]
                decision = json.loads(json_str)
                
                strategy = decision.get("strategy", "global")
                if strategy not in ["file_specific", "global"]:
                    strategy = "global"
                
                return {
                    "strategy": strategy,
                    "reasoning": decision.get("reasoning", "Model decision"),
                    "fileids": decision.get("fileids") if strategy == "file_specific" else None
                }
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
        
        # Fallback to rule-based
        return self._rule_based_decision(text, available_fileids)
    
    def _rule_based_decision(
        self,
        query: str,
        available_fileids: Optional[List[str]]
    ) -> Dict[str, any]:
        """
        Rule-based fallback decision making.
        Uses keyword matching and heuristics.
        """
        query_lower = query.lower()
        
        # Keywords that suggest file-specific retrieval
        file_specific_keywords = [
            "each file", "per file", "for each file", "every file",
            "compare", "comparison", "difference between",
            "specific file", "particular file", "which file",
            "file by file", "separate files", "individual files"
        ]
        
        # Keywords that suggest global retrieval
        global_keywords = [
            "all files", "across all", "overall", "combined",
            "aggregate", "total", "summarize all", "general"
        ]
        
        # Check for file-specific indicators
        file_specific_score = sum(1 for keyword in file_specific_keywords if keyword in query_lower)
        global_score = sum(1 for keyword in global_keywords if keyword in query_lower)
        
        # Check if query mentions specific files
        mentions_files = False
        if available_fileids:
            for fileid in available_fileids:
                if fileid.lower() in query_lower:
                    mentions_files = True
                    break
        
        # Decision logic
        if file_specific_score > global_score or mentions_files:
            strategy = "file_specific"
            fileids = available_fileids if available_fileids else None
            reasoning = "Query suggests file-specific information or comparisons"
        else:
            strategy = "global"
            fileids = None
            reasoning = "Query suggests general information across all files"
        
        return {
            "strategy": strategy,
            "reasoning": reasoning,
            "fileids": fileids
        }


# Global agent instance
_agent_instance: Optional[RetrievalAgent] = None


def get_agent() -> RetrievalAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RetrievalAgent()
    return _agent_instance


def initialize_agent(model_path: Optional[str] = None) -> RetrievalAgent:
    """Initialize the agent with a specific model path."""
    global _agent_instance
    _agent_instance = RetrievalAgent(model_path)
    return _agent_instance

