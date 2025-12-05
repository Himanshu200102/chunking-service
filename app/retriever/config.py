"""Configuration settings for the retriever service."""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenSearch configuration
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_username: Optional[str] = None
    opensearch_password: Optional[str] = None
    opensearch_index_name: str = "chunks_index"
    
    # LanceDB configuration
    lancedb_path: str = os.getenv("LANCEDB_PATH", "./lancedb_data")
    lancedb_table_name: str = os.getenv("LANCEDB_TABLE_NAME", "chunks")
    # Use same embedding model as rest of system (BAAI/bge-small-en-v1.5)
    lancedb_embedding_model: str = os.getenv("LANCEDB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Application configuration
    log_level: str = "INFO"
    max_chunks_per_query: int = 50
    
    # Agent configuration (decision-making)
    agent_model_path: Optional[str] = os.getenv("AGENT_MODEL_PATH", None)
    agent_enabled: bool = os.getenv("AGENT_ENABLED", "true").lower() == "true"
    
    # Summarizer configuration (LLM used for chunk summarization)
    summarizer_model_path: Optional[str] = os.getenv("SUMMARIZER_MODEL_PATH", None)
    
    def __init__(self, **kwargs):
        """Initialize settings and resolve relative paths."""
        super().__init__(**kwargs)
        # Resolve relative model paths to absolute if they exist
        if self.agent_model_path and not os.path.isabs(self.agent_model_path):
            # Get the directory where config.py is located
            config_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.join(config_dir, self.agent_model_path)
            if os.path.exists(resolved_path):
                self.agent_model_path = os.path.abspath(resolved_path)
        
        if self.summarizer_model_path and not os.path.isabs(self.summarizer_model_path):
            config_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.join(config_dir, self.summarizer_model_path)
            if os.path.exists(resolved_path):
                self.summarizer_model_path = os.path.abspath(resolved_path)
    
    # Chunking service integration
    chunking_service_url: str = os.getenv("CHUNKING_SERVICE_URL", "http://localhost:8002")
    port: int = int(os.getenv("PORT", "8001"))  # Default 8001 for retriever service
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
