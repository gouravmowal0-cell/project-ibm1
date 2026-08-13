"""
Centralized configuration using pydantic-settings.
Reads values from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # IBM watsonx.ai
    ibm_watsonx_api_key: str = ""
    ibm_watsonx_project_id: str = ""
    ibm_watsonx_url: str = "https://us-south.ml.cloud.ibm.com"

    # Primary model — Meta Llama 3.3 70B (best available on Lite plan)
    watsonx_model_id: str = "meta-llama/llama-3-3-70b-instruct"
    # Lighter model for classification / routing tasks
    watsonx_router_model_id: str = "ibm/granite-3-1-8b-base"

    # IBM Watson Discovery (optional)
    watson_discovery_api_key: str = ""
    watson_discovery_url: str = ""
    watson_discovery_env_id: str = ""
    watson_discovery_collection_id: str = ""

    # Vector store (ChromaDB)
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "legal_corpus"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True

    # Legal corpus data path
    legal_corpus_dir: str = "./data/legal_corpus"

    # RAG retrieval parameters
    retrieval_top_k: int = 6
    chunk_size: int = 800
    chunk_overlap: int = 120


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
