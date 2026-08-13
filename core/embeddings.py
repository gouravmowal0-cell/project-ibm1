"""
Embedding model setup.
Uses IBM watsonx.ai granite-embedding if configured, otherwise falls back to
a local sentence-transformers model so the system works offline too.
"""
from functools import lru_cache
from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Returns IBM watsonx.ai embeddings when credentials are present,
    otherwise falls back to a local HuggingFace model.
    IBM credentials are required on Render — the local fallback downloads
    ~90 MB and should only be used in local dev.
    """
    from config.settings import get_settings
    settings = get_settings()

    if settings.ibm_watsonx_api_key:
        try:
            from langchain_ibm import WatsonxEmbeddings
            return WatsonxEmbeddings(
                model_id="ibm/granite-embedding-278m-multilingual",
                url=settings.ibm_watsonx_url,
                apikey=settings.ibm_watsonx_api_key,
                project_id=settings.ibm_watsonx_project_id,
            )
        except Exception:
            pass  # Fall through to local model

    # Local fallback — only used when IBM credentials are absent
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
