"""
IBM watsonx.ai LLM client via langchain-ibm.
Provides a single factory function used by all agents.
"""
from functools import lru_cache
from langchain_ibm import WatsonxLLM
from config.settings import get_settings


def _build_llm(model_id: str, max_new_tokens: int = 1024, temperature: float = 0.1) -> WatsonxLLM:
    settings = get_settings()
    return WatsonxLLM(
        model_id=model_id,
        url=settings.ibm_watsonx_url,
        apikey=settings.ibm_watsonx_api_key,
        project_id=settings.ibm_watsonx_project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 10,
            "temperature": temperature,
            "repetition_penalty": 1.1,
            "stop_sequences": ["<|endoftext|>"],
        },
    )


@lru_cache(maxsize=1)
def get_primary_llm() -> WatsonxLLM:
    """Granite 13B — used for generation, summarization, analysis."""
    settings = get_settings()
    return _build_llm(settings.watsonx_model_id, max_new_tokens=1500)


@lru_cache(maxsize=1)
def get_router_llm() -> WatsonxLLM:
    """Granite 3-8B Instruct — used for fast classification/routing."""
    settings = get_settings()
    return _build_llm(settings.watsonx_router_model_id, max_new_tokens=256, temperature=0.0)
