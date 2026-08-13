"""
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.routes import chat, document, agents
from api.models import HealthResponse
from config.settings import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up heavy resources on startup."""
    logger.info("Starting LexAI Legal Aid System...")
    try:
        # Pre-initialize the LLM and vector store connections
        from core.llm import get_primary_llm, get_router_llm
        from core.vector_store import get_vector_store
        get_primary_llm()
        get_router_llm()
        get_vector_store()
        logger.info("Core services initialized.")
    except Exception as exc:
        logger.warning("Could not pre-initialize services: %s", exc)
    yield
    logger.info("Shutting down LexAI Legal Aid System.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="LexAI — AI Legal Aid System",
        description=(
            "IBM watsonx.ai-powered multi-agent system for legal information retrieval, "
            "document analysis, compliance checking, and conversational legal assistance."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router)
    app.include_router(document.router)
    app.include_router(agents.router)

    # Serve dashboard frontend
    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
    if os.path.isdir(dashboard_dir):
        app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(os.path.join(dashboard_dir, "index.html"))

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        ibm_configured = bool(settings.ibm_watsonx_api_key and settings.ibm_watsonx_project_id)
        return HealthResponse(
            status="ok" if ibm_configured else "degraded",
            models={
                "primary": settings.watsonx_model_id,
                "router": settings.watsonx_router_model_id,
                "ibm_credentials": "configured" if ibm_configured else "MISSING — set IBM_WATSONX_API_KEY and IBM_WATSONX_PROJECT_ID in Render environment",
            },
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
