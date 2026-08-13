"""
Legal corpus ingestion pipeline.

Usage (CLI):
    python -m rag.ingestion --source ./data/legal_corpus/

Supports: PDF, DOCX, TXT files.
Chunks documents and upserts them into ChromaDB.
"""
import argparse
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from core.vector_store import get_vector_store
from parsers.document_parser import parse_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _build_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )


def ingest_file(file_path: Path) -> int:
    """Parse a single file and upsert its chunks into the vector store.

    Returns the number of chunks added.
    """
    logger.info("Ingesting: %s", file_path)
    raw_text = parse_file(file_path)
    if not raw_text.strip():
        logger.warning("Empty file, skipping: %s", file_path)
        return 0

    splitter = _build_splitter()
    chunks: List[Document] = splitter.create_documents(
        texts=[raw_text],
        metadatas=[{
            "source": str(file_path),
            "filename": file_path.name,
            "doc_type": file_path.suffix.lstrip(".").upper(),
        }],
    )

    store = get_vector_store()
    store.add_documents(chunks)
    logger.info("  Added %d chunks from %s", len(chunks), file_path.name)
    return len(chunks)


def ingest_directory(directory: Path) -> int:
    """Recursively ingest all supported files in a directory."""
    total = 0
    files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning("No supported files found in %s", directory)
        return 0

    logger.info("Found %d files to ingest.", len(files))
    for f in files:
        try:
            total += ingest_file(f)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", f, exc)

    logger.info("Ingestion complete. Total chunks: %d", total)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal documents into ChromaDB")
    parser.add_argument("--source", required=True, help="Path to directory or single file")
    args = parser.parse_args()

    source = Path(args.source)
    if source.is_dir():
        ingest_directory(source)
    elif source.is_file():
        ingest_file(source)
    else:
        raise FileNotFoundError(f"Source not found: {source}")
