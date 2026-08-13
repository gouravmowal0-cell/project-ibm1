# AI Legal Aid Multi-Agent System

An IBM-powered Multi-Agent system for democratizing access to legal information.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     User Dashboard (Frontend)                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ REST / WebSocket
┌────────────────────────────▼─────────────────────────────────────┐
│              Orchestrator Agent (IBM Langflow / Orchestrate)      │
│         Routes queries to specialized sub-agents                  │
└──┬──────────┬──────────────┬──────────────┬───────────────────────┘
   │          │              │              │
   ▼          ▼              ▼              ▼
Contract   Compliance     Q&A RAG       Case Research
Reviewer   Checker        Agent         Agent
Agent      Agent
   │          │              │              │
   └──────────┴──────────────┴──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    IBM watsonx.ai (LLM)      │
              │  + Vector Store (ChromaDB)   │
              └─────────────────────────────┘
```

## Tech Stack

- **IBM watsonx.ai** — LLM backbone (`ibm/granite-13b-chat-v2`, `ibm/granite-3-8b-instruct`)
- **IBM Langflow** — Visual multi-agent orchestration pipeline
- **IBM Watson Discovery** — Legal corpus ingestion & semantic search
- **ChromaDB** — Local vector store for embeddings
- **LangChain** — Agent framework & tool calling
- **FastAPI** — REST API backend
- **Python 3.11+**

## Project Structure

```
legal-aid-system/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py            # Centralized config & IBM credentials
├── core/
│   ├── llm.py                 # IBM watsonx.ai LLM client
│   ├── embeddings.py          # IBM/HuggingFace embeddings
│   └── vector_store.py        # ChromaDB vector store setup
├── agents/
│   ├── orchestrator.py        # Master router agent
│   ├── contract_reviewer.py   # Contract risk analysis agent
│   ├── compliance_checker.py  # Regulatory compliance agent
│   ├── qa_rag_agent.py        # Q&A with RAG agent
│   └── case_research_agent.py # Case law research agent
├── rag/
│   ├── ingestion.py           # Document ingestion pipeline
│   ├── retriever.py           # Hybrid retrieval (semantic + keyword)
│   └── summarizer.py          # Legal text summarization
├── parsers/
│   └── document_parser.py     # PDF/DOCX/TXT contract parser
├── api/
│   ├── main.py                # FastAPI application entry point
│   ├── routes/
│   │   ├── chat.py            # Conversational assistant endpoint
│   │   ├── document.py        # Document upload & analysis endpoint
│   │   └── agents.py          # Direct agent invocation endpoints
│   └── models.py              # Pydantic request/response models
├── langflow/
│   └── legal_aid_flow.json    # Importable Langflow pipeline definition
└── dashboard/
    └── index.html             # Standalone user dashboard
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your IBM watsonx.ai API key and project ID

# 3. Ingest legal documents
python -m rag.ingestion --source ./data/legal_corpus/

# 4. Start the API server
uvicorn api.main:app --reload --port 8000

# 5. Open the dashboard
open dashboard/index.html
```

## Agent Descriptions

| Agent | Responsibility |
|---|---|
| **Orchestrator** | Classifies user intent and routes to the right agent |
| **Contract Reviewer** | Flags risky clauses, obligations, and inconsistencies |
| **Compliance Checker** | Checks documents against regulatory frameworks (GDPR, HIPAA, etc.) |
| **Q&A RAG** | Answers legal questions using retrieved corpus chunks |
| **Case Research** | Fetches relevant case summaries and legal precedents |

## Disclaimer

> This system provides legal **information**, not legal **advice**. Always consult a qualified attorney for specific legal matters.
