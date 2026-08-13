"""
Conversational Legal Assistant
================================
Maintains multi-turn conversation history and routes each turn
through the appropriate agent while keeping a warm, empathetic tone.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm

SYSTEM_PREAMBLE = """You are LexAI — a friendly, empathetic legal information assistant powered by IBM watsonx.ai.
Your role is to help people understand legal concepts, their rights, and how the law may apply to their situations.

Guidelines:
- Use plain, accessible language. Avoid jargon unless you explain it.
- Be empathetic and non-judgmental — legal situations can be stressful.
- Always include a disclaimer that you are not a substitute for professional legal advice.
- If a question is outside your knowledge, say so honestly.
- Keep responses focused and structured (use bullet points where helpful).
- If the user seems distressed, acknowledge their situation before diving into legal details.
"""

_CHAT_PROMPT = PromptTemplate(
    input_variables=["system", "history", "context", "user_message"],
    template="""{system}

CONVERSATION HISTORY:
{history}

RELEVANT LEGAL CONTEXT (from knowledge base):
{context}

USER: {user_message}

LEXAI:""",
)

DISCLAIMER = (
    "⚠️ I'm LexAI, an AI legal information assistant. My responses are for general information only "
    "and do not constitute legal advice. Laws vary by jurisdiction. Always consult a qualified attorney "
    "for advice specific to your situation."
)


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ConversationSession:
    session_id: str
    history: List[ChatMessage] = field(default_factory=list)
    max_history_turns: int = 10  # keep last N turns to stay within token budget

    def add_user(self, msg: str) -> None:
        self.history.append(ChatMessage(role="user", content=msg))

    def add_assistant(self, msg: str) -> None:
        self.history.append(ChatMessage(role="assistant", content=msg))

    def format_history(self) -> str:
        # Keep only the most recent max_history_turns * 2 messages
        recent = self.history[-(self.max_history_turns * 2):]
        lines = []
        for m in recent:
            prefix = "User" if m.role == "user" else "LexAI"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines) if lines else "No prior conversation."


# In-memory session store (replace with Redis for production)
_sessions: dict[str, ConversationSession] = {}


def get_or_create_session(session_id: str) -> ConversationSession:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSession(session_id=session_id)
    return _sessions[session_id]


def chat(session_id: str, user_message: str) -> dict:
    """
    Process a user message in a session and return the assistant response.
    Automatically retrieves relevant legal context from the RAG pipeline.
    """
    from rag.retriever import retrieve, format_context

    session = get_or_create_session(session_id)

    # Retrieve relevant legal context for this message
    docs = retrieve(user_message, top_k=4)
    context = format_context(docs)

    history_str = session.format_history()
    prompt = _CHAT_PROMPT.format(
        system=SYSTEM_PREAMBLE,
        history=history_str,
        context=context,
        user_message=user_message,
    )

    llm = get_primary_llm()
    response = llm.invoke(prompt).strip()

    # Update history
    session.add_user(user_message)
    session.add_assistant(response)

    sources = list({
        doc.metadata.get("filename", "Unknown")
        for doc in docs
    })

    return {
        "session_id": session_id,
        "response": response,
        "sources": sources,
        "disclaimer": DISCLAIMER,
        "turn": len(session.history) // 2,
    }


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
