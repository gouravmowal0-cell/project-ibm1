"""
Chat route — multi-turn conversational legal assistant.
"""
from fastapi import APIRouter
from api.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the conversational legal assistant.
    Maintains session history across turns using `session_id`.
    """
    from agents.conversational_assistant import chat
    result = chat(session_id=request.session_id, user_message=request.message)
    return ChatResponse(**result)


@router.delete("/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear the conversation history for a given session."""
    from agents.conversational_assistant import clear_session
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}
