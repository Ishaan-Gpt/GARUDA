"""
GARUDA API — Gemma AI Agent Router
====================================
Exposes endpoints to interact with local gemma3:1b model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import httpx

from ..core.config import get_settings
from ..core.database import UserModel
from ..core.agent_executor import execute_agent_loop
from .auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent")
settings = get_settings()


class ChatMessageInput(BaseModel):
    role: str # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessageInput] = []


class ChatResponse(BaseModel):
    text: str
    ui_action: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    body: ChatRequest,
    current_user: UserModel = Depends(get_current_user)
):
    """Process prompt with local gemma3:1b agent executor."""
    # Convert Pydantic schemas to standard dictionaries
    history_list = [{"role": msg.role, "content": msg.content} for msg in body.history]
    
    result = await execute_agent_loop(body.message, history_list)
    return ChatResponse(
        text=result["text"],
        ui_action=result["ui_action"]
    )


@router.get("/status")
async def check_agent_status():
    """Verify Ollama service and gemma3:1b availability."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{settings.OLLAMA_URL.rstrip('/')}/api/tags", timeout=3.0)
            if res.status_code == 200:
                models = res.json().get("models", [])
                gemma_available = any("gemma3:1b" in m.get("name", "") for m in models)
                return {
                    "ollama_running": True,
                    "gemma_loaded": gemma_available,
                    "models_registered": [m.get("name") for m in models]
                }
    except Exception as e:
        logger.warning("Ollama check failed: %s", e)
        
    return {
        "ollama_running": False,
        "gemma_loaded": False,
        "models_registered": []
    }
