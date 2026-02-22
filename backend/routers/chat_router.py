from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import asyncio
from ..auth import get_current_user
from ..models import User

router = APIRouter(tags=["chat"])


class ChatPayload(BaseModel):
    message: str
    chat_id: int | None = None
    smart_memory: bool | None = True
    history: list | None = None
    file_ids: list | None = None
    model: str | None = None
    persona: str | None = None


def _format_sse(event: str, data: str) -> str:
    lines = (data or "").splitlines() or [""]
    out = f"event: {event}\n"
    out += "".join([f"data: {ln}\n" for ln in lines])
    out += "\n"
    return out


async def _stream_generator(message: str) -> AsyncGenerator[bytes, None]:
    yield _format_sse("message", "[START]").encode("utf-8")
    await asyncio.sleep(0.05)
    for chunk in ["Thinking.", "Thinking..", "Thinking..."]:
        yield _format_sse("message", chunk).encode("utf-8")
        await asyncio.sleep(0.05)
    yield _format_sse("message", f"Echo: {message}").encode("utf-8")
    await asyncio.sleep(0.01)
    yield _format_sse("message", "[DONE]").encode("utf-8")


@router.post("/chat_stream")
async def chat_stream(request: Request, user: User = Depends(get_current_user)):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    message = (payload.get("message") if isinstance(payload, dict) else None) or ""
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    gen = _stream_generator(message)
    return StreamingResponse(gen, media_type="text/event-stream")
