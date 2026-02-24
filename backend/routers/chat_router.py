from typing import AsyncGenerator

import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatPayload(BaseModel):
    message: str
    user_id: str | int | None = None
    chat_id: str | int | None = None
    smart_memory: bool | None = True
    history: list | None = None
    file_ids: list | None = None
    model: str | None = None
    persona: str | None = None


def _sse_data(text: str) -> bytes:
    lines = (text or "").splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines) + "\n"
    return payload.encode("utf-8")


async def _stream_generator(message: str) -> AsyncGenerator[bytes, None]:
    for chunk in ["Thinking.", "Thinking..", "Thinking...", f"Echo: {message}"]:
        yield _sse_data(chunk)
        await asyncio.sleep(0.05)
    yield _sse_data("[DONE]")


@router.post("/chat_stream")
async def chat_stream(payload: ChatPayload):
    gen = _stream_generator(payload.message)
    return StreamingResponse(gen, media_type="text/event-stream")
