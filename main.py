from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    model: str | None = None


app = FastAPI(title="Devset IA API")

frontend_origin = "https://devset-backend1-production.up.railway.app"
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_origins=[frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "ok": True,
        "service": "Devset IA",
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/chat")
async def chat(payload: ChatRequest) -> dict[str, str]:
    chosen_model = payload.model or "devset-sim"
    return {
        "reply": f"Você disse: {payload.message}",
        "model_used": chosen_model,
    }


async def sse_tokens(message: str) -> AsyncGenerator[str, None]:
    content = f"Você disse: {message}"
    for token in content.split(" "):
        yield f"event: token\ndata: {token} \n\n"
    yield "event: done\ndata: [DONE]\n\n"


@app.post("/chat_stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_tokens(payload.message),
        media_type="text/event-stream",
        headers=headers,
    )
