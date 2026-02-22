import os
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .database import engine, Base
from .routers import auth_router, chat_router, upload_router
# ensure models imported for metadata
from . import models  # noqa: F401

logger = logging.getLogger("uvicorn.error")


def parse_origins(env_value: str | None) -> list[str]:
    if not env_value:
        return []
    parts = [s.strip() for s in env_value.split(",") if s.strip()]
    return [p for p in parts if p != "*"]


app = FastAPI(title="Devset Backend")

# configure CORS: read ALLOW_ORIGINS from env (comma separated)
origins = parse_origins(os.getenv("ALLOW_ORIGINS")) or [
    "https://devset-backend1-production-0b6f.up.railway.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root():
    return {"status": "ok"}


def _sse_data(text: str) -> str:
    lines = (text or "").splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"


@app.post("/chat_stream")
async def chat_stream(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    message = (payload.get("message") if isinstance(payload, dict) else None) or ""

    async def generator():
        yield _sse_data("[START]").encode("utf-8")
        await asyncio.sleep(0.05)
        for chunk in ["Thinking.", "Thinking..", "Thinking...", f"Echo: {message}"]:
            yield _sse_data(chunk).encode("utf-8")
            await asyncio.sleep(0.05)
        yield _sse_data("[DONE]").encode("utf-8")

    return StreamingResponse(generator(), media_type="text/event-stream")
