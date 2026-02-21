from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import requests
import json
import os
import uuid
import base64
import mimetypes
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

APP_TITLE = "Devset IA 👾"

# 🔥 IMPORTANTE: dentro do Railway container é 127.0.0.1
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

MODEL_DEFAULT_TEXT = "llama3.2:3b"
MODEL_DEFAULT_VISION = MODEL_DEFAULT_TEXT  # 🔥 desativa llava por segurança

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SYSTEM_PROMPT = """
Você é a Devset Growth Strategist 👾, especialista em Marketing Digital.

Regras:
- Seja profissional, estruturado e direto.
- Sempre use títulos, subtítulos e listas.
- Use parágrafos curtos.
- Nunca responda em bloco único.
""".strip()

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# UTIL
# =============================

def sse_data(text: str) -> str:
    lines = (text or "").splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"

def guess_mime(filename: str, content_type: Optional[str]) -> str:
    if content_type:
        return content_type
    mt = mimetypes.guess_type(filename)[0]
    return mt or "application/octet-stream"

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =============================
# MODELO REQUEST
# =============================

class ChatInput(BaseModel):
    message: str
    file_ids: Optional[List[str]] = None
    model: Optional[str] = None

FILES: Dict[str, Dict[str, Any]] = {}

# =============================
# ROTAS
# =============================

@app.get("/")
def home():
    return {
        "status": "Devset API online 🚀",
        "stream": "/chat_stream",
        "upload": "/upload"
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    raw = await file.read()

    path = os.path.join(UPLOAD_DIR, file_id)
    with open(path, "wb") as f:
        f.write(raw)

    mime = guess_mime(file.filename, file.content_type)
    FILES[file_id] = {"path": path, "mime": mime}

    return {"file_id": file_id}

# =============================
# STREAM OLLAMA
# =============================

def stream_ollama(messages: List[Dict[str, Any]], model_name: str):
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True
    }

    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            stream=True,
            timeout=240
        ) as r:

            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue

                if obj.get("done"):
                    break

                chunk = (obj.get("message") or {}).get("content", "")
                if chunk:
                    yield sse_data(chunk)

    except Exception as e:
        yield sse_data(f"Erro 👾: {str(e)}")

    yield "event: done\ndata: [DONE]\n\n"

@app.post("/chat_stream")
def chat_stream(payload: ChatInput):

    user_text = payload.message.strip()
    if not user_text:
        return StreamingResponse(
            iter([sse_data("Mensagem vazia."), "event: done\ndata: [DONE]\n\n"]),
            media_type="text/event-stream"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]

    chosen_model = payload.model or MODEL_DEFAULT_TEXT

    return StreamingResponse(
        stream_ollama(messages, chosen_model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )