from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import requests
import os
import json
import uuid
import base64
import mimetypes
from typing import Optional, List, Dict, Any

APP_TITLE = "Devset IA 👾"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_DEFAULT = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
Você é a Devset Growth Strategist 👾 especialista em Marketing Digital.

Regras:
- Seja profissional e estruturado.
- Use títulos e listas.
- Parágrafos curtos.
""".strip()

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FILES: Dict[str, Dict[str, Any]] = {}

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

# =============================
# MODELO INPUT
# =============================

class ChatInput(BaseModel):
    message: str
    model: Optional[str] = None

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
# STREAM GROQ
# =============================

def stream_groq(messages: List[Dict[str, str]], model: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    with requests.post(GROQ_URL, headers=headers, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield sse_data(delta)
                    except:
                        continue

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

    chosen_model = payload.model or MODEL_DEFAULT

    return StreamingResponse(
        stream_groq(messages, chosen_model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )