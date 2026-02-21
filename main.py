from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel
import requests
import os
import json
import uuid
import mimetypes
from typing import Optional, List, Dict, Any

APP_TITLE = "Devset IA 👾"

# =============================
# GROQ
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_DEFAULT = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
Você é a Devset Growth Strategist 👾 especialista em Marketing Digital.

Regras:
- Seja profissional e estruturado.
- Use títulos e listas.
- Parágrafos curtos.
""".strip()

# =============================
# APP
# =============================
app = FastAPI(title=APP_TITLE)

# ✅ CORS: libera apenas seu Cloudflare Workers (mais seguro e resolve o erro)
ALLOWED_ORIGINS = [
    "https://steep-disk-3924.guilhermexp0708.workers.dev",
    # Se você abrir o HTML localmente (file://), o browser pode mandar origin "null"
    # Se precisar testar local, descomenta:
    # "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ✅ Preflight OPTIONS (resolve "Response to preflight request doesn't pass...")
@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    return Response(status_code=204)

# =============================
# UPLOADS
# =============================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
FILES: Dict[str, Dict[str, Any]] = {}

def guess_mime(filename: str, content_type: Optional[str]) -> str:
    if content_type:
        return content_type
    mt = mimetypes.guess_type(filename)[0]
    return mt or "application/octet-stream"

# =============================
# SSE helpers
# =============================
def sse_data(text: str) -> str:
    """
    SSE exige 'data:' em cada linha e uma linha em branco ao final do evento.
    """
    lines = (text or "").splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"

def sse_done() -> str:
    return "event: done\ndata: [DONE]\n\n"

# =============================
# SCHEMAS
# =============================
class ChatInput(BaseModel):
    message: str
    model: Optional[str] = None

    # ✅ compat com seu frontend atual
    history: Optional[List[Dict[str, Any]]] = None
    file_ids: Optional[List[str]] = None

    # opcional (se o front mandar)
    user_id: Optional[str] = None
    chat_id: Optional[str] = None

# =============================
# ROUTES
# =============================
@app.get("/")
def home():
    return {
        "status": "Devset API online 🚀",
        "stream": "/chat_stream",
        "upload": "/upload",
        "default_model": MODEL_DEFAULT,
        "groq_key_present": bool(GROQ_API_KEY),
        "allowed_origins": ALLOWED_ORIGINS,
    }

@app.get("/health")
def health():
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "GROQ_API_KEY não configurada nas Variables do Railway."},
        )
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    raw = await file.read()

    path = os.path.join(UPLOAD_DIR, file_id)
    with open(path, "wb") as f:
        f.write(raw)

    mime = guess_mime(file.filename or "file", file.content_type)
    FILES[file_id] = {
        "path": path,
        "mime": mime,
        "name": file.filename or "file",
        "size": len(raw),
    }

    return {"file_id": file_id}

# =============================
# GROQ STREAM
# =============================
def build_messages(user_text: str, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ✅ aceita histórico do front (últimas mensagens)
    if history and isinstance(history, list):
        for h in history[-12:]:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content.strip()})

    msgs.append({"role": "user", "content": user_text})
    return msgs

def stream_groq(messages: List[Dict[str, str]], model: str):
    if not GROQ_API_KEY:
        yield sse_data("❌ ERRO: GROQ_API_KEY não configurada no Railway (Variables).")
        yield sse_done()
        return

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        # dá pra setar temperatura se quiser:
        # "temperature": 0.7,
    }

    try:
        with requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=120) as r:
            # se a Groq rejeitar (401/429/etc)
            if r.status_code >= 400:
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text[:500]}
                yield sse_data(f"❌ ERRO Groq HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)}")
                yield sse_done()
                return

            for line in r.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8", errors="ignore").strip()

                # a API stream manda linhas como: "data: {...}"
                if not decoded.startswith("data:"):
                    continue

                data_str = decoded[5:].strip()  # remove "data:"
                if data_str == "[DONE]":
                    break

                try:
                    obj = json.loads(data_str)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield sse_data(delta)
                except Exception:
                    # ignora pedaços inválidos
                    continue

    except requests.exceptions.RequestException as e:
        yield sse_data(f"❌ ERRO de rede ao chamar Groq: {str(e)}")

    yield sse_done()

@app.post("/chat_stream")
def chat_stream(payload: ChatInput):
    user_text = (payload.message or "").strip()
    if not user_text:
        def gen_empty():
            yield sse_data("Mensagem vazia.")
            yield sse_done()
        return StreamingResponse(gen_empty(), media_type="text/event-stream")

    chosen_model = (payload.model or MODEL_DEFAULT).strip() or MODEL_DEFAULT
    messages = build_messages(user_text, payload.history)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        stream_groq(messages, chosen_model),
        media_type="text/event-stream",
        headers=headers,
    )