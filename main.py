from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from groq import Groq
import json
import os
import uuid
import base64
import mimetypes
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

APP_TITLE = "Devset IA 👾"

# =============================
# GROQ CONFIG
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Modelos Groq (você pode trocar via env ou pela UI)
MODEL_DEFAULT_TEXT = os.getenv("MODEL_DEFAULT_TEXT", "llama-3.3-70b-versatile").strip()

# Groq (chat) não usa "images" do jeito que o Ollama usa.
# Se quiser visão, é outro fluxo/modelo/endpoint. Por agora, a gente bloqueia.
MODEL_DEFAULT_VISION = os.getenv("MODEL_DEFAULT_VISION", "").strip()

# Mapa opcional pra “alias” da sua UI
MODEL_MAP: Dict[str, str] = {
    "groq:llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "groq:llama-3.1-70b": "llama-3.1-70b-versatile",
    "groq:llama-3.1-8b": "llama-3.1-8b-instant",
    "groq:mixtral-8x7b": "mixtral-8x7b-32768",
    "groq:gemma2-9b": "gemma2-9b-it",
}

# Lista “base” (pra UI) caso você não queira bater em endpoint nenhum
GROQ_MODELS_FALLBACK = list(dict.fromkeys([MODEL_DEFAULT_TEXT] + list(MODEL_MAP.values())))

# Client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory_v2.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_HISTORY_MESSAGES = 12
FILES: Dict[str, Dict[str, Any]] = {}

# =============================
# SYSTEM PROMPT
# =============================
SYSTEM_PROMPT = """
Você é a Devset Growth Strategist 👾, especialista em Marketing Digital.

Regras:
- Seja profissional, estruturado e direto.
- Sempre use títulos, subtítulos e listas.
- Use parágrafos curtos.
- Nunca responda em bloco único.
""".strip()

# =============================
# CORS
# =============================
WORKER_ORIGIN = "https://steep-disk-3924.guilhermexp0708.workers.dev"
ALLOW_CREDENTIALS = True

ALLOWED_ORIGINS = [
    WORKER_ORIGIN,
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# =============================
# APP
# =============================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOW_CREDENTIALS else ["*"],
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# FAVICON
# =============================
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# =============================
# HELPERS
# =============================
def resolve_text_model(model_from_ui: Optional[str]) -> str:
    if not model_from_ui:
        return MODEL_DEFAULT_TEXT
    m = model_from_ui.strip()
    if m in MODEL_MAP:
        return MODEL_MAP[m]
    return m

def normalize_text(s: str) -> str:
    return (s or "").replace("\r", "").strip()

def guess_mime(filename: str, content_type: Optional[str]) -> str:
    if content_type:
        return content_type
    mt = mimetypes.guess_type(filename)[0]
    return mt or "application/octet-stream"

def meta_is_image(meta: Dict[str, Any]) -> bool:
    mime = (meta.get("mime") or "").lower()
    return mime.startswith("image/")

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def sse_data(text: str) -> str:
    """
    SSE exige que cada linha comece com 'data:'.
    Se o chunk tiver '\n', emitir múltiplas linhas 'data:'.
    """
    lines = (text or "").splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"

def get_groq_models_ui() -> List[str]:
    # Sem “listar modelos” via API (não é necessário e pode mudar).
    # Mantém uma lista previsível pra sua UI.
    out = GROQ_MODELS_FALLBACK[:]
    # também expõe aliases, se quiser:
    out_alias = list(MODEL_MAP.keys())
    return out + out_alias

# =============================
# MEMORY v2 (perfil + por chat)
# =============================
def memory_default() -> Dict[str, Any]:
    return {
        "version": 2,
        "profiles": {},  # user_id -> {name, goals, niche, prefs...}
        "chats": {},     # chat_id -> {summary, facts, updated_at, turns}
    }

def load_memory_v2() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return memory_default()
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return memory_default()
            if "profiles" not in data or not isinstance(data["profiles"], dict):
                data["profiles"] = {}
            if "chats" not in data or not isinstance(data["chats"], dict):
                data["chats"] = {}
            if "version" not in data:
                data["version"] = 2
            return data
    except:
        return memory_default()

def save_memory_v2(mem: Dict[str, Any]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def ensure_profile(mem: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    prof = mem["profiles"].get(user_id)
    if not isinstance(prof, dict):
        prof = {
            "name": user_id,
            "goals": [],
            "niche": "",
            "tone": "profissional e direto",
            "preferences": [],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        mem["profiles"][user_id] = prof
    return prof

def ensure_chat(mem: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
    ch = mem["chats"].get(chat_id)
    if not isinstance(ch, dict):
        ch = {
            "summary": "",
            "facts": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "turns": []
        }
        mem["chats"][chat_id] = ch
    return ch

def compact_list(items: List[str], max_items: int = 12, max_len_each: int = 140) -> List[str]:
    out = []
    for it in items or []:
        t = normalize_text(it)
        if not t:
            continue
        if len(t) > max_len_each:
            t = t[:max_len_each].rstrip() + "…"
        out.append(t)
    seen = set()
    uniq = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq[-max_items:]

# =============================
# “MEMÓRIA INTELIGENTE”
# =============================
MEMORY_EXTRACTOR_SYSTEM = """
Você é um extrator de memória para um chat.

Tarefa:
1) Atualize o PERFIL do usuário (user_profile) com informações persistentes (nome, objetivo, nicho, preferências de estilo).
2) Atualize a MEMÓRIA DA CONVERSA (chat_memory) com:
   - summary: resumo curto (máx 800 caracteres)
   - facts: lista de fatos úteis (máx 12 itens, cada item máx 140 caracteres)

Regras:
- Retorne APENAS JSON válido.
- Não inclua texto fora do JSON.
- Se não houver nada novo, mantenha os campos atuais.
""".strip()

def groq_json_call(model: str, messages: List[Dict[str, Any]], timeout: int = 120) -> Dict[str, Any]:
    if not groq_client:
        return {}
    # Groq SDK não expõe timeout “per request” do mesmo jeito do requests;
    # em caso de rede ruim, isso pode demorar. Em Railway normalmente ok.
    completion = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=False,
    )
    content = (completion.choices[0].message.content or "").strip()

    # Tenta parsear JSON direto; se vier lixo, tenta extrair {...}
    try:
        return json.loads(content)
    except:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except:
                return {}
        return {}

def update_intelligent_memory(
    user_id: str,
    chat_id: str,
    user_text: str,
    assistant_text: str,
    model_for_extractor: str,
) -> None:
    mem = load_memory_v2()
    prof = ensure_profile(mem, user_id)
    chat = ensure_chat(mem, chat_id)

    chat["turns"] = (chat.get("turns") or [])[-10:]
    chat["turns"].append({"user": user_text, "assistant": assistant_text})

    extractor_messages = [
        {"role": "system", "content": MEMORY_EXTRACTOR_SYSTEM},
        {"role": "user", "content": json.dumps({
            "current_user_profile": prof,
            "current_chat_memory": {
                "summary": chat.get("summary", ""),
                "facts": chat.get("facts", []),
            },
            "new_turn": {
                "user": user_text,
                "assistant": assistant_text
            }
        }, ensure_ascii=False)}
    ]

    data = {}
    try:
        data = groq_json_call(model_for_extractor, extractor_messages, timeout=120)
    except:
        data = {}

    new_prof = data.get("user_profile") if isinstance(data, dict) else None
    new_chat = data.get("chat_memory") if isinstance(data, dict) else None

    if isinstance(new_prof, dict):
        prof["name"] = normalize_text(new_prof.get("name", prof.get("name", user_id))) or prof.get("name", user_id)
        prof["goals"] = compact_list((new_prof.get("goals") or prof.get("goals") or []), max_items=10)
        prof["preferences"] = compact_list((new_prof.get("preferences") or prof.get("preferences") or []), max_items=12)

        niche = normalize_text(new_prof.get("niche", prof.get("niche", "")))
        if niche:
            prof["niche"] = niche

        tone = normalize_text(new_prof.get("tone", prof.get("tone", "")))
        if tone:
            prof["tone"] = tone

        prof["updated_at"] = datetime.now(timezone.utc).isoformat()

    if isinstance(new_chat, dict):
        summary = normalize_text(new_chat.get("summary", chat.get("summary", "")))
        if summary:
            if len(summary) > 800:
                summary = summary[:800].rstrip() + "…"
            chat["summary"] = summary

        facts = new_chat.get("facts", chat.get("facts", []))
        if isinstance(facts, list):
            chat["facts"] = compact_list([str(x) for x in facts], max_items=12, max_len_each=140)

        chat["updated_at"] = datetime.now(timezone.utc).isoformat()

    mem["profiles"][user_id] = prof
    mem["chats"][chat_id] = chat
    save_memory_v2(mem)

def build_memory_context(user_id: str, chat_id: str) -> str:
    mem = load_memory_v2()
    prof = ensure_profile(mem, user_id)
    chat = ensure_chat(mem, chat_id)

    parts = []
    parts.append("## Memória (Perfil do usuário)")
    parts.append(f"- Nome: {prof.get('name','')}")
    if prof.get("niche"):
        parts.append(f"- Nicho/Contexto: {prof.get('niche')}")
    if prof.get("tone"):
        parts.append(f"- Tom preferido: {prof.get('tone')}")
    goals = prof.get("goals") or []
    if goals:
        parts.append("- Objetivos:")
        for g in goals[:8]:
            parts.append(f"  - {g}")
    prefs = prof.get("preferences") or []
    if prefs:
        parts.append("- Preferências:")
        for p in prefs[:8]:
            parts.append(f"  - {p}")

    parts.append("\n## Memória (Conversa atual)")
    if chat.get("summary"):
        parts.append(f"- Resumo: {chat.get('summary')}")
    facts = chat.get("facts") or []
    if facts:
        parts.append("- Fatos úteis:")
        for f in facts[:12]:
            parts.append(f"  - {f}")

    return "\n".join(parts).strip()

# =============================
# BUILD MESSAGES
# =============================
def build_messages(
    user_text: str,
    user_id: str,
    chat_id: str,
    use_intelligent_memory: bool,
    history: Optional[List[Dict[str, Any]]],
    file_ids: Optional[List[str]]
) -> Tuple[List[Dict[str, Any]], bool]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if use_intelligent_memory:
        mem_context = build_memory_context(user_id, chat_id)
        if mem_context:
            messages.append({"role": "system", "content": mem_context})

    if history:
        for h in history[-MAX_HISTORY_MESSAGES:]:
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"]})

    # Se vier imagem anexada, a gente marca use_vision=True e trata na rota (bloqueia por enquanto)
    file_ids = file_ids or []
    images_b64: List[str] = []
    for fid in file_ids:
        meta = FILES.get(fid)
        if not meta:
            continue
        if meta_is_image(meta):
            images_b64.append(image_to_b64(meta["path"]))

    if images_b64:
        # Não adiciona "images" no payload do Groq, porque isso quebra.
        # Só sinaliza que tem imagem.
        messages.append({"role": "user", "content": user_text})
        return messages, True

    messages.append({"role": "user", "content": user_text})
    return messages, False

# =============================
# GROQ STREAM (SSE)
# =============================
def stream_groq(messages: List[Dict[str, Any]], model_name: str):
    if not groq_client:
        yield sse_data("Erro 👾: GROQ_API_KEY não configurada no Railway.")
        yield "event: done\ndata: [DONE]\n\n"
        return

    try:
        stream = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            stream=True,
        )

        for chunk in stream:
            # chunk.choices[0].delta.content em streaming
            delta = chunk.choices[0].delta
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield sse_data(content)

    except Exception as e:
        yield sse_data(f"Erro 👾: {str(e)}")

    yield "event: done\ndata: [DONE]\n\n"

# =============================
# SCHEMAS
# =============================
class ChatInput(BaseModel):
    message: str
    user_id: Optional[str] = "guilherme"
    chat_id: Optional[str] = None
    smart_memory: Optional[bool] = True

    file_ids: Optional[List[str]] = None
    history: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None

# =============================
# ROUTES
# =============================
@app.get("/")
def home():
    models_ui = get_groq_models_ui()
    return {
        "status": "Devset API online 🚀",
        "engine": "groq",
        "model_default_text": MODEL_DEFAULT_TEXT,
        "models_available_ui": models_ui,
        "stream": "/chat_stream",
        "upload": "/upload",
        "memory": "/memory",
        "memory_chat": "/memory/chat/{chat_id}",
        "cors": {
            "allow_credentials": ALLOW_CREDENTIALS,
            "allowed_origins": ALLOWED_ORIGINS if ALLOW_CREDENTIALS else ["*"],
        },
        "groq": {
            "api_key_configured": bool(GROQ_API_KEY),
        },
        "vision": {
            "enabled": False,
            "note": "Envio de imagens via Groq (chat) não está habilitado neste backend ainda."
        }
    }

@app.get("/memory")
def memory_overview():
    mem = load_memory_v2()
    return {
        "version": mem.get("version", 2),
        "profiles_count": len(mem.get("profiles", {})),
        "chats_count": len(mem.get("chats", {})),
    }

@app.get("/memory/chat/{chat_id}")
def memory_chat(chat_id: str):
    mem = load_memory_v2()
    ch = mem.get("chats", {}).get(chat_id, {})
    return ch or {}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    raw = await file.read()

    path = os.path.join(UPLOAD_DIR, file_id)
    with open(path, "wb") as f:
        f.write(raw)

    mime = guess_mime(file.filename, file.content_type)
    FILES[file_id] = {"path": path, "name": file.filename, "mime": mime}

    return {"file_id": file_id}

@app.post("/chat_stream")
def chat_stream(payload: ChatInput):
    user_text = (payload.message or "").strip()
    if not user_text:
        def gen_empty():
            yield sse_data("Mensagem vazia.")
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(gen_empty(), media_type="text/event-stream")

    user_id = (payload.user_id or "guilherme").strip() or "guilherme"
    chat_id = (payload.chat_id or "").strip() or f"chat_{uuid.uuid4().hex[:10]}"

    chosen_model = resolve_text_model(payload.model)
    messages, use_vision = build_messages(
        user_text=user_text,
        user_id=user_id,
        chat_id=chat_id,
        use_intelligent_memory=bool(payload.smart_memory),
        history=payload.history,
        file_ids=payload.file_ids
    )

    # Bloqueia visão por enquanto
    if use_vision:
        def gen_no_vision():
            yield sse_data("[meta] chat_id=" + chat_id)
            yield sse_data("Erro 👾: envio de imagem ainda não está habilitado no modo Groq.")
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(gen_no_vision(), media_type="text/event-stream")

    full_response: List[str] = []

    def generator():
        # meta opcional
        yield sse_data(f"[meta] chat_id={chat_id}")

        for sse in stream_groq(messages, chosen_model or MODEL_DEFAULT_TEXT):
            if sse.startswith("data: "):
                full_response.append(sse[6:])  # mantém sem strip
            yield sse

        assistant_text = normalize_text("".join(full_response))
        if payload.smart_memory:
            try:
                update_intelligent_memory(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    model_for_extractor=chosen_model or MODEL_DEFAULT_TEXT
                )
            except:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(generator(), media_type="text/event-stream", headers=headers)