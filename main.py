from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

APP_TITLE = "Devset IA API"

INDEX_HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Devset IA</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 24px auto; padding: 0 12px; }
    textarea { width: 100%; min-height: 80px; }
    .row { display:flex; gap:12px; align-items:center; margin:10px 0; flex-wrap: wrap; }
    button { padding:8px 14px; cursor:pointer; }
    #chat { border:1px solid #ddd; border-radius:8px; padding:12px; min-height:140px; }
    .msg { margin:8px 0; white-space:pre-wrap; }
    .user { color:#1b4fff; }
    .bot { color:#111; }
    #error { color:#c1121f; margin-top:10px; }
    .muted { color:#666; font-size:12px; }
  </style>
</head>
<body>
  <h1>Devset IA</h1>

  <label for="message">Mensagem</label>
  <textarea id="message" placeholder="Digite sua mensagem..."></textarea>

  <div class="row">
    <label><input type="checkbox" id="streamMode" checked /> Modo: Stream (SSE)</label>
    <button id="sendBtn">Enviar</button>
    <span class="muted" id="status"></span>
  </div>

  <div id="chat"></div>
  <div id="error"></div>

  <script>
    // ✅ MESMA ORIGEM (Railway). Nada de CORS.
    const API_BASE = "";

    const messageEl = document.getElementById("message");
    const streamEl  = document.getElementById("streamMode");
    const chatEl    = document.getElementById("chat");
    const errorEl   = document.getElementById("error");
    const statusEl  = document.getElementById("status");
    const sendBtn   = document.getElementById("sendBtn");

    function setStatus(text){ statusEl.textContent = text || ""; }

    function addMessage(role, text){
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = `${role === "user" ? "Você" : "Bot"}: ${text}`;
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
      return div;
    }

    async function sendNormal(message){
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ message })
      });

      if(!res.ok){
        const txt = await res.text().catch(()=> "");
        throw new Error(`HTTP ${res.status} ${res.statusText} ${txt}`);
      }

      const data = await res.json().catch(()=> ({}));
      addMessage("bot", data.reply || "Sem resposta");
    }

    function parseSSE(block){
      const lines = block.split("\\n").map(l => l.trimEnd());
      let event = "";
      const dataLines = [];
      for(const line of lines){
        if(line.startsWith("event:")) event = line.slice(6).trim();
        if(line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      return { event, data: dataLines.join("\\n") };
    }

    async function sendStream(message){
      const res = await fetch(`${API_BASE}/chat_stream`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ message })
      });

      if(!res.ok){
        const txt = await res.text().catch(()=> "");
        throw new Error(`HTTP ${res.status} ${res.statusText} ${txt}`);
      }
      if(!res.body) throw new Error("Streaming não suportado (res.body vazio)");

      const botLine = addMessage("bot", "");
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let botText = "";

      while(true){
        const { value, done } = await reader.read();
        if(done) break;

        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\\n\\n");
        buffer = blocks.pop() || "";

        for(const b of blocks){
          const { event, data } = parseSSE(b);
          if(!event) continue;

          if(event === "token"){
            botText += data; // backend já manda espaço no token
            botLine.textContent = `Bot: ${botText}`;
            chatEl.scrollTop = chatEl.scrollHeight;
          }

          if(event === "done" || data === "[DONE]") return;
        }
      }
    }

    async function onSend(){
      errorEl.textContent = "";
      const message = messageEl.value.trim();
      if(!message){
        errorEl.textContent = "Digite uma mensagem antes de enviar.";
        return;
      }

      addMessage("user", message);
      messageEl.value = "";

      sendBtn.disabled = true;
      setStatus("Enviando...");

      try{
        if(streamEl.checked){
          setStatus("Stream ligado...");
          await sendStream(message);
        } else {
          setStatus("Modo normal...");
          await sendNormal(message);
        }
        setStatus("OK");
      } catch(err){
        console.error(err);
        errorEl.textContent = `Falha de rede/API: ${err?.message || err}`;
        setStatus("Erro");
      } finally {
        sendBtn.disabled = false;
        setTimeout(()=> setStatus(""), 1200);
      }
    }

    sendBtn.addEventListener("click", onSend);
    messageEl.addEventListener("keydown", (e) => {
      if(e.key === "Enter" && !e.shiftKey){
        e.preventDefault();
        onSend();
      }
    });
  </script>
</body>
</html>
"""

class ChatRequest(BaseModel):
    message: str
    model: str | None = None

app = FastAPI(title=APP_TITLE)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
async def health():
    return {"status": "healthy", "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/chat")
async def chat(payload: ChatRequest):
    chosen_model = payload.model or "devset-sim"
    return {"reply": f"Você disse: {payload.message}", "model_used": chosen_model}

async def sse_tokens(message: str) -> AsyncGenerator[str, None]:
    content = f"Você disse: {message}"
    for token in content.split(" "):
        yield f"event: token\ndata: {token} \n\n"
    yield "event: done\ndata: [DONE]\n\n"

@app.post("/chat_stream")
async def chat_stream(payload: ChatRequest):
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