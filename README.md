DevSet FastAPI backend
======================

Features:
- Email OTP login (simulated send)
- Secure httpOnly session cookie
- SSE streaming endpoint `/chat_stream`
- File upload endpoint `/upload`
- SQLite/Postgres support via `DATABASE_URL`

Environment variables (important for CORS/cookies):
- `DATABASE_URL` (e.g. sqlite+aiosqlite:///./devset.db or postgres async URL)
- `SECRET_KEY` (recommended)
- `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS` (optional; `send_email` is simulated when missing)
- `SESSION_EXPIRE_MINUTES` defaults to 1440
- `SECURE_COOKIE` set to `true` in production to enforce Secure cookie
- `ALLOW_ORIGINS` comma-separated allowed CORS origins (NO WILDCARD when using credentials)
  - Example (Cloudflare Worker origin):
    `ALLOW_ORIGINS=https://steep-disk-3924.guilhermexp0708.workers.dev`
  - Important: `ALLOW_ORIGINS` must list the frontend origin(s) that will call the API.
    Do NOT set it to the backend domain itself (that does not help CORS for cross-site requests).

Run locally:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend example (fetch with credentials include):

Set the backend base URL to exactly:

`https://devset-backend1-production-0b6f.up.railway.app`

Example fetch for SSE (`/chat_stream`):

```js
fetch('https://devset-backend1-production-0b6f.up.railway.app/chat_stream', {
  method: 'POST',
  credentials: 'include',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream'
  },
  body: JSON.stringify({ message: 'Hello' })
})
```

Curl checks (replace `$BACKEND` and `$FRONT` as needed):

1) Preflight OPTIONS for `/chat_stream` from the worker origin:

```bash
BACKEND="https://devset-backend1-production-0b6f.up.railway.app"
FRONT="https://steep-disk-3924.guilhermexp0708.workers.dev"

curl -i -X OPTIONS "$BACKEND/chat_stream" \
  -H "Origin: $FRONT" \
  -H "Access-Control-Request-Method: POST"
```

Expect to see in the response headers:

- `Access-Control-Allow-Origin: https://steep-disk-3924.guilhermexp0708.workers.dev`
- `Access-Control-Allow-Credentials: true`

2) POST `/chat_stream` without a session (should return Not authenticated):

```bash
curl -i -X POST "$BACKEND/chat_stream" \
  -H "Origin: $FRONT" \
  -H "Content-Type: application/json" \
  -d '{"message":"hello"}'
```

You should receive `401 Unauthorized` / `Not authenticated` if no session cookie is present.

Railway deployment
------------------
- `Procfile` is included; Railway will use the `web` command to run the app.
- Add environment variables in Railway (see `.env.example`).
- Example: set `DATABASE_URL`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, and `SECURE_COOKIE=true`.

Notes
-----
- `send_email` uses SMTP via `aiosmtplib` when `SMTP_HOST` is set; otherwise it logs the message (safe for local testing).
- To enable real email delivery on Railway, set SMTP env vars in the project settings.
