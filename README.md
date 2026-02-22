DevSet FastAPI backend
======================

Features:
- Email OTP login (simulated send)
- Secure httpOnly session cookie
- SSE streaming endpoint `/chat_stream`
- File upload endpoint `/upload`
- SQLite/Postgres support via `DATABASE_URL`

Environment variables:
- `DATABASE_URL` (e.g. sqlite+aiosqlite:///./devset.db or postgres async URL)
- `SECRET_KEY` (not used for cookie storage in this simple example but recommended)
- `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS` (not used; `send_email` is simulated)
- `SESSION_EXPIRE_MINUTES` defaults to 1440
- `SECURE_COOKIE` set to true in production to enforce Secure cookie
- `ALLOW_ORIGINS` comma-separated allowed CORS origins (default `*`)

Run locally:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend example (fetch with credentials include):

```js
fetch('https://your-backend.example.com/chat_stream', {
  method: 'POST',
  credentials: 'include',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ message: 'Hello', chat_id: null })
})
```

Railway deployment
------------------
- `Procfile` is included; Railway will use the `web` command to run the app.
- Add environment variables in Railway (see `.env.example`).
- Example: set `DATABASE_URL`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, and `SECURE_COOKIE=true`.

Notes
-----
- `send_email` uses SMTP via `aiosmtplib` when `SMTP_HOST` is set; otherwise it logs the message (safe for local testing).
- To enable real email delivery on Railway, set SMTP env vars in the project settings.
