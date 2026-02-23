import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models  # noqa: F401
from .database import Base, engine
from .routers import auth_router, chat_router, upload_router

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Devset Backend")

# MVP without authentication on frontend requests:
# wildcard origins are allowed only when credentials are disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root():
    return {"status": "ok"}
