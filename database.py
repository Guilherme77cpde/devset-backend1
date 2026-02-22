from typing import AsyncGenerator
import os
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./devset.db")

# Async engine and session factory for SQLAlchemy
engine: AsyncEngine = create_async_engine(DATABASE_URL, future=True)
# Use generic sessionmaker configured for AsyncSession for compatibility
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
