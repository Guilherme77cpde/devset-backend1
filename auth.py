import os
from datetime import datetime, timedelta
from uuid import uuid4
from fastapi import Request, HTTPException, status, Response, Depends
from sqlalchemy import select, delete
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models import User, OTPCode, Session as SessionModel
from utils import generate_otp, send_email, now_utc
from typing import Optional

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session_id")
SESSION_EXPIRE_MINUTES = int(os.getenv("SESSION_EXPIRE_MINUTES", "1440"))
SECURE_COOKIE = os.getenv("SECURE_COOKIE", "false").lower() in ("1", "true", "yes")


async def start_login(email: str, db: AsyncSession, otp_length: int = 6, otp_ttl_minutes: int = 10) -> None:
    code = generate_otp(otp_length)
    expires_at = now_utc() + timedelta(minutes=otp_ttl_minutes)

    # Upsert OTPCode (replace any previous)
    stmt = delete(OTPCode).where(OTPCode.email == email)
    await db.execute(stmt)

    otp = OTPCode(email=email, code=code, expires_at=expires_at)
    db.add(otp)
    await db.commit()

    # Send email (async). If SMTP not configured, send_email logs instead.
    await send_email(email, "Your login code", f"Your code is: {code}")


async def verify_login(email: str, code: str, db: AsyncSession) -> str:
    # Validate OTP
    q = await db.execute(select(OTPCode).where(OTPCode.email == email))
    otp_row = q.scalar_one_or_none()
    if not otp_row:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid code")
    if otp_row.expires_at < now_utc():
        # remove expired
        await db.execute(delete(OTPCode).where(OTPCode.email == email))
        await db.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code expired")
    if otp_row.code != code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid code")

    # OTP valid: delete it
    await db.execute(delete(OTPCode).where(OTPCode.email == email))

    # Ensure user exists
    q2 = await db.execute(select(User).where(User.email == email))
    user = q2.scalar_one_or_none()
    if not user:
        user = User(email=email)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # create session
    session_id = uuid4().hex
    expires_at = now_utc() + timedelta(minutes=SESSION_EXPIRE_MINUTES)
    sess = SessionModel(id=session_id, user_id=user.id, expires_at=expires_at)
    db.add(sess)
    await db.commit()

    return session_id


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    q = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    sess = q.scalar_one_or_none()
    if not sess:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    if sess.expires_at < now_utc():
        # session expired - remove and fail
        await db.execute(delete(SessionModel).where(SessionModel.id == session_id))
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    q2 = await db.execute(select(User).where(User.id == sess.user_id))
    user = q2.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def set_session_cookie(response: Response, session_id: str, max_age: Optional[int] = None) -> None:
    if max_age is None:
        max_age = SESSION_EXPIRE_MINUTES * 60
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=SECURE_COOKIE,
        max_age=max_age,
        path="/",
    )
