from fastapi import APIRouter, Depends, HTTPException, Response, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from .. import auth as auth_logic
from ..models import Session as SessionModel

router = APIRouter(prefix="/auth", tags=["auth"])


class StartLoginSchema(BaseModel):
    email: EmailStr


class VerifyLoginSchema(BaseModel):
    email: EmailStr
    code: str


@router.post("/start")
async def start_login(payload: StartLoginSchema, db: AsyncSession = Depends(get_db)):
    await auth_logic.start_login(payload.email, db)
    return {"ok": True, "msg": "OTP sent if SMTP configured or logged"}


@router.post("/verify")
async def verify_login(payload: VerifyLoginSchema, response: Response, db: AsyncSession = Depends(get_db)):
    session_id = await auth_logic.verify_login(payload.email, payload.code, db)
    auth_logic.set_session_cookie(response, session_id)
    return {"ok": True}


@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    session_id = request.cookies.get(auth_logic.SESSION_COOKIE_NAME)
    if session_id:
        await db.execute(SessionModel.__table__.delete().where(SessionModel.id == session_id))
        await db.commit()
    response.delete_cookie(auth_logic.SESSION_COOKIE_NAME, path="/", domain=None)
    return {"ok": True}


@router.get("/me")
async def me(user=Depends(auth_logic.get_current_user)):
    return {"ok": True, "user": {"id": user.id, "email": user.email}}
