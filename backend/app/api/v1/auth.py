from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from app.api.deps import DB, CurrentUser
from app.core.config import get_settings
from app.core.security import make_session_token, verify_password
from app.models import User
from app.schemas import LoginIn, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response, db: DB) -> User:
    user = db.execute(select(User).where(User.email == body.email.lower())).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(user.password_hash, body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong email or password")
    s = get_settings()
    response.set_cookie(
        s.session_cookie_name, make_session_token(user.id), max_age=s.session_max_age_s,
        httponly=True, samesite="lax", secure=s.env != "dev",
    )
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(get_settings().session_cookie_name)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
