from collections.abc import Callable
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import read_session_token
from app.db.session import get_db
from app.models import User
from app.models.enums import Role

DB = Annotated[Session, Depends(get_db)]


def current_user(db: DB, wt_session: Annotated[str | None, Cookie()] = None) -> User:
    # The cookie name is fixed to the default "wt_session" here; keep WT_SESSION_COOKIE_NAME in sync.
    if not wt_session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not signed in")
    user_id = read_session_token(wt_session)
    user = db.get(User, user_id) if user_id is not None else None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session expired or user disabled")
    return user


CurrentUser = Annotated[User, Depends(current_user)]

_RANK = {Role.viewer: 0, Role.reviewer: 1, Role.admin: 2}


def require_role(minimum: Role) -> Callable[[User], User]:
    def check(user: CurrentUser) -> User:
        if _RANK[user.role] < _RANK[minimum]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role {minimum.value}")
        return user

    return check


Reviewer = Annotated[User, Depends(require_role(Role.reviewer))]
Admin = Annotated[User, Depends(require_role(Role.admin))]
