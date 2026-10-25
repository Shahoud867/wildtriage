from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.core.config import get_settings

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt="wt-session")


def make_session_token(user_id: int) -> str:
    return _serializer().dumps({"uid": user_id})


def read_session_token(token: str) -> int | None:
    try:
        data = _serializer().loads(token, max_age=get_settings().session_max_age_s)
    except (BadSignature, SignatureExpired):
        return None
    return int(data["uid"])
