"""Admin commands.

    python -m app.cli create-user --email you@uni.edu --name "You" --role admin   (prompts for a password)
    python -m app.cli init-db      # dev/SQLite only; use `alembic upgrade head` for Postgres
"""

import argparse
import getpass

from sqlalchemy import select

from app.core.security import hash_password
from app.db.base import Base
from app.db.session import get_engine, session_factory
from app.models import User
from app.models.enums import Role


def create_user(email: str, name: str, role: str, password: str | None) -> None:
    password = password or getpass.getpass("Password: ")
    with session_factory()() as db:
        if db.execute(select(User).where(User.email == email.lower())).scalar_one_or_none():
            raise SystemExit(f"user {email} already exists")
        db.add(User(email=email.lower(), name=name, role=Role(role), password_hash=hash_password(password)))
        db.commit()
    print(f"created {role} {email}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cu = sub.add_parser("create-user")
    cu.add_argument("--email", required=True)
    cu.add_argument("--name", required=True)
    cu.add_argument("--role", choices=[r.value for r in Role], default=Role.reviewer.value)
    cu.add_argument("--password", help="omit to be prompted (keeps it out of shell history)")
    sub.add_parser("init-db")
    args = parser.parse_args()
    if args.cmd == "create-user":
        create_user(args.email, args.name, args.role, args.password)
    elif args.cmd == "init-db":
        import app.models  # noqa: F401  (register tables)

        Base.metadata.create_all(get_engine())
        print("tables created")


if __name__ == "__main__":
    main()
