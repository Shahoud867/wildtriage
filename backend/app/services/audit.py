from sqlalchemy.orm import Session

from app.models import AuditLog


def record(
    db: Session,
    *,
    actor_id: int | None,
    action: str,
    entity_type: str,
    entity_id: int,
    before: dict | None = None,
    after: dict | None = None,
) -> None:
    """Append an audit entry in the caller's transaction (commit together with the change)."""
    db.add(
        AuditLog(
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            before=before,
            after=after,
        )
    )
