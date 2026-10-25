import logging

import structlog


def configure_logging(env: str) -> None:
    """JSON logs outside dev, readable console logs in dev."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    renderer = structlog.dev.ConsoleRenderer() if env == "dev" else structlog.processors.JSONRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
