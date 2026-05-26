import json
import logging
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra keyword args passed via the `extra` dict
        if hasattr(record, "ctx"):
            payload["ctx"] = record.ctx
        return json.dumps(payload)


def get_logger(name: str = "orchestrator") -> logging.Logger:
    logger = logging.getLogger(f"adaptive_agents.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


def log(logger: logging.Logger, level: str, msg: str, **ctx) -> None:
    """Helper for structured logging with a `ctx` dict attached."""
    method = getattr(logger, level.lower(), logger.info)
    method(msg, extra={"ctx": ctx} if ctx else {})
