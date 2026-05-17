import json
import logging
from typing import Any, Dict, Optional


def get_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_event(logger: Optional[logging.Logger], event: str, payload: Dict[str, Any]) -> None:
    if not logger:
        return
    safe_payload = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    logger.info("event=%s payload=%s", event, safe_payload)
