import logging
import os


def configure_langfuse_logging() -> None:
    if (os.getenv("LANGFUSE_DEBUG") or "").lower() != "true":
        return

    logger = logging.getLogger("langfuse")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = True

    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)


def get_langfuse(*, raise_if_configured: bool = False):
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    host = (os.getenv("LANGFUSE_HOST") or "").strip() or None

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse
    except Exception as exc:
        if raise_if_configured:
            raise RuntimeError(
                "Langfuse keys are set but the Langfuse SDK failed to import. "
                "Reinstall/upgrade the 'langfuse' package."
            ) from exc
        return None

    debug_env = (os.getenv("LANGFUSE_DEBUG") or "").lower() == "true"
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        debug=debug_env,
    )
