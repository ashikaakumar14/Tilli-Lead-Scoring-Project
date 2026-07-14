"""Langfuse 4.x tracing helpers for Tilli Lead Scoring.

All public helpers are safe no-ops when Langfuse credentials are missing
or initialization / authentication fails. The app runs fully without Langfuse.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
ENV_PATH = BASE_DIR / ".env"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_logger = logging.getLogger(__name__)

_initialized = False
_enabled = False
_client = None

F = TypeVar("F", bound=Callable[..., Any])


def _configure_langfuse_env() -> None:
    """Normalize Langfuse host env vars to LANGFUSE_BASE_URL (.env standard)."""
    if os.getenv("LANGFUSE_BASE_URL"):
        return
    host = os.getenv("LANGFUSE_HOST")
    if host:
        os.environ["LANGFUSE_BASE_URL"] = host


def _credentials_present() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def _disable_tracing(reason: str) -> None:
    global _enabled, _client
    _enabled = False
    _client = None
    _logger.debug("Langfuse tracing disabled: %s", reason)


def tracing_enabled() -> bool:
    """Return True when Langfuse tracing is active."""
    init_tracing()
    return _enabled


def get_langfuse_client():
    """Return the Langfuse client, or None when tracing is disabled."""
    init_tracing()
    return _client if _enabled else None


def init_tracing() -> None:
    """Initialize Langfuse if credentials are valid; otherwise no-op."""
    global _initialized, _enabled, _client
    if _initialized:
        return
    _initialized = True

    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
        _configure_langfuse_env()

        if not _credentials_present():
            _disable_tracing("missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY")
            return

        from langfuse import get_client
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        try:
            AnthropicInstrumentor().instrument()
        except Exception as exc:
            # Safe to continue if the SDK was already instrumented.
            _logger.debug("Anthropic instrumentation skipped: %s", exc)

        client = get_client()

        try:
            if hasattr(client, "auth_check") and not client.auth_check():
                _disable_tracing("auth_check returned False")
                return
        except Exception as exc:
            _disable_tracing(f"auth_check failed: {exc}")
            return

        _client = client
        _enabled = True
    except Exception as exc:
        _disable_tracing(f"initialization failed: {exc}")


def flush_tracing() -> None:
    """Send buffered traces to Langfuse. No-op when tracing is disabled."""
    if not _enabled or _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:
        _logger.debug("Langfuse flush failed: %s", exc)


def get_current_trace_id() -> Optional[str]:
    """Return the active trace ID, or None when tracing is disabled."""
    if not _enabled or _client is None:
        return None
    try:
        return _client.get_current_trace_id()
    except Exception as exc:
        _logger.debug("Langfuse get_current_trace_id failed: %s", exc)
        return None


def update_current_span(
    *,
    name: Optional[str] = None,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
) -> None:
    """Update the current observation span. No-op when tracing is disabled."""
    if not _enabled or _client is None:
        return

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if metadata is not None:
        kwargs["metadata"] = metadata
    if level is not None:
        kwargs["level"] = level
    if status_message is not None:
        kwargs["status_message"] = status_message
    if not kwargs:
        return

    try:
        _client.update_current_span(**kwargs)
    except Exception as exc:
        _logger.debug("Langfuse update_current_span failed: %s", exc)


@contextmanager
def propagate_attributes(**kwargs: Any) -> Iterator[None]:
    """Propagate Langfuse attributes, or no-op when tracing is disabled."""
    init_tracing()
    if not _enabled:
        yield
        return

    try:
        from langfuse import propagate_attributes as _langfuse_propagate

        with _langfuse_propagate(**kwargs):
            yield
    except Exception as exc:
        _logger.debug("Langfuse propagate_attributes failed: %s", exc)
        yield


def observe(*, name: str):
    """Decorator that creates a Langfuse span, or a passthrough when disabled."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            init_tracing()
            if not _enabled or _client is None:
                return func(*args, **kwargs)

            try:
                span_cm = _client.start_as_current_observation(name=name)
                observation = span_cm.__enter__()
            except Exception as exc:
                _logger.debug("Langfuse observation start failed: %s", exc)
                return func(*args, **kwargs)

            try:
                return func(*args, **kwargs)
            except Exception as app_exc:
                try:
                    observation.update(
                        level="ERROR",
                        status_message=str(app_exc) or type(app_exc).__name__,
                    )
                except Exception:
                    pass
                raise
            finally:
                try:
                    span_cm.__exit__(*sys.exc_info())
                except Exception as exc:
                    _logger.debug("Langfuse observation end failed: %s", exc)

        return wrapper  # type: ignore[return-value]

    return decorator


def _send_test_trace() -> Optional[str]:
    """Create a test observation, flush it, and return the trace ID."""
    init_tracing()
    if not _enabled or _client is None:
        return None

    try:
        with _client.start_as_current_observation(
            name="langfuse-connectivity-test",
            input={"source": "langfuse_tracing.py", "test": True},
        ) as observation:
            trace_id = _client.get_current_trace_id()
            observation.update(output={"status": "ok"})
        flush_tracing()
        return trace_id
    except Exception as exc:
        _logger.debug("Langfuse test trace failed: %s", exc)
        return None


if __name__ == "__main__":
    trace_id = _send_test_trace()
    if trace_id:
        print(f"Langfuse test trace sent. Trace ID: {trace_id}")
    else:
        print(
            "Langfuse tracing is disabled or unavailable. "
            "App will continue without tracing. "
            "Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL."
        )
