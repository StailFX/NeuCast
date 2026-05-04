"""SQLAlchemy session machinery — code-review L-2 (2026-05-04).

Previous version called ``os.environ["DATABASE_URL"]`` at IMPORT time,
which meant ``import app.main`` would raise ``KeyError`` on a fresh
deploy where the env wasn't yet exported. Worse, the import error
prevented FastAPI from starting at all — operator saw a hard 500 on
every page, with no way to render the friendly /error template.

Now the engine is built lazily on first access. Module import never
fails; if the env is missing at first DB use, the failure surfaces
where it can be handled gracefully (in the request handler, which
can return a 503 or render an "ingest not configured" template).
"""
import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _build_engine() -> Engine:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it via /etc/neucast/env or "
            "the local .env, then restart the service."
        )
    return create_engine(dsn)


def get_engine() -> Engine:
    """Lazy engine accessor — builds on first call, then cached."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    """Lazy SessionLocal accessor; mirrors ``get_engine``."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine(),
        )
    return _SessionLocal


class _LazyEngineProxy:
    """Backward-compatible drop-in for ``app.db.engine``.

    Existing call sites do ``from app.db import engine`` and then call
    ``engine.connect()``, ``engine.dispose()``, etc. The proxy forwards
    every attribute access to the real engine, building it on the first
    touch. Once the lazy build has happened, attribute access is just
    a single dict lookup — no measurable overhead.
    """
    __slots__ = ()

    def __getattr__(self, item):
        return getattr(get_engine(), item)

    def __repr__(self) -> str:
        return f"<LazyEngineProxy → {get_engine()!r}>"


class _LazySessionFactoryProxy:
    """Same shape as ``_LazyEngineProxy`` but for ``SessionLocal``.

    Call sites do ``SessionLocal()`` to mint a session — this hands
    that call through to the underlying ``sessionmaker`` instance
    after building it on first touch.
    """
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return get_session_factory()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(get_session_factory(), item)


# Backward-compatible exports — `from app.db import engine, SessionLocal`
# keeps working without import-time KeyError.
engine = _LazyEngineProxy()
SessionLocal = _LazySessionFactoryProxy()
