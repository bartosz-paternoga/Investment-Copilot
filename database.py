"""Database configuration and compatibility adapter for SQLite/PostgreSQL.

PostgreSQL deployments use one schema per authenticated user. A schema is selected
on every new connection from a validated server-side tenant context. Never accept
schema names directly from request parameters.
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any
import threading

_POSTGRES_POOLS: dict[str, Any] = {}
_POSTGRES_POOLS_LOCK = threading.Lock()


_ACTIVE_TENANT_SCHEMA: ContextVar[str | None] = ContextVar("investment_pilot_tenant_schema", default=None)
_SCHEMA_RE = re.compile(r"^tenant_[a-f0-9]{32}$")


def database_url() -> str | None:
    """Read DATABASE_URL from environment or Streamlit secrets without requiring Streamlit locally."""
    value = os.getenv("DATABASE_URL", "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get("DATABASE_URL", "")).strip()
        return value or None
    except Exception:
        return None


def set_active_tenant_schema(schema_name: str | None):
    """Select the tenant schema for this execution context; returns a reset token."""
    if schema_name is not None and not _SCHEMA_RE.fullmatch(str(schema_name)):
        raise ValueError("Invalid tenant schema name.")
    return _ACTIVE_TENANT_SCHEMA.set(schema_name)


def reset_active_tenant_schema(token) -> None:
    _ACTIVE_TENANT_SCHEMA.reset(token)


def active_tenant_schema() -> str | None:
    return _ACTIVE_TENANT_SCHEMA.get()


def get_postgres_pool(dsn: str):
    """Return a bounded, process-wide Psycopg pool for this database DSN."""
    with _POSTGRES_POOLS_LOCK:
        pool = _POSTGRES_POOLS.get(dsn)
        if pool is None:
            try:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL connection pooling requires psycopg[binary,pool]."
                ) from exc
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=5,
                kwargs={"row_factory": dict_row, "connect_timeout": 15},
                open=True,
            )
            _POSTGRES_POOLS[dsn] = pool
        return pool


class _HybridRow(dict):
    """Mapping row that also permits SQLite-style numeric indexing."""
    def __init__(self, values: dict[str, Any]):
        super().__init__(values)
        self._values = list(values.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


class _PostgresCursor:
    def __init__(self, cursor, lastrowid=None):
        self._cursor = cursor
        self.lastrowid = lastrowid
        self.rowcount = cursor.rowcount

    def fetchone(self):
        row = self._cursor.fetchone()
        return _HybridRow(row) if row is not None else None

    def fetchall(self):
        return [_HybridRow(row) for row in self._cursor.fetchall()]


class _PostgresConnection:
    """Minimal SQLite-ish API implemented on psycopg 3 for existing portfolio SQL."""
    def __init__(self, dsn: str, schema_name: str):
        self._pool = get_postgres_pool(dsn)
        self._connection = self._pool.getconn(timeout=10)
        try:
            # SET LOCAL is transaction-scoped: every checkout gets its validated
            # tenant schema, and COMMIT/ROLLBACK clears it before pool reuse.
            self._connection.execute(f'SET LOCAL search_path TO "{schema_name}", public')
        except Exception:
            self._pool.putconn(self._connection, close=True)
            raise

    @staticmethod
    def _translate(sql: str) -> str:
        sql = re.sub(r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", "BIGSERIAL PRIMARY KEY", sql, flags=re.I)
        # Preserve SQLite's double-precision numeric behavior for financial values.
        sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql, flags=re.I)
        # These application columns represent timestamps, but the legacy SQLite schema declared them TEXT.
        sql = re.sub(r"\b(created_at|updated_at|added_at|cached_at|imported_at|fetched_at|started_at|finished_at)\s+TEXT\b", r"\1 TIMESTAMPTZ", sql, flags=re.I)
        sql = re.sub(r"datetime\(\s*'now'\s*,\s*'-30 minutes'\s*\)", "(CURRENT_TIMESTAMP - INTERVAL '30 minutes')", sql, flags=re.I)
        # All placeholders in this app are positional parameters, not question marks in literals.
        sql = sql.replace("?", "%s")
        return sql

    def execute(self, sql: str, params=()):
        statement = self._translate(sql)
        wants_lastrowid = bool(re.match(r"\s*INSERT\s+INTO\s+transactions\b", statement, flags=re.I)) and " RETURNING " not in statement.upper()
        if wants_lastrowid:
            statement = statement.rstrip().rstrip(";") + " RETURNING id"
        cursor = self._connection.cursor()
        cursor.execute(statement, tuple(params or ()))
        lastrowid = None
        if wants_lastrowid:
            record = cursor.fetchone()
            if record is not None:
                lastrowid = record["id"]
        return _PostgresCursor(cursor, lastrowid=lastrowid)

    def executemany(self, sql: str, seq_of_params):
        cursor = self._connection.cursor()
        cursor.executemany(self._translate(sql), seq_of_params)
        return _PostgresCursor(cursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        except Exception:
            self._pool.putconn(self._connection, close=True)
            raise
        else:
            self._pool.putconn(self._connection)
        return False


def connect_database(sqlite_path: Path):
    """Open the configured durable database, or local SQLite in explicit dev mode."""
    dsn = database_url()
    if dsn:
        schema = active_tenant_schema()
        if schema is None:
            raise RuntimeError("No authenticated tenant is active; refusing to access PostgreSQL without tenant isolation.")
        if not _SCHEMA_RE.fullmatch(schema):
            raise RuntimeError("Invalid active tenant schema.")
        return _PostgresConnection(dsn, schema)

    if os.getenv("ETF_APP_LOCAL_DEV_MODE", "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError("DATABASE_URL is required. Set ETF_APP_LOCAL_DEV_MODE=1 only for local, single-user development.")
    import sqlite3
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    class _ClosingConnection(sqlite3.Connection):
        def __exit__(self, exc_type, exc, traceback):
            try:
                return super().__exit__(exc_type, exc, traceback)
            finally:
                self.close()

    con = sqlite3.connect(sqlite_path, timeout=20, factory=_ClosingConnection)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con
