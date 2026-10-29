"""Opt-in PostgreSQL isolation, pool-reset, and backup/restore integration tests.

Set TEST_DATABASE_URL to a disposable PostgreSQL database. The restore test also
requires TEST_RESTORE_DATABASE_URL to point to a second, empty disposable database.
Never point either variable at production.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import database

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
TEST_RESTORE_DATABASE_URL = os.getenv("TEST_RESTORE_DATABASE_URL", "").strip()
try:
    import psycopg
    import psycopg_pool  # noqa: F401
except ImportError:  # pragma: no cover - exercised when optional PG deps are absent
    psycopg = None

POSTGRES_READY = bool(TEST_DATABASE_URL and psycopg is not None)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    POSTGRES_READY,
    "Set TEST_DATABASE_URL and install psycopg[binary,pool] to run PostgreSQL integration tests.",
)
class PostgreSQLTenantPoolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dsn = TEST_DATABASE_URL
        cls._database_url_patch = patch.dict(os.environ, {"DATABASE_URL": cls.dsn})
        cls._database_url_patch.start()
        suffix = secrets.token_hex(16)
        cls.schema_a = f"tenant_{suffix}"
        cls.schema_b = f"tenant_{secrets.token_hex(16)}"
        cls._schemas = (cls.schema_a, cls.schema_b)

        with psycopg.connect(cls.dsn, autocommit=True) as con:
            for schema, owner in ((cls.schema_a, "user_a"), (cls.schema_b, "user_b")):
                con.execute(f'CREATE SCHEMA "{schema}"')
                con.execute(
                    f'''CREATE TABLE "{schema}".pool_isolation_probe (
                           owner TEXT NOT NULL,
                           iteration INTEGER NOT NULL,
                           PRIMARY KEY (owner, iteration)
                       )'''
                )
                con.execute(
                    f'INSERT INTO "{schema}".pool_isolation_probe VALUES (%s, %s)',
                    (owner, -1),
                )

    @classmethod
    def tearDownClass(cls):
        pool = database._POSTGRES_POOLS.get(cls.dsn)
        if pool is not None:
            pool.close()
            with database._POSTGRES_POOLS_LOCK:
                database._POSTGRES_POOLS.pop(cls.dsn, None)
        try:
            with psycopg.connect(cls.dsn, autocommit=True) as con:
                for schema in cls._schemas:
                    con.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            cls._database_url_patch.stop()

    def _assert_no_tenant_search_path_on_idle_pool_connection(self):
        pool = database.get_postgres_pool(self.dsn)
        pool_size = pool.get_stats()["pool_size"]
        checked_out = []
        try:
            for _ in range(pool_size):
                raw = pool.getconn(timeout=10)
                checked_out.append(raw)
                schemas = raw.execute("SELECT current_schemas(false)").fetchone()[0]
                leaked = set(schemas).intersection(self._schemas)
                self.assertFalse(leaked, f"A pooled connection retained tenant search_path: {leaked}")
        finally:
            for raw in checked_out:
                try:
                    raw.rollback()
                except Exception:
                    pool.putconn(raw, close=True)
                else:
                    pool.putconn(raw)

    def test_commit_and_rollback_reset_tenant_search_path_before_pool_reuse(self):
        token = database.set_active_tenant_schema(self.schema_a)
        try:
            with database.connect_database(Path("unused.sqlite3")) as con:
                row = con.execute("SELECT current_schema() AS schema").fetchone()
                self.assertEqual(row["schema"], self.schema_a)
                con.execute(
                    "INSERT INTO pool_isolation_probe(owner, iteration) VALUES (?, ?)",
                    ("user_a", 1000),
                )
        finally:
            database.reset_active_tenant_schema(token)

        self._assert_no_tenant_search_path_on_idle_pool_connection()

        token = database.set_active_tenant_schema(self.schema_b)
        try:
            with self.assertRaisesRegex(RuntimeError, "force rollback"):
                with database.connect_database(Path("unused.sqlite3")) as con:
                    row = con.execute("SELECT current_schema() AS schema").fetchone()
                    self.assertEqual(row["schema"], self.schema_b)
                    con.execute(
                        "INSERT INTO pool_isolation_probe(owner, iteration) VALUES (?, ?)",
                        ("user_b", 1001),
                    )
                    raise RuntimeError("force rollback")
        finally:
            database.reset_active_tenant_schema(token)

        self._assert_no_tenant_search_path_on_idle_pool_connection()
        with psycopg.connect(self.dsn) as con:
            count = con.execute(
                f'SELECT count(*) FROM "{self.schema_b}".pool_isolation_probe WHERE iteration=1001'
            ).fetchone()[0]
        self.assertEqual(count, 0, "A failed tenant transaction must roll back its writes")

    def test_two_users_can_query_and_write_concurrently_without_tenant_leakage(self):
        iterations = 24
        barrier = threading.Barrier(2)

        def exercise_tenant(schema: str, owner: str) -> int:
            for iteration in range(iterations):
                token = database.set_active_tenant_schema(schema)
                try:
                    self.assertEqual(database.active_tenant_schema(), schema)
                    with database.connect_database(Path("unused.sqlite3")) as con:
                        selected = con.execute(
                            "SELECT current_schema() AS schema, owner FROM pool_isolation_probe "
                            "WHERE iteration=-1"
                        ).fetchone()
                        if selected["schema"] != schema or selected["owner"] != owner:
                            raise AssertionError(
                                f"Tenant {owner} saw schema={selected['schema']!r}, "
                                f"owner={selected['owner']!r}"
                            )
                        barrier.wait(timeout=15)
                        con.execute(
                            "INSERT INTO pool_isolation_probe(owner, iteration) VALUES (?, ?)",
                            (owner, iteration),
                        )
                finally:
                    database.reset_active_tenant_schema(token)
                if database.active_tenant_schema() is not None:
                    raise AssertionError("Tenant context was not reset after a request")
            return iterations

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(exercise_tenant, self.schema_a, "user_a"),
                executor.submit(exercise_tenant, self.schema_b, "user_b"),
            ]
            self.assertEqual([future.result(timeout=90) for future in futures], [iterations, iterations])

        with psycopg.connect(self.dsn) as con:
            for schema, owner in ((self.schema_a, "user_a"), (self.schema_b, "user_b")):
                rows = con.execute(
                    f'SELECT owner, count(*) FROM "{schema}".pool_isolation_probe GROUP BY owner'
                ).fetchall()
                self.assertEqual(rows, [(owner, iterations + 1)])

        self._assert_no_tenant_search_path_on_idle_pool_connection()


@unittest.skipUnless(
    POSTGRES_READY and TEST_RESTORE_DATABASE_URL,
    "Set TEST_DATABASE_URL and a separate TEST_RESTORE_DATABASE_URL for the backup restore drill.",
)
class PostgreSQLBackupRestoreIntegrationTests(unittest.TestCase):
    def test_custom_dump_restores_shared_and_tenant_rows_into_separate_empty_database(self):
        for binary in ("pg_dump", "pg_restore"):
            if shutil.which(binary) is None:
                self.skipTest(f"{binary} is not installed; install PostgreSQL client tools to run the restore drill.")

        source_schema = f"tenant_{secrets.token_hex(16)}"
        shared_schema = f"shared_restore_{secrets.token_hex(12)}"
        marker = secrets.token_hex(24)
        source_db_patch = patch.dict(os.environ, {"DATABASE_URL": TEST_DATABASE_URL})
        source_db_patch.start()
        try:
            with psycopg.connect(TEST_DATABASE_URL) as source:
                source_name = source.execute("SELECT current_database()").fetchone()[0]
                source.execute(f'CREATE SCHEMA "{source_schema}"')
                source.execute(f'CREATE SCHEMA "{shared_schema}"')
                source.execute(
                    f'CREATE TABLE "{source_schema}".restore_probe (marker TEXT PRIMARY KEY)'
                )
                source.execute(
                    f'CREATE TABLE "{shared_schema}".restore_probe (marker TEXT PRIMARY KEY)'
                )
                source.execute(
                    f'INSERT INTO "{source_schema}".restore_probe VALUES (%s)', (marker,)
                )
                source.execute(
                    f'INSERT INTO "{shared_schema}".restore_probe VALUES (%s)', (marker,)
                )

            with psycopg.connect(TEST_RESTORE_DATABASE_URL) as target:
                target_name = target.execute("SELECT current_database()").fetchone()[0]
                self.assertNotEqual(source_name, target_name, "Restore target must be a different database")
                user_schemas = target.execute(
                    "SELECT nspname FROM pg_namespace "
                    "WHERE nspname NOT LIKE 'pg_%' AND nspname <> 'information_schema' "
                    "AND nspname <> 'public'"
                ).fetchall()
                public_objects = target.execute(
                    "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S','f')"
                ).fetchone()[0]
                self.assertFalse(user_schemas or public_objects, "Restore database must be empty")

            with tempfile.TemporaryDirectory(prefix="investment-copilot-pg-restore-") as tmp:
                backup_dir = Path(tmp) / "backups"
                backup_env = os.environ.copy()
                backup_env.update({"DATABASE_URL": TEST_DATABASE_URL, "PG_BACKUP_DIR": str(backup_dir)})
                backup = subprocess.run(
                    ["bash", str(PROJECT_ROOT / "scripts/backup_postgres.sh")],
                    cwd=PROJECT_ROOT,
                    env=backup_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(backup.returncode, 0, f"PostgreSQL backup script failed: {backup.stderr[-2000:]}")
                archives = list(backup_dir.glob("*.dump"))
                self.assertEqual(len(archives), 1, "Backup script should create exactly one archive")
                archive = archives[0]
                checksum_line = archive.with_suffix(archive.suffix + ".sha256").read_text(encoding="utf-8")
                recorded_checksum = checksum_line.split()[0]
                actual_checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
                self.assertEqual(recorded_checksum, actual_checksum)

                restore_env = os.environ.copy()
                restore_env["RESTORE_DATABASE_URL"] = TEST_RESTORE_DATABASE_URL
                restore = subprocess.run(
                    ["bash", str(PROJECT_ROOT / "scripts/restore_postgres.sh"), str(archive)],
                    cwd=PROJECT_ROOT,
                    env=restore_env,
                    input="RESTORE\n",
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(restore.returncode, 0, f"PostgreSQL restore script failed: {restore.stderr[-2000:]}")

            with psycopg.connect(TEST_RESTORE_DATABASE_URL) as restored:
                tenant_marker = restored.execute(
                    f'SELECT marker FROM "{source_schema}".restore_probe'
                ).fetchone()[0]
                shared_marker = restored.execute(
                    f'SELECT marker FROM "{shared_schema}".restore_probe'
                ).fetchone()[0]
            self.assertEqual(tenant_marker, marker)
            self.assertEqual(shared_marker, marker)
        finally:
            try:
                with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as source:
                    source.execute(f'DROP SCHEMA IF EXISTS "{source_schema}" CASCADE')
                    source.execute(f'DROP SCHEMA IF EXISTS "{shared_schema}" CASCADE')
            finally:
                try:
                    with psycopg.connect(TEST_RESTORE_DATABASE_URL, autocommit=True) as target:
                        target.execute(f'DROP SCHEMA IF EXISTS "{source_schema}" CASCADE')
                        target.execute(f'DROP SCHEMA IF EXISTS "{shared_schema}" CASCADE')
                finally:
                    source_db_patch.stop()


if __name__ == "__main__":
    unittest.main()
