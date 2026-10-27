import os
import unittest
from unittest.mock import patch

import auth
from database import _PostgresConnection, active_tenant_schema, reset_active_tenant_schema, set_active_tenant_schema


class DeploymentSecurityTests(unittest.TestCase):
    def test_postgres_sql_translates_sqlite_ddl_and_time_query(self):
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, ticker TEXT PRIMARY KEY)"
        translated = _PostgresConnection._translate(ddl)
        self.assertIn("BIGSERIAL PRIMARY KEY", translated)
        self.assertIn("created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP", translated)
        self.assertIn("DOUBLE PRECISION", _PostgresConnection._translate("CREATE TABLE t (amount REAL NOT NULL)"))
        sql = _PostgresConnection._translate("SELECT * FROM fx_rates WHERE requested_date=? AND cached_at >= datetime('now','-30 minutes')")
        self.assertIn("requested_date=%s", sql)
        self.assertIn("CURRENT_TIMESTAMP - INTERVAL '30 minutes'", sql)

    def test_tenant_schema_must_be_generated_format(self):
        token = set_active_tenant_schema("tenant_0123456789abcdef0123456789abcdef")
        try:
            self.assertEqual(active_tenant_schema(), "tenant_0123456789abcdef0123456789abcdef")
        finally:
            reset_active_tenant_schema(token)
        with self.assertRaises(ValueError):
            set_active_tenant_schema('public; DROP SCHEMA public')

    def test_password_hashing_uses_salt_and_verifiable_digest(self):
        salt, digest = auth._password_hash("example long password")
        self.assertEqual(len(bytes.fromhex(salt)), 16)
        self.assertEqual(len(digest), 64)
        _, same_digest = auth._password_hash("example long password", bytes.fromhex(salt))
        _, other_digest = auth._password_hash("a different password", bytes.fromhex(salt))
        self.assertEqual(digest, same_digest)
        self.assertNotEqual(digest, other_digest)
        self.assertTrue(auth._valid_password("a sufficiently long password"))
        self.assertFalse(auth._valid_password("short"))


if __name__ == "__main__":
    unittest.main()
