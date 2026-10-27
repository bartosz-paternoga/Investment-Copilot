# Online deployment: GitHub + Streamlit Community Cloud + PostgreSQL

This build uses PostgreSQL for hosted storage, a shared authentication table in `app_private.auth_users`, and a separate `tenant_<32 hex characters>` schema for each account. Streamlit Cloud does not need local disk persistence. The application refuses to access PostgreSQL until an authenticated user's tenant schema has been selected. Self-registration is disabled.

## 1. Create a PostgreSQL database

Use a managed PostgreSQL provider such as Supabase or Neon. Create a dedicated database/project for Investment Pilot. Use a server-side PostgreSQL connection string (`DATABASE_URL`) with TLS enabled and a database role permitted to create schemas/tables. Do not put a database password or privileged connection string in code, a GitHub commit, a public issue, or a client-side/browser variable. Use the provider's direct or session-mode connection string if its transaction-pooling connection does not support session-level `search_path` consistently.

The first connection needs permission to create `app_private.auth_users`, a tenant schema, and tables/indexes inside tenant schemas. Portfolio rows are not migrated automatically: each new tenant begins with an empty investment ledger. The previous SQLite database/archive stays separate and is never imported by hosted mode.

## 2. Push this project to GitHub

1. Create a **private** GitHub repository unless the source code is intentionally public.
2. Unzip the delivered project, then from its root run:

   ```bash
   git init
   git add .
   git commit -m "Prepare Investment Pilot for hosted PostgreSQL"
   git branch -M main
   git remote add origin https://github.com/YOUR-ORG/YOUR-REPO.git
   git push -u origin main
   ```

3. Verify that `.env`, `.streamlit/secrets.toml`, `data/`, SQLite databases, backup dumps, and exports are not committed. If a credential was ever committed, rotate it; deleting it in a later commit is not sufficient.
4. GitHub Actions compiles the app and runs unit tests on pushes and pull requests.

## 3. Deploy on Streamlit Community Cloud

1. Open <https://share.streamlit.io/> and choose the private repository, `main` branch, and `app.py` as the entry point. In Advanced settings, select Python 3.12 (the currently documented Community Cloud default) unless you have a tested reason to use another supported version. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
2. In **Advanced settings / Secrets**, add this TOML (use a new, unique bootstrap password of at least 12 characters):

   ```toml
   DATABASE_URL = "postgresql://DB_USER:DB_PASSWORD@DB_HOST:5432/DB_NAME?sslmode=require"
   BOOTSTRAP_ADMIN_USERNAME = "admin"
   BOOTSTRAP_ADMIN_PASSWORD = "REPLACE_WITH_A_LONG_RANDOM_PASSWORD"
   ```

   Use the exact connection string supplied by your database provider, with URL-escaped credentials if needed. These are server-side secrets; do not use a public/anonymous client API key as `DATABASE_URL`. Streamlit documents this secrets workflow in its [Community Cloud secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
3. Deploy the app. On first startup, it creates `app_private.auth_users` and the bootstrap administrator's empty tenant schema. Sign in with the configured admin username/password.
4. After confirming access, remove or rotate `BOOTSTRAP_ADMIN_PASSWORD` in Streamlit Secrets. The app creates the bootstrap administrator only when `app_private.auth_users` is empty, so the secret is not a routine password-reset mechanism.
5. In the sidebar **User management** expander, create each account. The temporary password appears once in the administrator's session and is required to be changed on the user's first sign-in. Deliver it to the user through a secure channel. Disable unused accounts or reset forgotten passwords from the same panel. Password resets create a temporary password and require a new password at next sign-in.

### Authentication and data boundaries

- Passwords are stored as PBKDF2-HMAC-SHA256 hashes with individual random salts; plaintext passwords are not stored.
- New accounts are standard users. Self-registration is disabled. First login requires changing the generated temporary password.
- Each active user has a server-generated tenant schema, validated before use. All portfolio DB connections require a tenant context; there is no default/shared hosted portfolio schema.
- An administrator can manage accounts but this interface does not grant access to another user's portfolio records.
- Deactivation blocks future sign-ins and active-session revalidation. It does not delete the user's tenant schema or stored data.
- Use a unique database and strong database credentials. Restrict project/database-console access to trusted operators; anyone holding the privileged DB connection can access the data.

## 4. Backups and recovery

**Configure provider-managed backups/PITR according to the database plan's actual retention and recovery guarantees.** Verify that the plan includes the recovery window you need; do not assume that every free tier offers automated backups or point-in-time recovery. For additional independent backup copies, run from a trusted machine with PostgreSQL client tools installed:

```bash
export DATABASE_URL='postgresql://...'
./scripts/backup_postgres.sh
```

The script uses `pg_dump` custom format, verifies that the archive listing is readable, and creates a SHA-256 checksum. Dumps include all account metadata and portfolio schemas; store them encrypted, limit access, and transfer them to storage separate from the database host. Do not commit backup files to GitHub. Schedule the script using a trusted, appropriately secured scheduler/backup service and retain copies according to your recovery policy.

To test recovery, create a **separate empty recovery database** and run:

```bash
export RESTORE_DATABASE_URL='postgresql://.../investment_pilot_recovery'
./scripts/restore_postgres.sh backups/postgres/investment-pilot-YYYYMMDDTHHMMSSZ.dump
```

The restore script requires typing `RESTORE` and deliberately uses `RESTORE_DATABASE_URL`, not `DATABASE_URL`, to reduce the chance of overwriting production. After restoration, verify sign-in, user activation, portfolio transactions, watchlists, history caches, and snapshots. Perform a restore drill before relying on backups for production recovery.

## 5. Clean start / keeping old SQLite data separate

Hosted PostgreSQL starts empty. No SQLite file is imported. Keep your original ZIP and any existing `data/portfolio.sqlite3` or holdings exports in a separate private archive or encrypted backup location, outside the Git repository. The original upload ZIP did not contain a `data/` database folder, so there was no SQLite database to migrate from the provided archive. Local `data/` files, if you have them elsewhere, are left untouched by this deployment package.

## 6. Local development and tests

For local single-user SQLite mode only:

```bash
./start.sh
```

This launcher explicitly enables local development mode if `DATABASE_URL` is not set. Do not use local mode for a public deployment; it bypasses authentication and uses local disk. To run tests:

```bash
ETF_APP_LOCAL_DEV_MODE=1 python -m unittest discover -s tests -v
```

For hosted mode, set `DATABASE_URL`, `BOOTSTRAP_ADMIN_USERNAME`, and `BOOTSTRAP_ADMIN_PASSWORD` in Streamlit Secrets and do not enable `ETF_APP_LOCAL_DEV_MODE`.

## Operational caveats

- Deploying/restarting Streamlit does not delete PostgreSQL schemas; data durability depends on the database provider and backup policy.
- A database can be durable while the web app is temporarily unavailable; uptime and data retention are separate concerns.
- The current market-data worker is designed for local SQLite/Docker use. It is not automatically launched on Streamlit Community Cloud. The UI can fetch/cache quotes when users view or refresh a portfolio. If a scheduled hosted market-data worker is required, deploy a separate scheduled job that explicitly iterates tenants with audited access rather than choosing an implicit default schema.
- This work prepares secure account isolation at the app layer but is not a formal security audit. Before using high-value financial records, test recovery and authorization, rotate secrets, restrict database privileges, and review provider retention/compliance terms.
