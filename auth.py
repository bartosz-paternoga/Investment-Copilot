"""Authentication and administrator-provisioned accounts for hosted deployments."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import uuid
from typing import Any
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from database import database_url, get_postgres_pool

LOGGER = logging.getLogger("investment_copilot.auth")

_ITERATIONS = 310_000
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,40}$")
_SESSION_COOKIE_NAME = "investment_copilot_session"
_SHORT_SESSION_HOURS = 12
_REMEMBER_SESSION_DAYS = 30


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


@contextmanager
def _connect():
    dsn = database_url()
    if not dsn:
        raise RuntimeError("DATABASE_URL must point to the managed PostgreSQL database.")
    # Reuse the app's bounded pool; pool.connection() commits/rolls back and
    # returns the connection to the pool safely, including on exceptions.
    pool = get_postgres_pool(dsn)
    with pool.connection(timeout=10) as con:
        yield con


def _password_hash(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return salt.hex(), digest.hex()


def _valid_password(password: str) -> bool:
    return len(password) >= 12 and len(password) <= 256


def initialize_auth_db() -> None:
    """Create only the shared authentication table; portfolio data is in tenant schemas."""
    with _connect() as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS app_private")
        con.execute("""CREATE TABLE IF NOT EXISTS app_private.auth_users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            email TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL CHECK (role IN ('admin','user')),
            tenant_schema TEXT NOT NULL UNIQUE,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_active ON app_private.auth_users(active, username)")
        con.execute("""CREATE TABLE IF NOT EXISTS app_private.auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES app_private.auth_users(user_id) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            revoked_at TIMESTAMPTZ
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON app_private.auth_sessions(user_id, expires_at)")


def ensure_bootstrap_admin() -> str | None:
    """Create the first admin only when configured secrets exist and no account exists."""
    initialize_auth_db()
    username = _setting("BOOTSTRAP_ADMIN_USERNAME", "admin").strip()
    password = _setting("BOOTSTRAP_ADMIN_PASSWORD", "")
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) AS n FROM app_private.auth_users").fetchone()["n"]
        if total:
            return None
        if not _USERNAME_RE.fullmatch(username):
            return "Set BOOTSTRAP_ADMIN_USERNAME to 3–40 letters, numbers, dots, underscores, or hyphens."
        if not _valid_password(password):
            return "No administrator exists yet. Configure BOOTSTRAP_ADMIN_PASSWORD as a unique password of at least 12 characters in Streamlit Secrets."
        user_id = uuid.uuid4().hex
        schema = f"tenant_{user_id}"
        salt, digest = _password_hash(password)
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        con.execute("""INSERT INTO app_private.auth_users
            (user_id,username,display_name,email,role,tenant_schema,password_salt,password_hash,active,must_change_password)
            VALUES (%s,%s,%s,%s,'admin',%s,%s,%s,TRUE,FALSE)""",
            (user_id, username, "Administrator", "", schema, salt, digest))
        return f"Created bootstrap administrator '{username}'. Remove/rotate the bootstrap password secret after confirming login."


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    username = str(username or "").strip()
    with _connect() as con:
        row = con.execute("SELECT * FROM app_private.auth_users WHERE lower(username)=lower(%s)", (username,)).fetchone()
    if row is None:
        # Reduce username-enumeration timing differences.
        _password_hash(password or "", bytes(16))
        return None
    _, expected = _password_hash(str(password or ""), bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(expected, row["password_hash"]) or not row["active"]:
        return None
    return dict(row)


def load_active_user(user_id: str) -> dict[str, Any] | None:
    if not user_id:
        return None
    with _connect() as con:
        row = con.execute("SELECT * FROM app_private.auth_users WHERE user_id=%s AND active=TRUE", (str(user_id),)).fetchone()
    return dict(row) if row else None


def _session_token_hash(token: str) -> str:
    """Store only a digest of the random browser session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_auth_session(user_id: str, stay_signed_in: bool) -> tuple[str, datetime]:
    """Create a random browser token and store only its hash in PostgreSQL."""
    token = secrets.token_urlsafe(32)
    duration = (
        timedelta(days=_REMEMBER_SESSION_DAYS)
        if stay_signed_in
        else timedelta(hours=_SHORT_SESSION_HOURS)
    )
    expires_at = datetime.now(timezone.utc) + duration
    with _connect() as con:
        con.execute(
            "DELETE FROM app_private.auth_sessions "
            "WHERE expires_at <= CURRENT_TIMESTAMP OR revoked_at IS NOT NULL"
        )
        con.execute(
            """INSERT INTO app_private.auth_sessions (token_hash,user_id,expires_at)
               VALUES (%s,%s,%s)""",
            (_session_token_hash(token), user_id, expires_at),
        )
    return token, expires_at


def load_auth_session(token: str) -> dict[str, Any] | None:
    """Validate a browser token against an unexpired, non-revoked active account."""
    if not token or len(token) > 256:
        return None
    with _connect() as con:
        row = con.execute(
            """SELECT u.*
               FROM app_private.auth_sessions s
               JOIN app_private.auth_users u ON u.user_id=s.user_id
               WHERE s.token_hash=%s
                 AND s.expires_at > CURRENT_TIMESTAMP
                 AND s.revoked_at IS NULL
                 AND u.active=TRUE""",
            (_session_token_hash(token),),
        ).fetchone()
    return dict(row) if row else None


def revoke_auth_session(token: str | None) -> None:
    """Invalidate one browser token on the server."""
    if not token:
        return
    with _connect() as con:
        con.execute(
            "UPDATE app_private.auth_sessions SET revoked_at=CURRENT_TIMESTAMP "
            "WHERE token_hash=%s AND revoked_at IS NULL",
            (_session_token_hash(token),),
        )


def _request_session_cookie(st, *, browser_result_only: bool = False) -> str | None:
    """Read the session token from the request or the browser cookie component."""
    try:
        # Streamlit's request snapshot is available in the supported versions.
        request_cookies = st.context.cookies
    except Exception:
        request_cookies = None
    token = request_cookies.get(_SESSION_COOKIE_NAME) if request_cookies is not None else None
    if token:
        return str(token)

    # A completed CookieManager call is already stored under its Streamlit key.
    # Read that result directly on the follow-up run instead of mounting the
    # browser component again beside the login form.
    if browser_result_only:
        cookies = st.session_state.get("auth_cookie_manager")
        token = cookies.get(_SESSION_COOKIE_NAME) if isinstance(cookies, dict) else None
        return str(token) if token else None

    # Streamlit Community Cloud may filter cookies from the initial request.
    # Read from the browser component whenever the request has no session token.
    try:
        token = _cookie_manager(st).get(_SESSION_COOKIE_NAME)
        return str(token) if token else None
    except Exception:
        return None

def _cookie_manager(st):
    """Return the cookie component for the current script run."""
    import extra_streamlit_components as stx

    return stx.CookieManager(key="auth_cookie_manager")


def _set_auth_cookie(
    st, token: str, expires_at: datetime, *, persistent: bool
) -> None:
    """Set a session or persistent cookie without a component-value rerun."""
    import streamlit.components.v1 as components

    cookie_name = json.dumps(_SESSION_COOKIE_NAME)
    cookie_value = json.dumps(token)
    expires = json.dumps(expires_at.isoformat())
    persistent_json = json.dumps(bool(persistent))
    components.html(
        f"""<!doctype html>
        <script>
        try {{
            const name = {cookie_name};
            const value = {cookie_value};
            const persistent = {persistent_json};
            const cookie = `${{encodeURIComponent(name)}}=${{encodeURIComponent(value)}}; ` +
                `Path=/; Secure; SameSite=Lax`;
            if (persistent) {{
                const expires = new Date({expires}).toUTCString();
                document.cookie = cookie + `; Expires=${{expires}}`;
            }} else {{
                // Replace any older remembered cookie with a true browser-
                // session cookie. No Expires/Max-Age means it ends at session close.
                document.cookie = `${{encodeURIComponent(name)}}=; Path=/; Max-Age=0; Secure; SameSite=Lax`;
                document.cookie = cookie;
            }}
        }} catch (error) {{}}
        </script>""",
        height=0,
        scrolling=False,
    )


def _delete_auth_cookie(st, key: str) -> bool:
    try:
        from extra_streamlit_components.CookieManager import _component_func

        _component_func(
            method="delete",
            cookie=_SESSION_COOKIE_NAME,
            key=f"{key}_delete",
            default=False,
        )
        return True
    except Exception:
        # Server-side revocation still prevents a stale cookie from authenticating.
        return False


def set_password(user_id: str, password: str) -> None:
    if not _valid_password(password):
        raise ValueError("Password must be between 12 and 256 characters.")
    salt, digest = _password_hash(password)
    with _connect() as con:
        con.execute("""UPDATE app_private.auth_users SET password_salt=%s,password_hash=%s,
            must_change_password=FALSE,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s AND active=TRUE""",
            (salt, digest, user_id))
        con.execute(
            "UPDATE app_private.auth_sessions SET revoked_at=CURRENT_TIMESTAMP "
            "WHERE user_id=%s AND revoked_at IS NULL",
            (user_id,),
        )


def create_user(username: str, display_name: str, email: str = "") -> tuple[dict[str, Any], str]:
    username = str(username or "").strip()
    display_name = str(display_name or "").strip()
    email = str(email or "").strip()
    if not _USERNAME_RE.fullmatch(username):
        raise ValueError("Username must be 3–40 characters: letters, numbers, dot, underscore, or hyphen.")
    if not display_name or len(display_name) > 100:
        raise ValueError("Display name is required and must be at most 100 characters.")
    if len(email) > 254 or (email and ("@" not in email or "\n" in email)):
        raise ValueError("Enter a valid email address or leave it blank.")
    temp_password = secrets.token_urlsafe(15)
    salt, digest = _password_hash(temp_password)
    user_id = uuid.uuid4().hex
    schema = f"tenant_{user_id}"
    with _connect() as con:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        con.execute("""INSERT INTO app_private.auth_users
            (user_id,username,display_name,email,role,tenant_schema,password_salt,password_hash,active,must_change_password)
            VALUES (%s,%s,%s,%s,'user',%s,%s,%s,TRUE,TRUE)""",
            (user_id, username, display_name, email, schema, salt, digest))
        row = con.execute("SELECT * FROM app_private.auth_users WHERE user_id=%s", (user_id,)).fetchone()
    return dict(row), temp_password


def list_users() -> list[dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("""SELECT user_id,username,display_name,email,role,active,must_change_password,created_at
            FROM app_private.auth_users ORDER BY role DESC, lower(username)""").fetchall()
    return [dict(row) for row in rows]


def reset_user_password(user_id: str) -> str:
    """Issue a one-time password that must be changed at the user's next sign-in."""
    temp_password = secrets.token_urlsafe(15)
    salt, digest = _password_hash(temp_password)
    with _connect() as con:
        cur = con.execute("""UPDATE app_private.auth_users SET password_salt=%s,password_hash=%s,
            must_change_password=TRUE,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s""",
            (salt, digest, user_id))
        if cur.rowcount != 1:
            raise ValueError("Account not found.")
        con.execute(
            "UPDATE app_private.auth_sessions SET revoked_at=CURRENT_TIMESTAMP "
            "WHERE user_id=%s AND revoked_at IS NULL",
            (user_id,),
        )
    return temp_password


def set_user_active(user_id: str, active: bool, *, actor_user_id: str) -> None:
    if user_id == actor_user_id and not active:
        raise ValueError("You cannot deactivate your own administrator account.")
    with _connect() as con:
        target = con.execute("SELECT role,active FROM app_private.auth_users WHERE user_id=%s", (user_id,)).fetchone()
        if not target:
            raise ValueError("Account not found.")
        if target["role"] == "admin" and not active:
            admins = con.execute("SELECT COUNT(*) AS n FROM app_private.auth_users WHERE role='admin' AND active=TRUE").fetchone()["n"]
            if admins <= 1:
                raise ValueError("The last active administrator cannot be deactivated.")
        con.execute("UPDATE app_private.auth_users SET active=%s,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s", (bool(active), user_id))
        if not active:
            con.execute(
                "UPDATE app_private.auth_sessions SET revoked_at=CURRENT_TIMESTAMP "
                "WHERE user_id=%s AND revoked_at IS NULL",
                (user_id,),
            )


def render_auth_gate(st) -> dict[str, Any] | None:
    """Render login/reset screen and return a freshly validated active user."""
    auth_screen = st.empty()
    # Read cookies opportunistically, but never gate the login form on the
    # browser component returning. Streamlit Cloud can delay that response;
    # the form must remain usable while it is in flight.
    skip_cookie_once = st.session_state.pop("_auth_skip_cookie_once", False)
    session_token = st.session_state.get("auth_session_token")
    if not session_token and not skip_cookie_once:
        session_token = _request_session_cookie(st)

    # CookieManager reports its browser-side set result by triggering one more
    # Streamlit run. Reuse the account already verified by password auth during
    # that single handoff, then require normal server-session validation again.
    login_transition = st.session_state.get("_auth_login_transition")
    transition_user = None
    if session_token and isinstance(login_transition, dict):
        transition_hash = str(login_transition.get("token_hash", ""))
        transition_deadline = float(login_transition.get("deadline", 0.0))
        if (
            transition_hash
            and hmac.compare_digest(transition_hash, _session_token_hash(session_token))
            and time.monotonic() <= transition_deadline
        ):
            candidate = login_transition.get("user")
            if isinstance(candidate, dict):
                transition_user = candidate
    if transition_user is None:
        st.session_state.pop("_auth_login_transition", None)
    current = transition_user or (load_auth_session(session_token) if session_token else None)
    if session_token and not current:
        st.session_state.pop("auth_session_token", None)
        st.session_state.pop("auth_user_id", None)
    if current and session_token:
        st.session_state["auth_session_token"] = session_token
        st.session_state["auth_user_id"] = current["user_id"]

        if current["must_change_password"]:
            with auth_screen.container():
                st.title("Investment Copilot")
                st.caption("Private investment workspace · administrator-provisioned accounts")
                st.subheader("Set your password")
                st.warning("Your administrator issued a temporary password. Choose a new password to continue.")
                with st.form("required_password_change"):
                    password_1 = st.text_input("New password", type="password")
                    password_2 = st.text_input("Confirm new password", type="password")
                    submit = st.form_submit_button("Save password", type="primary")
            if submit:
                if password_1 != password_2:
                    st.error("The passwords do not match.")
                else:
                    try:
                        set_password(current["user_id"], password_1)
                        _delete_auth_cookie(st, "auth_cookie_password_changed")
                        st.session_state.pop("auth_user_id", None)
                        st.session_state.pop("auth_session_token", None)
                        st.session_state["_auth_skip_cookie_once"] = True
                        st.success("Password updated. Please sign in again with your new password.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            return None

        auth_screen.empty()
        return current

    # Do not trust an old user ID kept in Streamlit session state by itself.
    st.session_state.pop("auth_user_id", None)
    st.session_state.pop("auth_session_token", None)

    with auth_screen.container():
        st.title("Investment Copilot")
        st.caption("Private investment workspace · administrator-provisioned accounts")
        with st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            stay_signed_in = st.checkbox(
                "Stay signed in on this device (30 days)",
                value=False,
                key="login_stay_signed_in",
                help="When unchecked, the browser-session cookie ends when you close the browser and the server session expires after 12 hours. When checked, the sign-in persists for up to 30 days.",
            )
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        st.caption("Accounts are created by an administrator. Self-registration is disabled.")

    if submitted:
        # Set up the bootstrap account only when someone actually submits the
        # form. Database DDL during an ordinary page open delayed the login
        # screen and made cookie-check failures look like a blank page.
        if not st.session_state.get("_bootstrap_checked", False):
            bootstrap_message = ensure_bootstrap_admin()
            if bootstrap_message and not bootstrap_message.startswith("Created bootstrap administrator"):
                st.error(bootstrap_message)
                return None
            st.session_state["_bootstrap_checked"] = True
        user = authenticate(username, password)
        if user:
            LOGGER.info("Password check accepted; creating an app session.")
            token, expires_at = create_auth_session(user["user_id"], stay_signed_in)
            # Save the verified session before mounting the browser cookie
            # component. Its response can cause an immediate Streamlit rerun;
            # that rerun must already have enough state to stay authenticated.
            st.session_state["auth_session_token"] = token
            st.session_state["auth_user_id"] = user["user_id"]
            st.session_state["_auth_login_transition"] = {
                "token_hash": _session_token_hash(token),
                "user": {
                    key: user[key]
                    for key in (
                        "user_id", "username", "display_name", "email", "role",
                        "tenant_schema", "active", "must_change_password",
                    )
                    if key in user
                },
                "deadline": time.monotonic() + 15,
            }
            try:
                _set_auth_cookie(st, token, expires_at, persistent=stay_signed_in)
            except Exception:
                LOGGER.exception("Could not request the browser session-cookie write.")
                revoke_auth_session(token)
                st.session_state.pop("auth_session_token", None)
                st.session_state.pop("auth_user_id", None)
                st.session_state.pop("_auth_login_transition", None)
                st.error("Could not establish the browser session. Please refresh and try again.")
            else:
                # Continue into the app in this run instead of showing the old
                # login form while a second full portfolio run is starting.
                auth_screen.empty()
                if user["must_change_password"]:
                    with auth_screen.container():
                        st.title("Investment Copilot")
                        st.caption("Private investment workspace · administrator-provisioned accounts")
                        st.subheader("Set your password")
                        st.warning("Your administrator issued a temporary password. Choose a new password to continue.")
                        with st.form("required_password_change"):
                            password_1 = st.text_input("New password", type="password")
                            password_2 = st.text_input("Confirm new password", type="password")
                            submit = st.form_submit_button("Save password", type="primary")
                    return None
                return user
        else:
            LOGGER.warning("Password check rejected.")
            st.error("Sign-in failed. Check your credentials or contact the administrator.")
    return None


def render_account_sidebar(st, user: dict[str, Any]) -> None:
    """Render account details and guarded admin account management controls."""
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Signed in as **{user['display_name']}** · {user['role']}")
        if st.button("Sign out", use_container_width=True, key="sign_out"):
            token = st.session_state.get("auth_session_token") or _request_session_cookie(st)
            revoke_auth_session(token)
            _delete_auth_cookie(st, "auth_cookie_logout")
            st.session_state.pop("auth_user_id", None)
            st.session_state.pop("auth_session_token", None)
            st.session_state.pop("created_user_credentials", None)
            # Avoid restoring the cookie from the current request during logout.
            st.session_state["_auth_skip_cookie_once"] = True
            st.rerun()
        if user["role"] != "admin":
            return
        with st.expander("User management", expanded=False):
            st.caption("New accounts receive a one-time temporary password and must change it at first sign-in.")
            with st.form("create_user_form", clear_on_submit=True):
                username = st.text_input("Username", key="new_user_username")
                display = st.text_input("Display name", key="new_user_display")
                email = st.text_input("Email (optional)", key="new_user_email")
                submit = st.form_submit_button("Create user", type="primary")
            if submit:
                try:
                    created, temp_password = create_user(username, display, email)
                    st.session_state["created_user_credentials"] = {"username": created["username"], "password": temp_password}
                    st.success(f"Created account {created['username']}. Copy the temporary password now.")
                except Exception as exc:
                    # Database exception messages may include SQL details; show safe common-case errors only.
                    message = str(exc)
                    if isinstance(exc, ValueError):
                        st.error(message)
                    elif "duplicate key" in message.lower() or "unique constraint" in message.lower():
                        st.error("That username already exists. Choose another username.")
                    else:
                        st.error("Could not create account. Check the username and database logs.")
            credentials = st.session_state.get("created_user_credentials")
            if credentials:
                st.info(f"Temporary password for {credentials['username']} (shown only to this administrator session):")
                st.code(credentials["password"])
                if st.button("Hide temporary password", key="hide_temp_password"):
                    st.session_state.pop("created_user_credentials", None)
                    st.rerun()
            st.markdown("**Existing accounts**")
            try:
                accounts = list_users()
                for account in accounts:
                    status = "Active" if account["active"] else "Disabled"
                    st.write(f"**{account['username']}** · {account['role']} · {status}")
                    if account["user_id"] == user["user_id"]:
                        continue
                    col1, col2, col3 = st.columns([1.5, 1, 1])
                    with col1:
                        st.caption("Password change required" if account["must_change_password"] else "Ready")
                    with col2:
                        if st.button("Reset password", key=f"reset_{account['user_id']}"):
                            try:
                                temp = reset_user_password(account["user_id"])
                                st.session_state["created_user_credentials"] = {"username": account["username"], "password": temp}
                                st.rerun()
                            except Exception:
                                st.error("Could not reset password. Check the database logs.")
                    with col3:
                        label = "Disable" if account["active"] else "Reactivate"
                        if st.button(label, key=f"toggle_{account['user_id']}"):
                            try:
                                set_user_active(account["user_id"], not account["active"], actor_user_id=user["user_id"])
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))
            except Exception:
                st.error("Could not load accounts. Check the database connection and permissions.")
