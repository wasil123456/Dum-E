"""SQLite-backed authentication helpers for the DUM-E API."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7


@dataclass(frozen=True)
class AuthUser:
    """Authenticated user identity."""

    id: int
    email: str
    name: str


def initialize_auth_database(database_path: Path) -> None:
    """Create the users and sessions tables if they do not exist."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "name" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN name TEXT NOT NULL DEFAULT 'DUM-E learner'"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_expires_at ON sessions(expires_at)"
            )


def validate_credentials(email: str, password: str) -> tuple[str, str]:
    """Validate and normalize registration credentials."""
    normalized_email = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized_email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    return normalized_email, password


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        salt.hex(),
        digest.hex(),
    )


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_user(database_path: Path, name: str, email: str, password: str) -> AuthUser:
    """Create a user and return its identity."""
    normalized_email, valid_password = validate_credentials(email, password)
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Name is required.")
    if len(normalized_name) > 120:
        raise ValueError("Name must be 120 characters or fewer.")
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash, created_at)
                    VALUES (?, ?, ?, datetime('now'))
                    """,
                    (
                        normalized_name,
                        normalized_email,
                        _hash_password(valid_password),
                    ),
                )
                return AuthUser(
                    id=int(cursor.lastrowid),
                    email=normalized_email,
                    name=normalized_name,
                )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email already exists.") from exc


def authenticate_user(
    database_path: Path, email: str, password: str
) -> AuthUser | None:
    """Return a user for valid credentials, otherwise None."""
    normalized_email = email.strip().lower()
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT id, email, name, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None or not _verify_password(password, row[3]):
        return None
    return AuthUser(id=int(row[0]), email=str(row[1]), name=str(row[2]))


def create_session(database_path: Path, user: AuthUser) -> str:
    """Create an expiring opaque bearer token for a user."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM sessions
                WHERE expires_at <= ? OR user_id = ?
                """,
                (time.time(), user.id),
            )
            connection.execute(
                """
                INSERT INTO sessions (token_hash, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (token_hash, user.id, time.time() + SESSION_TTL_SECONDS),
            )
    return token


def get_user_for_token(database_path: Path, token: str) -> AuthUser | None:
    """Resolve a bearer token to a user and remove expired sessions."""
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (time.time(),)
            )
            row = connection.execute(
                """
                SELECT users.id, users.email, users.name
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
                """,
                (token_hash, time.time()),
            ).fetchone()
    if row is None:
        return None
    return AuthUser(id=int(row[0]), email=str(row[1]), name=str(row[2]))
