# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-only board snapshot import into a NEW sandbox database, never the mesh."""

import argparse
import gzip
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import psycopg
from psycopg import sql

from trial.probe_deepseek import private_value

TABLES = ("board_cards", "card_obligations", "board_projects", "card_obligation_events")
DATABASE = "auto_ontology_board"
READER = "board_reader"


def newest_backup(directory: Path) -> Path:
    """UTC-stamped backup names sort chronologically, independent of copy mtimes."""
    backups = sorted(directory.glob("claude_mesh.*.db.gz"))
    if not backups:
        raise ValueError("no mesh SQLite backup found")
    return backups[-1]


def column_type(declaration: str) -> str:
    """Preserve SQLite's scalar types; JSON/stage history remain TEXT."""
    declaration = declaration.upper()
    if "INT" in declaration:
        return "BIGINT"
    if any(name in declaration for name in ("REAL", "FLOA", "DOUB")):
        return "DOUBLE PRECISION"
    if "BLOB" in declaration:
        return "BYTEA"
    return "TEXT"


def restore(
    source: sqlite3.Connection, admin: psycopg.Connection, password: str
) -> None:
    """Create only the fixed trial DB, refusing existing DBs/roles rather than overwriting."""
    definitions = {
        table: source.execute(f'PRAGMA table_info("{table}")').fetchall()
        for table in TABLES
    }
    if any(not columns for columns in definitions.values()):
        raise ValueError("backup is missing an allowlisted table")
    if admin.execute(
        "SELECT 1 FROM pg_database WHERE datname=%s", (DATABASE,)
    ).fetchone():
        raise ValueError("trial database already exists; refusing to overwrite it")
    if admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (READER,)).fetchone():
        raise ValueError("trial reader role already exists; refusing to modify it")
    admin.execute(
        sql.SQL(
            "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT"
        ).format(sql.Identifier(READER), sql.Literal(password))
    )
    admin.execute(
        sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
            sql.Identifier(READER)
        )
    )
    admin.execute(
        sql.SQL("ALTER ROLE {} SET statement_timeout = '10s'").format(
            sql.Identifier(READER)
        )
    )
    admin.execute(
        sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(
            sql.Identifier(DATABASE)
        )
    )
    connection = dict(admin.info.get_parameters())
    connection["dbname"] = DATABASE
    # libpq parameters omit passwords; retain the operator's PG environment.
    with psycopg.connect(**connection) as destination:
        destination.execute(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(
                sql.Identifier(DATABASE)
            )
        )
        destination.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        for table, columns in definitions.items():
            fields = [
                sql.SQL("{} {}").format(
                    sql.Identifier(c[1]), sql.SQL(column_type(c[2]))
                )
                for c in columns
            ]
            destination.execute(
                sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(table), sql.SQL(", ").join(fields)
                )
            )
            names = sql.SQL(", ").join(sql.Identifier(c[1]) for c in columns)
            with destination.cursor().copy(
                sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(table), names)
            ) as copy:
                for row in source.execute(f'SELECT * FROM "{table}"'):
                    copy.write_row(row)
            count = destination.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
            ).fetchone()[0]
            print(f"{table}: {count} rows")
        destination.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(DATABASE), sql.Identifier(READER)
            )
        )
        destination.execute(
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(READER))
        )
        destination.execute(
            sql.SQL("GRANT SELECT ON {} TO {}").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in TABLES),
                sql.Identifier(READER),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("/home/ubuntu/backups/mesh")
    )
    parser.add_argument("--reader-password-file", type=Path, required=True)
    parser.add_argument("--confirm-new-sandbox", action="store_true", required=True)
    args = parser.parse_args()
    print("ONLY tables: " + ", ".join(TABLES), flush=True)
    backup = newest_backup(args.backup_dir)
    print("Source: " + backup.name, flush=True)
    password = private_value(args.reader_password_file)
    # libpq PGHOST/PGPORT/PGUSER/PGPASSWORD address ONLY sandbox Postgres.
    with psycopg.connect(autocommit=True) as admin:
        if admin.info.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("restore requires sandbox Postgres on localhost")
        if admin.info.dbname != "auto_ontology":
            raise ValueError(
                "admin connection must address sandbox auto_ontology database"
            )
        with tempfile.TemporaryDirectory(prefix="ontology-board-") as temporary:
            snapshot = Path(temporary) / "snapshot.db"
            with gzip.open(backup, "rb") as compressed, snapshot.open("wb") as output:
                snapshot.chmod(0o600)
                shutil.copyfileobj(compressed, output)
            with closing(
                sqlite3.connect(f"file:{snapshot}?mode=ro&immutable=1", uri=True)
            ) as source:
                restore(source, admin, password)
    print(
        "Created auto_ontology_board; board_reader SELECT only, statement_timeout=10s"
    )


if __name__ == "__main__":
    main()
