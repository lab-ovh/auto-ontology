# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic regressions for card #1641 v2; never talks to a live service."""

import io
import json
import os
import signal
import subprocess
import sys
import socket
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trial import probe_deepseek as probe
from trial import restore_board as board


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(sock: socket.socket, address: object) -> None:
        raise AssertionError("offline trial tests must not connect to a service")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.mark.parametrize(
    "method,expected_exit", [("function_calling", 0), ("json_schema", 1)]
)
@pytest.mark.parametrize("failed_case", ["json_content", "tools_auto"])
def test_malformed_json_case_reports_and_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
    expected_exit: int,
    failed_case: str,
) -> None:
    credential = tmp_path / "key"
    credential.write_text("synthetic")
    credential.chmod(0o600)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--key-file", str(credential), "--model", "deepseek-flash"],
    )
    monkeypatch.setenv("DEEPSEEK_STRUCTURED_METHOD", method)
    calls = []

    def response(
        opener: urllib.request.OpenerDirector,
        request: urllib.request.Request,
        *args: object,
        **kwargs: object,
    ) -> io.BytesIO:
        body = json.loads(request.data)
        calls.append(body)
        message = {"content": '{"ok":true}'}
        if "tools" in body:
            message = {
                "tool_calls": [
                    {"function": {"name": "report", "arguments": '{"ok":true}'}}
                ]
            }
            if failed_case == "tools_auto" and body["tool_choice"] == "auto":
                message["tool_calls"][0]["function"]["arguments"] = (
                    '{"ok":tru<|endoftext|>'
                )
        elif (
            failed_case == "json_content"
            and body["response_format"]["type"] == "json_object"
        ):
            message = {"content": '{"ok":tru<|endoftext|>'}
        return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", response)
    try:
        probe.main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    assert code == (1 if failed_case == "tools_auto" else expected_exit)
    assert len(calls) == 4
    output = capsys.readouterr()
    assert f"{failed_case}: FAIL" in output.out
    assert ("informational" in output.out) == (method == "function_calling")
    assert ("Step 0 PASS" in output.out) == (
        method == "function_calling" and failed_case == "json_content"
    )
    assert "Traceback" not in output.err
    assert "endoftext" not in output.out


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_never_forwards_authorization(
    code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener = urllib.request.build_opener(probe.NoRedirect())
    target = MagicMock()
    monkeypatch.setattr(opener, "open", target)
    headers = Message()
    headers["Location"] = "https://not-the-provider.example/redirect"
    request = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=b"{}",
        headers={"Authorization": "Bearer synthetic"},
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        opener.error("http", request, io.BytesIO(), code, "redirect", headers)
    assert error.value.code == code
    target.assert_not_called()


@pytest.fixture
def restore_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    (tmp_path / "claude_mesh.20261117T031926Z.db.gz").touch()
    password = tmp_path / "password"
    password.write_text("synthetic-password")
    password.chmod(0o600)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore",
            "--backup-dir",
            str(tmp_path),
            "--reader-password-file",
            str(password),
            "--confirm-new-sandbox",
        ],
    )
    for name, value in {
        "PGHOST": "localhost",
        "PGHOSTADDR": "127.0.0.1",
        "PGPORT": "55432",
        "PGDATABASE": "auto_ontology",
    }.items():
        monkeypatch.setenv(name, value)
    connect = MagicMock()
    connect.return_value.__enter__.return_value.info = SimpleNamespace(
        host="localhost", hostaddr="127.0.0.1", port=55432, dbname="auto_ontology"
    )
    monkeypatch.setattr(board.psycopg, "connect", connect)
    return connect


@pytest.mark.parametrize(
    "field,value",
    [
        ("PGHOSTADDR", "127.0.0.2"),
        ("PGHOSTADDR", "203.0.113.1"),
        ("PGHOST", "other-host"),
        ("PGHOST", "/tmp"),
        ("PGPORT", "5432"),
        ("PGDATABASE", "postgres"),
    ],
)
def test_bad_restore_target_refused_before_connection(
    restore_cli: MagicMock, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match="sandbox"):
        board.main()
    restore_cli.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hostaddr", "127.0.0.2"),
        ("hostaddr", "203.0.113.1"),
        ("port", 5432),
        ("dbname", "postgres"),
    ],
)
def test_connected_server_checked_before_restore(
    restore_cli: MagicMock, field: str, value: object
) -> None:
    admin = restore_cli.return_value.__enter__.return_value
    setattr(admin.info, field, value)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(ValueError, match="sandbox"):
        board.main()
    admin.execute.assert_not_called()
    assert restore_cli.call_args.kwargs["hostaddr"] == "127.0.0.1"
    assert restore_cli.call_args.kwargs["port"] == 55432
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_restore_signal_removes_snapshot(tmp_path: Path, signum: int) -> None:
    import gzip

    backup = tmp_path / "claude_mesh.20261117T031926Z.db.gz"
    with gzip.open(backup, "wb") as output:
        output.write(b"synthetic SQLite fixture")
    password = tmp_path / "password"
    password.write_text("synthetic-password")
    password.chmod(0o600)
    temporary = tmp_path / "disk-temp"
    temporary.mkdir()
    harness = """
import os, signal, sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from trial import restore_board as board
admin = MagicMock()
admin.info = SimpleNamespace(host="127.0.0.1", hostaddr="127.0.0.1", port=55432, dbname="auto_ontology")
board.psycopg.connect = lambda **kwargs: admin
admin.__enter__.return_value = admin
signum = int(sys.argv[1])
def interrupt(source, output):
    output.write(b"synthetic snapshot")
    output.flush()
    os.kill(os.getpid(), signum)
board.shutil.copyfileobj = interrupt
sys.argv = ["restore", "--backup-dir", sys.argv[2], "--reader-password-file", sys.argv[3], "--confirm-new-sandbox"]
board.main()
"""
    env = dict(
        os.environ,
        TMPDIR=str(temporary),
        PGHOST="127.0.0.1",
        PGHOSTADDR="127.0.0.1",
        PGPORT="55432",
        PGDATABASE="auto_ontology",
    )
    result = subprocess.run(
        [sys.executable, "-c", harness, str(signum), str(tmp_path), str(password)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 128 + signum, result.stderr
    assert list(temporary.iterdir()) == []
