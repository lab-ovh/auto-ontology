# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline trial checks: synthetic data only, no service startup or compile."""

import json
import sqlite3
import socket
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from auto_ontology.infra import feature_flags
from auto_ontology.utils import embedding, llm_invoke, model_config
from auto_ontology.vdb.postgres import PostgresVDB
from trial import probe_deepseek as probe
from trial import restore_board as board


class Result(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(sock: socket.socket, address: object) -> None:
        raise AssertionError("offline trial tests must not connect to a service")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_deepseek_real_wire_bare_model_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.host == "api.deepseek.com"
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        if "response_format" in body:
            assert body["response_format"]["json_schema"]["strict"] is True
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "response_format json_schema unavailable now",
                        "type": "invalid_request_error",
                    }
                },
            )
        assert body["tool_choice"]["function"]["name"] == "Result"
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Result",
                                        "arguments": '{"ok":true}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    monkeypatch.setenv("DEEPSEEK_STRUCTURED_METHOD", "json_schema")
    monkeypatch.setattr(llm_invoke, "RETRY_MAX_ATTEMPTS", 1)
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
    client = llm_invoke._build_client(
        model="openai/deepseek-flash",
        api_key="synthetic",
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=128,
    )
    assert isinstance(client, ChatOpenAI)
    client.http_client = httpx.Client(transport=httpx.MockTransport(handler))
    # Rebuild the SDK on the same explicit mock transport, never a live network.
    from openai import OpenAI

    client.root_client = OpenAI(
        api_key="synthetic",
        base_url="https://api.deepseek.com/v1",
        http_client=client.http_client,
    )
    client.client = client.root_client.chat.completions
    assert llm_invoke.safe_invoke_with_structured_output(
        client, [HumanMessage(content="synthetic")], Result
    ).ok
    assert len(calls) == 2


def test_default_method_and_configurable_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_STRUCTURED_METHOD", raising=False)
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    client = llm_invoke._build_client(
        model="deepseek-flash",
        api_key="synthetic",
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=128,
    )
    assert client.extra_body == {"thinking": {"type": "enabled"}}
    assert llm_invoke._structured_output_kwargs(client) == {
        "method": "function_calling",
        "tool_choice": None,
    }
    monkeypatch.setenv("DEEPSEEK_THINKING", "invalid")
    with pytest.raises(ValueError):
        llm_invoke._build_client(
            model="deepseek-flash",
            api_key="synthetic",
            base_url="https://api.deepseek.com/v1",
            temperature=0,
            max_tokens=128,
        )


@pytest.mark.parametrize(
    "status,detail",
    [(401, "json_schema"), (429, "json_schema"), (400, "invalid model")],
)
def test_no_schema_fallback_on_other_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, detail: str
) -> None:
    monkeypatch.setenv("DEEPSEEK_STRUCTURED_METHOD", "json_schema")
    monkeypatch.setattr(llm_invoke, "RETRY_MAX_ATTEMPTS", 1)
    error = RuntimeError(detail)
    error.status_code = status
    client = MagicMock(
        openai_api_base="https://api.deepseek.com/v1", model_name="deepseek-flash"
    )
    client.with_structured_output.return_value.invoke.side_effect = error
    with pytest.raises(RuntimeError):
        llm_invoke.safe_invoke_with_structured_output(
            client, [HumanMessage(content="synthetic")], Result
        )
    assert client.with_structured_output.call_count == 1


@pytest.mark.parametrize("value,expected", [(None, 2048), ("768", 768)])
def test_dimensions_reach_shared_table_factory(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    if value is None:
        monkeypatch.delenv("EMBED_DIMENSIONS", raising=False)
    else:
        monkeypatch.setenv("EMBED_DIMENSIONS", value)
    store = PostgresVDB(connection_string="postgresql://synthetic")
    engine = MagicMock()
    monkeypatch.setattr(store, "_get_engine", lambda: engine)
    monkeypatch.setattr(store, "_table_exists", lambda: False)
    monkeypatch.setattr(store, "_ensure_schema", lambda: None)
    monkeypatch.setattr(store, "_create_vector_index", lambda: None)
    factory = MagicMock()
    monkeypatch.setattr("auto_ontology.vdb.postgres.PGVectorStore.create_sync", factory)
    store._get_store()
    assert engine.init_vectorstore_table.call_args.kwargs["vector_size"] == expected
    factory.assert_called_once()


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_dimensions_refused(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EMBED_DIMENSIONS", value)
    with pytest.raises(ValueError):
        PostgresVDB(connection_string="postgresql://synthetic")


def test_nomic_prefix_and_existing_prefixes() -> None:
    assert (
        embedding._text_prefix("nomic-embed-text:latest", "query") == "search_query: "
    )
    assert embedding._text_prefix("nomic-embed-text", "passage") == "search_document: "
    assert embedding._text_prefix("nemotron-3.5-embed-1b-nvfp4", "query") == "query: "
    assert embedding._text_prefix("other", "query") == ""


def test_privacy_off_even_if_saved_toggle_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTINCT_VALUE_PROBING_ENABLED", "false")
    monkeypatch.setenv("PII_DETECTION_ENABLED", "false")
    reader = MagicMock(side_effect=AssertionError("must not read saved toggle"))
    monkeypatch.setattr(feature_flags, "read_configuration_flag", reader)
    assert not feature_flags.is_distinct_value_probing_enabled()
    assert not feature_flags.is_pii_detection_enabled()
    reader.assert_not_called()


def test_runtime_env_key_not_sourced_or_interpolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = tmp_path / "private.env"
    credential.write_text(
        "DEEPSEEK_API_KEY='synthetic-${HOME}-$(false)'\nOTHER=value\n"
    )
    credential.chmod(0o600)
    monkeypatch.delenv("REASONING_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_MODELS_API_KEY_FILE", str(credential))
    assert model_config.resolve("REASONING", "API_KEY") == "synthetic-${HOME}-$(false)"
    assert (
        probe.private_value(credential, deepseek_env=True)
        == "synthetic-${HOME}-$(false)"
    )
    credential.chmod(0o644)
    with pytest.raises(ValueError):
        probe.private_value(credential, deepseek_env=True)


@pytest.mark.parametrize(
    "case", ["json_schema", "tools_auto", "tools_forced", "json_content"]
)
def test_synthetic_probe(case: str) -> None:
    body = probe.request_body("deepseek-flash", case)
    assert body["thinking"] == {"type": "disabled"}
    assert "board" not in json.dumps(body)
    message = {"content": '{"ok":true}'}
    if case.startswith("tools_"):
        message = {
            "tool_calls": [{"function": {"name": "report", "arguments": '{"ok":true}'}}]
        }
    probe.check_response({"choices": [{"message": message}]}, case)
    message["reasoning_content"] = "preamble"
    with pytest.raises(ValueError):
        probe.check_response({"choices": [{"message": message}]}, case)
    with pytest.raises(ValueError):
        probe.request_body("openai/deepseek-flash", case)


def test_restore_exact_allowlist_and_read_only_role() -> None:
    assert board.TABLES == (
        "board_cards",
        "card_obligations",
        "board_projects",
        "card_obligation_events",
    )
    source = sqlite3.connect(":memory:")
    for table in (*board.TABLES, "claude_messages", "peer_tokens"):
        source.execute(f'CREATE TABLE "{table}" (id INTEGER, stage_history TEXT)')
        source.execute(f'INSERT INTO "{table}" VALUES (1, ?) ', ('[{"stage":"test"}]',))
    observed = []
    source.set_trace_callback(observed.append)
    admin = MagicMock()
    admin.execute.return_value.fetchone.return_value = None
    admin.info.get_parameters.return_value = {
        "host": "127.0.0.1",
        "dbname": "auto_ontology",
    }
    destination = MagicMock()
    destination.execute.return_value.fetchone.return_value = (1,)
    from unittest.mock import patch

    with patch.object(board.psycopg, "connect") as connect:
        connect.return_value.__enter__.return_value = destination
        board.restore(source, admin, "synthetic-password")
    assert not any(
        "claude_messages" in query or "peer_tokens" in query for query in observed
    )
    statements = "\n".join(
        str(call.args[0])
        for call in admin.execute.call_args_list + destination.execute.call_args_list
    )
    assert "statement_timeout = '10s'" in statements
    assert "default_transaction_read_only = on" in statements
    assert "GRANT SELECT" in statements
    assert "GRANT ALL" not in statements
    admin_sql = [
        call.args[0].as_string()
        for call in admin.execute.call_args_list
        if hasattr(call.args[0], "as_string")
    ]
    assert 'REVOKE CONNECT ON DATABASE "auto_ontology" FROM PUBLIC' in admin_sql
    assert 'REVOKE CONNECT ON DATABASE "postgres" FROM PUBLIC' in admin_sql
    copied = destination.cursor.return_value.copy.return_value.__enter__.return_value.write_row
    assert copied.call_count == 4
    assert copied.call_args.args[0] == (1, '[{"stage":"test"}]')
    admin.execute.return_value.fetchone.return_value = (1,)
    with pytest.raises(ValueError, match="already exists"):
        board.restore(source, admin, "synthetic-password")


def test_newest_backup_name_not_mtime(tmp_path: Path) -> None:
    for name in (
        "claude_mesh.20261116T031700Z.db.gz",
        "claude_mesh.20261117T031926Z.db.gz",
        "other.db.gz",
    ):
        (tmp_path / name).touch()
    assert board.newest_backup(tmp_path).name == "claude_mesh.20261117T031926Z.db.gz"


@pytest.mark.parametrize(
    "content",
    [
        '{"ok":1}',
        '{"ok":true,"extra":1}',
        'prefix {"ok":true}',
        '```json\n{"ok":true}\n```',
        "",
    ],
)
def test_probe_rejects_unclean_json(content: str) -> None:
    with pytest.raises(ValueError):
        probe.check_response(
            {"choices": [{"message": {"content": content}}]}, "json_content"
        )


def test_missing_allowlist_table_refuses_before_create() -> None:
    with sqlite3.connect(":memory:") as source:
        admin = MagicMock()
        with pytest.raises(ValueError, match="missing"):
            board.restore(source, admin, "synthetic")
        admin.execute.assert_not_called()


def test_probe_cli_offline_four_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import sys
    import urllib.error

    credential = tmp_path / "private.env"
    credential.write_text("DEEPSEEK_API_KEY=synthetic\n")
    credential.chmod(0o600)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--key-file", str(credential), "--model", "deepseek-flash"],
    )
    calls = []

    def open_mock(
        opener: urllib.request.OpenerDirector,
        request: urllib.request.Request,
        timeout: int,
    ) -> io.BytesIO:
        body = json.loads(request.data)
        calls.append(body)
        if (
            "response_format" in body
            and body["response_format"]["type"] == "json_schema"
        ):
            raise urllib.error.HTTPError(request.full_url, 400, "unavailable", {}, None)
        message = {"content": '{"ok":true}'}
        if "tools" in body:
            message = {
                "tool_calls": [
                    {"function": {"name": "report", "arguments": '{"ok":true}'}}
                ]
            }
        return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr(probe.urllib.request.OpenerDirector, "open", open_mock)
    probe.main()
    assert len(calls) == 4
    output = capsys.readouterr().out
    assert "fallback required" in output and "Step 0 PASS" in output
    assert "synthetic" not in output


def test_real_compose_render_has_no_public_or_internal_service_ports() -> None:
    import os

    env = dict(os.environ, TRIAL_ENV_FILE="trial/board.env.example")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "trial/board.env.example",
            "-f",
            "docker-compose.yml",
            "-f",
            "trial/compose.board.yaml",
            "--profile",
            "disabled-admin",
            "config",
            "--format",
            "json",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    services = json.loads(result.stdout)["services"]
    ports = [
        (name, port)
        for name, service in services.items()
        for port in service.get("ports", [])
    ]
    assert {name for name, _ in ports} == {"postgres", "auto-ontology-frontend"}
    assert all(port["host_ip"] == "127.0.0.1" for _, port in ports)
    assert "disabled-admin" in services["pgadmin"]["profiles"]
    for name in ("auto-ontology", "ingestion-service"):
        assert services[name]["environment"]["EMBED_DIMENSIONS"] == "768"
        assert services[name]["environment"]["PII_DETECTION_ENABLED"] == "false"
        assert (
            services[name]["environment"]["DISTINCT_VALUE_PROBING_ENABLED"] == "false"
        )
        assert services[name]["environment"]["POSTGRES_HOST"] == "postgres"
