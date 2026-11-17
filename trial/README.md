<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# card #1641 — board-only DeepSeek trial v2

Fork base: NVIDIA/auto-ontology a630ebd97aa4b4410b2f41f5a8af06260af2f7b9.
This is a patch, not an installation/readiness or compilation receipt. The
builder must NOT start containers, restore real data, or run ontology compilation.
Lab runs the following steps; stop on any mandatory failure. No OpenCode Go endpoint,
header workaround, fund key, fund environment, or fund checkout is used.

## Step 0 — dedicated-key synthetic provider probe FIRST

Lab confirmed swarph's existing `/home/ubuntu/deepseek/.env`, mode0600,
variable DEEPSEEK_API_KEY (not OMEGA's credential). Do not echo/source it or pass
its value on the command line. Dotenv parsing disables variable interpolation.
Set a bare model id available on that account. The template uses `deepseek-flash`
with `thinking.type=disabled`; the probe must verify this before any board data.

```sh
uv run python -m trial.probe_deepseek --key-file /home/ubuntu/deepseek/.env --model deepseek-flash
```

This makes four SYNTHETIC requests: strict json_schema, tool choice auto,
forced named tool, and clean JSON content (no reasoning preamble). A strict
schema HTTP400/422 is reported as requiring the function-calling fallback,
not called a schema success. Both tool calls must pass. The json_content case
is informational when DEEPSEEK_STRUCTURED_METHOD=function_calling: application
callers do not use json_object. It remains mandatory in json_schema mode.
Lab initially measured JSON content passing, then malformed json_object content
three times (including a leaked special token); the probe now prints a sanitized
FAIL per case, continues all four, and exits nonzero for mandatory failures.
It never echoes response bodies or exception messages. Redirects are refused,
so credentials cannot follow a Location header. Therefore
DEEPSEEK_STRUCTURED_METHOD defaults function_calling, DEEPSEEK_THINKING defaults
disabled. Both are configurable; thinking enabled uses automatic tool choice.
With the optional json_schema method, runtime retries unsupported response_format/json_schema
using function_calling; authentication, quota and transport errors do not trigger
that fallback. No board data or arbitrary prompt input enters this script.
An exit0 is the operator's prerequisite for Step1, not permission to compile.
Current API reference: https://api-docs.deepseek.com/api/create-chat-completion/
and https://api-docs.deepseek.com/guides/tool_calls/ .

## Step 1 — prepare a fresh isolated sandbox

Copy `trial/board.env.example` to ignored `trial/board.env`, chmod0600, and fill
fresh sandbox Postgres/admin/auth secrets. Do not copy a production `.env`.
Unset inherited model/API variables when launching: the file must not specify
REASONING/NON_REASONING overrides or DEFAULT_MODELS_API_KEY; the mounted dedicated
key is resolved only at process runtime. Docker inspection exposes the key FILE
path, not the key. The image user must be able to read its0600 mount; lab must
check uid ownership before starting (do not broaden file permissions).

Compose2.24.4+ supports the port-replacement tags used here. Every command uses:

```sh
docker compose --env-file trial/board.env -f docker-compose.yml -f trial/compose.board.yaml config
```

Review config without sharing environment/credential values. Only frontend3000
and sandbox Postgres55432 publish, BOTH127.0.0.1. Backend3001 and ingestion3002
remain network-only. pgAdmin has a disabled profile AND no published port.
Ollama is network-only at `http://ollama:11434/v1`; its one-shot model service
pulls nomic-embed-text before ingestion/backend starts. Never rebind host Ollama.
Use a fresh Compose project/volumes (e.g. `-p ontology-board-trial`); never reuse
production volumes. Start sandbox Postgres alone first, NOT ingestion yet.

## Step 2 — operator-only restore (physical table isolation)

Post this exact allowlist on card1641 BEFORE any compile:

* board_cards (includes the JSON `stage_history` column; no separate history table)
* card_obligations
* board_projects
* card_obligation_events

NO claude_messages, claude_threads, peer_tokens, or any other table is copied.
The newest UTC-stamped `claude_mesh.*.db.gz` under `/home/ubuntu/backups/mesh`
is selected; lab observed `claude_mesh.20261117T031926Z.db.gz` at build time.
If MESH_BACKUP_DIR was overridden, supply `--backup-dir` explicitly. Lab must set
TMPDIR to a private directory on disk, not the shared tmpfs `/tmp`, before the
restore: the extracted backup includes all source data until the allowlist copy.
The full gzip is extracted only in a0700 temporary directory, snapshot0600,
opened read-only, then deleted on success, Python failure, SIGINT, SIGTERM and
SIGHUP. SIGKILL/power loss cannot unwind cleanup: inspect the private TMPDIR
and remove leftover snapshots manually before retrying. Only fixed-table SELECTs
are issued. SQLite schema/scalars are mapped to PostgreSQL without copying
SQLite SQL, triggers, credentials, message tables or sequences. JSON/history
and timestamp values remain text. This is an analytical snapshot, not a live
mesh replica; source constraints/indexes are intentionally not replayed.

Set libpq PGHOST=127.0.0.1, PGPORT=55432, PGDATABASE=auto_ontology, PGUSER and
PGPASSWORD locally (not command-line values or logs). PGHOSTADDR, when present,
must be exactly127.0.0.1 or::1. The script pins the connection target explicitly
and checks libpq's actual connected address/port (not just the hostname);
PGSERVICE cannot change it. This works with Docker port forwarding, where
inet_server_addr() inside Postgres need not be a loopback address. Create a separate0600
reader-password file, then:

```sh
uv run python -m trial.restore_board --reader-password-file /path/to/private/board-reader-password --confirm-new-sandbox
```

The script prints its table list BEFORE copying. It refuses non-localhost,
non-sandbox admin DBs and existing `auto_ontology_board` DB/`board_reader` role;
it NEVER overwrites or drops either. Host port55432 is required by default;
if changing the Compose mapping, explicitly pass matching --sandbox-port.
A failure after mutation can leave a partial sandbox DB/role or CONNECT revokes:
inspect it manually, do not rerun into existing objects.
Only `board_reader` SELECT grants on the four tables, schema USAGE and database
CONNECT are added. PUBLIC CONNECT is revoked on the sandbox's auto_ontology and
postgres databases, so the reader cannot enter them; this intentionally affects
PUBLIC grants in this fresh sandbox ONLY. Role has default_transaction_read_only=on
and statement_timeout=10s. The timeout is a session default, not an unchangeable
hard cap; table grants provide write isolation. Do not use an admin URL for answering. Configure ONLY
this database in CONNECTION_STRINGS with the reader's URL-encoded password,
in-network host `postgres:5432/auto_ontology_board`.

## Step 3 — lab-only startup and compilation

After Step0 passes, restore completes, the allowlist is posted, and the commander
confirms the dedicated credential, lab can start the remaining services. Check
source connection and embeddings before compiling. Distinct-value probing and
PII detection are hard OFF in both services even if a saved UI toggle says ON;
semantic-FK source queries are also prevented. Bounded row sampling still exists:
the physically isolated board snapshot is the ONLY permitted source.

EMBED_DIMENSIONS defaults2048 in shared PostgresVDB construction; this trial sets
768 for nomic-embed-text. Never reuse existing2048-dimensional VDB tables. All
PGVectorStore objects come through that constructor. Other2048 code hits:
server/chat/router.py and retrieval/interactive/coordinator.py are chat token
budgets; postgres.py's halfvec/HNSW docstrings describe the original model/index,
not extra dimension settings. Nomic prefixes are search_query:/search_document:
(not NVIDIA's query:/passage:). Ingest AND query must use the same model/prefix.

Reranking has no deployed service and points at loopback dead port9. Bridge-owner
selection during compilation can call it, catches the error and falls back to
the first target Term (semantic/bridge_tables.py); this may reduce accuracy and
has NOT been verified by a compile run. Retrieval paths needing it also remain
UNVERIFIED and will fail until lab explicitly configures a supported reranker. Do not silently send
rerank traffic/keys to an unrelated vendor. This trial does not claim end-to-end
query readiness. Lab records compile tokens, DeepSeek balance before/after, and
10 real board questions with expected SQL and swarph comparisons separately.
