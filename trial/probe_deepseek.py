# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator Step 0: synthetic requests only; never accepts board input."""

import argparse
import json
import os
import stat
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def private_value(path: Path, *, deepseek_env: bool = False) -> str:
    """Read an operator-owned private file without displaying it."""
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("credential file must be owned by this user and mode 0600")
    value = path.read_text().strip()
    if deepseek_env and ("=" in value or "\n" in value):
        value = (
            dotenv_values(stream=StringIO(value), interpolate=False).get(
                "DEEPSEEK_API_KEY"
            )
            or ""
        )
    if not value or "\n" in value:
        raise ValueError("credential file must contain exactly one nonempty line")
    return value


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def request_body(model: str, case: str) -> dict[str, Any]:
    """Use the same bare model and disabled thinking as the trial client."""
    if not model or "/" in model:
        raise ValueError("a bare model id is required")
    body: dict[str, Any] = {
        "model": model,
        "thinking": {"type": "disabled"},
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": 'Return JSON {"ok":true}. For tools, call report with ok=true.',
            }
        ],
    }
    if case == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "report", "strict": True, "schema": SCHEMA},
        }
    elif case == "json_content":
        body["response_format"] = {"type": "json_object"}
    else:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "report",
                    "description": "Report the synthetic check",
                    "parameters": SCHEMA,
                },
            }
        ]
        body["tool_choice"] = (
            "auto"
            if case == "tools_auto"
            else {"type": "function", "function": {"name": "report"}}
        )
    return body


def check_response(data: dict[str, Any], case: str) -> None:
    """Reject reasoning/preamble and malformed or missing output."""
    choice = data["choices"][0]
    message = choice["message"]
    if choice.get("finish_reason") == "length" or message.get("reasoning_content"):
        raise ValueError("probe truncated or thinking was not disabled")
    if case.startswith("tools_"):
        calls = message.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "report":
            raise ValueError("probe did not return the requested tool")
        content = calls[0]["function"]["arguments"]
    else:
        content = message["content"]
    parsed = json.loads(content)
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"ok"}
        or parsed["ok"] is not True
    ):
        raise ValueError("probe did not return clean expected JSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    key = private_value(args.key_file, deepseek_env=True)
    schema_supported = True
    for case in ("json_schema", "tools_auto", "tools_forced", "json_content"):
        payload = request_body(args.model, case)
        request = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            # Strict chat schemas may not be served; tools MUST still pass.
            if case == "json_schema" and exc.code in {400, 422}:
                schema_supported = False
                print("json_schema: refused; function_calling fallback required")
                continue
            raise RuntimeError(f"{case}: HTTP {exc.code}") from None
        check_response(data, case)
        print(f"{case}: PASS")
    print(
        "Step 0 PASS; structured method="
        + ("json_schema" if schema_supported else "function_calling")
    )


if __name__ == "__main__":
    main()
