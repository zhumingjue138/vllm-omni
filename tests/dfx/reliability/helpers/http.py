"""HTTP client helpers for reliability tests."""

from __future__ import annotations

import http.client
import json
from typing import Any


def post_chat_completions_raw(
    host: str,
    port: int,
    body: bytes | str,
    *,
    content_type: str = "application/json",
    timeout_sec: int = 120,
) -> tuple[int, bytes]:
    """POST /v1/chat/completions with raw bytes; returns (status, response_body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        headers = {"Content-Type": content_type}
        payload = body.encode("utf-8") if isinstance(body, str) else body
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def get_health_raw(host: str, port: int, *, timeout_sec: int = 20) -> tuple[int, bytes]:
    """GET /health with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def post_json_raw_http_client(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    """POST JSON to one endpoint with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def post_json_raw(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    """POST JSON to one endpoint; returns (status, body)."""
    return (
        post_chat_completions_raw(
            host,
            port,
            json.dumps(payload),
            content_type="application/json",
            timeout_sec=timeout_sec,
        )
        if path == "/v1/chat/completions"
        else post_json_raw_http_client(
            host,
            port,
            path,
            payload,
            timeout_sec=timeout_sec,
        )
    )


def extract_openai_error_contract_from_bytes(response_body: bytes) -> dict[str, Any] | None:
    """Best-effort parse OpenAI-style error object from raw response bytes."""
    try:
        payload = json.loads(response_body.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    return extract_openai_error_contract_from_payload(payload)


def extract_openai_error_contract_from_payload(payload: Any) -> dict[str, Any] | None:
    """Best-effort parse OpenAI-style error object from decoded JSON payload."""
    if not isinstance(payload, dict):
        return None
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return None
    if not isinstance(error_obj.get("message"), str):
        return None
    return error_obj
