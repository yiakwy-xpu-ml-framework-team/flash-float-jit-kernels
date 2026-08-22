"""
opencode_client.py — Python REST client for opencode headless server.

Wraps the opencode HTTP API (session, message, command, event) for use by
the kernel agent orchestrator.  Designed to mirror the JS client in
wechat-opencode-bot/src/opencode.js.

Usage:
    from tools.opencode_client import OpenCodeClient

    client = OpenCodeClient("http://localhost:8096")
    session = client.create_session()
    client.send_prompt(session["id"], "write a vector add kernel")
    reply = client.wait_for_reply(session["id"])
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlencode

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level HTTP helpers (stdlib only, no requests dependency required)
# ---------------------------------------------------------------------------

def _request(
    method: str,
    url: str,
    body: Optional[dict] = None,
    timeout: float = 30.0,
) -> Any:
    """Send an HTTP request and return parsed JSON (or None)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode()
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode()[:500] if exc.fp else ""
        raise RuntimeError(
            f"opencode {method} {url} -> {exc.code} {err_body}"
        ) from exc


def _get(base: str, path: str, query: Optional[dict] = None, **kw: Any) -> Any:
    qs = f"?{urlencode(query)}" if query else ""
    return _request("GET", f"{base}{path}{qs}", **kw)


def _post(base: str, path: str, body: Optional[dict] = None, **kw: Any) -> Any:
    return _request("POST", f"{base}{path}", body=body, **kw)


def _delete(base: str, path: str, **kw: Any) -> Any:
    return _request("DELETE", f"{base}{path}", **kw)


def _patch(base: str, path: str, body: Optional[dict] = None, **kw: Any) -> Any:
    return _request("PATCH", f"{base}{path}", body=body, **kw)


# ---------------------------------------------------------------------------
# SSE event stream (blocking iterator)
# ---------------------------------------------------------------------------

def _sse_stream(url: str, timeout: float = 300.0) -> Iterator[dict]:
    """Yield SSE events from *url* as dicts with keys ``event`` and ``data``."""
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    event_type = ""
    data_lines: list[str] = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode().rstrip("\r\n")
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "":
            # End of event
            payload = "\n".join(data_lines)
            try:
                data = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                data = payload
            yield {"event": event_type or "message", "data": data}
            event_type = ""
            data_lines = []


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class OpenCodeClient:
    """High-level client for the opencode headless server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8096") -> None:
        self.base = base_url.rstrip("/")

    # -- health / config ----------------------------------------------------

    def ping(self) -> bool:
        """Return True if the server is reachable."""
        try:
            _get(self.base, "/config", timeout=5)
            return True
        except Exception:
            return False

    def get_config(self) -> dict:
        return _get(self.base, "/config")

    def get_model(self) -> Optional[dict]:
        """Return ``{providerID, modelID}`` from server config."""
        cfg = self.get_config()
        model_str = cfg.get("model", "")
        if isinstance(model_str, str) and "/" in model_str:
            prov, mid = model_str.split("/", 1)
            return {"providerID": prov, "modelID": mid}
        return None

    # -- sessions ------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        return _get(self.base, "/session") or []

    def create_session(self, title: Optional[str] = None) -> dict:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        return _post(self.base, "/session", body=body)

    def get_session(self, session_id: str) -> dict:
        return _get(self.base, f"/session/{session_id}")

    def delete_session(self, session_id: str) -> bool:
        return _delete(self.base, f"/session/{session_id}")

    def abort_session(self, session_id: str) -> bool:
        try:
            return _post(self.base, f"/session/{session_id}/abort")
        except Exception as exc:
            logger.warning("abort failed: %s", exc)
            return False

    def fork_session(self, session_id: str, message_id: Optional[str] = None) -> dict:
        body: dict[str, Any] = {}
        if message_id:
            body["messageID"] = message_id
        return _post(self.base, f"/session/{session_id}/fork", body=body)

    def share_session(self, session_id: str) -> dict:
        return _post(self.base, f"/session/{session_id}/share")

    # -- permissions (human-in-the-loop approval) -------------------------

    def list_permissions(self) -> list[dict]:
        """List ALL pending permission requests (across sessions).

        Each item: ``{id, sessionID, permission, patterns, metadata, always, tool}``.
        Filter by ``sessionID`` for the session you drive.
        """
        return _get(self.base, "/permission") or []

    def reply_permission(self, request_id: str, reply: str) -> Any:
        """Respond to a pending permission request.

        ``reply`` is one of ``"once"`` (approve this call), ``"always"``
        (approve this call and auto-approve identical ones) or ``"reject"``.
        """
        return _post(self.base, f"/permission/{request_id}/reply", body={"reply": reply})

    def revert_message(
        self, session_id: str, message_id: str, part_id: Optional[str] = None
    ) -> bool:
        body: dict[str, Any] = {"messageID": message_id}
        if part_id:
            body["partID"] = part_id
        return _post(self.base, f"/session/{session_id}/revert", body=body)

    def unrevert_session(self, session_id: str) -> bool:
        return _post(self.base, f"/session/{session_id}/unrevert")

    # -- messages ------------------------------------------------------------

    def list_messages(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        query = {"limit": limit} if limit else None
        return _get(self.base, f"/session/{session_id}/message", query=query) or []

    def get_message(self, session_id: str, message_id: str) -> dict:
        return _get(self.base, f"/session/{session_id}/message/{message_id}")

    def send_prompt(
        self,
        session_id: str,
        text: str,
        *,
        model: Optional[dict] = None,
        agent: Optional[str] = None,
        no_reply: bool = False,
    ) -> dict:
        """Send a prompt synchronously (waits for full response).

        Returns the assistant message dict ``{info, parts}``.
        """
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": text}],
        }
        if model is None:
            model = self.get_model()
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent
        if no_reply:
            body["noReply"] = True
        return _post(self.base, f"/session/{session_id}/message", body=body)

    def send_prompt_async(
        self,
        session_id: str,
        text: str,
        *,
        model: Optional[dict] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Send a prompt asynchronously (returns immediately with 204)."""
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": text}],
        }
        if model is None:
            model = self.get_model()
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent
        _post(self.base, f"/session/{session_id}/prompt_async", body=body)

    # -- slash commands ------------------------------------------------------

    def send_command(
        self,
        session_id: str,
        command: str,
        arguments: str = "",
        *,
        model: Optional[dict] = None,
        agent: Optional[str] = None,
        timeout: float = 300.0,
    ) -> dict:
        """Execute a slash command (e.g. ``/kernel-opt ...``).

        ``timeout`` (seconds) bounds the whole round trip — the agent loop can
        run for minutes, so the default 30s ``_request`` timeout is too short.
        """
        body: dict[str, Any] = {"command": command, "arguments": arguments}
        if model is None:
            model = self.get_model()
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent
        return _post(self.base, f"/session/{session_id}/command", body=body, timeout=timeout)

    # -- shell ----------------------------------------------------------------

    def run_shell(
        self,
        session_id: str,
        command: str,
        *,
        agent: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"command": command}
        if agent:
            body["agent"] = agent
        return _post(self.base, f"/session/{session_id}/shell", body=body)

    # -- file / search --------------------------------------------------------

    def find_text(self, pattern: str) -> list[dict]:
        return _get(self.base, "/find", query={"pattern": pattern}) or []

    def find_files(self, query_str: str, file_type: Optional[str] = None) -> list[str]:
        params: dict[str, str] = {"query": query_str}
        if file_type:
            params["type"] = file_type
        return _get(self.base, "/find/file", query=params) or []

    def read_file(self, path: str) -> dict:
        return _get(self.base, "/file/content", query={"path": path})

    def file_status(self) -> list[dict]:
        return _get(self.base, "/file/status") or []

    # -- events (SSE) ---------------------------------------------------------

    def subscribe_events(self) -> Iterator[dict]:
        """Yield SSE events from the server."""
        return _sse_stream(f"{self.base}/event")

    # -- agents ---------------------------------------------------------------

    def list_agents(self) -> list[dict]:
        return _get(self.base, "/agent") or []

    # -- commands -------------------------------------------------------------

    def list_commands(self) -> list[dict]:
        return _get(self.base, "/command") or []

    # -- diff -----------------------------------------------------------------

    def get_diff(self, session_id: str, message_id: Optional[str] = None) -> list[dict]:
        query = {"messageID": message_id} if message_id else None
        return _get(self.base, f"/session/{session_id}/diff", query=query) or []

    # -- waiting helpers ------------------------------------------------------

    @staticmethod
    def _extract_text(message: dict) -> str:
        """Extract plain text from a message envelope ``{info, parts}``."""
        parts = message.get("parts", [])
        return "\n".join(
            p.get("text", "")
            for p in parts
            if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored")
        )

    @staticmethod
    def _find_reply(messages: list[dict], user_message_id: Optional[str]) -> Optional[dict]:
        """Find the assistant message that replies to *user_message_id*."""
        if user_message_id:
            # Direct parent match
            for m in reversed(messages):
                info = m.get("info", m)
                if info.get("role") == "assistant" and info.get("parentID") == user_message_id:
                    return m
            # Positional match
            idx = next(
                (i for i, m in enumerate(messages)
                 if (m.get("info", m)).get("id") == user_message_id),
                -1,
            )
            if idx >= 0:
                for m in reversed(messages[idx + 1:]):
                    info = m.get("info", m)
                    if info.get("role") == "assistant" and not info.get("parentID"):
                        return m
            return None
        # No user message id — return last assistant
        for m in reversed(messages):
            info = m.get("info", m)
            if info.get("role") == "assistant":
                return m
        return None

    def wait_for_reply(
        self,
        session_id: str,
        user_message_id: Optional[str] = None,
        *,
        timeout_ms: int = 300_000,
        poll_interval: float = 1.5,
        poll_hook: Optional[Callable[[], None]] = None,
    ) -> str:
        """Poll until the assistant finishes replying. Returns the text.

        ``poll_hook`` (if given) runs before every poll — e.g. the
        orchestrator uses it to service pending permission requests while
        the agent is working.
        """
        start = time.monotonic()
        deadline = start + timeout_ms / 1000.0
        last_len = 0

        while time.monotonic() < deadline:
            if poll_hook is not None:
                poll_hook()
            messages = self.list_messages(session_id)
            assistant = self._find_reply(messages, user_message_id)
            if assistant:
                info = assistant.get("info", assistant)
                text = self._extract_text(assistant)
                if len(text) > last_len:
                    last_len = len(text)
                    logger.info("streaming… (%d chars)", len(text))
                finish = info.get("finish", "")
                if finish == "stop":
                    return text
                if finish == "error":
                    raise RuntimeError(f"opencode finished with error: {text[:500]}")
            time.sleep(poll_interval)

        raise TimeoutError(f"timeout waiting for reply ({timeout_ms}ms)")

    def send_and_wait(
        self,
        session_id: str,
        text: str,
        *,
        agent: Optional[str] = None,
        timeout_ms: int = 300_000,
        poll_hook: Optional[Callable[[], None]] = None,
    ) -> str:
        """Send prompt asynchronously, then wait for reply. Returns text."""
        before = len(self.list_messages(session_id))
        self.send_prompt_async(session_id, text, agent=agent)
        # Discover the user message id created by this prompt
        time.sleep(1.5)
        after = self.list_messages(session_id)
        new_msgs = after[before:]
        user_msg_id = None
        for m in reversed(new_msgs):
            info = m.get("info", m)
            if info.get("role") == "user":
                user_msg_id = info.get("id")
                break
        return self.wait_for_reply(
            session_id, user_msg_id, timeout_ms=timeout_ms, poll_hook=poll_hook
        )
