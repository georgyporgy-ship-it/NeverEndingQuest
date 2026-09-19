"""Official Codex app-server adapter for ChatGPT-authenticated inference.

The adapter deliberately does not inspect Codex credential files or handle
OAuth tokens.  A persistent ``codex app-server`` child owns authentication,
credential persistence, refresh, model discovery, and inference.  Each game
request uses a fresh thread that is deleted after completion so
NeverEndingQuest's message list is the only conversation history seen by the
model.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


DEFAULT_TIMEOUT = 120.0
STARTUP_TIMEOUT = 15.0


class CodexError(RuntimeError):
    """Base error for the Codex provider."""


class CodexUnavailableError(CodexError):
    """The Codex CLI or app-server process is unavailable."""


class CodexAuthenticationError(CodexError):
    """No ChatGPT-managed Codex account is authenticated."""


class CodexProtocolError(CodexError):
    """The app-server returned an invalid or failed protocol response."""


class CodexTimeoutError(CodexError):
    """The app-server did not finish before the request deadline."""


class CodexTransportError(CodexError):
    """The persistent app-server process exited or disconnected."""


def _positive_timeout(value: Any, default: float = DEFAULT_TIMEOUT) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    for attr in ("read", "timeout"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return default


class CodexAppServer:
    """Small synchronous JSONL client around one persistent app-server child."""

    def __init__(self, binary: Optional[str] = None) -> None:
        self.binary = binary or os.environ.get("CODEX_BINARY") or "codex"
        self._process: Optional[subprocess.Popen[str]] = None
        self._pending: Dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._notifications: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stderr = deque(maxlen=80)
        self._stderr_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._next_id = 0
        self._generation = 0
        atexit.register(self.stop)

    @property
    def installed(self) -> bool:
        if os.path.sep in self.binary or (os.path.altsep and os.path.altsep in self.binary):
            return os.path.isfile(self.binary)
        return shutil.which(self.binary) is not None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.installed:
                raise CodexUnavailableError(
                    "Codex CLI was not found. Install it or set CODEX_BINARY to its path."
                )
            self.stop()
            try:
                process = subprocess.Popen(
                    [self.binary, "app-server", "--listen", "stdio://"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except (OSError, ValueError) as exc:
                raise CodexUnavailableError(f"Could not start Codex app-server: {exc}") from exc
            self._process = process
            self._generation += 1
            generation = self._generation
            threading.Thread(
                target=self._read_stdout,
                args=(process, generation),
                name="neq-codex-stdout",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_stderr,
                args=(process, generation),
                name="neq-codex-stderr",
                daemon=True,
            ).start()
            try:
                self.request(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "neverendingquest",
                            "title": "NeverEndingQuest",
                            "version": "1.0",
                        }
                    },
                    timeout=STARTUP_TIMEOUT,
                    ensure_started=False,
                )
                self.notify("initialized", {}, ensure_started=False)
            except Exception:
                self.stop()
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            process = self._process
            self._process = None
            self._generation += 1
            if process is not None:
                try:
                    if process.stdin:
                        process.stdin.close()
                except OSError:
                    pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
            self._fail_pending(CodexTransportError("Codex app-server stopped"))
            self._drain_notifications()

    def restart(self) -> None:
        self.stop()
        self.start()

    def _read_stdout(self, process: subprocess.Popen[str], generation: int) -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                if generation != self._generation:
                    return
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    error = CodexProtocolError(f"Malformed app-server JSON: {exc}")
                    self._fail_pending(error)
                    self._notifications.put({"_exception": error})
                    continue
                if not isinstance(message, dict):
                    self._fail_pending(CodexProtocolError("App-server message was not an object"))
                    continue
                response_id = message.get("id")
                if response_id is not None:
                    try:
                        response_id = int(response_id)
                    except (TypeError, ValueError):
                        response_id = None
                if response_id is not None:
                    with self._pending_lock:
                        waiter = self._pending.pop(response_id, None)
                    if waiter is not None:
                        waiter.put(message)
                elif isinstance(message.get("method"), str):
                    self._notifications.put(message)
        finally:
            if generation == self._generation:
                detail = self.stderr_tail()
                suffix = f": {detail}" if detail else ""
                error = CodexTransportError(f"Codex app-server exited{suffix}")
                self._fail_pending(error)
                self._notifications.put({"_exception": error})

    def _read_stderr(self, process: subprocess.Popen[str], generation: int) -> None:
        assert process.stderr is not None
        for raw_line in process.stderr:
            if generation != self._generation:
                return
            line = raw_line.strip()
            if line:
                with self._stderr_lock:
                    self._stderr.append(line)

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return " | ".join(list(self._stderr)[-4:])

    def _fail_pending(self, error: Exception) -> None:
        with self._pending_lock:
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            waiter.put(error)

    def _drain_notifications(self) -> None:
        while True:
            try:
                self._notifications.get_nowait()
            except queue.Empty:
                return

    def _send(self, message: Dict[str, Any], ensure_started: bool = True) -> None:
        if ensure_started:
            self.start()
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise CodexTransportError("Codex app-server is not running")
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
        try:
            with self._write_lock:
                process.stdin.write(payload)
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CodexTransportError(f"Could not write to Codex app-server: {exc}") from exc

    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        ensure_started: bool = True,
    ) -> Dict[str, Any]:
        # Start the transport before registering this request.  start() cleans
        # up a previous child via stop(), and stop() deliberately fails every
        # pending waiter.  Registering first therefore made the first request
        # on a new provider fail locally with "Codex app-server stopped" even
        # though the newly initialized child was healthy and authenticated.
        if ensure_started:
            self.start()
        with self._pending_lock:
            self._next_id += 1
            request_id = self._next_id
            waiter: queue.Queue = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter
        try:
            message: Dict[str, Any] = {"id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._send(message, ensure_started=False)
            try:
                response = waiter.get(timeout=timeout)
            except queue.Empty as exc:
                raise CodexTimeoutError(f"Timed out waiting for {method}") from exc
            if isinstance(response, Exception):
                raise response
            if "error" in response:
                error = response.get("error") or {}
                if isinstance(error, dict):
                    detail = error.get("message") or json.dumps(error, ensure_ascii=False)
                else:
                    detail = str(error)
                raise CodexProtocolError(f"{method} failed: {detail}")
            result = response.get("result")
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise CodexProtocolError(f"{method} returned a non-object result")
            return result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def notify(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        ensure_started: bool = True,
    ) -> None:
        message: Dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        self._send(message, ensure_started=ensure_started)

    def wait_notification(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        timeout: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        skipped: List[Dict[str, Any]] = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CodexTimeoutError("Timed out waiting for Codex turn completion")
                try:
                    message = self._notifications.get(timeout=remaining)
                except queue.Empty as exc:
                    raise CodexTimeoutError("Timed out waiting for Codex notification") from exc
                notification_error = message.get("_exception")
                if isinstance(notification_error, Exception):
                    raise notification_error
                if predicate(message):
                    return message
                skipped.append(message)
        finally:
            for message in skipped:
                self._notifications.put(message)


def _model_id(entry: Dict[str, Any]) -> str:
    value = entry.get("model") or entry.get("id")
    return value.strip() if isinstance(value, str) else ""


def _normalize_model(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model_id = _model_id(entry)
    if not model_id or entry.get("hidden") is True:
        return None
    efforts: List[str] = []
    for item in entry.get("supportedReasoningEfforts") or []:
        value = item.get("reasoningEffort") if isinstance(item, dict) else item
        if isinstance(value, str) and value and value not in efforts:
            efforts.append(value)
    return {
        "id": model_id,
        "display_name": entry.get("displayName") or model_id,
        "supported_reasoning_efforts": efforts,
        "default_reasoning_effort": entry.get("defaultReasoningEffort"),
        "is_default": entry.get("isDefault") is True,
    }


def _response_items(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            role = "developer"
        if role not in ("developer", "user", "assistant"):
            role = "user"
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content or "")
        content_type = "output_text" if role == "assistant" else "input_text"
        items.append(
            {"type": "message", "role": role, "content": [{"type": content_type, "text": text}]}
        )
    return items


def _schema_from_format(response_format: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    wrapper = response_format.get("json_schema")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("schema"), dict):
        return wrapper["schema"]
    return None


def _usage_from_event(params: Dict[str, Any]) -> Dict[str, int]:
    usage = params.get("tokenUsage") or params.get("usage") or {}
    total = usage.get("totalTokenUsage") if isinstance(usage, dict) else {}
    if not isinstance(total, dict):
        total = usage if isinstance(usage, dict) else {}
    prompt = total.get("inputTokens", total.get("promptTokens", 0))
    completion = total.get("outputTokens", total.get("completionTokens", 0))
    cached = total.get("cachedInputTokens", total.get("cachedTokens", 0))
    reasoning = total.get("reasoningOutputTokens", total.get("reasoningTokens", 0))
    values = {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "cached_tokens": int(cached or 0),
        "reasoning_tokens": int(reasoning or 0),
    }
    values["total_tokens"] = int(total.get("totalTokens") or (values["prompt_tokens"] + values["completion_tokens"]))
    return values


class CodexProvider:
    """High-level account, catalogue, and isolated inference operations."""

    def __init__(self, rpc: Optional[CodexAppServer] = None) -> None:
        self.rpc = rpc or CodexAppServer()
        self._call_lock = threading.RLock()
        self._warning_callback: Optional[Callable[[str], None]] = None

    def set_warning_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._warning_callback = callback

    def _warning(self, message: str) -> None:
        if self._warning_callback:
            try:
                self._warning_callback(message)
            except Exception:
                pass

    def close(self) -> None:
        self.rpc.stop()

    def account(self, refresh: bool = False) -> Dict[str, Any]:
        result = self.rpc.request("account/read", {"refreshToken": bool(refresh)}, timeout=30)
        account = result.get("account")
        if not isinstance(account, dict):
            return {"authenticated": False, "account": None}
        return {
            "authenticated": account.get("type") == "chatgpt",
            "account": account,
            "email": account.get("email"),
            "plan_type": account.get("planType"),
        }

    def status(self, refresh_models: bool = False) -> Dict[str, Any]:
        base = {
            "installed": self.rpc.installed,
            "authenticated": False,
            "email": None,
            "plan_type": None,
            "models": [],
            "error": None,
        }
        if not base["installed"]:
            base["error"] = "Codex CLI was not found on PATH"
            return base
        try:
            account = self.account(refresh=False)
            base.update({key: account.get(key) for key in ("authenticated", "email", "plan_type")})
            if base["authenticated"]:
                base["models"] = self.list_models(force=refresh_models)
        except CodexError as exc:
            base["error"] = str(exc)
        return base

    def begin_device_login(self) -> Dict[str, Any]:
        result = self.rpc.request(
            "account/login/start", {"type": "chatgptDeviceCode"}, timeout=30
        )
        url = result.get("verificationUrl")
        code = result.get("userCode")
        if not isinstance(url, str) or not url.startswith("https://") or not isinstance(code, str):
            raise CodexProtocolError("Codex did not return a valid device login URL and code")
        return {
            "login_id": result.get("loginId"),
            "verification_url": url,
            "user_code": code,
        }

    def logout(self) -> None:
        self.rpc.request("account/logout", None, timeout=30)

    def list_models(self, force: bool = False) -> List[Dict[str, Any]]:
        del force  # app-server remains authoritative; persistence is only a UI cache.
        models: List[Dict[str, Any]] = []
        cursor = None
        seen = set()
        while True:
            params: Dict[str, Any] = {"limit": 100, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            result = self.rpc.request("model/list", params, timeout=30)
            data = result.get("data")
            if not isinstance(data, list):
                raise CodexProtocolError("model/list returned a malformed catalogue")
            for raw in data:
                normalized = _normalize_model(raw) if isinstance(raw, dict) else None
                if normalized and normalized["id"] not in seen:
                    seen.add(normalized["id"])
                    models.append(normalized)
            cursor = result.get("nextCursor")
            if not cursor:
                break
        if not models:
            raise CodexProtocolError("No visible Codex models are available for this account")
        import model_config

        model_config.cache_codex_models(models)
        return models

    def _require_chatgpt(self) -> None:
        try:
            # Managed ChatGPT auth refreshes automatically.  Forcing a token
            # refresh before every inference can temporarily report a valid
            # persisted CLI login as unauthenticated.
            account = self.account(refresh=False)
        except CodexProtocolError as exc:
            lowered = str(exc).lower()
            if "unauthorized" in lowered or "authentication" in lowered or "401" in lowered:
                raise CodexAuthenticationError(
                    "ChatGPT authentication expired. Reconnect in Settings."
                ) from exc
            raise
        if not account.get("authenticated"):
            raise CodexAuthenticationError(
                "ChatGPT is not signed in through Codex. Open Settings and sign in."
            )

    @staticmethod
    def _effort(requested: Optional[str], model: Dict[str, Any]) -> Optional[str]:
        supported = model.get("supported_reasoning_efforts") or []
        if not supported:
            return requested or model.get("default_reasoning_effort")
        if requested in supported:
            return requested
        order = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        target = order.index(requested) if requested in order else order.index("medium")
        return min(supported, key=lambda item: abs(order.index(item) - target) if item in order else 99)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tier: str,
        preferred_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Any = None,
        response_format_provided: bool = False,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        deadline = _positive_timeout(timeout)
        with self._call_lock:
            for attempt in range(2):
                try:
                    return self._complete_once(
                        messages,
                        tier,
                        preferred_model,
                        reasoning_effort,
                        response_format,
                        response_format_provided,
                        deadline,
                    )
                except (CodexTransportError, CodexUnavailableError):
                    if attempt:
                        raise
                    self.rpc.restart()
        raise CodexTransportError("Codex request failed after reconnect")

    def _complete_once(
        self,
        messages: List[Dict[str, Any]],
        tier: str,
        preferred_model: Optional[str],
        reasoning_effort: Optional[str],
        response_format: Any,
        response_format_provided: bool,
        timeout: float,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        self._require_chatgpt()
        catalogue = self.list_models(force=True)
        import model_config

        selected, warning = model_config.select_codex_model(
            tier, catalogue, preferred_model=preferred_model
        )
        if warning:
            self._warning(warning)
        selected_entry = next(item for item in catalogue if item["id"] == selected)
        effort = self._effort(reasoning_effort, selected_entry)
        temp_dir = tempfile.TemporaryDirectory(prefix="neq-codex-")
        thread_id = None
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        try:
            thread_result = self.rpc.request(
                "thread/start",
                {
                    "model": selected,
                    "cwd": str(Path(temp_dir.name).resolve()),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                },
                timeout=min(30, timeout),
            )
            thread = thread_result.get("thread") or {}
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id:
                raise CodexProtocolError("thread/start returned no thread id")

            request_messages = [{
                "role": "system",
                "content": (
                    "Answer only from the supplied conversation. Do not inspect files, "
                    "run commands, use tools, or rely on any prior thread history."
                ),
            }] + list(messages or [])
            final_user = None
            if request_messages and request_messages[-1].get("role") == "user":
                final_user = request_messages.pop()
            injected = _response_items(request_messages)
            if injected:
                self.rpc.request(
                    "thread/inject_items",
                    {"threadId": thread_id, "items": injected},
                    timeout=min(30, timeout),
                )
            final_text = str((final_user or {}).get("content") or "Please respond to the supplied conversation.")
            schema = _schema_from_format(response_format)
            generic_json = (
                response_format is not None
                and isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            ) or not response_format_provided
            if generic_json and schema is None:
                final_text += "\n\nReturn only one valid JSON value with no Markdown fence or commentary."

            turn_params: Dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": final_text}],
                "model": selected,
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "readOnly"},
            }
            if effort:
                turn_params["effort"] = effort
            if schema is not None:
                turn_params["outputSchema"] = schema
            turn_result = self.rpc.request(
                "turn/start",
                turn_params,
                timeout=max(1, timeout - (time.monotonic() - started)),
            )
            turn = turn_result.get("turn") or {}
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if not isinstance(turn_id, str) or not turn_id:
                raise CodexProtocolError("turn/start returned no turn id")

            content_parts: List[str] = []

            def terminal(message: Dict[str, Any]) -> bool:
                nonlocal usage
                method = message.get("method")
                params = message.get("params") or {}
                if not isinstance(params, dict) or params.get("threadId") not in (None, thread_id):
                    return False
                if method == "thread/tokenUsage/updated":
                    usage = _usage_from_event(params)
                elif method == "item/completed":
                    item = params.get("item") or {}
                    if isinstance(item, dict) and item.get("type") == "agentMessage":
                        text = item.get("text")
                        if isinstance(text, str):
                            content_parts.append(text)
                if method != "turn/completed":
                    return False
                completed = params.get("turn") or {}
                return not isinstance(completed, dict) or completed.get("id") in (None, turn_id)

            completed_message = self.rpc.wait_notification(
                terminal, max(1, timeout - (time.monotonic() - started))
            )
            completed = (completed_message.get("params") or {}).get("turn") or {}
            status = completed.get("status") if isinstance(completed, dict) else None
            if status not in (None, "completed"):
                error = completed.get("error") if isinstance(completed, dict) else None
                if "auth" in str(error).lower() or "unauthorized" in str(error).lower():
                    raise CodexAuthenticationError(f"Codex authentication was lost: {error}")
                raise CodexProtocolError(f"Codex turn ended with status {status}: {error}")
            content = "".join(content_parts).strip()
            if not content:
                raise CodexProtocolError("Codex turn completed without an agent message")
            return {
                "content": content,
                "model": selected,
                "usage": usage,
                "finish_reason": "stop",
                "id": turn_id,
                "raw_response": completed_message,
            }
        finally:
            # Root threads are persisted by app-server. Delete every request
            # thread so the game never leaves hidden conversation history.
            if thread_id:
                try:
                    self.rpc.request("thread/delete", {"threadId": thread_id}, timeout=5)
                except CodexError:
                    pass
            temp_dir.cleanup()


_provider: Optional[CodexProvider] = None
_provider_lock = threading.Lock()


def get_codex_provider() -> CodexProvider:
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = CodexProvider()
    return _provider
