"""Lightweight observations shared by ordinary and experimental hooks."""
import json
import re

from .contracts import State


def classify(event: dict) -> str | None:
    tool = str(event.get("tool_name", "")).lower()
    command = str(event.get("tool_input", {})).lower()
    if "apply_patch" in tool or tool in ("edit", "write"):
        return "patch"
    if any(x in command for x in ("pytest", "unittest", "npm test", "cargo test")):
        return "focused_test" if any(x in command for x in ("test_", " -k ", "::")) else "broad_test"
    if any(x in command for x in ("get-content", "read_file", "type ", "cat ")):
        return "inspect_file"
    if any(x in command for x in ("rg ", "grep ", "search")) or "search" in tool:
        return "search_code"
    return None


def response_summary(event: dict) -> tuple[bool, bool]:
    response = event.get("tool_response", event.get("tool_result", {}))
    # Some hosts wrap the command result in MCP content blocks.
    if isinstance(response, dict) and isinstance(response.get("content"), list):
        for block in response["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    nested = json.loads(block.get("text", ""))
                    if isinstance(nested, dict) and "exit_code" in nested:
                        response = nested
                        break
                except (ValueError, TypeError):
                    pass
    text = json.dumps(response, default=str).lower() if not isinstance(response, str) else response.lower()
    failed = bool(re.search(r"(?:failed|failures?)[ (:]|traceback|assertionerror", text))
    passed = bool(re.search(r"\b\d+ passed\b|\bok\b", text)) and not failed
    exit_code = response.get("exit_code") if isinstance(response, dict) else None
    if exit_code is not None and exit_code != 0:
        failed, passed = True, False
    return failed, passed


def update_state(state: State, event: dict) -> tuple[State, str | None, bool]:
    kind = classify(event)
    response = event.get("tool_response", event.get("tool_result", {}))
    text = json.dumps(response, default=str).lower() if not isinstance(response, str) else response.lower()
    f = state.flags
    if kind == "patch":
        f.update(patched=True, focused_passed=False, broad_passed=False)
    elif kind == "search_code":
        f["path_known"] = bool(re.search(r"\.(?:py|js|ts|rs)\b", text)) or f.get("path_known", False)
    elif kind == "inspect_file":
        f.update(path_known=True, source_seen=True)
    failed, passed = response_summary(event)
    if kind in ("focused_test", "broad_test"):
        if failed:
            f.update(failure_seen=True, failure_inspected="traceback" in text,
                     focused_passed=False, broad_passed=False)
        elif passed:
            f["focused_passed"] = True
            if kind == "broad_test":
                f["broad_passed"] = True
    if kind:
        state.previous_actions = (state.previous_actions + [kind])[-16:]
    state.observations = ["Observed workflow flags: " + ", ".join(k for k, v in f.items() if v)]
    return state, kind, passed
