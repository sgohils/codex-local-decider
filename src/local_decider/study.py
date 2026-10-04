"""Prospective chat-randomized study. Baseline hooks use only the standard library.

Raw transcripts are read incrementally in place, never copied. Unknown formats fail open.
The five-day clock starts only after a real probe hook observes valid nonzero usage.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .contracts import State, default_actions
from .observations import response_summary, update_state
from .storage import Lock, read_json, write_json

SCHEMA = 1
MARKER = "codex-local-decider-randomized-study"
EVENTS = ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "Interrupt",
          "SessionEnd", "SubagentStart", "SubagentStop")
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens")
MAX_RECORD_BYTES = 256_000
MAX_STREAM_READ_BYTES = 8_000_000
# Only top-level response_item content is ignored by every token accounting adapter.
IGNORED_CONTENT_HEADER = re.compile(
    rb'^\s*\{\s*(?:"timestamp"\s*:\s*"[^"\r\n]*"\s*,\s*)?'
    rb'"type"\s*:\s*"response_item"\s*[,}]')
CODING = re.compile(r"\b(fix|debug|bug|implement|refactor|build|code|coding|test|tests|function|regression|feature)\b", re.I)
RESEARCH = re.compile(r"research paper|literature review|study maintenance|randomized study|write.*preprint", re.I)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def utc(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class StudyLock(Lock):
    """Short bounded serialization; dropped events are diagnosed, never block Codex."""
    def __enter__(self):
        deadline = time.monotonic() + 1.5
        while True:
            try:
                return super().__enter__()
            except (FileExistsError, PermissionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.015)


def project_root(cwd: str, excluded: list[str]) -> Path | None:
    path = Path(cwd).resolve()
    if not path.is_dir() or str(path).startswith("\\\\"):
        return None
    for item in excluded:
        if path.is_relative_to(Path(item).resolve()):
            return None
    for parent in (path, *path.parents):
        if (parent / ".git").exists() or any((parent / name).is_file() for name in
                ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "setup.py")):
            return parent
    return None


def allowed_transcript(path: str, roots: list[str]) -> Path | None:
    resolved = Path(path).resolve()
    if resolved.suffix != ".jsonl" or not resolved.is_file():
        return None
    return resolved if any(resolved.is_relative_to(Path(root).resolve()) for root in roots) else None


def metadata(path: Path) -> dict:
    with path.open("rb") as file:
        for _ in range(8):
            line = file.readline(256_000)
            if not line:
                break
            try:
                record = json.loads(line)
                if record.get("type") == "session_meta":
                    value = record["payload"]
                    created = datetime.fromisoformat(value.get("timestamp", record.get("timestamp", "")).replace("Z", "+00:00"))
                    source = value.get("source", "unknown")
                    return {"id": str(value["id"]), "created": created.timestamp(),
                            "subagent": isinstance(source, dict) and "subagent" in source,
                            "client_version": str(value.get("cli_version", "unknown"))[:64]}
            except (ValueError, KeyError, TypeError):
                continue
    return {}


def prepare(data_dir: Path, candidate: Path, days: float = 5, seed: int = 20261001,
            excluded: list[str] | None = None, transcript_roots: list[str] | None = None) -> dict:
    if days != 5:
        raise ValueError("this preregistered study lasts five days")
    if (data_dir / "study.json").exists():
        raise ValueError("study already exists; use a new directory rather than overwrite assignments")
    from .models import Model
    model = Model.load(candidate)
    if model.kind != "linear" or model.version != "linear-s7-g40":
        raise ValueError("study requires the frozen linear-s7-g40 checkpoint")
    raw = candidate.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    frozen = data_dir.resolve() / "models" / f"{sha}.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(raw)
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    value = {"schema_version": SCHEMA, "status": "armed", "prepared_at": time.time(),
             "started_at": None, "deadline": None, "days": days, "seed": seed,
             "candidate": str(frozen), "candidate_sha256": sha, "model_version": model.version,
             "excluded_roots": [str(Path(p).resolve()) for p in [*(excluded or []), str(data_dir)]],
             "transcript_roots": transcript_roots or [str(home / "sessions")],
             "strata": {}, "sessions": [], "completed_episodes": 0,
             "last_hook_at": None, "verification": None,
             "protocol": "chat-randomized-block4-v1", "target_chats_per_arm": 20}
    write_json(data_dir / "study.json", value)
    return status(data_dir)


def assign(study: dict, project: str, model: str) -> str:
    key = digest(project + "\0" + model)
    position = study["strata"].get(key, 0)
    block, index = divmod(position, 4)
    rng = random.Random(digest(f"{study['seed']}:{key}:{block}"))
    arms = ["baseline", "baseline", "assisted", "assisted"]
    rng.shuffle(arms)
    study["strata"][key] = position + 1
    return arms[index]


def new_turn(turn_id: str, now: float, model: str) -> dict:
    return {"turn_hash": digest(turn_id), "started_at": now, "ended_at": None,
            "status": "open", "model": model, "usage": None, "usage_valid": False,
            "models_seen": [model],
            "tool_calls": 0, "tool_failures": 0, "test_runs": 0, "test_passes": 0,
            "test_failures": 0, "repair_cycles": 0, "advice_count": 0,
            "model_failures": 0, "inference_seconds": 0.0, "collection_seconds": 0.0,
            "model_decisions": 0, "model_abstentions": 0,
            "test_verified_after_last_patch": False, "outcome": "unknown", "evidence_sha256": None}


def token_values(value: dict) -> dict | None:
    if not isinstance(value, dict):
        return None
    if not all(type(value.get(k)) is int and value[k] >= 0 for k in TOKEN_KEYS):
        return None
    if value["cached_input_tokens"] > value["input_tokens"]:
        return None
    return {k: value[k] for k in TOKEN_KEYS}


def read_stream(session: dict, path: Path, cutoff: float | None = None,
                parent_turn: str | None = None) -> None:
    key = digest(str(path))
    stream = session["streams"].setdefault(key, {"path": str(path), "offset": 0,
                      "total": dict.fromkeys(TOKEN_KEYS, 0), "current_turn": session.get("current_turn"),
                      "format": None, "warnings": [], "backlog": False,
                      "subagent": metadata(path).get("subagent", False), "request_mode": False})
    stream["cutoff"] = cutoff
    if parent_turn:
        stream["parent_turn"] = parent_turn
    if path.stat().st_size < stream["offset"]:
        stream["warnings"] = ["transcript_truncated"]
        return  # Do not silently reset cumulative counters and double-count.
    with path.open("rb") as file:
        file.seek(stream["offset"])
        consumed = 0
        while consumed < MAX_STREAM_READ_BYTES:
            line = file.readline(min(MAX_RECORD_BYTES + 1, MAX_STREAM_READ_BYTES - consumed))
            if not line:
                break
            consumed += len(line)
            if len(line) > MAX_RECORD_BYTES or "discarding_record" in stream:
                # Stream large content in bounded chunks, without parsing or retaining it.
                # Partial large lines resume on the next hook, without losing the boundary.
                if "discarding_record" not in stream:
                    stream["discarding_record"] = bool(IGNORED_CONTENT_HEADER.match(line))
                stream["offset"] = file.tell()
                if line.endswith(b"\n"):
                    ignored_content = stream.pop("discarding_record")
                    if ignored_content:
                        stream["skipped_content_records"] = stream.get("skipped_content_records", 0) + 1
                    else:
                        stream["warnings"] = ["oversized_record"]
                continue
            if not line.endswith(b"\n"):
                break
            stream["offset"] = file.tell()
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                stream["warnings"] = ["malformed_record"]
                continue
            if not isinstance(record, dict):
                continue
            if stream.get("cutoff"):
                try:
                    stamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")).timestamp()
                    if stamp > stream["cutoff"]:
                        continue
                except (ValueError, KeyError, TypeError):
                    stream["warnings"] = ["missing_record_timestamp_at_cutoff"]
                    continue
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "turn_context" and payload.get("turn_id"):
                stream["current_turn"] = digest(str(payload["turn_id"]))
                current = session["turns"].get(stream["current_turn"])
                if current and payload.get("model") and payload["model"] not in current["models_seen"]:
                    current["models_seen"].append(str(payload["model"])[:128])
            if record.get("type") == "event_msg" and payload.get("type") == "task_started":
                if payload.get("turn_id"):
                    stream["current_turn"] = digest(str(payload["turn_id"]))
            if record.get("type") == "token_usage_record":
                values = token_values(payload.get("usage"))
                response_id = payload.get("response_id")
                turn = session["turns"].get(digest(str(payload.get("root_turn_id", ""))))
                turn = turn or session["turns"].get(digest(str(payload.get("turn_id", ""))))
                if turn is None and stream["subagent"]:
                    turn = session["turns"].get(stream.get("parent_turn", session.get("current_turn")))
                if values is not None and response_id and turn is not None:
                    stream["request_mode"] = True
                    stream["format"] = "token_usage_record.usage.v1"
                    requests = turn.setdefault("requests", {})
                    response_hash = digest(str(response_id))
                    prior = requests.get(response_hash)
                    if prior and any(values[k] < prior[k] for k in TOKEN_KEYS):
                        stream["warnings"] = ["nonmonotonic_response_usage"]
                        continue
                    requests[response_hash] = values
                    threads = turn.setdefault("usage_threads", [])
                    thread = digest(str(payload.get("thread_id", "")))
                    if thread not in threads:
                        threads.append(thread)
                    turn["usage"] = {k: sum(v[k] for v in requests.values()) for k in TOKEN_KEYS}
                    turn["usage_source"] = "response_ids"
                    turn["usage_valid"] = turn["usage"]["input_tokens"] + turn["usage"]["output_tokens"] > 0
                continue
            if record.get("type") != "event_msg" or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            values = token_values(info.get("total_token_usage") if isinstance(info, dict) else None)
            if values is None:
                continue
            previous = stream["total"]
            if any(values[k] < previous[k] for k in TOKEN_KEYS):
                if not stream["request_mode"]:
                    stream["warnings"] = ["nonmonotonic_usage"]
                continue
            delta = {k: values[k] - previous[k] for k in TOKEN_KEYS}
            stream["total"] = values
            if not stream["request_mode"]:
                stream["format"] = "event_msg.token_count.info.total_token_usage.v1"
            turn = session["turns"].get(stream.get("current_turn"))
            # Child turns contribute to the active root turn, never form independent chats.
            if turn is None and stream["subagent"]:
                turn = session["turns"].get(stream.get("parent_turn", session.get("current_turn")))
            if turn is not None and turn.get("usage_source") != "response_ids" and any(delta.values()):
                usage = turn["usage"] or dict.fromkeys(TOKEN_KEYS, 0)
                turn["usage"] = {k: usage[k] + delta[k] for k in TOKEN_KEYS}
                turn["usage_source"] = "cumulative_delta"
                turn["usage_valid"] = turn["usage"]["input_tokens"] + turn["usage"]["output_tokens"] > 0
        stream["backlog"] = "discarding_record" in stream or stream["offset"] < path.stat().st_size


def maybe_start(study: dict, session: dict, now: float) -> None:
    if study["status"] != "armed" or session["arm"] != "probe":
        return
    verified = [t for t in session["turns"].values()
                if t["status"] in ("completed", "interrupted") and usable_turn(session, t)]
    if not verified or not session.get("post_tool_seen"):
        return
    if any(s.get("warnings") or s.get("backlog") for s in session["streams"].values()):
        return
    study.update(status="running", started_at=now, deadline=now + 5 * 86400,
                 verification={"session_hash": session["session_hash"], "verified_at": now,
                               "source": "ordinary Codex hooks and supported transcript usage",
                               "probe_excluded_from_comparison": True,
                               "probe_advice_enabled": bool(session.get("probe_advice_enabled"))})


def set_probe_advice(data_dir: Path, session_hash: str, enabled: bool = True) -> dict:
    """Explicit demonstration in an excluded probe; never change randomized arms."""
    if not isinstance(session_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", session_hash):
        raise ValueError("probe advice requires a valid chat hash from the local registry")
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        now = time.time()
        if study["status"] not in ("armed", "running") or (study["deadline"] and now >= study["deadline"]):
            raise ValueError("study is stopped or expired")
        if session_hash not in study["sessions"]:
            raise ValueError("chat is not enrolled")
        file = data_dir / "sessions" / f"{session_hash}.json"
        session = read_json(file)
        if session["arm"] != "probe":
            raise ValueError("manual advice is restricted to excluded probe chats")
        if enabled:
            raw = Path(study["candidate"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != study["candidate_sha256"]:
                raise ValueError("frozen checkpoint changed")
            from .models import Model
            Model.load(Path(study["candidate"]))
        if bool(session.get("probe_advice_enabled")) != enabled:
            session.update(probe_advice_enabled=enabled, probe_advice_changed_at=now)
            study.setdefault("protocol_amendments", []).append({
                "at": now, "change": "excluded probe advice enabled" if enabled else "excluded probe advice disabled",
                "comparison_assignments_unchanged": True, "probe_excluded_from_comparison": True})
            write_json(file, session)
            write_json(data_dir / "study.json", study)
    return {"probe_advice_enabled": enabled, "probe_excluded_from_comparison": True,
            "study_status": study["status"], "clock_manually_started": False}


def advise(study: dict, session: dict, turn: dict, state: State, kind: str | None) -> dict:
    allowed = session["arm"] == "assisted" or (session["arm"] == "probe" and session.get("probe_advice_enabled"))
    if not allowed or not kind:
        return {}
    started = time.perf_counter()
    try:
        raw = Path(study["candidate"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != study["candidate_sha256"]:
            raise ValueError("frozen checkpoint changed")
        from .models import Model
        model = Model.load(Path(study["candidate"]))
        decision = model.decide(state, default_actions())
        turn["model_decisions"] = turn.get("model_decisions", 0) + 1
        if decision["uncertain"] or decision["action_kind"] == "defer":
            turn["model_abstentions"] = turn.get("model_abstentions", 0) + 1
            return {}
        key = digest(json.dumps([state.flags, decision["action_kind"]], sort_keys=True))
        if key == session.get("last_advice"):
            return {}
        label = next(a.label for a in default_actions() if a.id == decision["action_id"])
        session["last_advice"] = key
        turn["advice_count"] += 1
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                "additionalContext": f"Local advisor: {label}. Use this only if relevant."}}
    except Exception:
        turn["model_failures"] += 1
        return {}
    finally:
        turn["inference_seconds"] += time.perf_counter() - started


def handle(event: dict, data_dir: Path, now: float | None = None) -> dict:
    started = time.perf_counter()
    now = time.time() if now is None else now
    if not isinstance(event, dict) or event.get("hook_event_name") not in EVENTS:
        return {}
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        if study["status"] in ("stopped", "expired"):
            return {}
        if study["deadline"] is not None and now >= study["deadline"]:
            study["expiry_observed_at"] = now
            write_json(data_dir / "study.json", study)
            return {}
        study["last_hook_at"] = now
        name = event["hook_event_name"]
        identity = str(event.get("session_id", ""))
        if not identity:
            return {}
        session_hash = digest(identity)
        session_file = data_dir / "sessions" / f"{session_hash}.json"
        session = read_json(session_file) if session_file.exists() else None
        path = allowed_transcript(str(event.get("transcript_path") or ""), study["transcript_roots"])
        if session is None:
            if name != "UserPromptSubmit" or not path or not event.get("turn_id"):
                write_json(data_dir / "study.json", study)
                return {}
            meta = metadata(path)
            root = project_root(str(event.get("cwd", "")), study["excluded_roots"])
            prompt = str(event.get("prompt", ""))
            if (not root or not meta or meta["id"] != identity or meta["subagent"]
                    or meta["created"] < study["prepared_at"] or not CODING.search(prompt)
                    or RESEARCH.search(prompt) or event.get("permission_mode") == "plan"):
                write_json(data_dir / "study.json", study)
                return {}
            model = str(event.get("model", "unknown"))
            project = digest(str(root))
            arm = "probe" if study["status"] == "armed" else assign(study, project, model)
            session = {"session_hash": session_hash, "project_hash": project, "arm": arm,
                       "assigned_at": now, "initial_model": model, "turns": {}, "streams": {},
                       "client_version": meta["client_version"],
                       "state": State("Fix code and verify tests").to_dict(), "current_turn": None,
                       "last_advice": None, "closed": False, "post_tool_seen": False,
                       "subagents": [], "subagent_turns": {}}
            study["sessions"].append(session_hash)
            # Commit the random assignment before examining any tool outcome.
            write_json(session_file, session)
            write_json(data_dir / "study.json", study)
        if path:
            read_stream(session, path, study["deadline"])
        child_path = allowed_transcript(str(event.get("agent_transcript_path") or ""), study["transcript_roots"])
        if child_path and name == "SubagentStop":
            parent_turn = session["subagent_turns"].get(digest(str(event.get("agent_id", ""))))
            read_stream(session, child_path, study["deadline"], parent_turn)
        turn_hash = digest(str(event.get("turn_id", ""))) if event.get("turn_id") else session.get("current_turn")
        if name == "UserPromptSubmit":
            prompt = str(event.get("prompt", ""))
            if RESEARCH.search(prompt) or event.get("permission_mode") == "plan":
                session["current_turn"] = None
                write_json(session_file, session)
                write_json(data_dir / "study.json", study)
                return {}
            if turn_hash not in session["turns"]:
                session["turns"][turn_hash] = new_turn(str(event["turn_id"]), now, str(event.get("model", "unknown")))
                # Only a fixed vocabulary survives; no full prompt or identifier is saved.
                words = sorted({m.group().lower() for m in CODING.finditer(prompt)})
                state = State.from_dict(session["state"])
                state.task = "Fix code and verify tests " + " ".join(words)
                session["state"] = state.to_dict()
            session["current_turn"] = turn_hash
            session["closed"] = False
            if path:
                session["streams"][digest(str(path))]["current_turn"] = turn_hash
        turn = session["turns"].get(turn_hash) or session["turns"].get(session.get("current_turn"))
        inference_before = turn["inference_seconds"] if turn else 0
        if turn and event.get("model") and event["model"] not in turn["models_seen"]:
            turn["models_seen"].append(str(event["model"])[:128])
        result = {}
        if turn and name == "PostToolUse" and turn["status"] == "open":
            session["post_tool_seen"] = True
            # Tool call ids deduplicate replayed completion notifications without saving inputs.
            call_id = event.get("tool_use_id", event.get("call_id"))
            event_hash = digest(str(call_id)) if call_id else None
            seen = turn.setdefault("event_hashes", [])
            if event_hash is None or event_hash not in seen:
                if event_hash:
                    seen.append(event_hash)
                turn["tool_calls"] += 1
                failed, _ = response_summary(event)
                turn["tool_failures"] += int(failed)
                state = State.from_dict(session["state"])
                previous = state.to_dict()
                state, kind, passed = update_state(state, event)
                if kind == "patch":
                    turn["repair_cycles"] += int(turn["test_failures"] > 0)
                    turn["test_verified_after_last_patch"] = False
                if kind in ("focused_test", "broad_test"):
                    turn["test_runs"] += 1
                    turn["test_passes"] += int(passed)
                    turn["test_failures"] += int(failed)
                    if failed:
                        turn["test_verified_after_last_patch"] = False
                    elif passed and state.flags.get("patched"):
                        turn["test_verified_after_last_patch"] = True
                if passed and kind == "focused_test" and previous["flags"].get("patched"):
                    sample = {"state": previous, "action_kind": kind, "verified_test_pass": True,
                              "project_hash": session["project_hash"], "note": "Observed action, not optimality"}
                    with (data_dir / "experience.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(sample) + "\n")
                result = advise(study, session, turn, state, kind)
                session["state"] = state.to_dict()
        if turn and name in ("Stop", "Interrupt"):
            if turn["status"] == "open":
                turn.update(status="interrupted" if name == "Interrupt" else "completed", ended_at=now)
                if name == "Stop" and session["arm"] != "probe":
                    study["completed_episodes"] += 1
        if name in ("SubagentStart", "SubagentStop") and event.get("agent_id"):
            child = digest(str(event["agent_id"]))
            if child not in session["subagents"]:
                session["subagents"].append(child)
            if name == "SubagentStart":
                session["subagent_turns"][child] = session.get("current_turn")
            else:
                session["subagent_turns"].setdefault(child, session.get("current_turn"))
        if name == "SessionEnd":
            session["closed"] = True
        if turn:
            elapsed = time.perf_counter() - started
            turn["collection_seconds"] += max(0, elapsed - (turn["inference_seconds"] - inference_before))
        maybe_start(study, session, now)
        write_json(session_file, session)
        write_json(data_dir / "study.json", study)
        return result


def refresh(data_dir: Path) -> None:
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        if study["status"] == "stopped" or (data_dir / "frozen.json").exists():
            return
        now = time.time()
        for identity in study["sessions"]:
            file = data_dir / "sessions" / f"{identity}.json"
            session = read_json(file)
            for stream in list(session["streams"].values()):
                stream["cutoff"] = study["deadline"]
                path = allowed_transcript(stream["path"], study["transcript_roots"])
                if path:
                    read_stream(session, path, study["deadline"], stream.get("parent_turn"))
            maybe_start(study, session, now)
            write_json(file, session)
        if study["deadline"] and now >= study["deadline"]:
            study.update(status="expired", closed_at=now)
        write_json(data_dir / "study.json", study)


def usable_turn(session: dict, turn: dict) -> bool:
    expected = [child for child, parent in session["subagent_turns"].items() if parent == turn["turn_hash"]]
    return bool(turn["usage_valid"] and turn["usage"] is not None
                and all(child in turn.get("usage_threads", []) for child in expected)
                and not any(s.get("warnings") or s.get("backlog") for s in session["streams"].values()))


def status(data_dir: Path, reconcile: bool = False) -> dict:
    if reconcile:
        refresh(data_dir)
    study = read_json(data_dir / "study.json")
    counts = {arm: {"chats": 0, "turns": 0, "measured_turns": 0, "advice": 0,
                    "model_decisions": 0, "model_abstentions": 0, "skipped_content_records": 0}
              for arm in ("probe", "baseline", "assisted")}
    warnings = set()
    for identity in study["sessions"]:
        session = read_json(data_dir / "sessions" / f"{identity}.json")
        row = counts[session["arm"]]
        row["chats"] += 1
        row["turns"] += len(session["turns"])
        row["measured_turns"] += sum(usable_turn(session, t) for t in session["turns"].values())
        row["advice"] += sum(t["advice_count"] for t in session["turns"].values())
        row["model_decisions"] += sum(t.get("model_decisions", 0) for t in session["turns"].values())
        row["model_abstentions"] += sum(t.get("model_abstentions", 0) for t in session["turns"].values())
        row["skipped_content_records"] += sum(s.get("skipped_content_records", 0) for s in session["streams"].values())
        for stream in session["streams"].values():
            warnings.update(stream["warnings"])
    diagnostics = {}
    for file in (data_dir / "diagnostics").glob("*.json"):
        try:
            kind = read_json(file)["error_type"]
            diagnostics[kind] = diagnostics.get(kind, 0) + 1
        except (ValueError, KeyError, OSError):
            continue
    training_seconds = 0.0
    history = data_dir / "worker-history.jsonl"
    if history.exists():
        for line in history.read_text().splitlines():
            try:
                training_seconds += float(json.loads(line)["elapsed_seconds"])
            except (ValueError, KeyError):
                continue
    return {"status": study["status"], "prepared_at": utc(study["prepared_at"]),
            "hook_configured": bool(study.get("hook_config")),
            "hook_file_present": bool(study.get("hook_config") and Path(study["hook_config"]).is_file()),
            "last_hook_at": utc(study["last_hook_at"]) if study["last_hook_at"] else None,
            "hook_failures": diagnostics, "background_training_wall_seconds": training_seconds,
            "started_at": utc(study["started_at"]) if study["started_at"] else None,
            "deadline": utc(study["deadline"]) if study["deadline"] else None,
            "candidate_sha256": study["candidate_sha256"], "groups": counts,
            "completed_episodes": study["completed_episodes"], "warnings": sorted(warnings),
            "probe_advice_chats": sum(bool(read_json(data_dir / "sessions" / f"{identity}.json").get("probe_advice_enabled"))
                                      for identity in study["sessions"]),
            "next": "Complete the probe turn to verify collection" if study["status"] == "armed" and counts["probe"]["chats"]
                    else "Trust installed hooks and use a new coding chat" if study["status"] == "armed" else None}


def export(data_dir: Path) -> dict:
    study = read_json(data_dir / "study.json")
    rows = []
    for identity in study["sessions"]:
        session = read_json(data_dir / "sessions" / f"{identity}.json")
        if session["arm"] == "probe":
            continue
        turns = []
        for t in session["turns"].values():
            row = {k: v for k, v in t.items() if k not in ("event_hashes", "requests", "usage_threads")}
            expected = [child for child, parent in session["subagent_turns"].items() if parent == t["turn_hash"]]
            missing = sum(child not in t.get("usage_threads", []) for child in expected)
            row["missing_subagent_usage"] = missing
            row["usage_valid"] = usable_turn(session, t)
            turns.append(row)
        rows.append({"session_hash": identity, "project_hash": session["project_hash"],
                     "arm": session["arm"], "initial_model": session["initial_model"],
                     "client_version": session["client_version"],
                     "assigned_at": session["assigned_at"], "closed": session["closed"],
                     "skipped_content_records": sum(s.get("skipped_content_records", 0) for s in session["streams"].values()),
                     "subagent_count": len(session["subagents"]), "turns": turns,
                     "usage_warnings": sorted({w for s in session["streams"].values() for w in s["warnings"]})})
    return {"schema_version": SCHEMA, "protocol": study["protocol"], "status": study["status"],
            "seed": study["seed"], "started_at": study["started_at"], "deadline": study["deadline"],
            "candidate_sha256": study["candidate_sha256"], "verification": bool(study["verification"]),
            "protocol_amendments": study.get("protocol_amendments", []),
            "sessions": rows, "metric": "reported input + output tokens; cached input included once"}


def analyze(value: dict, bootstrap_samples: int = 10000) -> dict:
    groups = {}
    chat_values = {}
    for arm in ("baseline", "assisted"):
        chats = [s for s in value["sessions"] if s["arm"] == arm]
        turns = [t for s in chats for t in s["turns"]]
        measured = [t for t in turns if t["usage_valid"] and t["usage"] is not None]
        total = sum(t["usage"]["input_tokens"] + t["usage"]["output_tokens"] for t in measured)
        confirmed = [t for t in turns if t["outcome"] != "unknown"]
        groups[arm] = {"chats": len(chats), "turns": len(turns), "measured_turns": len(measured),
                      "skipped_content_records": sum(s.get("skipped_content_records", 0) for s in chats),
                      "missing_usage_turns": len(turns) - len(measured), "total_tokens": total,
                      "mean_tokens_per_measured_turn": total / len(measured) if measured else None,
                      "cached_input_tokens": sum(t["usage"]["cached_input_tokens"] for t in measured),
                      "interrupted_turns": sum(t["status"] == "interrupted" for t in turns),
                      "unfinished_chats": sum(not s["closed"] for s in chats),
                      "tool_failures": sum(t["tool_failures"] for t in turns),
                      "test_verified_turns": sum(t["test_verified_after_last_patch"] for t in turns),
                      "confirmed_solved": sum(t["outcome"] == "solved" for t in confirmed),
                      "confirmed_outcomes": len(confirmed), "unknown_outcomes": len(turns) - len(confirmed),
                      "follow_up_turns": sum(max(0, len(s["turns"]) - 1) for s in chats),
                      "repair_cycles": sum(t["repair_cycles"] for t in turns),
                      "advice_count": sum(t["advice_count"] for t in turns),
                      "model_failures": sum(t["model_failures"] for t in turns),
                      "model_decisions": sum(t.get("model_decisions", 0) for t in turns),
                      "model_abstentions": sum(t.get("model_abstentions", 0) for t in turns),
                      "inference_seconds": sum(t["inference_seconds"] for t in turns),
                      "collection_seconds": sum(t["collection_seconds"] for t in turns),
                      "elapsed_turn_seconds": sum(max(0, t["ended_at"] - t["started_at"])
                                                   for t in turns if t["ended_at"] is not None)}
        chat_values[arm] = []
        for session in chats:
            valid = [t for t in session["turns"] if t["usage_valid"] and t["usage"] is not None]
            if valid:
                chat_values[arm].append((sum(t["usage"]["input_tokens"] + t["usage"]["output_tokens"]
                                             for t in valid), len(valid)))
    before = groups["baseline"]["mean_tokens_per_measured_turn"]
    after = groups["assisted"]["mean_tokens_per_measured_turn"]
    reduction = 1 - after / before if before and after is not None else None
    enough = min(len(chat_values[a]) for a in chat_values) >= 10
    interval = None
    if enough and reduction is not None:
        rng = random.Random(value["seed"])
        samples = []
        for _ in range(bootstrap_samples):
            means = {}
            for arm, rows in chat_values.items():
                chosen = rng.choices(rows, k=len(rows))  # Whole chat, all its follow-up turns.
                means[arm] = sum(x[0] for x in chosen) / sum(x[1] for x in chosen)
            if means["baseline"]:
                samples.append(1 - means["assisted"] / means["baseline"])
        samples.sort()
        if samples:
            interval = [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]]
    per_chat = []
    composition = {}
    for session in value["sessions"]:
        measured = [t for t in session["turns"] if t["usage_valid"] and t["usage"] is not None]
        per_chat.append({"session_hash": session["session_hash"], "arm": session["arm"],
                         "measured_turns": len(measured), "turns": len(session["turns"]),
                         "total_tokens": sum(t["usage"]["input_tokens"] + t["usage"]["output_tokens"] for t in measured) if measured else None,
                         "models_seen": sorted({m for t in session["turns"] for m in t.get("models_seen", [t["model"]])})})
        key = (session["project_hash"], session["initial_model"], session["arm"])
        composition[key] = composition.get(key, 0) + 1
    return {"schema_version": SCHEMA, "study_status": value["status"], "groups": groups,
            "per_chat": per_chat,
            "project_model_composition": [{"project_hash": p, "model": m, "arm": a, "chats": n}
                                          for (p, m, a), n in sorted(composition.items())],
            "token_reduction_fraction": reduction, "chat_bootstrap_95_interval": interval,
            "interpretation": "exploratory" if enough else "inconclusive: fewer than ten measured chats in a group",
            "target_reduction_fraction": .1, "ordinary_activation_changed": False,
            "limitations": ["Single user and five-day observation window", "Chats have different task difficulty",
                            "Test passes do not establish task completion", "Missing usage is unknown, not zero",
                            "Reported tokens are not billed dollars or subscription allowance",
                            "Timer excludes interpreter startup and final record writes; turn elapsed includes hook delays"],
            "candidate_sha256": value["candidate_sha256"], "protocol": value["protocol"]}


def outcome(data_dir: Path, session_hash: str, turn_hash: str, result: str, evidence: Path | None) -> dict:
    if result not in ("solved", "unsolved", "unknown"):
        raise ValueError("invalid outcome")
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        if study["status"] in ("stopped", "expired"):
            raise ValueError("dataset is frozen")
        if session_hash not in study["sessions"]:
            raise ValueError("unknown chat")
        path = data_dir / "sessions" / f"{session_hash}.json"
        session = read_json(path)
        turn = session["turns"][turn_hash]
        turn["outcome"] = result
        turn["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest() if evidence else None
        turn["outcome_source"] = "operator annotation"
        write_json(path, session)
    return {"outcome": result, "evidence_recorded": evidence is not None}


def install(data_dir: Path, config_file: Path) -> dict:
    config = read_json(config_file) if config_file.exists() else {"hooks": {}}
    if not (data_dir / "hooks-before.json").exists():
        write_json(data_dir / "hooks-before.json", {"existed": config_file.exists(), "config": config})
    args = [sys.executable, "-m", "local_decider.study", "hook", "--data-dir", str(data_dir.resolve())]
    command = subprocess_command(args)
    for groups in config.get("hooks", {}).values():
        if any(g.get("description") == MARKER and any(h.get("command") != command for h in g.get("hooks", [])) for g in groups):
            raise ValueError("another study installation exists; stop it before installing a new study")
    for name in EVENTS:
        groups = config.setdefault("hooks", {}).setdefault(name, [])
        groups = [g for g in groups if g.get("description") != MARKER]
        handler = {"type": "command", "command": command, "timeout": 3}
        if name == "PostToolUse":
            handler["additionalContextLimit"] = 100
        groups.append({"description": MARKER, "hooks": [handler]})
        config["hooks"][name] = groups
    write_config(config_file, config, data_dir)
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        study["hook_config"] = str(config_file.resolve())
        study["hook_command"] = command
        write_json(data_dir / "study.json", study)
    return {"hooks": str(config_file), "status": "requires hook trust", "clock_started": False}


def subprocess_command(args: list[str]) -> str:
    import subprocess
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def write_config(config_file: Path, config: dict, data_dir: Path) -> None:
    """Stage inside our directory; only the exact external configuration needs write access."""
    staged = data_dir / f"hooks-config-{uuid.uuid4().hex}.json"
    write_json(staged, config)
    try:
        for attempt in range(20):
            try:
                os.replace(staged, config_file)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)
    finally:
        staged.unlink(missing_ok=True)


def stop(data_dir: Path, remove_hooks: bool = True) -> dict:
    refresh(data_dir)
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        if study["status"] != "expired":
            study.update(status="stopped", closed_at=time.time())
        write_json(data_dir / "study.json", study)
        snapshot = export(data_dir)
        write_json(data_dir / "frozen.json", snapshot)
    if remove_hooks and study.get("hook_config"):
        file = Path(study["hook_config"])
        if file.exists():
            config = read_json(file)
            for name in list(config.get("hooks", {})):
                groups = config["hooks"][name]
                remaining = [g for g in groups if not (g.get("description") == MARKER
                             and any(h.get("command") == study.get("hook_command") for h in g.get("hooks", [])))]
                if remaining:
                    config["hooks"][name] = remaining
                else:
                    del config["hooks"][name]
            write_config(file, config, data_dir)  # Preserve hooks added during the study.
    return status(data_dir)


def maybe_worker(data_dir: Path) -> None:
    """Bounded candidate-only learning, with the study's fixed checkpoint untouched."""
    import subprocess
    with StudyLock(data_dir / "registry.lock"):
        study = read_json(data_dir / "study.json")
        if study["status"] != "running":
            return
        counter_file = data_dir / "counter.json"
        counter = read_json(counter_file) if counter_file.exists() else {"last_trained": 0, "pending_since": 0}
        counter["completed"] = study["completed_episodes"]
        due = counter["completed"] - counter["last_trained"] >= 25 and time.time() - counter.get("pending_since", 0) > 1200
        if due:
            counter["pending_since"] = time.time()
        write_json(counter_file, counter)
    if due:
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        with (data_dir / "worker.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-m", "local_decider", "worker", "--data-dir", str(data_dir)],
                             stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=environment,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Five-day randomized local Codex study")
    parser.add_argument("action", choices=("start", "install", "hook", "status", "stop", "export", "analyze", "outcome", "probe-advice"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--keep-hooks", action="store_true")
    parser.add_argument("--disable", action="store_true", help="Disable optional advice for the specified probe chat")
    parser.add_argument("--session")
    parser.add_argument("--turn")
    parser.add_argument("--result", choices=("solved", "unsolved", "unknown"), default="unknown")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    if args.action == "hook":
        result = {}
        try:
            raw = sys.stdin.read(256001)
            if len(raw) <= 256000:
                event = json.loads(raw)
                result = handle(event, args.data_dir)
                if read_json(args.data_dir / "study.json")["status"] == "armed":
                    # Catch up the enrolled probe after delayed writes on an unrelated hook.
                    # Only its real completion and supported usage can verify the clock.
                    refresh(args.data_dir)
                if event.get("hook_event_name") == "Stop":
                    maybe_worker(args.data_dir)
        except Exception as error:
            try:
                folder = args.data_dir / "diagnostics"
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / f"{time.time_ns()}-{os.getpid()}.json", {"error_type": type(error).__name__, "at": time.time()})
            except Exception:
                pass
        print(json.dumps(result))
        return
    try:
        if args.action == "start":
            if not args.candidate:
                raise ValueError("start requires --candidate")
            result = prepare(args.data_dir, args.candidate, excluded=args.exclude)
        elif args.action == "install":
            if not args.config:
                raise ValueError("install requires --config")
            result = install(args.data_dir, args.config)
        elif args.action == "status":
            result = status(args.data_dir, args.refresh)
        elif args.action == "probe-advice":
            result = set_probe_advice(args.data_dir, args.session, not args.disable)
        elif args.action == "stop":
            result = stop(args.data_dir, not args.keep_hooks)
        elif args.action in ("export", "analyze"):
            value = read_json(args.data_dir / "frozen.json") if (args.data_dir / "frozen.json").exists() else export(args.data_dir)
            result = value if args.action == "export" else analyze(value)
        else:
            result = outcome(args.data_dir, args.session, args.turn, args.result, args.evidence)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"local-decider study: {error}\n")


if __name__ == "__main__":
    main()
