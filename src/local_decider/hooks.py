"""Advisory hooks; they never run suggested commands or change approval decisions."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .contracts import State, default_actions
from .models import Model
from .observations import classify as classify, update_state
from .storage import Lock, read_json, write_json


def redact(text: str) -> str:
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    return re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]+)\b", "[REDACTED]", text)


def handle(event: dict, data_dir: Path, candidate: Path | None = None,
           experimental: bool = False, spawn_worker: bool = True) -> dict:
    name = event.get("hook_event_name", "")
    if name not in ("UserPromptSubmit", "PostToolUse", "Stop"):
        return {}
    data_dir.mkdir(parents=True, exist_ok=True)
    session = hashlib.sha256(str(event.get("session_id", "unknown")).encode()).hexdigest()[:24]
    turn = str(event.get("turn_id", "default"))
    session_file = data_dir / "sessions" / f"{session}.json"
    trigger = False
    response = {}
    with Lock(data_dir / "events.lock"):
        saved = read_json(session_file) if session_file.exists() else {}
        state = State.from_dict(saved.get("state", {"task": ""}))
        if name == "UserPromptSubmit":
            state = State(redact(str(event.get("prompt", "")))[:4000])
            saved = {"turn": turn, "last_advice": None, "closed_turns": saved.get("closed_turns", [])}
        if name == "PostToolUse":
            previous = state.to_dict()
            state, kind, passed = update_state(state, event)
            if (passed and kind == "focused_test" and previous["flags"].get("patched")
                    and not previous["flags"].get("focused_passed")):
                sample = {"state": previous, "action_kind": kind, "verified_test_pass": True,
                          "project_hash": hashlib.sha256(str(event.get("cwd", "")).encode()).hexdigest(),
                          "note": "Observed successful action, not a counterfactual optimal choice"}
                with (data_dir / "experience.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(sample) + "\n")
            model_path = candidate
            if not experimental:
                manifest = read_json(data_dir / "active.json") if (data_dir / "active.json").exists() else {}
                model_path = Path(manifest["model"]) if manifest.get("live_validated") else None
            if model_path and model_path.is_file() and kind and state.task:
                model = Model.load(model_path)
                decision = model.decide(state, default_actions())
                if not decision["uncertain"] and decision["action_kind"] != "defer":
                    key = hashlib.sha256(json.dumps([state.flags, decision["action_kind"]], sort_keys=True).encode()).hexdigest()
                    if key != saved.get("last_advice"):
                        action = next(a for a in default_actions() if a.id == decision["action_id"])
                        response = {"hookSpecificOutput": {"hookEventName": name,
                                    "additionalContext": f"Local advisor: {action.label}. Use this only if relevant."}}
                        saved["last_advice"] = key
                        saved["advice_count"] = saved.get("advice_count", 0) + 1
        if name == "Stop" and turn not in saved.get("closed_turns", []):
            saved["closed_turns"] = (saved.get("closed_turns", []) + [turn])[-100:]
            counter_path = data_dir / "counter.json"
            counter = read_json(counter_path) if counter_path.exists() else {"completed": 0, "last_trained": 0}
            counter["completed"] += 1
            write_json(counter_path, counter)
            trigger = (counter["completed"] - counter["last_trained"] >= 25
                       and time.time() - counter.get("pending_since", 0) > 1200)
            if trigger and spawn_worker:
                counter["pending_since"] = time.time()
                write_json(counter_path, counter)
        saved["state"] = state.to_dict()
        write_json(session_file, saved)
    if trigger and spawn_worker:
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        with (data_dir / "worker.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-m", "local_decider", "worker", "--data-dir", str(data_dir)],
                             stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=environment,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return response


def run_hook(data_dir: Path, candidate: Path | None, experimental: bool) -> None:
    try:
        raw = sys.stdin.read(256_001)
        if len(raw) > 256_000:
            raise ValueError("hook input exceeds limit")
        event = json.loads(raw)
        result = handle(event, data_dir, candidate, experimental)
    except Exception:
        # Failure never blocks Codex, invokes a shell, or leaks payloads to stdout.
        result = {}
    print(json.dumps(result))


def install(project: Path, data_dir: Path, candidate: Path | None = None,
            experimental: bool = False) -> Path:
    import shlex

    config_file = project.resolve() / ".codex" / "hooks.json"
    config = read_json(config_file) if config_file.exists() else {"hooks": {}}
    args = [sys.executable, "-m", "local_decider", "hook", "--data-dir", str(data_dir.resolve())]
    if experimental:
        if candidate is None:
            raise ValueError("experimental hooks require a candidate model")
        args += ["--candidate", str(candidate.resolve()), "--experimental"]
    command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
    for name in ("UserPromptSubmit", "PostToolUse", "Stop"):
        groups = config.setdefault("hooks", {}).setdefault(name, [])
        groups = [g for g in groups if g.get("description") != "codex-local-decider"]
        groups.append({"description": "codex-local-decider", "hooks": [{"type": "command",
                       "command": command, "timeout": 3, "additionalContextLimit": 100}]})
        config["hooks"][name] = groups
    write_json(config_file, config)
    return config_file


def worker(data_dir: Path) -> dict:
    from .evolution import Config, compare, train

    with Lock(data_dir / "training.lock"):
        with Lock(data_dir / "events.lock"):
            counter = read_json(data_dir / "counter.json")
            if counter["completed"] - counter["last_trained"] < 25:
                return {"status": "not_due"}
            episodes_at_start = counter["completed"]
        # Both candidates share one 15-minute background budget, not 15 minutes each.
        started = time.perf_counter()
        output = data_dir / "candidates"
        trained = []
        for kind in ("linear", "neural"):
            trained.append(train(kind, output, Config(minutes=7.5, seed=7 + episodes_at_start),
                                 experience_path=data_dir / "experience.jsonl"))
        result = compare(output, seed=7 + episodes_at_start)
        write_json(output / "comparison.json", result)
        training_record = {"elapsed_seconds": time.perf_counter() - started,
                   "completed_episodes_at_start": episodes_at_start,
                   "models": [{k: v for k, v in model.items() if k != "history"} for model in trained],
                   "active_changed": False}
        write_json(output / "worker-training.json", training_record)
        with (data_dir / "worker-history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(training_record) + "\n")
        with Lock(data_dir / "events.lock"):
            current = read_json(data_dir / "counter.json")
            current["last_trained"] = episodes_at_start
            current["pending_since"] = 0
            write_json(data_dir / "counter.json", current)
        return {"status": "candidate_ready", "candidate": result["candidate"],
                "activated": False, "reason": "Requires a new live validation; active model unchanged"}
