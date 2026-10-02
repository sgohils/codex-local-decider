"""Paired fresh-checkout evaluations via the authenticated Codex CLI's JSON events."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .exercises import TEMPLATES, Exercise
from .hooks import install
from .storage import read_json, write_json


def parse_events(text: str) -> dict:
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    completed, commands, failures = False, 0, 0
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if event.get("type") == "turn.completed":
            completed = True
            for key in usage:
                usage[key] += int(event.get("usage", {}).get(key, 0))
        if event.get("type") in ("turn.failed", "error"):
            failures += 1
        if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "command_execution":
            commands += 1
    return {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "turn_completed": completed, "command_count": commands, "failures": failures}


def qualifies(result: dict) -> bool:
    rows = result.get("pairs", [])
    if result.get("status") != "complete" or len(rows) < 12:
        return False
    if any(not r[arm].get("usage_valid") for r in rows for arm in ("baseline", "assisted")):
        return False
    baseline = sum(r["baseline"]["success"] for r in rows)
    assisted = sum(r["assisted"]["success"] for r in rows)
    before = sum(r["baseline"]["total_tokens"] for r in rows)
    after = sum(r["assisted"]["total_tokens"] for r in rows)
    # At least one actual recommendation must have been delivered, not just a no-op arm.
    return bool(baseline and assisted >= baseline and before > 0 and after <= before * 0.9
                and sum(r["assisted"].get("advice_count", 0) for r in rows) > 0)


def run(candidate: Path, output: Path, git: str, codex: str = "codex",
        pairs: int = 12, max_tokens: int = 100_000, timeout: int = 180,
        model: str | None = None, progress=None) -> dict:
    if not 1 <= pairs <= len(TEMPLATES) or max_tokens <= 0 or timeout <= 0:
        raise ValueError("invalid benchmark limits")
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "running", "pairs": [], "max_tokens": max_tokens,
              "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "codex_model": model or "client default", "metric": "input + output tokens; cached input included",
              "limitations": ["Small paired pilot; no claim of statistical significance",
                              "Does not directly measure billable dollars or subscription allowance"]}
    consumed = 0
    for index in range(pairs):
        row = {"task": TEMPLATES[index][0]}
        # Alternate arm order to reduce warm-cache/order bias.
        for arm in (("baseline", "assisted") if index % 2 == 0 else ("assisted", "baseline")):
            if consumed >= max_tokens:
                result["status"] = "budget_exhausted"
                write_json(output / "live.json", result)
                return result
            target = output / f"task-{index}" / arm
            if target.exists():
                raise ValueError("benchmark output must be fresh; existing checkouts will not be overwritten")
            Exercise(f"live-{index}", index, 0, "live", 7).materialize(target)
            original_test = (target / "test_logic.py").read_bytes()
            subprocess.run([git, "init", "-b", "main"], cwd=target, capture_output=True, check=True)
            subprocess.run([git, "add", "."], cwd=target, capture_output=True, check=True)
            subprocess.run([git, "-c", "user.name=Benchmark", "-c", "user.email=benchmark@example.invalid",
                            "commit", "-m", "Seeded exercise"], cwd=target, capture_output=True, check=True)
            if arm == "assisted":
                install(target, target / ".local-decider", candidate, experimental=True)
            prompt = ("Fix the defect in logic.py, preserving test_logic.py. Verify using "
                      f"{sys.executable} -B -m unittest -q. This is a small local exercise; "
                      "do not install packages or use the network. Give a brief result.")
            executable = shutil.which(codex) or codex
            command = [executable, "exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
                       "--sandbox", "workspace-write", "--cd", str(target.resolve())]
            # Only our reviewed, generated hooks are used in the assisted evaluation.
            if arm == "assisted":
                command += ["--dangerously-bypass-hook-trust"]
            command += ["-c", f'projects.{json.dumps(str(target.resolve()))}.trust_level="trusted"']
            # Benchmark databases/logs are isolated from existing personal Codex runtime state.
            runtime = target / ".benchmark-runtime"
            runtime.mkdir()
            command += ["-c", f"sqlite_home={json.dumps(str(runtime.resolve()))}",
                        "-c", f"log_dir={json.dumps(str((runtime / 'logs').resolve()))}"]
            if model:
                command += ["--model", model]
            command += ["-"]
            if os.name == "nt" and executable.lower().endswith((".cmd", ".ps1")):
                # Invoke the installed JS CLI directly, avoiding shell command interpolation.
                base = Path(executable).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
                node = shutil.which("node")
                if not base.is_file() or not node:
                    raise RuntimeError("Cannot resolve Codex JS CLI and Node runtime")
                command = [node, str(base), *command[1:]]
            started = time.perf_counter()
            try:
                process = subprocess.run(command, input=prompt, text=True, encoding="utf-8",
                                         errors="replace", capture_output=True, timeout=timeout,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                events, error, returncode = process.stdout, process.stderr, process.returncode
            except subprocess.TimeoutExpired as exc:
                events = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
                error, returncode = "Evaluation timeout", -1
            metrics = parse_events(events)
            elapsed = time.perf_counter() - started
            consumed += metrics["total_tokens"]
            # Logs contain only our synthetic fixture tasks and stay in ignored runs/ by default.
            (target / "events.jsonl").write_text(events, encoding="utf-8")
            (target / "stderr.log").write_text(error, encoding="utf-8")
            verification = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q"], cwd=target,
                                           capture_output=True, timeout=20)
            advice = 0
            for path in (target / ".local-decider" / "sessions").glob("*.json"):
                advice += read_json(path).get("advice_count", 0)
            metrics.update(elapsed_seconds=elapsed, success=bool(
                returncode == 0 and metrics["turn_completed"] and verification.returncode == 0
                and (target / "test_logic.py").read_bytes() == original_test),
                usage_valid=metrics["turn_completed"] and metrics["total_tokens"] > 0,
                advice_count=advice, returncode=returncode)
            row[arm] = metrics
            if progress:
                progress({"task": index + 1, "arm": arm, **metrics})
            if returncode != 0 and not metrics["turn_completed"]:
                result["status"] = "blocked"
                result["blocking_error"] = error[-2000:]
                result["incomplete_pair"] = row
                write_json(output / "live.json", result)
                return result
        result["pairs"].append(row)
        write_json(output / "live.json", result)
    result["status"] = "complete"
    result["qualifies"] = qualifies(result)
    write_json(output / "live.json", result)
    return result


def promote(candidate: Path, live_result: Path, data_dir: Path) -> dict:
    from .models import Model

    model = Model.load(candidate)
    result = read_json(live_result)
    if result.get("candidate_sha256") != hashlib.sha256(candidate.read_bytes()).hexdigest():
        raise ValueError("live evaluation belongs to a different candidate checkpoint")
    if not qualifies(result):
        raise ValueError("candidate has not met the live token and success acceptance criteria")
    active = data_dir / "active.json"
    if active.exists():
        write_json(data_dir / "previous-active.json", read_json(active))
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    snapshot = data_dir / "models" / f"{digest}.json"
    model.save(snapshot)
    manifest = {"model": str(snapshot.resolve()), "version": model.version,
                "live_validated": True, "evaluation": str(live_result.resolve())}
    write_json(active, manifest)
    return manifest
