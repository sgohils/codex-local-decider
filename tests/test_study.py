import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from local_decider.features import DIM, TEXT_DIM
from local_decider.models import Model
from local_decider.storage import read_json, write_json
from local_decider.study import (MARKER, MAX_RECORD_BYTES, MAX_STREAM_READ_BYTES,
                                analyze, assign, digest, export, handle, install, main,
                                outcome, prepare, refresh, set_probe_advice, status, stop, utc)


def append(path, record):
    record.setdefault("timestamp", utc(time.time()))
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


@pytest.fixture
def setup(tmp_path):
    candidate = tmp_path / "candidate.json"
    weights = np.zeros(DIM)
    weights[TEXT_DIM + 9] = 1  # Confident file inspection advice.
    Model("linear", weights, "linear-s7-g40").save(candidate)
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    data = tmp_path / "data"
    prepare(data, candidate, transcript_roots=[str(transcripts)])
    return data, root, transcripts


def session(setup, name, created=None):
    data, root, transcripts = setup
    now = time.time() + 1
    path = transcripts / f"{name}.jsonl"
    append(path, {"type": "session_meta", "timestamp": utc(now), "payload": {
        "id": name, "timestamp": utc(now if created is None else created), "source": "vscode"}})
    event = {"session_id": name, "turn_id": f"{name}-turn", "cwd": str(root),
             "model": "test-model", "transcript_path": str(path), "hook_event_name": "UserPromptSubmit",
             "prompt": "Fix a code bug password=DO_NOT_STORE this_identifier"}
    handle(event, data, now=now + 1)
    return event, path


def use(event, data, name, **kwargs):
    return handle({**event, "hook_event_name": name, **kwargs}, data)


def usage(path, event, response="r1", inputs=100, outputs=10, stamp=None):
    append(path, {"timestamp": stamp or utc(time.time()), "type": "token_usage_record", "payload": {
        "thread_id": event["session_id"], "turn_id": event["turn_id"], "root_turn_id": event["turn_id"],
        "response_id": response, "usage": {"input_tokens": inputs, "cached_input_tokens": 40,
                                            "output_tokens": outputs}}})


def start(setup):
    data, _, _ = setup
    event, path = session(setup, "probe")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
        tool_response="logic.py: solve", tool_use_id="probe-call")
    usage(path, event)
    use(event, data, "Stop")
    assert status(data)["status"] == "running"
    return data


def test_real_probe_is_required_and_excluded(setup):
    data, _, _ = setup
    event, path = session(setup, "probe")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"}, tool_response="logic.py")
    use(event, data, "Stop")
    assert status(data)["status"] == "armed"
    usage(path, event)
    refresh(data)  # Stop's record may arrive just after the hook itself.
    current = read_json(data / "study.json")
    assert current["status"] == "running"
    assert current["deadline"] - current["started_at"] == 5 * 86400
    assert export(data)["sessions"] == []


def test_explicit_probe_advice_persists_and_stays_excluded(setup):
    data, _, _ = setup
    event, path = session(setup, "demo-probe")
    assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
               tool_response="logic.py: solve", tool_use_id="before") == {}
    identity = digest(event["session_id"])
    set_probe_advice(data, identity)
    set_probe_advice(data, identity)  # Idempotent, without duplicate protocol amendments.
    result = use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
                 tool_response="logic.py: solve", tool_use_id="after")
    assert result["hookSpecificOutput"]["additionalContext"].startswith("Local advisor:")
    saved = read_json(data / "sessions" / f"{identity}.json")
    assert saved["arm"] == "probe" and saved["probe_advice_enabled"]
    current = status(data)
    assert current["status"] == "armed" and current["started_at"] is None
    assert current["groups"]["probe"]["model_decisions"] == 1
    assert current["groups"]["probe"]["advice"] == 1
    assert len(export(data)["protocol_amendments"]) == 1
    assert export(data)["sessions"] == []
    usage(path, event)
    use(event, data, "Stop")
    assert status(data)["status"] == "running"
    assert read_json(data / "study.json")["verification"]["probe_advice_enabled"]
    assert export(data)["sessions"] == []


def test_probe_opt_in_cannot_override_randomized_arms_or_expiry(setup):
    data = start(setup)
    for i in range(4):
        event, _ = session(setup, f"protected-{i}")
        identity = digest(event["session_id"])
        file = data / "sessions" / f"{identity}.json"
        before = read_json(file)
        with pytest.raises(ValueError, match="restricted to excluded probe"):
            set_probe_advice(data, identity)
        assert read_json(file) == before
    study = read_json(data / "study.json")
    study["deadline"] = time.time() - 1
    write_json(data / "study.json", study)
    with pytest.raises(ValueError, match="expired"):
        set_probe_advice(data, digest("probe"))
    with pytest.raises(ValueError, match="valid chat hash"):
        set_probe_advice(data, "../study")


def test_probe_advice_disable_and_model_failure_allow_normal_work(setup):
    data, _, _ = setup
    event, _ = session(setup, "toggle-probe")
    identity = digest(event["session_id"])
    set_probe_advice(data, identity)
    set_probe_advice(data, identity, False)
    assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
               tool_response="logic.py", tool_use_id="disabled") == {}
    assert status(data)["groups"]["probe"]["model_decisions"] == 0
    set_probe_advice(data, identity)
    Path(read_json(data / "study.json")["candidate"]).write_text("broken")
    assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
               tool_response="logic.py", tool_use_id="corrupt") == {}
    turn = next(iter(read_json(data / "sessions" / f"{identity}.json")["turns"].values()))
    assert turn["model_failures"] == 1 and turn["advice_count"] == 0
    with pytest.raises(ValueError, match="checkpoint changed"):
        set_probe_advice(data, identity)


def test_block_randomization_reproducible_balanced_and_stratified():
    one = {"seed": 12, "strata": {}}
    two = {"seed": 12, "strata": {}}
    arms = [assign(one, "project", "model") for _ in range(12)]
    assert arms == [assign(two, "project", "model") for _ in range(12)]
    assert all(arms[i:i + 4].count("assisted") == 2 for i in range(0, 12, 4))
    assign(one, "another-project", "model")
    assert len(one["strata"]) == 2


def test_arm_persists_and_baseline_never_injects(setup):
    data = start(setup)
    for i in range(4):
        event, _ = session(setup, f"chat-{i}")
        file = data / "sessions" / f"{digest(event['session_id'])}.json"
        arm = read_json(file)["arm"]
        result = use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
                     tool_response="logic.py: solve", tool_use_id="one")
        assert bool(result) == (arm == "assisted")
        assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"},
                   tool_response="logic.py: solve", tool_use_id="one") == {}
        use(event, data, "SessionStart", source="compact")
        use(event, data, "UserPromptSubmit", turn_id="follow-up", prompt="Run the tests")
        assert read_json(file)["arm"] == arm
        assert len(read_json(file)["turns"]) == 2
    assert status(data)["groups"]["baseline"]["chats"] == 2


def test_request_ids_override_cumulative_records_and_deduplicate(setup):
    data = start(setup)
    event, path = session(setup, "cost")
    append(path, {"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10}}}})
    usage(path, event)
    usage(path, event)
    usage(path, event, "r2", 200, 20)
    use(event, data, "Stop")
    use(event, data, "Stop")
    row = export(data)["sessions"][0]["turns"][0]
    assert row["usage"] == {"input_tokens": 300, "cached_input_tokens": 80, "output_tokens": 30}
    assert read_json(data / "study.json")["completed_episodes"] == 1
    assert analyze(export(data))["groups"][export(data)["sessions"][0]["arm"]]["total_tokens"] == 330


def test_cumulative_fallback_and_partial_record_recovery(setup):
    data = start(setup)
    event, path = session(setup, "legacy")
    row = {"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10}}}}
    append(path, row)
    append(path, row)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row))
    use(event, data, "Stop")
    assert not export(data)["sessions"][0]["turns"][0]["usage_valid"]
    with path.open("a", encoding="utf-8") as file:
        file.write("\n")
    refresh(data)
    row = export(data)["sessions"][0]["turns"][0]
    assert row["usage_valid"] and row["usage"]["input_tokens"] == 100


def test_existing_research_and_out_of_scope_chats_are_excluded(setup):
    data = start(setup)
    session(setup, "old", created=0)
    event, path = session(setup, "placeholder", created=0)
    for name, change in (("research", {"prompt": "Write a research paper about code"}),
                         ("plan", {"permission_mode": "plan"}),
                         ("outside", {"cwd": str(data)})):
        new = {**event, "session_id": name, **change}
        fresh = path.with_name(f"{name}.jsonl")
        append(fresh, {"type": "session_meta", "payload": {"id": name,
                      "timestamp": utc(time.time() + 1), "source": "vscode"}})
        new["transcript_path"] = str(fresh)
        handle(new, data)
    assert len(export(data)["sessions"]) == 0


def test_privacy_and_explicit_outcome_not_inferred_from_test_pass(setup, tmp_path):
    data = start(setup)
    event, path = session(setup, "privacy")
    use(event, data, "PostToolUse", tool_name="apply_patch", tool_input={"patch": "SECRET SOURCE CODE"}, tool_use_id="patch")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "pytest test_logic.py"},
        tool_response={"exit_code": 0, "output": "1 passed SECRET SOURCE CODE"}, tool_use_id="test")
    usage(path, event)
    use(event, data, "Stop")
    encoded = json.dumps(export(data))
    assert all(secret not in encoded for secret in ("DO_NOT_STORE", "SECRET SOURCE CODE", "this_identifier", str(path)))
    row = export(data)["sessions"][0]["turns"][0]
    assert row["test_verified_after_last_patch"] and row["outcome"] == "unknown"
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("operator verified the acceptance criteria")
    outcome(data, digest(event["session_id"]), digest(event["turn_id"]), "solved", evidence)
    assert export(data)["sessions"][0]["turns"][0]["outcome"] == "solved"
    for file in (data / "sessions").glob("*.json"):
        text = file.read_text()
        assert "DO_NOT_STORE" not in text and "SECRET SOURCE CODE" not in text


def test_model_corruption_does_not_change_assignment_or_block(setup):
    data = start(setup)
    for i in range(4):
        event, _ = session(setup, f"broken-{i}")
        file = data / "sessions" / f"{digest(event['session_id'])}.json"
        if read_json(file)["arm"] == "assisted":
            study = read_json(data / "study.json")
            Path(study["candidate"]).write_text("broken")
            assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"}, tool_response="logic.py") == {}
            turn = next(iter(read_json(file)["turns"].values()))
            assert turn["model_failures"] == 1
            break


def test_concurrent_chats_keep_unique_balanced_assignments(setup):
    data = start(setup)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: session(setup, f"concurrent-{i}"), range(8)))
    value = export(data)
    assert len(value["sessions"]) == 8
    assert sum(s["arm"] == "assisted" for s in value["sessions"]) == 4
    assert len(set(s["session_hash"] for s in value["sessions"])) == 8


def test_expiry_cutoff_and_preserving_other_hooks(setup, tmp_path):
    data = start(setup)
    config = tmp_path / "hooks.json"
    write_json(config, {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "before"}]}]}})
    install(data, config)
    install(data, config)
    assert len(read_json(config)["hooks"]["Stop"]) == 2
    event, path = session(setup, "expiry")
    value = read_json(data / "study.json")
    value["deadline"] = time.time() - 1
    write_json(data / "study.json", value)
    usage(path, event, "before", stamp=utc(value["deadline"] - 1))
    usage(path, event, "after", inputs=999, stamp=utc(value["deadline"] + 1))
    assert use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"}) == {}
    changed = read_json(config)
    changed["hooks"]["Stop"].append({"hooks": [{"command": "added-later"}]})
    write_json(config, changed)
    stop(data)
    frozen = read_json(data / "frozen.json")
    assert frozen["sessions"][0]["turns"][0]["usage"]["input_tokens"] == 100
    remaining = read_json(config)["hooks"]["Stop"]
    assert [g["hooks"][0]["command"] for g in remaining] == ["before", "added-later"]
    assert all(g.get("description") != MARKER for g in remaining)


def test_subagent_usage_is_grouped_and_not_double_counted(setup):
    data = start(setup)
    event, path = session(setup, "parent")
    use(event, data, "SubagentStart", agent_id="child")
    child_path = path.with_name("child.jsonl")
    append(child_path, {"type": "session_meta", "payload": {"id": "child", "timestamp": utc(time.time()),
                       "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent"}}}}})
    usage(path, event, "parent-response")
    child = {**event, "session_id": "child", "turn_id": "child-turn"}
    usage(child_path, child, "child-response", 200, 20)
    use(event, data, "SubagentStop", agent_id="child", agent_transcript_path=str(child_path))
    use(event, data, "SubagentStop", agent_id="child", agent_transcript_path=str(child_path))
    use(event, data, "Stop")
    value = export(data)
    assert len(value["sessions"]) == 1
    row = value["sessions"][0]["turns"][0]
    assert row["usage_valid"] and row["usage"]["input_tokens"] == 300
    assert row["missing_subagent_usage"] == 0


def test_missing_child_usage_is_unknown_not_zero(setup):
    data = start(setup)
    event, path = session(setup, "missing-child")
    use(event, data, "SubagentStart", agent_id="child")
    usage(path, event)
    use(event, data, "Stop")
    row = export(data)["sessions"][0]["turns"][0]
    assert not row["usage_valid"] and row["missing_subagent_usage"] == 1
    arm = export(data)["sessions"][0]["arm"]
    assert status(data)["groups"][arm]["measured_turns"] == 0


def test_analysis_resamples_chats_and_rejects_small_sample(setup):
    data = start(setup)
    event, path = session(setup, "one")
    usage(path, event)
    use(event, data, "Interrupt")
    small = analyze(export(data))
    assert small["chat_bootstrap_95_interval"] is None and "inconclusive" in small["interpretation"]
    row = export(data)["sessions"][0]
    sessions = []
    for arm, tokens in (("baseline", 1000), ("assisted", 800)):
        for i in range(10):
            item = json.loads(json.dumps(row))
            item.update(arm=arm, session_hash=f"{arm}-{i}")
            item["turns"][0]["usage"] = {"input_tokens": tokens, "output_tokens": 0, "cached_input_tokens": 0}
            sessions.append(item)
    result = analyze({**export(data), "sessions": sessions}, bootstrap_samples=100)
    assert result["token_reduction_fraction"] == pytest.approx(.2)
    assert result["chat_bootstrap_95_interval"] == pytest.approx([.2, .2])


def test_response_revisions_update_once_and_invalid_corrections_are_unknown(setup):
    data = start(setup)
    event, path = session(setup, "revised")
    usage(path, event, "same-response", 100, 10)
    usage(path, event, "same-response", 100, 20)
    use(event, data, "Stop")
    row = export(data)["sessions"][0]["turns"][0]
    assert row["usage_valid"] and row["usage"]["output_tokens"] == 20
    usage(path, event, "same-response", 100, 5)
    refresh(data)
    assert not export(data)["sessions"][0]["turns"][0]["usage_valid"]


def test_crashed_hook_releases_registry_lock(tmp_path):
    from local_decider.storage import Lock
    from local_decider.study import StudyLock
    path = tmp_path / "registry.lock"
    code = ("import os,sys; from pathlib import Path; from local_decider.storage import Lock; "
            "lock=Lock(Path(sys.argv[1])); lock.__enter__(); print('locked',flush=True); os._exit(0)")
    process = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, timeout=10)
    assert process.returncode == 0 and b"locked" in process.stdout
    with StudyLock(path):
        with pytest.raises(FileExistsError):
            with Lock(path):
                pass
    with StudyLock(path):
        pass


def test_large_content_record_does_not_stall_or_count_embedded_usage(setup):
    data, _, _ = setup
    event, path = session(setup, "large-content")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    embedded = json.dumps({"type": "token_usage_record", "usage": {"input_tokens": 999999}})
    append(path, {"timestamp": utc(time.time()), "type": "response_item",
                  "payload": {"output": embedded + "PRIVATE_SOURCE" * 22000}})
    usage(path, event)
    use(event, data, "Stop")
    assert status(data)["status"] == "running"
    saved = read_json(data / "sessions" / f"{digest(event['session_id'])}.json")
    stream = next(iter(saved["streams"].values()))
    turn = next(iter(saved["turns"].values()))
    assert stream["skipped_content_records"] == 1 and not stream["backlog"]
    assert not stream["warnings"] and turn["usage"]["input_tokens"] == 100
    assert "PRIVATE_SOURCE" not in json.dumps(saved)


def test_partial_large_content_resumes_at_the_line_boundary(setup):
    data, _, _ = setup
    event, path = session(setup, "partial-large")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    encoded = json.dumps({"timestamp": utc(time.time()), "type": "response_item",
                          "payload": {"output": "x" * (MAX_RECORD_BYTES * 3)}})
    split = MAX_RECORD_BYTES + 500
    with path.open("a", encoding="utf-8") as file:
        file.write(encoded[:split])
    use(event, data, "Stop")
    assert status(data)["status"] == "armed"
    with path.open("a", encoding="utf-8") as file:
        file.write(encoded[split:] + "\n")
    usage(path, event)
    refresh(data)
    current = status(data)
    assert current["status"] == "running"
    assert current["groups"]["probe"]["skipped_content_records"] == 1


def test_large_content_recovers_across_the_bounded_read_budget(setup):
    data, _, _ = setup
    event, path = session(setup, "bounded-large")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    append(path, {"timestamp": utc(time.time()), "type": "response_item",
                  "payload": {"output": "x" * (MAX_STREAM_READ_BYTES + 500000)}})
    usage(path, event)
    use(event, data, "Stop")
    assert status(data)["status"] == "armed"
    saved = read_json(data / "sessions" / f"{digest(event['session_id'])}.json")
    stream = next(iter(saved["streams"].values()))
    assert stream["offset"] < MAX_STREAM_READ_BYTES + 2000
    assert stream["backlog"] and stream["discarding_record"]
    refresh(data)
    assert status(data)["status"] == "running"
    assert status(data)["groups"]["probe"]["skipped_content_records"] == 1


def test_oversized_usage_record_remains_unknown_without_stalling(setup):
    data, _, _ = setup
    event, path = session(setup, "unknown-large")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    append(path, {"timestamp": utc(time.time()), "type": "token_usage_record",
                  "payload": {"padding": "x" * (MAX_RECORD_BYTES + 10)}})
    usage(path, event)
    assert use(event, data, "Stop") == {}
    current = status(data)
    assert current["status"] == "armed" and current["warnings"] == ["oversized_record"]
    saved = read_json(data / "sessions" / f"{digest(event['session_id'])}.json")
    assert not next(iter(saved["streams"].values()))["backlog"]
    assert current["groups"]["probe"]["measured_turns"] == 0


def test_unrelated_installed_hook_reconciles_a_completed_probe(setup, monkeypatch, capsys):
    data, _, _ = setup
    event, path = session(setup, "late-probe")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    use(event, data, "Stop")
    assert status(data)["status"] == "armed"
    usage(path, event)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "hook_event_name": "SessionStart", "session_id": "unrelated-maintenance"})))
    main(["hook", "--data-dir", str(data)])
    assert json.loads(capsys.readouterr().out) == {}
    assert status(data)["status"] == "running"
    assert len(read_json(data / "study.json")["sessions"]) == 1
    assert export(data)["sessions"] == []


def test_probe_waits_for_complete_subagent_usage(setup):
    data, _, _ = setup
    event, path = session(setup, "probe-child")
    use(event, data, "PostToolUse", tool_name="Bash", tool_input={"command": "rg solve"})
    use(event, data, "SubagentStart", agent_id="child")
    usage(path, event)
    use(event, data, "Stop")
    assert status(data)["status"] == "armed"
    assert status(data)["groups"]["probe"]["measured_turns"] == 0
    child_path = path.with_name("probe-child-agent.jsonl")
    append(child_path, {"type": "session_meta", "payload": {
        "id": "child", "timestamp": utc(time.time()),
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": "probe-child"}}}}})
    usage(child_path, {**event, "session_id": "child", "turn_id": "child-turn"}, "child-response")
    use(event, data, "SubagentStop", agent_id="child", agent_transcript_path=str(child_path))
    assert status(data)["status"] == "running"
    assert status(data)["groups"]["probe"]["measured_turns"] == 1
    assert export(data)["sessions"] == []
