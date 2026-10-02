import json
import subprocess
import sys

import numpy as np
import pytest

from local_decider.benchmark import parse_events, promote, qualifies
from local_decider.contracts import Action, State, actions_from_dict, default_actions
from local_decider.evolution import Config, compare, evaluate, fitness, train
from local_decider.exercises import dataset, verify_fixtures
from local_decider.features import DIM, encode
from local_decider.hooks import handle, install, redact
from local_decider.models import Model, parameter_count
from local_decider.storage import read_json, write_json


def test_template_and_repository_isolation():
    data = dataset()
    groups = {key: {e.template for e in value} for key, value in data.items()}
    assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert len({e.id for values in data.values() for e in values}) == 120


def test_all_seeded_defects_fail_and_reference_repairs_pass(tmp_path):
    assert verify_fixtures(tmp_path)["passed"]


@pytest.mark.parametrize("kind", ["linear", "neural"])
def test_model_round_trip_and_bounded_actions(kind, tmp_path):
    model = Model(kind, np.random.default_rng(8).normal(size=parameter_count(kind)), "v1")
    path = tmp_path / "model.json"
    model.save(path)
    loaded = Model.load(path)
    available = [Action("read-main", "inspect_file", "Read main"), Action("handoff", "defer", "Defer")]
    state = State("Find an incorrect result", flags={"path_known": True})
    assert loaded.decide(state, available) == model.decide(state, available)
    assert loaded.decide(state, available)["action_id"] in {a.id for a in available}
    assert loaded.decide(State(""), available)["uncertain"]


def test_invalid_models_inputs_and_nan_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        actions_from_dict([{"id": "a", "kind": "invented", "label": "x"}])
    with pytest.raises(ValueError):
        actions_from_dict([{"id": "a", "kind": "defer", "label": "x"}] * 2)
    with pytest.raises(ValueError):
        State.from_dict({"task": "x", "flags": {"patched": "yes"}})
    model = Model("linear", np.full(DIM, np.nan))
    with pytest.raises(ValueError):
        model.decide(State("x"), default_actions())


def test_features_are_stable_across_processes():
    code = "from local_decider.features import encode; from local_decider.contracts import State,default_actions; print(encode(State('read code'),default_actions()).tolist())"
    first = subprocess.check_output([sys.executable, "-c", code])
    second = subprocess.check_output([sys.executable, "-c", code])
    assert first == second
    assert encode(State("x"), default_actions()).shape == (6, DIM)


def test_success_precedes_token_reduction():
    assert fitness({"successes": 1, "mean_proxy_tokens": 10000, "mean_redundant_steps": 30}) > fitness(
        {"successes": 0, "mean_proxy_tokens": 1, "mean_redundant_steps": 0})
    assert evaluate("rules", dataset()["test"])["success_rate"] == 1


@pytest.mark.parametrize("kind", ["linear", "neural"])
def test_seeded_training_resume_equals_uninterrupted(kind, tmp_path):
    full, partial = tmp_path / "full", tmp_path / "partial"
    train(kind, full, Config(population=4, parents=2, generations=4, workers=1))
    train(kind, partial, Config(population=4, parents=2, generations=2, workers=1))
    train(kind, partial, Config(population=4, parents=2, generations=4, workers=1), resume=True)
    assert np.array_equal(Model.load(full / f"{kind}.json").weights,
                          Model.load(partial / f"{kind}.json").weights)
    assert read_json(full / f"{kind}-population.json")["rng_state"] == read_json(partial / f"{kind}-population.json")["rng_state"]
    with pytest.raises(ValueError):
        train(kind, partial, Config(population=5, parents=2, generations=5), resume=True)


def test_limits_and_validation_selection(tmp_path):
    for config in (Config(workers=3), Config(minutes=16), Config(population=2, parents=2)):
        with pytest.raises(ValueError):
            config.validate()
    for kind in ("linear", "neural"):
        train(kind, tmp_path, Config(population=4, parents=2, generations=1, workers=1))
    result = compare(tmp_path)
    assert result["live_validated"] is False
    assert set(result["metrics"]) == {"linear", "neural", "rules", "random"}


def event(name, **kwargs):
    return {"hook_event_name": name, "session_id": "s", "turn_id": "t", **kwargs}


def test_hook_missing_model_is_noop_and_stop_is_deduplicated(tmp_path):
    handle(event("UserPromptSubmit", prompt="Fix a bug"), tmp_path)
    assert handle(event("PostToolUse", tool_name="Bash", tool_input={"command": "rg foo"},
                        tool_response="main.py: foo"), tmp_path,
                  tmp_path / "missing.json", experimental=True) == {}
    handle(event("Stop"), tmp_path, spawn_worker=False)
    handle(event("Stop"), tmp_path, spawn_worker=False)
    assert read_json(tmp_path / "counter.json")["completed"] == 1


def test_hook_recommendation_suppression_and_activation_gate(tmp_path):
    weights = np.zeros(DIM)
    from local_decider.features import TEXT_DIM, FLAGS
    weights[TEXT_DIM + 9] = 1  # inspect_file action bias
    model = Model("linear", weights, "test")
    checkpoint = tmp_path / "candidate.json"
    model.save(checkpoint)
    handle(event("UserPromptSubmit", prompt="Fix bug"), tmp_path)
    observation = event("PostToolUse", tool_name="Bash", tool_input={"command": "rg foo"},
                        tool_response="logic.py: foo")
    assert handle(observation, tmp_path, checkpoint) == {}  # no live-validated active manifest
    advice = handle(observation, tmp_path, checkpoint, experimental=True)
    assert "Inspect" in advice["hookSpecificOutput"]["additionalContext"]
    assert handle(observation, tmp_path, checkpoint, experimental=True) == {}
    assert len(FLAGS) == 7


def test_hook_collects_verified_test_decisions_and_preserves_other_hooks(tmp_path):
    handle(event("UserPromptSubmit", prompt="Fix bug token=abc123"), tmp_path)
    handle(event("PostToolUse", tool_name="apply_patch", tool_input={}), tmp_path)
    handle(event("PostToolUse", tool_name="Bash", tool_input={"command": "pytest test_logic.py"},
                 tool_response={"output": "1 passed", "exit_code": 0}), tmp_path)
    sample = json.loads((tmp_path / "experience.jsonl").read_text())
    assert sample["verified_test_pass"] and "abc123" not in json.dumps(sample)
    assert redact("password=secret sk-exampletoken") == "password=[REDACTED] [REDACTED]"
    project = tmp_path / "project"
    write_json(project / ".codex/hooks.json", {"hooks": {"Stop": [{"hooks": [{"command": "other"}]}]}})
    install(project, tmp_path)
    install(project, tmp_path)
    assert len(read_json(project / ".codex/hooks.json")["hooks"]["Stop"]) == 2


def test_hook_malformed_payload_does_not_block(tmp_path):
    process = subprocess.run([sys.executable, "-m", "local_decider", "hook", "--data-dir", str(tmp_path)],
                             input="not JSON", text=True, capture_output=True)
    assert process.returncode == 0 and json.loads(process.stdout) == {}


def test_live_metrics_and_promotion_reject_incomplete_or_failed_trial(tmp_path):
    metrics = parse_events('\n'.join([json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}}), "bad json"]))
    assert metrics["total_tokens"] == 120 and metrics["cached_input_tokens"] == 40
    pair = {"baseline": {"success": True, "total_tokens": 1000, "usage_valid": True},
            "assisted": {"success": True, "total_tokens": 850, "usage_valid": True, "advice_count": 1}}
    result = {"status": "complete", "pairs": [pair] * 12}
    assert qualifies(result)
    result["status"] = "budget_exhausted"
    assert not qualifies(result)
    candidate = tmp_path / "candidate.json"
    Model("linear", np.zeros(DIM)).save(candidate)
    result_path = tmp_path / "live.json"
    write_json(result_path, result)
    with pytest.raises(ValueError):
        promote(candidate, result_path, tmp_path / "active")


def test_verified_promotion_has_rollback_snapshot(tmp_path):
    import hashlib
    candidate = tmp_path / "candidate.json"
    Model("linear", np.zeros(DIM), "v1").save(candidate)
    row = {"baseline": {"success": True, "total_tokens": 1000, "usage_valid": True},
           "assisted": {"success": True, "total_tokens": 800, "usage_valid": True, "advice_count": 1}}
    result = {"status": "complete", "pairs": [row] * 12,
              "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    write_json(tmp_path / "live.json", result)
    write_json(tmp_path / "active.json", {"version": "old"})
    assert promote(candidate, tmp_path / "live.json", tmp_path)["version"] == "v1"
    assert read_json(tmp_path / "previous-active.json")["version"] == "old"


def test_background_training_threshold_is_coalesced(tmp_path, monkeypatch):
    import local_decider.hooks as hooks
    launched = []
    monkeypatch.setattr(hooks.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))
    for index in range(24):
        handle({**event("Stop"), "turn_id": str(index)}, tmp_path)
    assert not launched
    handle({**event("Stop"), "turn_id": "24"}, tmp_path)
    handle({**event("Stop"), "turn_id": "25"}, tmp_path)
    assert len(launched) == 1


def test_verified_experience_is_frozen_on_resume(tmp_path):
    sample = {"state": State("Fix code", flags={"patched": True}).to_dict(),
              "action_kind": "focused_test", "verified_test_pass": True, "project_hash": "00000000"}
    experience = tmp_path / "experience.jsonl"
    experience.write_text(json.dumps(sample) + "\n")
    output = tmp_path / "trained"
    train("linear", output, Config(population=4, parents=2, generations=1), experience_path=experience)
    assert Model.load(output / "linear.json").metadata["verified_real_decisions"] == 1
    experience.write_text("")
    train("linear", output, Config(population=4, parents=2, generations=2), resume=True,
          experience_path=experience)
    assert Model.load(output / "linear.json").metadata["verified_real_decisions"] == 1
