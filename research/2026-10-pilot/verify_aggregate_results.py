"""Verify the public aggregate without access to task rows or transcripts."""

import json
import sys
from pathlib import Path


def verify(aggregate):
    data = json.loads(Path(aggregate).read_text(encoding="utf-8"))
    assert data["ordinary_activation_changed"] is False
    assert data["target_achieved"] is None
    assert data["token_reduction_fraction"] is None
    assert data["chat_bootstrap_95_interval"] is None
    for arm, g in data["groups"].items():
        assert (
            g["turns"]
            == g["completed_turns"] + g["interrupted_turns"] + g["open_turns"]
        )
        assert g["turns"] == g["measured_turns"] + g["missing_usage_turns"]
        assert g["turns"] == g["confirmed_outcomes"] + g["unknown_outcomes"]
        if g["measured_turns"] == 0:
            assert all(
                g[k] is None
                for k in [
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "mean_tokens_per_measured_turn",
                ]
            )
        assert g["confirmed_solved"] <= g["confirmed_outcomes"]
    text = json.dumps(data)
    assert all(
        key not in text
        for key in ["session_hash", "project_hash", "transcript_path", "per_chat"]
    )
    assert data["groups"]["baseline"]["chats"] == 1
    assert data["groups"]["baseline"]["turns"] == 5
    assert data["groups"]["baseline"]["missing_usage_turns"] == 5
    assert data["groups"]["assisted"]["chats"] == 0
    assert data["excluded_probes"] == {
        "chats": 3,
        "turns": 9,
        "decisions": 14,
        "recommendations": 1,
        "truncated_streams": 3,
    }
    synthetic = json.loads(
        Path(aggregate).with_name("synthetic-results.json").read_text()
    )
    reduction = (
        1
        - synthetic["metrics"]["linear"]["mean_proxy_tokens"]
        / synthetic["metrics"]["rules"]["mean_proxy_tokens"]
    )
    assert abs(reduction - 0.058577405857740586) < 1e-12
    print(
        json.dumps(
            {
                "aggregate_checks": "passed",
                "synthetic_context_reduction_fraction": reduction,
                "field_comparison": "inconclusive",
                "actual_token_reduction": None,
            }
        )
    )


if __name__ == "__main__":
    verify(sys.argv[1])
