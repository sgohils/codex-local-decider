from dataclasses import asdict, dataclass, field
from typing import Any

KINDS = ("search_code", "inspect_file", "inspect_failure", "focused_test", "broad_test", "defer")


@dataclass(frozen=True)
class Action:
    id: str
    kind: str
    label: str
    target: str = ""


@dataclass
class State:
    task: str
    observations: list[str] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "State":
        if not isinstance(value, dict) or not isinstance(value.get("task", ""), str):
            raise ValueError("state must contain a text task")
        observations = value.get("observations", [])
        history = value.get("previous_actions", [])
        flags = value.get("flags", {})
        if not isinstance(observations, list) or not all(isinstance(x, str) for x in observations):
            raise ValueError("observations must be a list of strings")
        if not isinstance(history, list) or not all(isinstance(x, str) for x in history):
            raise ValueError("previous_actions must be a list of strings")
        if not isinstance(flags, dict) or not all(isinstance(v, bool) for v in flags.values()):
            raise ValueError("flags must contain booleans")
        return cls(value.get("task", "")[:4000], [x[:2000] for x in observations[-8:]],
                   history[-16:], flags)


def actions_from_dict(values: list[dict]) -> list[Action]:
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("supply 1 to 64 available actions")
    try:
        actions = [Action(**v) for v in values]
    except (TypeError, ValueError) as error:
        raise ValueError("each available action must be an action object") from error
    if any(a.kind not in KINDS or not a.id
           or not all(isinstance(x, str) for x in (a.id, a.kind, a.label, a.target)) for a in actions):
        raise ValueError("unknown action kind or invalid action")
    if len({a.id for a in actions}) != len(actions):
        raise ValueError("action identifiers must be unique")
    return actions


def default_actions() -> list[Action]:
    labels = ("Search for the relevant symbol", "Inspect the relevant source file",
              "Inspect the latest failing test or traceback", "Run the relevant focused tests",
              "Run the broader test suite", "Use Codex's own judgment")
    return [Action(kind, kind, label) for kind, label in zip(KINDS, labels, strict=True)]
