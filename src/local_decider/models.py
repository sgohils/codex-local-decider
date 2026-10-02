from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from .contracts import Action, State
from .features import DIM, FEATURE_VERSION, encode
from .storage import read_json, write_json

HIDDEN = 16


def parameter_count(kind: str) -> int:
    if kind == "linear":
        return DIM
    if kind == "neural":
        return DIM * HIDDEN + HIDDEN + HIDDEN + 1
    raise ValueError("model kind must be linear or neural")


@dataclass
class Model:
    kind: str
    weights: np.ndarray
    version: str = "unversioned"
    metadata: dict = field(default_factory=dict)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            return features @ self.weights
        offset = DIM * HIDDEN
        hidden = np.tanh(features @ self.weights[:offset].reshape(DIM, HIDDEN)
                         + self.weights[offset:offset + HIDDEN])
        return hidden @ self.weights[offset + HIDDEN:offset + 2 * HIDDEN] + self.weights[-1]

    def save(self, path: Path) -> None:
        write_json(path, {"schema_version": 1, "feature_version": FEATURE_VERSION,
                          "kind": self.kind, "weights": self.weights.tolist(),
                          "version": self.version, "metadata": self.metadata})

    @classmethod
    def load(cls, path: Path) -> "Model":
        value = read_json(path)
        if value.get("schema_version") != 1 or value.get("feature_version") != FEATURE_VERSION:
            raise ValueError("unsupported checkpoint schema or features")
        weights = np.asarray(value["weights"], dtype=np.float64)
        if weights.shape != (parameter_count(value["kind"]),) or not np.isfinite(weights).all():
            raise ValueError("checkpoint has invalid weights")
        return cls(value["kind"], weights, value["version"], value.get("metadata", {}))

    def decide(self, state: State, actions: list[Action], min_margin: float = 0.05) -> dict:
        if not actions or len({a.id for a in actions}) != len(actions):
            raise ValueError("actions must be nonempty and unique")
        scores = self.score(encode(state, actions))
        if not np.isfinite(scores).all():
            raise ValueError("non-finite model scores")
        order = np.argsort(-scores, kind="stable")
        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None
        # Score margins are not calibrated probabilities or guarantees of correctness.
        coding = bool(re.search(r"\b(fix|debug|bug|code|test|function|incorrect|regression|error)\b", state.task, re.I))
        uncertain = not coding or (margin is not None and margin < min_margin)
        selected = actions[int(order[0])]
        if uncertain:
            selected = next((a for a in actions if a.kind == "defer"), selected)
        return {"action_id": selected.id, "action_kind": selected.kind,
                "scores": {a.id: float(s) for a, s in zip(actions, scores, strict=True)},
                "uncertain": uncertain, "score_margin": margin, "model_version": self.version}
