"""Stable hashed text plus observable workflow features, shared by both policies."""
import hashlib
import re

import numpy as np

from .contracts import Action, KINDS, State

FEATURE_VERSION = 1
TEXT_DIM = 64
FLAGS = ("path_known", "source_seen", "failure_seen", "failure_inspected",
         "patched", "focused_passed", "broad_passed")
DIM = TEXT_DIM + len(KINDS) * (len(FLAGS) + 2)


def encode(state: State, actions: list[Action]) -> np.ndarray:
    result = np.zeros((len(actions), DIM), dtype=np.float64)
    for index, action in enumerate(actions):
        text = " ".join([state.task, *state.observations, action.label, action.target])
        for token in re.findall(r"[a-z_][a-z_0-9]+", text.lower())[:1024]:
            digest = hashlib.blake2s(token.encode(), digest_size=4).digest()
            result[index, int.from_bytes(digest, "little") % TEXT_DIM] += 1
        norm = np.linalg.norm(result[index, :TEXT_DIM])
        if norm:
            result[index, :TEXT_DIM] *= 0.2 / norm
        base = TEXT_DIM + KINDS.index(action.kind) * (len(FLAGS) + 2)
        result[index, base:base + len(FLAGS) + 2] = [
            1.0, *(float(state.flags.get(key, False)) for key in FLAGS),
            min(state.previous_actions.count(action.kind), 3) / 3,
        ]
    return result
