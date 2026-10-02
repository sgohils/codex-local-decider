"""Seeded real Python defects plus an explicit, bounded workflow simulator.

Simulation supplies the reference repair at handoff; the decision model never fixes code.
Its output-token estimates are proxies, not measured Codex usage.
"""
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .contracts import KINDS, State, default_actions

TEMPLATES = (
    ("sum", "def solve(xs):\n    return sum(xs) + 1\n", "def solve(xs):\n    return sum(xs)\n", "solve([2, 3])", "5"),
    ("count", "def solve(xs):\n    return len(xs) - 1\n", "def solve(xs):\n    return len(xs)\n", "solve([1, 2, 3])", "3"),
    ("maximum", "def solve(xs):\n    return min(xs)\n", "def solve(xs):\n    return max(xs)\n", "solve([2, 9, 4])", "9"),
    ("minimum", "def solve(xs):\n    return max(xs)\n", "def solve(xs):\n    return min(xs)\n", "solve([2, 9, 4])", "2"),
    ("reverse", "def solve(xs):\n    return xs\n", "def solve(xs):\n    return xs[::-1]\n", "solve([1, 2])", "[2, 1]"),
    ("unique", "def solve(xs):\n    return sorted(xs)\n", "def solve(xs):\n    return sorted(set(xs))\n", "solve([2, 1, 2])", "[1, 2]"),
    ("double", "def solve(x):\n    return x + 2\n", "def solve(x):\n    return x * 2\n", "solve(7)", "14"),
    ("square", "def solve(x):\n    return x * 2\n", "def solve(x):\n    return x * x\n", "solve(7)", "49"),
    ("absolute", "def solve(x):\n    return x\n", "def solve(x):\n    return abs(x)\n", "solve(-7)", "7"),
    ("lower", "def solve(x):\n    return x.upper()\n", "def solve(x):\n    return x.lower()\n", "solve('Hello')", "'hello'"),
    ("trim", "def solve(x):\n    return x.lstrip()\n", "def solve(x):\n    return x.strip()\n", "solve(' x ')", "'x'"),
    ("join", "def solve(xs):\n    return ''.join(xs)\n", "def solve(xs):\n    return ','.join(xs)\n", "solve(['a', 'b'])", "'a,b'"),
    ("first", "def solve(xs):\n    return xs[-1]\n", "def solve(xs):\n    return xs[0]\n", "solve([3, 8])", "3"),
    ("last", "def solve(xs):\n    return xs[0]\n", "def solve(xs):\n    return xs[-1]\n", "solve([3, 8])", "8"),
    ("even", "def solve(x):\n    return x % 2 == 1\n", "def solve(x):\n    return x % 2 == 0\n", "solve(4)", "True"),
)


@dataclass(frozen=True)
class Exercise:
    id: str
    template: int
    variant: int
    split: str
    seed: int

    def initial_state(self) -> State:
        name = TEMPLATES[self.template][0]
        flags = {"path_known": bool(self.variant & 1), "failure_seen": bool(self.variant & 2)}
        observations = [f"Investigate incorrect {name} behavior"]
        if flags["path_known"]:
            observations.append("Relevant source: logic.py; focused tests: test_logic.py")
        if flags["failure_seen"]:
            observations.append("AssertionError in test_logic.py; inspect the failure details")
        return State(f"Fix incorrect {name} behavior and verify the result", observations, [], flags)

    def materialize(self, directory: Path, repaired: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _, broken, correct, expression, expected = TEMPLATES[self.template]
        (directory / "logic.py").write_text(correct if repaired else broken, encoding="utf-8")
        tests = ("import unittest\nfrom logic import solve\n\nclass TestLogic(unittest.TestCase):\n"
                 f"    def test_regression(self):\n        self.assertEqual({expression}, {expected})\n"
                 "\nif __name__ == '__main__':\n    unittest.main()\n")
        (directory / "test_logic.py").write_text(tests, encoding="utf-8")
        (directory / "AGENTS.md").write_text(
            "Fix the reported defect in logic.py. Preserve test_logic.py. "
            "Use Python unittest to verify it. Avoid unrelated files.\n", encoding="utf-8")


def dataset(seed: int = 7) -> dict[str, list[Exercise]]:
    # Group by template: variants of a defect never cross dataset boundaries.
    result = {"train": [], "validation": [], "test": []}
    for index in range(len(TEMPLATES)):
        split = "train" if index < 9 else "validation" if index < 12 else "test"
        for variant in range(8):
            result[split].append(Exercise(f"{split}-{index}-{variant}", index, variant, split, seed))
    return result


def dataset_digest(seed: int = 7) -> str:
    value = repr(TEMPLATES) + repr(dataset(seed))
    return hashlib.sha256(value.encode()).hexdigest()


def preferred_kind(state: State) -> str:
    f = state.flags
    if f.get("broad_passed"):
        return "defer"
    if f.get("patched"):
        return "broad_test" if f.get("focused_passed") else "focused_test"
    if f.get("source_seen") and f.get("failure_inspected"):
        return "defer"
    if f.get("failure_seen") and not f.get("failure_inspected"):
        return "inspect_failure"
    if not f.get("path_known"):
        return "search_code"
    if not f.get("source_seen"):
        return "inspect_file"
    return "focused_test"


class Episode:
    def __init__(self, exercise: Exercise):
        self.state = exercise.initial_state()
        self.done = False
        self.tokens = 0
        self.steps = 0
        self.redundant = 0

    def step(self, kind: str) -> None:
        if kind not in KINDS or self.done:
            raise ValueError("invalid episode action")
        f = self.state.flags
        before = dict(f)
        self.steps += 1
        self.tokens += {"search_code": 90, "inspect_file": 120, "inspect_failure": 80,
                        "focused_test": 70, "broad_test": 230, "defer": 20}[kind]
        if kind == "search_code":
            f["path_known"] = True
        elif kind == "inspect_file" and f.get("path_known"):
            f["source_seen"] = True
        elif kind == "inspect_failure" and f.get("failure_seen"):
            f.update(failure_inspected=True, path_known=True)
        elif kind == "focused_test" and f.get("path_known"):
            if f.get("patched"):
                f["focused_passed"] = True
            else:
                f["failure_seen"] = True
        elif kind == "broad_test":
            if f.get("patched"):
                f.update(focused_passed=True, broad_passed=True)
            else:
                f.update(failure_seen=True, path_known=True)
        elif kind == "defer":
            if f.get("broad_passed"):
                self.done = True
            elif f.get("source_seen") and f.get("failure_inspected"):
                f["patched"] = True  # Scripted oracle repair, never a model-generated repair.
        if before == f and not self.done:
            self.redundant += 1
        self.state.previous_actions.append(kind)
        self.state.observations = ["Observed workflow flags: " + ", ".join(k for k, v in f.items() if v)]


def verify_fixtures(directory: Path) -> dict:
    results = []
    for index in range(len(TEMPLATES)):
        exercise = Exercise(f"fixture-{index}", index, 0, "verification", 7)
        target = directory / str(index)
        exercise.materialize(target)
        broken = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=target,
                                capture_output=True, timeout=20)
        exercise.materialize(target, repaired=True)
        # Python bytecode can retain the previous same-size source in a rapid rewrite.
        fixed = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q"], cwd=target,
                               capture_output=True, timeout=20,
                               env={**__import__("os").environ, "PYTHONPYCACHEPREFIX": str(target / "fixed-cache")})
        results.append({"template": TEMPLATES[index][0], "broken_fails": broken.returncode != 0,
                        "repair_passes": fixed.returncode == 0})
    return {"passed": all(r["broken_fails"] and r["repair_passes"] for r in results),
            "fixtures": results}


def actions():
    return default_actions()
