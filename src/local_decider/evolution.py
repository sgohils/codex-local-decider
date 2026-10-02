import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import KINDS, State
from .exercises import Episode, dataset, dataset_digest, preferred_kind
from .features import encode
from .models import Model, parameter_count
from .storage import read_json, write_json


@dataclass
class Config:
    population: int = 24
    parents: int = 4
    generations: int = 50
    workers: int = 2
    minutes: float = 15.0
    seed: int = 7
    mutation: float = 0.15
    mutation_fraction: float = 0.1
    max_steps: int = 12

    def validate(self):
        if not 2 <= self.population <= 256 or not 1 <= self.parents < self.population:
            raise ValueError("invalid population or parent count")
        if not 1 <= self.generations <= 10000 or not 1 <= self.workers <= 2:
            raise ValueError("generations must be positive; workers must be 1 or 2")
        if not 0 < self.minutes <= 15 or not 0 < self.mutation <= 2 or not 1 <= self.max_steps <= 64:
            raise ValueError("invalid time, mutation, or step limit")
        if not 0 < self.mutation_fraction <= 1:
            raise ValueError("mutation_fraction must be in (0, 1]")


def evaluate(policy: Model | str, exercises: list, seed: int = 7, max_steps: int = 12) -> dict:
    from .contracts import default_actions

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    episodes = [Episode(exercise) for exercise in exercises]
    available = default_actions()
    # Batched inference bounds CPU work and avoids one matrix operation per exercise.
    for _ in range(max_steps):
        active = [e for e in episodes if not e.done]
        if not active:
            break
        if isinstance(policy, Model):
            inputs = np.concatenate([encode(e.state, available) for e in active])
            choices = policy.score(inputs).reshape(len(active), len(available)).argmax(axis=1)
        elif policy == "rules":
            choices = [KINDS.index(preferred_kind(e.state)) for e in active]
        elif policy == "random":
            choices = rng.integers(0, len(available), len(active))
        else:
            raise ValueError("unknown policy")
        for episode, choice in zip(active, choices, strict=True):
            episode.step(available[int(choice)].kind)
    count = len(episodes)
    if not count:
        raise ValueError("empty exercise set")
    completed = [e for e in episodes if e.done]
    return {"episodes": count, "successes": len(completed), "success_rate": len(completed) / count,
            "mean_progress": sum(sum(e.state.flags.values()) for e in episodes) / count,
            "mean_proxy_tokens": sum(e.tokens for e in episodes) / count,
            "mean_success_proxy_tokens": (sum(e.tokens for e in completed) / len(completed)
                                           if completed else None),
            "mean_steps": sum(e.steps for e in episodes) / count,
            "mean_redundant_steps": sum(e.redundant for e in episodes) / count,
            "elapsed_seconds": time.perf_counter() - started,
            "token_metric": "simulated tool-output estimates, not Codex tokens"}


def fitness(metrics: dict) -> tuple:
    # Lexicographic: one extra solved episode always outranks any token reduction.
    return (metrics["successes"], metrics.get("mean_progress", 0),
            -metrics["mean_proxy_tokens"], -metrics["mean_redundant_steps"])


def decision_examples(exercises: list) -> tuple[np.ndarray, np.ndarray]:
    """Reference trajectories yield cheap dense rewards from verified exercise state."""
    from .contracts import default_actions

    inputs, labels = [], []
    available = default_actions()
    for exercise in exercises:
        episode = Episode(exercise)
        for _ in range(12):
            if episode.done:
                break
            kind = preferred_kind(episode.state)
            inputs.append(encode(episode.state, available))
            labels.append(KINDS.index(kind))
            episode.step(kind)
    return np.stack(inputs), np.asarray(labels)


def dense_fitness(model: Model, inputs: np.ndarray, labels: np.ndarray) -> tuple:
    scores = model.score(inputs.reshape(-1, inputs.shape[-1])).reshape(len(labels), len(KINDS))
    correct = scores[np.arange(len(labels)), labels]
    other = scores.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    margins = correct - other.max(axis=1)
    # Balance rare actions (such as search) so the population cannot ignore a necessary step.
    weights = 1.0 / np.bincount(labels, minlength=len(KINDS))[labels]
    shifted = scores - scores.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    loss = float(np.average(losses, weights=weights)) + 0.0001 * float(np.mean(model.weights ** 2))
    return (-loss, float(np.average(margins > 0, weights=weights)), int(np.sum(margins > 0)))


def train(kind: str, output: Path, config: Config, resume: bool = False,
          progress=None, experience_path: Path | None = None) -> dict:
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{kind}-population.json"
    rng = np.random.default_rng(config.seed)
    data = dataset(config.seed)
    digest = dataset_digest(config.seed)
    inputs, labels = decision_examples(data["train"])
    experience = []
    if experience_path and experience_path.is_file():
        import json
        from .contracts import default_actions
        # Only explicitly observed, useful test passes enter training. Unknown outcomes are excluded.
        for line in experience_path.read_text(encoding="utf-8")[-2_000_000:].splitlines()[-500:]:
            try:
                sample = json.loads(line)
                state = State.from_dict(sample["state"])
                bucket = int(sample["project_hash"][:8], 16) % 5
                if (sample.get("verified_test_pass") is True and sample.get("action_kind") == "focused_test"
                        and state.flags.get("patched") and not state.flags.get("focused_passed")
                        and bucket < 3):
                    experience.append(sample)
            except (ValueError, TypeError, KeyError):
                continue
    history = []
    generation = 0
    if resume:
        saved = read_json(checkpoint)
        if saved.get("algorithm_version") != 3:
            raise ValueError("population uses an older evolution algorithm; start a fresh run")
        experience = saved.get("experience", [])  # Freeze the original sample snapshot on resume.
        if saved["kind"] != kind or saved["dataset_digest"] != digest:
            raise ValueError("resume checkpoint uses a different model or dataset")
        previous = saved["config"]
        for key in ("population", "parents", "seed", "mutation", "max_steps"):
            if previous[key] != asdict(config)[key]:
                raise ValueError(f"resume configuration changed: {key}")
        if previous.get("mutation_fraction", 1.0) != config.mutation_fraction:
            raise ValueError("resume configuration changed: mutation_fraction")
        population = np.asarray(saved["population"], dtype=float)
        if population.shape != (config.population, parameter_count(kind)):
            raise ValueError("invalid population checkpoint shape")
        rng.bit_generator.state = saved["rng_state"]
        history = saved["history"]
        generation = saved["generation"]
        champion = Model.load(output / f"{kind}.json")
        best_validation = saved["best_validation"]
    else:
        population = rng.normal(0, 0.2, (config.population, parameter_count(kind)))
        champion = Model(kind, population[0].copy())
        best_validation = evaluate(champion, data["validation"], max_steps=config.max_steps)
    if experience:
        from .contracts import default_actions
        extra = [encode(State.from_dict(sample["state"]), default_actions()) for sample in experience]
        inputs = np.concatenate([inputs, np.stack(extra)])
        labels = np.concatenate([labels, np.full(len(extra), KINDS.index("focused_test"))])
    started = time.perf_counter()
    deadline = started + config.minutes * 60
    generation_limit = min(config.generations, generation + 50)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        while generation < generation_limit and time.perf_counter() < deadline:
            candidates = [Model(kind, weights) for weights in population]
            ranked = list(executor.map(lambda m: dense_fitness(m, inputs, labels), candidates))
            order = sorted(range(len(ranked)), key=lambda i: ranked[i], reverse=True)
            parents = population[order[:config.parents]].copy()
            candidate = candidates[order[0]]
            validation = evaluate(candidate, data["validation"], max_steps=config.max_steps)
            if fitness(validation) > fitness(best_validation) or generation == 0:
                champion = Model(kind, candidate.weights.copy(), f"{kind}-s{config.seed}-g{generation + 1}")
                best_validation = validation
            generation += 1
            record = {"generation": generation, "correct_training_decisions": ranked[order[0]][2],
                      "balanced_training_loss": -ranked[order[0]][0],
                      "training_decisions": len(labels), "validation": validation}
            history.append(record)
            if progress and (generation == 1 or generation % 10 == 0):
                progress({"kind": kind, **record})
            # Elites remain bit-for-bit unchanged; every other candidate inherits a parent.
            children = []
            for i in range(config.population - config.parents):
                noise = rng.normal(0, config.mutation, parameter_count(kind))
                noise *= rng.random(parameter_count(kind)) < config.mutation_fraction
                children.append(parents[i % config.parents] + noise)
            population = np.vstack([parents, children])
            champion.metadata = {"dataset_digest": digest, "seed": config.seed,
                                 "validation": best_validation, "trained_on": "synthetic exercises",
                                 "verified_real_decisions": len(experience),
                                 "live_validated": False}
            champion.save(output / f"{kind}.json")
            write_json(checkpoint, {"schema_version": 1, "algorithm_version": 3, "kind": kind, "config": asdict(config),
                                   "dataset_digest": digest, "generation": generation,
                                   "population": population.tolist(), "rng_state": rng.bit_generator.state,
                                   "experience": experience,
                                   "history": history, "best_validation": best_validation})
    if generation == 0:
        raise RuntimeError("time limit expired before a generation completed")
    return {"kind": kind, "parameters": parameter_count(kind), "generations": generation,
            "elapsed_seconds": time.perf_counter() - started, "config": asdict(config),
            "validation": best_validation, "history": history}


def compare(output: Path, seed: int = 7) -> dict:
    data = dataset(seed)
    policies = {kind: Model.load(output / f"{kind}.json") for kind in ("linear", "neural")}
    policies.update(rules="rules", random="random")
    metrics = {name: evaluate(policy, data["test"], seed) for name, policy in policies.items()}
    winner = max(("linear", "neural"), key=lambda name: fitness(metrics[name]))
    # Winner selection must use validation, never test scores.
    validation_winner = max(("linear", "neural"), key=lambda name: fitness(
        policies[name].metadata["validation"]))
    return {"schema_version": 1, "dataset_digest": dataset_digest(seed), "seed": seed,
            "split_counts": {key: len(value) for key, value in data.items()}, "metrics": metrics,
            "candidate": validation_winner, "descriptive_test_best": winner,
            "live_validated": False,
            "limitations": ["Small scripted Python workflows with oracle repairs",
                            "No calibrated probability or general coding capability",
                            "Proxy token counts do not demonstrate Codex token savings"]}


def report(result: dict) -> str:
    lines = ["# Local evolutionary comparison", "",
             "Measured on held-out synthetic workflows. Token counts below are simulation estimates.",
             "The harness supplies reference repairs; the decision model does not write code.", "",
             "| Policy | Completed | Proxy tokens per episode | Steps | Redundant steps |",
             "|---|---:|---:|---:|---:|"]
    for name, metrics in result["metrics"].items():
        lines.append(f"| {name} | {metrics['successes']}/{metrics['episodes']} | "
                     f"{metrics['mean_proxy_tokens']:.1f} | {metrics['mean_steps']:.2f} | "
                     f"{metrics['mean_redundant_steps']:.2f} |")
    lines += ["", f"Validation-selected candidate: **{result['candidate']}**.", "",
              "No live Codex savings are established by this comparison.", "",
              *[f"- {value}" for value in result["limitations"]], ""]
    return "\n".join(lines)
