# Codex Local Decider

A CPU-first experiment in evolving small decision models that can advise Codex during coding and debugging. Inspired by Jev's typed decision interface; this is an original implementation, not a reproduction of TypeSafe's proprietary model.

**Train first, integrate second, measure before activating.** Local evolution does not call Codex or any external model. Live evaluation uses your authenticated Codex CLI and consumes its usage allowance.

## What it does

- Evolves a 118-parameter linear controller and a 1,921-parameter neural network.
- Selects among supplied search, inspection, testing, and defer actions.
- Preserves the best parents and produces offspring with sparse random weight mutations (10% of coordinates on average).
- Supports deterministic seeds, resumable populations, bounded CPU work, and comparison reports.
- Supplies short advisory hooks, verified-observation collection, background candidate training, and an activation gate with rollback snapshots.

The models do not generate code. Training workflows include a scripted reference repair at handoff. Soft score margins are **not calibrated probabilities**. This is a small workflow research prototype; it is not a general-purpose reasoning model or a guarantee of token savings.

## Install

Requires Python 3.13. On Windows, in this repository:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS, use `python3.13 -m venv .venv` and `.venv/bin/python` instead. The commands below assume the environment is activated; otherwise prefix them with `.venv\Scripts\` on Windows.

## Train and compare

```powershell
local-decider verify-fixtures
local-decider train --kind both --output runs/initial
local-decider compare --output runs/initial
local-decider report --input runs/initial/comparison.json --output runs/initial/report.md
local-decider evaluate --model runs/initial/linear.json --split test
```

Defaults: 24 candidates, four parents, two workers, seed 7, at most 50 new generations and 15 minutes per invocation shared across both models. Use `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to keep numerical-library threads bounded. `--generations` specifies the cumulative target, so continuation runs can target a larger total while still generating at most 50 new generations per model per invocation:

```powershell
local-decider train --output runs/initial --resume --generations 100
```

Parent selection minimizes class-balanced cross-entropy on reference workflow decisions, providing a smooth dense reward before full episodes succeed. Validation selects the checkpoint using complete episodes first, then observed progress, then tool-output estimates and repeated actions. The test split is used for reporting, not evolutionary rewards or winner selection. All baselines receive the same exercises and maximum step count; only learned policies receive a training budget.

The 120 exercises use 15 actual seeded Python defects. Defect templates are split 9/3/3 into training/validation/test groups; variants never cross groups. The exercise verifier checks that each defect fails and its reference repair passes. The simulator models tool choices and evidence gathering, not arbitrary repository behavior.

See [example results](examples/results/local-report.md) for measured local outcomes and [evaluation notes](docs/EVALUATION.md) for limitations.

## Make a decision

```powershell
local-decider decide --model examples/models/candidate.json --input examples/decision.json
```

Input: `state` with task, observations, action history, and observable boolean flags; `available_actions` with unique identifiers, supported kinds, labels, and optional targets. Output: selected identifier/kind, scores, uncertainty flag, score margin, and model version. Only supplied actions can be selected. The host supplies command arguments and executes actions.

## Evaluate with Codex before enabling advice

```powershell
local-decider benchmark --candidate examples/models/candidate.json --git "C:\path\to\git.exe" --output runs/live
local-decider promote --candidate examples/models/candidate.json --live-result runs/live/live.json
local-decider install-hooks --project .
```

The benchmark creates fresh Git repositories, alternates arm order, preserves original tests, and counts JSON-reported input plus output tokens (including cached input). Default limits are 12 pairs, 100,000 reported tokens, and 180 seconds per invocation. A single in-flight invocation may exceed the observed token budget; the next run is prevented. Use `--model` to pin the same available Codex model for both arms. Both arms ignore user configuration to avoid unrelated hooks and plugin differences.

Each benchmark arm uses its own `sqlite_home` and `log_dir` under its checkout. The initial live attempt on this Windows environment failed before model execution with an app-server access error; it established no token savings.

Only our generated, inspected benchmark hooks run using the CLI's one-off hook-trust flag. Sandbox protections remain enabled. Normal installation requires reviewing the generated definitions through `/hooks`; it does not alter your global Codex configuration or approvals. Official reference: [Codex hooks](https://learn.chatgpt.com/docs/hooks).

Promotion requires a complete 12-pair trial, valid usage measurements, at least one delivered recommendation, no decrease in observed successes, and at least 10% fewer aggregate tokens. Otherwise the candidate remains inactive. Promoted models are copied to immutable snapshots and the previous active manifest is retained.

You can inspect experimental advice in a specific disposable project with `install-hooks --experimental --candidate MODEL`; this intentionally bypasses the activation gate only for that installation, and normal hook trust still applies.

## Continued local learning

Hooks save redacted task descriptions, flags, and summaries under `.local-decider/`, never full tool output. Review logs before sharing; automatic redaction is best effort. A successful focused test immediately after a patch can supply a verified decision example. This records what succeeded, not proof that it was the optimal action. Project hashes separate eligible training examples from reserved buckets.

After 25 newly closed turns, one hidden worker trains both candidates locally under a shared 15-minute budget. It freezes verified samples for reproducible resume and never automatically replaces the active model. New candidates require new live validation. No hosted Codex calls occur in background training.

Git ignores environments, real task observations, run logs, credentials, local hook configuration, and temporary assets. Only explicitly selected synthetic checkpoints and summary results belong under `examples/`.

## Five-day randomized study

The separate experimental collector assigns new coding chats equally to baseline
measurement or assisted measurement using persistent randomized blocks. It freezes
`linear-s7-g40`, records actual transcript token usage, and never replaces the normal
active model. The first ordinary coding chat verifies collection and is excluded
from comparison; only then does the five-day clock start. Follow-ups and compaction
retain the same assignment. No extra hosted coding trials are launched.

```powershell
local-decider study start --data-dir work/study --candidate examples/models/candidate.json
local-decider study install --data-dir work/study --config C:\Users\YOU\.codex\hooks.json
local-decider study status --data-dir work/study --refresh
local-decider study analyze --data-dir work/study --output work/study-analysis.json
local-decider study stop --data-dir work/study
```

Review installed hooks using `/hooks`. Missing usage is unknown, never zero. Test
passes and confirmed task outcomes are separate. See [the registered study protocol](docs/STUDY.md)
for enrollment, privacy, clustered analysis, failure handling, and the paper workflow.

## Development

```powershell
python -m ruff check src tests
python -m pytest -q
```

GitHub Actions runs checks on Windows and Linux with Python 3.13. Dependency versions used for the published example are recorded in `requirements-tested.txt`.

MIT licensed.
