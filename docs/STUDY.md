# Five-day randomized study protocol (version 1)

This is a single-user prospective pilot, not a guarantee of savings. The existing
synthetic run is separate from this experiment. The exact `linear-s7-g40` checkpoint
is frozen by SHA-256. The repository stays private.

## Enrollment and assignment

`study start` prepares an **armed** collector. The initial newly created coding chat
is a probe: it receives no advice and never enters the comparison. Its ordinary
PostToolUse and Stop/Interrupt hooks must observe supported, nonzero token usage
before the registry sets the start time and a deadline exactly five days later.
Delayed transcript writes can be reconciled using `study status --refresh`.
No additional hosted coding trials are launched.

Only new chats whose transcript creation time is after preparation qualify.
The working folder must belong to a local Git or recognizable coding project.
Plan-mode turns, research prompts, the decider repository, and study maintenance
are excluded. Eligibility uses an explicit keyword heuristic; it can miss tasks.

After verification, each new eligible chat receives a persistent assignment in
blocks of four (two baseline, two assisted), stratified by hashed project and
initial model. Block shuffles use seed 20261001 and SHA-256-derived stratum seeds.
Assignments are committed before tool outcomes are inspected. Incomplete blocks
can yield unequal observed group sizes. Compaction, resume, and follow-up turns
never rerandomize a chat. Both groups have the same measurement hooks.

Baseline hooks emit `{}` and do not load a decision model. Assisted hooks load the
frozen checkpoint and may supply concise, repeated-advice-suppressed recommendations.
Uncertainty, deferred decisions, corruption, and unavailable models emit `{}`.
Such chats remain in their assigned group regardless of whether advice was delivered.
The normal promotion gate is unchanged; the study uses a separate experimental path.

## Measurements and privacy

The installed Codex 0.159.3 transcript was inspected for **schema only**, without
importing existing chats into the study. It contains `token_usage_record` entries
with response IDs, root turn IDs, and per-response usage. These are preferred.
Hashed response IDs prevent duplicates across parent/child streams. The supported
older adapter uses differences between monotonic `event_msg.token_count` totals.
If both formats appear, per-response records replace provisional cumulative totals.
Cached input is part of input and is never added again to total tokens.

The collector opens only the transcript path supplied by an enrolled hook, under
the configured session roots. It never writes Codex databases. Subagent transcript
paths supplied by SubagentStop are grouped with the parent chat. Missing child
usage, malformed records, truncation, and pending transcript data invalidate the
relevant usage measurement rather than imply zero cost. Unknown formats cannot
verify the start of the study. The format is versioned because Codex warns it is
not stable: [official hooks documentation](https://learn.chatgpt.com/docs/hooks).

Records contain hashes, model names, observable flags, numeric metrics, and a
small fixed coding vocabulary. They do not retain full prompts, identifiers,
source code, suggested command arguments, tool output, or copied transcripts.
Private registry files contain local transcript paths for reconciliation; public
exports omit these paths and request IDs. Exported per-chat rows still require
review before sharing. Publish only aggregate research results by default.

For each turn, record usage availability, token counts, tool failures, test passes
and failures, post-failure editing events (a retry proxy), advice count, model
failures, inference time, collection time, elapsed turn time, and interruption.
Follow-up turns are reported separately: they are not automatically retries.
Elapsed time includes tool and hook delays and can include time waiting for approval.
Internal hook timers exclude interpreter startup and final record writes; inference
timing includes model import, loading, integrity checks, and scoring. Do not claim
these internal timings are complete process CPU measurements.

Test passes after a recorded patch are **test-verified observations**, not proof
that the user's task is solved. Confirmed outcomes require operator annotations:

```powershell
local-decider study outcome --data-dir DATA --session CHAT_HASH --turn TURN_HASH --result solved --evidence acceptance.txt
```

An evidence file is hashed, never copied. Unannotated outcomes remain unknown.
Candidate-only background retraining is triggered after 25 closed eligible turns,
using the existing shared 15-minute budget. It cannot replace the study checkpoint.
Its wall time is recorded separately and may affect local machine load.

## Analysis and stopping

Primary descriptive estimate: `1 - assisted_mean_tokens / baseline_mean_tokens`,
where each mean is over measured coding turns, including failed/interrupted turns
with usable records. Every randomized chat is counted; report missing usage by
group, open turns/chats, and advice delivery instead of silently dropping them.
Report per-chat totals and model/project composition alongside per-turn means.

Bootstrap entire chats independently within each arm, keeping all their turns
together (10,000 draws, registered seed; percentile 95% interval). This is an
exploratory bootstrap, not an exact stratified randomization test. With fewer than
ten measured independent chats in either group, omit the interval and label the
comparison inconclusive. The target is 20 chats per arm, not a stopping rule.
Ten chats is a reporting threshold, not a statistical power guarantee.

The aspiration remains at least 10% fewer tokens without reduced observed task
success. Unknown outcomes cannot substantiate preserved success. There is no
automatic promotion based on this small ordinary-work study. Differences in task
difficulty, model settings, shared memories, user behavior, and machine load limit
causal interpretation even with random assignment.

At the deadline, recommendation hooks stop immediately, even while the app is
offline between scheduled checks. Reconciliation accepts only records timestamped
within the five-day window. `study stop` freezes a snapshot and removes only this
installation's hook entries, preserving unrelated entries added during the study.
If global hook removal is unavailable to an unattended run, `--keep-hooks` still
disables all recommendations through the frozen registry.

## Commands

```powershell
local-decider study start --data-dir DATA --candidate examples/models/candidate.json --exclude MAINTENANCE_FOLDER
local-decider study install --data-dir DATA --config C:\Users\YOU\.codex\hooks.json
local-decider study status --data-dir DATA --refresh
local-decider study export --data-dir DATA --output work/private-export.json
local-decider study analyze --data-dir DATA --output work/analysis.json
local-decider study stop --data-dir DATA
```

Installed definitions must be reviewed and trusted with Codex's hook controls
(`/hooks` in the CLI). Preparation and installation do not bypass trust or start
the study clock. If hooks or measurements are unavailable, report the limitation
and resolve it before claiming collection has begun.

## Paper and scheduled follow-up

A heartbeat attached to the original chat checks every six hours. It remains quiet
when the state is unchanged and reports missing data, important failures, required
actions, or completion. At expiry it freezes the data and prepares the technical
preprint with abstract, related work, equations, methods, randomized protocol,
results, limitations, conclusion, and reproducibility details.

The paper reports the prior synthetic result separately, including repeated
initial states, three test templates, oracle repairs, one seed, and development
iterations. No claim of a pristine untouched test set across development is made.
The private implementation and aggregate results can be published to the repository;
raw task records remain local. Deliver PDF, standalone LaTeX, and Markdown in the
original chat's outputs directory. Use `sgohils` as provisional author with no
invented affiliation. Small, missing, or adverse results are publishable findings.
