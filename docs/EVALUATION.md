# Evaluation protocol

## Local experiment

Evolution receives synthetic reference workflow decisions as a dense reward. These references are derived from observable exercise state. No LLM inference occurs. Validation runs full bounded episodes and prioritizes task completion over token estimates. Both policy architectures use the same features, exercises, population, and generation budget.

The script supplies the known repair when enough evidence has been gathered and the policy hands off. Successful completion means reaching the verified workflow terminal state, not independently diagnosing or generating an unknown patch. An actual subprocess verifier separately confirms the broken and repaired Python fixtures.

The rules baseline encodes the reference workflow and is deliberately strong. Evolution is useful here only if it learns comparable decisions or transfers to live tasks. Failure to beat rules is a valid experimental result.

The eight variants per template repeat four initial flag combinations, so the 24
test episodes cover only 12 distinct template/state combinations across three
templates. Prior development iterations examined test scores; the final saved run
is a preliminary development result, not an untouched confirmatory benchmark.
Synthetic evaluation uses raw score argmax. Deployment additionally applies an
uncertainty margin and may abstain, so live behavior needs separate measurement.

## Live experiment

Each task has baseline and assisted fresh checkouts with identical fixtures, prompts, model settings, and sandbox policies. Arm order alternates. Tests must remain byte-for-byte unchanged and pass after Codex finishes. Usage comes from every `turn.completed` JSON event; cached input is included once as part of input, never added twice.

We report actual successes, total input/output tokens, cached input separately, command counts, event failures, elapsed time, and advice delivery counts. Failed or incomplete trials cannot qualify a candidate. The token cap is enforced between calls because final usage is reported after a turn completes.

Twelve small pairs are an exploratory pilot, not statistical proof of broad savings. Fixed order within each pair and alternate ordering reduce but do not eliminate cache effects. Aggregate reported tokens are not equivalent to billable dollars or ChatGPT subscription usage units.

The pilot fixtures are separate fresh checkouts but can overlap local exercise templates. This tests integration behavior, not unseen semantic bug understanding. Larger independent repositories and repeated trials are needed for generalization claims.

## Improvement boundaries

Real logs reveal only observed actions and outcomes. They do not contain reliable rewards for untried alternatives. Only verified post-patch test decisions can supplement training; no automatic labeling based on a model's own confidence is used. Validation remains independent of those examples.

Candidate snapshots are inactive by default. An active model is only replaced after its exact checkpoint passes the live gate. Existing active manifests are preserved for rollback.
