# Manuscript brief for the study's scheduled completion

Provisional title: **Evolving Local Decision Controllers for Coding Assistants:
A Five-Day Randomized Pilot**. Author: **sgohils**, no invented affiliation.
Produce a full technical preprint, not a promotional summary. This is an original
prototype inspired by typed decision interfaces; do not call it a Jev reproduction
or claim a new neuroevolution algorithm.

## Evidence and sections

Write abstract, introduction and research questions, related work, system design,
evolution methods, synthetic experiment, preregistered randomized field study,
results, threats to validity, privacy and reproducibility, discussion, conclusion,
references, and a methods appendix. Separate hypotheses, targets, observations,
and conclusions. Approximately 10–14 pages is appropriate; accuracy takes priority
over length. Cite verified primary sources and provide a brief AI assistance disclosure.

Use `examples/results/local-comparison.json`, `run-manifest.json`, and
`learning-curves.csv` for synthetic measurements. The linear controller completed
24/24 episodes, neural 12/24, rules 24/24, random 1/24. Proxy costs were 562.5,
690.0, 597.5, and 1130.4167 per episode, respectively. The 5.858% reduction relative
to rules is simulated context, not measured Codex tokens. Completion includes an
oracle repair and is not autonomous patch generation. The failed CLI live attempt
completed zero pairs and has no valid usage measurements.

Eight variants encode four distinct starting flag combinations, each repeated.
There are nine training, three validation, and three test templates. There are 12
distinct test template/initial-state combinations, not 24 independent tasks.
Seed affects evolution but does not randomize the template split or fixture code.
Algorithm development inspected synthetic test metrics in previous iterations;
the final saved run must not be described as a pristine untouched evaluation.
The simulator uses raw score argmax, whereas actual advisory hooks also use the
uncertainty margin and abstention. Discuss this deployment difference explicitly.

Use the frozen study export for randomized measurements, generated analysis for
statistics, and the installed protocol in `docs/STUDY.md`. Never import old chats
as observations or fill missing data with zero. If collection never starts, report
that integration verification remains pending; do not invent a five-day trial.
If sample size is small or outcomes unknown, publish an inconclusive pilot paper.
Report instrument coverage, group composition, missing child usage, failure and
interruption counts, advice delivery, and measured timing limitations.

Show the linear score `w^T phi(s,a)` and neural score
`v^T tanh(W^T phi(s,a)+b)+c`. Features have 64 normalized hashed-text coordinates
and six action blocks with one bias, seven observed flags, and one repetition
feature: 118 dimensions. The 16-unit neural model has 1,921 parameters.
Training uses 24 candidates, four unchanged parents, 50 generations, initialization
N(0,0.2^2), mutations N(0,0.15^2) masked independently with probability 0.1, and
round-robin parent inheritance. Parent selection minimizes inverse-class-frequency
weighted softmax cross entropy plus 0.0001 times mean squared parameter magnitude.
Validation ranks completion, progress, negative proxy cost, and negative redundant
steps lexicographically. There is no crossover, backpropagation, or topology growth.
Background candidates remain separate; the field study model never changes.

## Figures and sources

Generate publication-quality figures with standard plotting tools: system and
study flow, saved evolution curves, synthetic completion/proxy cost comparisons,
randomized token distributions and uncertainty if available, and collection coverage.
Omit unavailable field plots with an explicit explanation rather than draw fake data.
Preserve plot data and SVG/PDF or high-resolution PNG artifacts.

Verify and cite at least the following primary sources, expanding related work as
needed with evolutionary optimization, policy routing, and coding-agent evaluation:

- TypeSafe AI, *Introducing System One Models & Jev* (2026):
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Salimans et al., *Evolution Strategies as a Scalable Alternative to Reinforcement
  Learning* (2017): https://arxiv.org/abs/1703.03864
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*:
  https://arxiv.org/abs/2310.06770
- Official Codex hooks: https://learn.chatgpt.com/docs/hooks
- Official non-interactive JSON usage: https://learn.chatgpt.com/docs/non-interactive-mode

## Deliverables and finalization

Create the PDF, standalone `.tex` source with embedded references/plot data, and
Markdown from the same manuscript in the original chat's `outputs/` folder.
Open and validate the standalone LaTeX using Codex's built-in editor/compiler.
That compiler returns diagnostics, not a saved PDF export; use an available export
route, or generate the matching PDF from the same manuscript using the PDF skill.
Visually inspect the final PDF and verify all tables against the frozen results.
Do not claim the PDF was compiled from LaTeX if it used another renderer.

Publish the tested implementation and aggregate research artifacts to the existing
private `sgohils/codex-local-decider` repository. Do not publish per-task logs,
private transcript paths, evidence files, tokens/credentials, or source from other
projects. Report final study dates in America/New_York, actual sample sizes,
limitations, repository URL, and links to delivered files. Finish the heartbeat
after delivery; report any actual compilation or publication blocker candidly.
