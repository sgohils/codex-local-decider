import argparse
import json
from pathlib import Path

from .contracts import State, actions_from_dict
from .evolution import Config, compare, evaluate, report, train
from .exercises import dataset, verify_fixtures
from .models import Model
from .storage import read_json, write_json


def main(argv=None):
    import sys
    selected = list(sys.argv[1:] if argv is None else argv)
    if selected and selected[0] == "study":
        from .study import main as study_main
        return study_main(selected[1:])
    parser = argparse.ArgumentParser(description="Train bounded local decision models before Codex integration")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--kind", choices=("linear", "neural", "both"), default="both")
    training.add_argument("--output", type=Path, default=Path("runs/initial"))
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--generations", type=int, default=50)
    training.add_argument("--population", type=int, default=24)
    training.add_argument("--parents", type=int, default=4)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--minutes", type=float, default=15)
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "validation", "test"), default="test")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--output", type=Path, default=Path("runs/initial"))
    comparison.add_argument("--seed", type=int, default=7)
    decision = commands.add_parser("decide")
    decision.add_argument("--model", type=Path, required=True)
    decision.add_argument("--input", type=Path, required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument("--input", type=Path, required=True)
    reporting.add_argument("--output", type=Path, required=True)
    fixtures = commands.add_parser("verify-fixtures")
    fixtures.add_argument("--output", type=Path, default=Path("work/fixtures"))
    for name in ("hook", "install-hooks"):
        hook = commands.add_parser(name)
        hook.add_argument("--data-dir", type=Path, default=Path(".local-decider"))
        hook.add_argument("--candidate", type=Path)
        hook.add_argument("--experimental", action="store_true")
        if name == "install-hooks":
            hook.add_argument("--project", type=Path, default=Path.cwd())
    background = commands.add_parser("worker")
    background.add_argument("--data-dir", type=Path, default=Path(".local-decider"))
    live = commands.add_parser("benchmark")
    live.add_argument("--candidate", type=Path, required=True)
    live.add_argument("--output", type=Path, default=Path("runs/live"))
    live.add_argument("--git", required=True)
    live.add_argument("--codex", default="codex")
    live.add_argument("--pairs", type=int, default=12)
    live.add_argument("--max-tokens", type=int, default=100_000)
    live.add_argument("--timeout", type=int, default=180)
    live.add_argument("--model")
    activation = commands.add_parser("promote")
    activation.add_argument("--candidate", type=Path, required=True)
    activation.add_argument("--live-result", type=Path, required=True)
    activation.add_argument("--data-dir", type=Path, default=Path(".local-decider"))
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            results = []
            kinds = ("linear", "neural") if args.kind == "both" else (args.kind,)
            for kind in kinds:
                config = Config(population=args.population, parents=args.parents,
                                generations=args.generations, workers=args.workers,
                                minutes=args.minutes / len(kinds), seed=args.seed)
                results.append(train(kind, args.output, config, args.resume,
                                     lambda value: print(json.dumps(value), flush=True)))
            write_json(args.output / "training.json", {"results": results})
            result = {"models": [r["kind"] for r in results], "output": str(args.output)}
        elif args.command == "evaluate":
            result = evaluate(Model.load(args.model), dataset()[args.split])
        elif args.command == "compare":
            result = compare(args.output, args.seed)
            write_json(args.output / "comparison.json", result)
            (args.output / "report.md").write_text(report(result), encoding="utf-8")
        elif args.command == "decide":
            value = read_json(args.input)
            result = Model.load(args.model).decide(State.from_dict(value["state"]),
                                                   actions_from_dict(value["available_actions"]))
        elif args.command == "report":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report(read_json(args.input)), encoding="utf-8")
            result = {"report": str(args.output)}
        elif args.command == "verify-fixtures":
            result = verify_fixtures(args.output)
            if not result["passed"]:
                print(json.dumps(result))
                raise SystemExit(1)
        elif args.command in ("hook", "install-hooks", "worker"):
            from .hooks import install, run_hook, worker
            if args.command == "hook":
                run_hook(args.data_dir, args.candidate, args.experimental)
                return
            if args.command == "worker":
                result = worker(args.data_dir)
            else:
                result = {"hooks": str(install(args.project, args.data_dir, args.candidate, args.experimental)),
                          "next": "Review these hooks with /hooks in Codex; untrusted hooks are skipped"}
        elif args.command in ("benchmark", "promote"):
            from .benchmark import promote, run
            if args.command == "promote":
                result = promote(args.candidate, args.live_result, args.data_dir)
            else:
                result = run(args.candidate, args.output, args.git, args.codex, args.pairs,
                             args.max_tokens, args.timeout, args.model,
                             lambda value: print(json.dumps(value), flush=True))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"local-decider: {error}\n")
