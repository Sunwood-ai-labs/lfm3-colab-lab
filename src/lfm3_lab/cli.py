from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .lab import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, Lab, LabError
from .models import SUITES


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lfm3_lab",
        description="Run the six pinned LFM3 experiments on a Tesla T4 and summarize bounded results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare deterministic fixed image, audio, and video inputs")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    run = commands.add_parser("run", help="run one fixed model suite")
    run.add_argument("--model", choices=tuple(SUITES), required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    summarize = commands.add_parser("summarize", help="refresh and print a compact six-suite summary")
    summarize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    export = commands.add_parser("export", help="create a ZIP containing run artifacts and process logs")
    export.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            lab = Lab(args.output_dir, cache_dir=args.cache_dir)
            manifest = lab.prepare()
            print(f"Prepared fixed inputs under {args.output_dir / 'inputs'}")
            if manifest.get("speech_reference"):
                speech = manifest["speech_reference"]
                print(f"Public speech reference: {speech.get('duration_seconds', 'unknown')} seconds")
            return 0
        if args.command == "run":
            result = Lab(args.output_dir).run(args.model, repeats=args.repeats)
            print(
                f"{result.suite}: {result.status}; passed {result.passed}/{result.attempted}; "
                f"functional failures {result.functional_failures}; infrastructure errors {result.infrastructure_errors}"
            )
            print(f"Summary: {result.summary_path}")
            print(f"Logs: {result.stdout_path}, {result.stderr_path}")
            return 0
        if args.command == "summarize":
            rows = Lab(args.output_dir).summarize()
            print("suite\tmodel\tstatus\tpassed/attempted\tfunctional\tformat\tungraded\tinfrastructure")
            for row in rows:
                print(
                    f"{row['suite_key']}\t{row['model']}\t{row['status']}\t"
                    f"{row['passed']}/{row['attempted']}\t{row['functional_failures']}\t"
                    f"{row['format_failures']}\t{row['ungraded_count']}\t{row['infrastructure_errors']}"
                )
            return 0
        if args.command == "export":
            path = Lab(args.output_dir).export()
            print(path)
            return 0
    except LabError as exc:
        print(f"lfm3-lab: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
