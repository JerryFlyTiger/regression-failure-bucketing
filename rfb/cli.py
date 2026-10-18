"""Command-line entry point for regr_fail_bucketing."""

import argparse
import sys

from rfb import pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regr_fail_bucketing")
    parser.add_argument("--input", required=True, help="path to input.csv")
    parser.add_argument("--output", required=True, help="path to output.csv")
    parser.add_argument("--k", required=True, type=int, help="hint: number of buckets")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline.run(args.input, args.output, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
