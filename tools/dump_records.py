#!/usr/bin/env python3
"""Dump FailureRecord JSON (one per line) for a given input.csv.

Usage: dump_records.py <input.csv> [--case N]

Requires rfb.parse_regr/parse_sim/parse_trace (milestone M1); exits 2 with
a message if they are not available yet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfb.pipeline import parse_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dump_records")
    parser.add_argument("input", help="path to input.csv")
    parser.add_argument("--case", type=int, default=None, help="only print this case_id")
    args = parser.parse_args(argv)

    records = parse_all(args.input)
    for record in records:
        if args.case is not None and record.case_id != args.case:
            continue
        print(record.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
