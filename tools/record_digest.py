#!/usr/bin/env python3
"""Dump per-case FailureRecord.to_json() digests for equivalence checking
(M4 perf work). For a given input.csv, calls rfb.pipeline.parse_all and
writes one <outdir>/case_<id>.json per record (sorted by case id).

Usage:
    record_digest.py --input INPUT_CSV --out OUTDIR
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from rfb.pipeline import parse_all

    records = parse_all(args.input)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    for record in sorted(records, key=lambda r: r.case_id):
        out_path = os.path.join(out_dir, f"case_{record.case_id}.json")
        with open(out_path, "w") as f:
            f.write(record.to_json())
            f.write("\n")

    print(f"record_digest: wrote {len(records)} case digests to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
