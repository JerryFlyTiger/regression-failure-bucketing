#!/usr/bin/env python3
"""M4 stage-level timing breakdown for the rfb pipeline, without touching
rfb/ itself: (a) parse (multiprocessing pool, same as pipeline.parse_all),
(b) signature build, (c) clustering. Also runs cProfile on parsing a single
case (useful to pick out the biggest case in a --bench dir) and dumps the
top-25 functions by cumulative time.

Usage:
    profile_stages.py --bench DIR --k K [--tag TAG]
    profile_stages.py --bench DIR --cprofile-case N [--tag TAG]
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import os
import pstats
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _stage_timing(bench_dir: str, k: int) -> dict:
    from rfb import pipeline, signature, cluster

    input_csv = os.path.join(bench_dir, "input.csv")

    t0 = time.perf_counter()
    records = pipeline.parse_all(input_csv)
    t1 = time.perf_counter()
    signatures = signature.build_signatures(records)
    t2 = time.perf_counter()
    labels = cluster.cluster(signatures, k)
    t3 = time.perf_counter()

    return {
        "n_cases": len(records),
        "parse_s": t1 - t0,
        "signature_s": t2 - t1,
        "cluster_s": t3 - t2,
        "total_s": t3 - t0,
        "n_labels": len(labels),
    }


def _largest_case(bench_dir: str) -> int:
    """Pick the case with the largest trace.log.gz on disk (proxy for the
    biggest case, without decompressing everything)."""
    input_csv = os.path.join(bench_dir, "input.csv")
    best_case, best_size = None, -1
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["Case"])
            trace_path = os.path.join(bench_dir, row["Trace Log"].strip())
            try:
                size = os.path.getsize(trace_path)
            except OSError:
                continue
            if size > best_size:
                best_size = size
                best_case = case_id
    return best_case


def _cprofile_case(bench_dir: str, case_id: int, out_path: str) -> None:
    from rfb.pipeline import _parse_one_case

    input_csv = os.path.join(bench_dir, "input.csv")
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        row = next(r for r in reader if int(r["Case"]) == case_id)

    regr_path = os.path.join(bench_dir, row["Regr Log"].strip())
    sim_path = os.path.join(bench_dir, row["Sim Log"].strip())
    trace_path = os.path.join(bench_dir, row["Trace Log"].strip())

    pr = cProfile.Profile()
    pr.enable()
    _parse_one_case(case_id, regr_path, sim_path, trace_path)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(40)
    with open(out_path, "w") as f:
        f.write(stream.getvalue())
    print(f"profile_stages: cProfile for case {case_id} written to {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cprofile-case", type=int, default=None,
                     help="case id to cProfile; if omitted but --stages-and-profile "
                          "is set, auto-picks the case with the largest trace.log.gz")
    ap.add_argument("--stages-and-profile", action="store_true",
                     help="run both the stage timing and a cProfile of the largest case")
    args = ap.parse_args(argv)

    bench_dir = os.path.abspath(args.bench)
    tag = args.tag or os.path.basename(bench_dir)
    out_dir = os.path.join(REPO_ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)

    if args.k is not None:
        timing = _stage_timing(bench_dir, args.k)
        print(
            f"stage_timing[{tag}]: n_cases={timing['n_cases']} "
            f"parse_s={timing['parse_s']:.2f} signature_s={timing['signature_s']:.3f} "
            f"cluster_s={timing['cluster_s']:.3f} total_s={timing['total_s']:.2f}"
        )
        import json
        with open(os.path.join(out_dir, f"m4_stage_timing_{tag}.json"), "w") as f:
            json.dump(timing, f, indent=2)

    if args.cprofile_case is not None or args.stages_and_profile:
        case_id = args.cprofile_case
        if case_id is None:
            case_id = _largest_case(bench_dir)
        out_path = os.path.join(out_dir, f"m4_profile_{tag}.txt")
        _cprofile_case(bench_dir, case_id, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
