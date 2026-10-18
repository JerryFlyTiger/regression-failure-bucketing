#!/usr/bin/env python3
"""Synthesize a stress-test benchmark from public set 2 (M4 perf work only).

Usage:
    stress_gen.py --cases N --total-lines L --out DIR [--seed S]

Design: see PLAN.md M4 spec / task description. Cases are assigned
round-robin over a seed-shuffled copy of the 25 source cases in set 2.
A single global amplification factor f = L / base_total_lines is computed;
if f <= 1 the source log files are symlinked directly (no distortion,
under target). If f > 1, sim.log and trace.log body lines are replicated
to approximately reach f times their original line count; regr.log is
never amplified.

golden.csv is produced here (from the source golden.csv) purely as a
benchmarking convenience for tools/score.py; rfb/ itself never reads it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rfb.isa import is_control_flow  # noqa: E402

SOURCE_DIR_DEFAULT = "testcase/2026_08_20_samples/problem/benchmark_set_2"
_CHUNK_LINES = 20000


def _read_source_golden(source_dir: str) -> dict[int, str]:
    path = os.path.join(source_dir, "golden.csv")
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[int(row["Case"])] = row["Bug"]
    return result


def _read_source_input(source_dir: str) -> dict[int, tuple[str, str, str]]:
    path = os.path.join(source_dir, "input.csv")
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = int(row["Case"])
            result[case] = (row["Regr Log"], row["Sim Log"], row["Trace Log"])
    return result


class SourceCase:
    """Lazily-loaded, cached content of one source case's three logs."""

    __slots__ = (
        "case_dir",
        "regr_lines",
        "sim_lines",
        "sim_report_idx",
        "sim_dup_line",
        "trace_header",
        "trace_body",
        "n_regr",
        "n_sim",
        "n_trace",
    )

    def __init__(self, case_dir: str, regr_path: str, sim_path: str, trace_path: str):
        self.case_dir = case_dir
        with open(regr_path, "rt", encoding="utf-8", errors="replace") as f:
            self.regr_lines = f.read().splitlines()
        with gzip.open(sim_path, "rt", encoding="utf-8", errors="replace") as f:
            self.sim_lines = f.read().splitlines()
        with gzip.open(trace_path, "rt", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        self.trace_header = lines[0] if lines else ""
        self.trace_body = lines[1:]

        self.n_regr = len(self.regr_lines)
        self.n_sim = len(self.sim_lines)
        self.n_trace = len(lines)

        # locate "--- UVM Report Summary ---" to know where to insert
        # duplicate lines, and pick a duplication candidate line (prefer a
        # UVM_ERROR line, else the first non-empty line).
        report_idx = len(self.sim_lines)
        for i, line in enumerate(self.sim_lines):
            if line.strip() == "--- UVM Report Summary ---":
                report_idx = i
                break
        self.sim_report_idx = report_idx

        dup_line = None
        for line in self.sim_lines[:report_idx]:
            if line.startswith("UVM_ERROR"):
                dup_line = line
                break
        if dup_line is None:
            for line in self.sim_lines:
                if line.strip():
                    dup_line = line
                    break
        self.sim_dup_line = dup_line or ""


def _write_regr(src: SourceCase, out_path: str) -> None:
    with open(out_path, "wt", encoding="utf-8") as f:
        for line in src.regr_lines:
            f.write(line)
            f.write("\n")


def _write_sim_scaled(src: SourceCase, out_path: str, f: float) -> None:
    target_lines = max(src.n_sim, round(f * src.n_sim))
    n_extra = max(0, target_lines - src.n_sim)
    before = src.sim_lines[: src.sim_report_idx]
    after = src.sim_lines[src.sim_report_idx :]
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=1) as f_out:
        buf = []
        for line in before:
            buf.append(line)
        for _ in range(n_extra):
            buf.append(src.sim_dup_line)
            if len(buf) >= _CHUNK_LINES:
                f_out.write("\n".join(buf))
                f_out.write("\n")
                buf = []
        for line in after:
            buf.append(line)
        if buf:
            f_out.write("\n".join(buf))
            f_out.write("\n")


def _line_mnemonic(line: str) -> str:
    """Extract the mnemonic from a raw trace body line (see
    docs/ARCHITECTURE.md 4.6: Decoded column, first whitespace token)."""
    parts = line.split("\t")
    if len(parts) < 5:
        return ""
    decoded = parts[4]
    toks = decoded.split()
    return toks[0] if toks else ""


def _seam_trimmed_body(trace_body: list[str]) -> list[str]:
    """Drop trailing control-flow instructions from a replica so the PC
    discontinuity at the seam with the next replica lands after a
    non-control-flow instruction (gets counted as an implicit jump / trap
    entry and excluded from semantic checks, instead of being misjudged as
    e.g. BRANCH_WRONG_TARGET on the control-flow instruction itself)."""
    n = len(trace_body)
    while n > 0 and is_control_flow(_line_mnemonic(trace_body[n - 1])):
        n -= 1
    return trace_body[:n] if n > 0 else trace_body


def _write_trace_scaled(src: SourceCase, out_path: str, f: float) -> None:
    body_target = max(len(src.trace_body), round(f * len(src.trace_body)))
    trimmed_body = _seam_trimmed_body(src.trace_body)
    # Number of replicas needed using the trimmed body length for all but
    # the final (possibly partial) replica, which uses the full body since
    # nothing follows it (no seam to protect).
    reps = math.ceil(body_target / max(1, len(trimmed_body)))
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=1) as f_out:
        f_out.write(src.trace_header)
        f_out.write("\n")
        written = 0
        buf = []
        for rep_i in range(reps):
            is_last_rep = rep_i == reps - 1
            body = src.trace_body if is_last_rep else trimmed_body
            for line in body:
                if written >= body_target:
                    break
                buf.append(line)
                written += 1
                if len(buf) >= _CHUNK_LINES:
                    f_out.write("\n".join(buf))
                    f_out.write("\n")
                    buf = []
            if written >= body_target:
                break
        if buf:
            f_out.write("\n".join(buf))
            f_out.write("\n")


def _symlink(source_abs: str, out_path: str) -> None:
    if os.path.lexists(out_path):
        os.remove(out_path)
    os.symlink(source_abs, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, required=True)
    ap.add_argument("--total-lines", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--source", type=str, default=SOURCE_DIR_DEFAULT)
    args = ap.parse_args(argv)

    t_start = time.time()
    source_dir = os.path.abspath(args.source)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    golden_map = _read_source_golden(source_dir)
    input_map = _read_source_input(source_dir)
    source_cases = sorted(input_map.keys())
    n_source = len(source_cases)
    if n_source == 0:
        print("stress_gen: source input.csv has no cases", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shuffled = list(source_cases)
    rng.shuffle(shuffled)

    n = args.cases
    assign = [shuffled[i % n_source] for i in range(n)]

    # First pass: compute base_total (line counts without amplification) by
    # reading each *distinct* source case's three logs once (small, ~16MB
    # total for set 2).
    print(f"stress_gen: loading {n_source} source cases ...", file=sys.stderr)
    cache: dict[int, SourceCase] = {}
    for sc in sorted(set(assign)):
        regr_rel, sim_rel, trace_rel = input_map[sc]
        case_dir = os.path.join(source_dir, os.path.dirname(regr_rel) or ".")
        cache[sc] = SourceCase(
            case_dir,
            os.path.join(source_dir, regr_rel),
            os.path.join(source_dir, sim_rel),
            os.path.join(source_dir, trace_rel),
        )

    base_total = sum(
        cache[sc].n_regr + cache[sc].n_sim + cache[sc].n_trace for sc in assign
    )
    f = args.total_lines / base_total if base_total > 0 else 1.0
    mode = "symlink" if f <= 1.0 else "amplify"
    print(
        f"stress_gen: cases={n} base_total_lines={base_total} target={args.total_lines} "
        f"f={f:.4f} mode={mode}",
        file=sys.stderr,
    )

    rows = []
    golden_rows = []
    actual_total = 0
    progress_step = max(1, n // 10)
    for i in range(1, n + 1):
        sc = assign[i - 1]
        src = cache[sc]
        case_out_dir = os.path.join(out_dir, f"case_{i}")
        os.makedirs(case_out_dir, exist_ok=True)

        regr_out = os.path.join(case_out_dir, "regr.log")
        sim_out = os.path.join(case_out_dir, "sim.log.gz")
        trace_out = os.path.join(case_out_dir, "trace.log.gz")

        if mode == "symlink":
            actual_total += src.n_regr + src.n_sim + src.n_trace
            regr_src_abs = os.path.join(source_dir, input_map[sc][0])
            sim_src_abs = os.path.join(source_dir, input_map[sc][1])
            trace_src_abs = os.path.join(source_dir, input_map[sc][2])
            _symlink(regr_src_abs, regr_out)
            _symlink(sim_src_abs, sim_out)
            _symlink(trace_src_abs, trace_out)
        else:
            regr_src_abs = os.path.join(source_dir, input_map[sc][0])
            _symlink(regr_src_abs, regr_out)  # regr.log never amplified
            sim_lines_written = max(src.n_sim, round(f * src.n_sim))
            trace_body_target = max(len(src.trace_body), round(f * len(src.trace_body)))
            _write_sim_scaled(src, sim_out, f)
            _write_trace_scaled(src, trace_out, f)
            actual_total += src.n_regr + sim_lines_written + 1 + trace_body_target

        rows.append((i, f"case_{i}/regr.log", f"case_{i}/sim.log.gz", f"case_{i}/trace.log.gz"))
        golden_rows.append((i, golden_map[sc]))

        if i % progress_step == 0 or i == n:
            pct = round(100 * i / n)
            print(f"stress_gen: {pct}% ({i}/{n})", file=sys.stderr)

    with open(os.path.join(out_dir, "input.csv"), "w", newline="") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        w.writerows(rows)

    with open(os.path.join(out_dir, "golden.csv"), "w", newline="") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(["Case", "Bug"])
        w.writerows(golden_rows)

    elapsed = time.time() - t_start
    du = 0
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            fp = os.path.join(root, fname)
            if not os.path.islink(fp):
                du += os.path.getsize(fp)

    import json

    meta = {
        "cases": n,
        "requested_total_lines": args.total_lines,
        "actual_total_lines": actual_total,
        "mode": mode,
        "f": f,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "on_disk_bytes": du,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f_meta:
        json.dump(meta, f_meta, indent=2)

    print(
        f"stress_gen: done in {elapsed:.1f}s, out_dir={out_dir}, "
        f"actual_total_lines={actual_total}, "
        f"on-disk (non-symlink) bytes={du} ({du / 1e6:.1f} MB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
