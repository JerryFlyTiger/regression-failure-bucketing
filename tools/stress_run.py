#!/usr/bin/env python3
"""Run bin/regr_fail_bucketing against a stress benchmark, measuring wall
time and peak RSS (including multiprocessing worker children), then score
it against the benchmark's golden.csv.

Usage: stress_run.py --bench DIR --k K [--workers W] [--tag TAG]

Prints one table row to stdout:
    cases | total_lines | wall_s | peak_rss_MB | BA

RSS measurement: prefers `/usr/bin/time -v` if present. Otherwise polls
/proc for the RSS of the whole process tree rooted at the child (this
correctly captures multiprocessing.Pool worker RSS, which plain
resource.getrusage(RUSAGE_CHILDREN) would miss since pool workers are
grandchildren, not direct children, of this script). If /proc is not
available either, falls back to resource.getrusage(RUSAGE_CHILDREN)
(direct-child-only, a known undercount for this pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
PROGRAM = os.path.join(REPO_ROOT, "bin", "regr_fail_bucketing")
SCORE_PY = os.path.join(REPO_ROOT, "tools", "score.py")


def _proc_tree_rss_kb(root_pid: int) -> int:
    """Sum VmRSS (kB) of root_pid and all its descendants, via /proc."""
    try:
        pids_by_ppid: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
                # comm field may contain spaces/parens; ppid is after the
                # last ')'.
                after = stat.rsplit(")", 1)[1].split()
                ppid = int(after[1])
            except (OSError, IndexError, ValueError):
                continue
            pids_by_ppid.setdefault(ppid, []).append(pid)

        total_kb = 0
        stack = [root_pid]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                with open(f"/proc/{pid}/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            total_kb += int(line.split()[1])
                            break
            except OSError:
                pass
            stack.extend(pids_by_ppid.get(pid, []))
        return total_kb
    except OSError:
        return 0


def _run_with_proc_sampler(cmd: list[str], interval: float = 0.2) -> tuple[int, float, int]:
    """Run cmd, sampling /proc RSS of its process tree. Returns
    (returncode, wall_s, peak_rss_kb)."""
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    peak_kb = 0
    stop = threading.Event()

    def sampler():
        nonlocal peak_kb
        while not stop.is_set():
            kb = _proc_tree_rss_kb(proc.pid)
            if kb > peak_kb:
                peak_kb = kb
            stop.wait(interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    rc = proc.wait()
    stop.set()
    th.join(timeout=5)
    # one last sample in case the process ballooned right before exit
    kb = _proc_tree_rss_kb(proc.pid)
    if kb > peak_kb:
        peak_kb = kb
    wall_s = time.time() - t0
    return rc, wall_s, peak_kb


def _time_v_supported(time_bin: str) -> bool:
    """Whether ``time_bin -v`` is a valid invocation (GNU time supports
    ``-v``; e.g. busybox's ``time`` does not)."""
    try:
        rc = subprocess.call(
            [time_bin, "-v", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return rc == 0


def _run_with_usr_bin_time(cmd: list[str], time_log: str, time_bin: str) -> tuple[int, float, int]:
    full_cmd = [time_bin, "-v", "-o", time_log] + cmd
    t0 = time.time()
    rc = subprocess.call(full_cmd)
    wall_s = time.time() - t0
    peak_kb = 0
    try:
        with open(time_log) as f:
            for line in f:
                if "Maximum resident set size" in line:
                    peak_kb = int(line.strip().split()[-1])
    except OSError:
        pass
    return rc, wall_s, peak_kb


def _run_with_rusage_fallback(cmd: list[str]) -> tuple[int, float, int]:
    import resource

    t0 = time.time()
    rc = subprocess.call(cmd)
    wall_s = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is in kB on Linux
    return rc, wall_s, ru.ru_maxrss


def _count_input_cases(bench_dir: str) -> int:
    import csv

    with open(os.path.join(bench_dir, "input.csv"), newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _total_lines(bench_dir: str) -> int | None:
    meta_path = os.path.join(bench_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("actual_total_lines")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, help="benchmark dir (has input.csv, golden.csv)")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None, help="unused hook, reserved")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args(argv)

    bench_dir = os.path.abspath(args.bench)
    tag = args.tag or os.path.basename(bench_dir)
    out_dir = os.path.join(REPO_ROOT, "out", "stress", tag)
    os.makedirs(out_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, "output.csv")

    cmd = [PY, PROGRAM, "--input", os.path.join(bench_dir, "input.csv"),
           "--output", output_csv, "--k", str(args.k)]

    time_bin = shutil.which("time") or (
        "/usr/bin/time" if os.path.exists("/usr/bin/time") else None
    )
    if time_bin and _time_v_supported(time_bin):
        time_log = os.path.join(out_dir, "time_v.log")
        rc, wall_s, peak_kb = _run_with_usr_bin_time(cmd, time_log, time_bin)
        method = "usr_bin_time"
    elif os.path.isdir("/proc"):
        rc, wall_s, peak_kb = _run_with_proc_sampler(cmd)
        method = "proc_sampler"
    else:
        rc, wall_s, peak_kb = _run_with_rusage_fallback(cmd)
        method = "rusage_children"

    peak_mb = peak_kb / 1024.0

    if rc != 0:
        print(f"stress_run: program exited rc={rc} (method={method})", file=sys.stderr)

    golden_csv = os.path.join(bench_dir, "golden.csv")
    ba = None
    ba_display = "NA"
    if rc == 0 and os.path.exists(golden_csv) and os.path.exists(output_csv):
        score_out = subprocess.run(
            [PY, SCORE_PY, golden_csv, output_csv],
            capture_output=True, text=True,
        )
        for line in score_out.stdout.splitlines():
            if line.startswith("BA="):
                ba = float(line.split("=", 1)[1])
        with open(os.path.join(out_dir, "score.log"), "w") as f:
            f.write(score_out.stdout)
            f.write(score_out.stderr)
        ba_display = ba if ba is not None else "NA"
    elif rc != 0:
        # Program failed: scoring an incomplete/absent output.csv would
        # silently mix a garbage BA into the table, so don't score at all.
        ba_display = f"FAILED(rc={rc})"

    n_cases = _count_input_cases(bench_dir)
    total_lines = _total_lines(bench_dir)

    row = {
        "tag": tag,
        "cases": n_cases,
        "total_lines": total_lines,
        "wall_s": round(wall_s, 2),
        "peak_rss_MB": round(peak_mb, 1),
        "BA": ba,
        "rss_method": method,
        "rc": rc,
        "ok": rc == 0,
    }
    with open(os.path.join(out_dir, "stress_run_result.json"), "w") as f:
        json.dump(row, f, indent=2)

    print("tag\tcases\ttotal_lines\twall_s\tpeak_rss_MB\tBA\trss_method")
    print(
        f"{tag}\t{n_cases}\t{total_lines}\t{wall_s:.2f}\t{peak_mb:.1f}\t"
        f"{ba_display}\t{method}"
    )
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
