"""End-to-end pipeline.run() against the two public benchmark sets, scored
against golden.csv (read only here in tests, never inside rfb/ itself).

See docs/ARCHITECTURE.md section 8 (M3 acceptance): set 1/2 balanced
accuracy >= 0.95, and output must be identical regardless of input case
order.
"""

import csv
import os
import random

import pytest

from rfb import pipeline
from tools.score import _read_case_label_csv, balanced_accuracy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEM_ROOT = os.path.join(REPO_ROOT, "testcase", "2026_08_20_samples", "problem")

_SETS = {
    "benchmark_set_1": 2,
    "benchmark_set_2": 4,
}


def _score(set_name: str, k: int, input_csv: str, output_csv: str) -> float:
    pipeline.run(input_csv, output_csv, k)
    golden = _read_case_label_csv(
        os.path.join(PROBLEM_ROOT, set_name, "golden.csv"), ("bug",)
    )
    output = _read_case_label_csv(output_csv, ("bucket",))
    _, _, _, _, _, _, ba = balanced_accuracy(golden, output)
    return ba


@pytest.mark.parametrize("set_name,k", list(_SETS.items()))
def test_public_set_balanced_accuracy(set_name, k, tmp_path):
    input_csv = os.path.join(PROBLEM_ROOT, set_name, "input.csv")
    output_csv = str(tmp_path / "output.csv")
    ba = _score(set_name, k, input_csv, output_csv)
    assert ba >= 0.95


def _write_shuffled_input(src: str, dest: str, seed: int) -> None:
    base_dir = os.path.dirname(os.path.abspath(src))
    with open(src, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rng = random.Random(seed)
    rng.shuffle(rows)

    for row in rows:
        for i in (1, 2, 3):
            row[i] = os.path.join(base_dir, row[i])

    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.mark.parametrize("set_name,k", list(_SETS.items()))
def test_public_set_output_stable_under_shuffle(set_name, k, tmp_path):
    orig_input = os.path.join(PROBLEM_ROOT, set_name, "input.csv")

    orig_output = str(tmp_path / "orig_output.csv")
    pipeline.run(orig_input, orig_output, k)

    shuffled_input = str(tmp_path / "shuffled_input.csv")
    _write_shuffled_input(orig_input, shuffled_input, seed=20260928)
    shuffled_output = str(tmp_path / "shuffled_output.csv")
    pipeline.run(shuffled_input, shuffled_output, k)

    orig = _read_case_label_csv(orig_output, ("bucket",))
    # normalize labels: compare partitions, not label names (bucket_N
    # numbering is by min case_id and is order-independent by
    # construction, but comparing partitions directly is the real
    # invariant under test).
    shuffled = _read_case_label_csv(shuffled_output, ("bucket",))
    assert orig == shuffled


# hknegu (PLAN.md 4c 公開合併 0.96 根因): set 1 + set 2 merged into one bench
# (set 1 ids re-based to 101.. as in tools/mixed_eval.py). Same-bug cases from
# different tests collide on hard_key; the CSR X-origin signature separates
# bug_7021 from bug_7023. Without R1x- / xsig the merged BA is 0.9600.
MERGED_BA_FLOOR = 0.976


def test_public_merged_balanced_accuracy_regression(tmp_path):
    rows, golden = [], {}
    for set_name, id_base in (("benchmark_set_2", 0), ("benchmark_set_1", 100)):
        d = os.path.join(PROBLEM_ROOT, set_name)
        gold = _read_case_label_csv(os.path.join(d, "golden.csv"), ("bug",))
        with open(os.path.join(d, "input.csv"), newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                c = int(row[0].strip())
                rows.append([str(id_base + c)] + [os.path.join(d, p.strip()) for p in row[1:4]])
                golden[str(id_base + c)] = gold[str(c)]
    merged = str(tmp_path / "merged_input.csv")
    with open(merged, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        w.writerows(rows)
    output = str(tmp_path / "merged_output.csv")
    pipeline.run(merged, output, len(set(golden.values())))
    got = _read_case_label_csv(output, ("bucket",))
    ba = balanced_accuracy(golden, got)[-1]
    assert ba >= MERGED_BA_FLOOR, ba
