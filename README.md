# Regression Failure Bucketing

An independent solution to **Problem B, "Regression Failure Bucketing"**, of the
[2026 CAD Contest @ ICCAD](https://www.iccad-contest.org/Problems.html)
(problem provided by Synopsys).

Given a set of failing simulations of a modified Ibex RISC-V core, the program
groups the failures into buckets so that failures caused by the same injected
bug end up in the same bucket. Each failing case comes with three logs: the
regression comparator output (`regr.log`), the UVM console log (`sim.log.gz`),
and the per-instruction retire trace (`trace.log.gz`).

This is not a contest entry. It was developed independently, as a personal
project, and has only been evaluated on the public sample benchmarks.

## Public and private versions

| Version | Repository | Contents |
|---|---|---|
| Public (this repo) | `JerryFlyTiger/regression-failure-bucketing` | Source code and tests only |
| Private | [`JerryFlyTiger/Regressio_Failure_Bucketing`](https://github.com/JerryFlyTiger/Regressio_Failure_Bucketing) (not publicly accessible) | Full development history, design notes, review records |

- Development happens in the private repository. **The private version is
  ahead of this one.** Reviewed milestones are copied here as periodic
  snapshots.
- The current snapshot corresponds to private commit `6872edb`.
- Not included here:
  - The problem description, Q&A and leaderboards. See the official contest
    website.
  - The benchmark data (logs, `input.csv`, `golden.csv`). Download it from
    the official contest website (see [Test data](#test-data)).
  - Design documents. Some code comments refer to `docs/ARCHITECTURE.md`,
    which lives only in the private repository.

## Approach

```
regr.log ─┐
sim.log ──┼─> parse ──> ISA semantic check ──> signature ──> clustering ──> output.csv
trace.log ┘   (rfb/parse_*)   (rfb/isa, diverge)   (rfb/signature)  (rfb/cluster)
```

- **Parse** the three logs as streams. Cases are processed in parallel.
- **Re-check the retire trace against ISA semantics.** Branch conditions,
  ALU results, register reads against the last write, loads against a memory
  shadow, CSR reads (including predicted `mstatus` fields), trap/xret targets,
  and decoder register fields against the printed operands. The histogram of violations over the whole trace acts
  as the bug's signature. The divergence point reported by the comparator is
  used as a precise anchor.
- **Build a signature** per case: failure family (MISMATCH / FATAL / ASSERT),
  the deviating instruction class, the systematic violation classes, and
  UVM message features.
- **Cluster** with hard keys plus hierarchical merging. Cross-family rules
  link, for example, a mismatch case and a UVM fatal case that share the same
  trace evidence. The `--k` hint is used as a target bucket count.

The program never reads `golden.csv`. It also ignores metadata such as file
paths, seeds, timestamps and case order, so shuffling the cases does not
change the result.

## Results

Scored with pairwise balanced accuracy, the contest metric
(`tools/score.py`):

| Benchmark | Cases | Bugs | Original order | Shuffled |
|---|---|---|---|---|
| Public set 1 | 10 | 2 | 1.000 | 1.000 |
| Public set 2 | 30 | 4 | 1.000 | 1.000 |

The hidden benchmarks (sets 3–10, up to 3000 cases and 64 bugs) are not
public, so performance on them is unknown. The public sets are small, so a
perfect score there says little about the hidden ones.

## Requirements

- Python 3 (developed and tested with Python 3.14 on macOS)
- `numpy` (the program itself); `pytest` and `coverage` for the tests

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Some tests and scripts expect the virtual environment at `.venv/` in the
repository root.

## Test data

The problem description and benchmark data are published on the
[official contest website](https://www.iccad-contest.org/Problems.html)
and are not redistributed here. After downloading the samples, place them
like this (`testcase/` is ignored by git):

```
testcase/2026_08_20_samples/problem/
├── benchmark_set_1/
│   ├── input.csv
│   ├── golden.csv
│   └── case_1/ … case_10/
└── benchmark_set_2/
    └── …
```

## Usage

Run on one benchmark (paths in `input.csv` are relative to the CSV file):

```sh
.venv/bin/python bin/regr_fail_bucketing --input input.csv --output output.csv --k 4
```

The output is a CSV with columns `Case,bucket`.

Score both public sets, in the original and a shuffled order:

```sh
tools/run_public.sh
```

Run the tests:

```sh
.venv/bin/python -m pytest tests
```

Without the test data, the tests that read the public benchmarks fail
(about 110 of 1426). The others still run. With the data in place, all
tests pass.

## Repository layout

| Path | Contents |
|---|---|
| `bin/regr_fail_bucketing` | Command-line entry point |
| `rfb/` | The bucketing program |
| `tests/` | Unit, regression and end-to-end tests |
| `tools/` | Scoring, benchmark runner, bug-injection and stress-test experiments, mutation-testing helper |

## License

The code in this repository is released under the [MIT License](LICENSE).
The license does not cover the contest problem description or benchmark data,
which belong to the contest organizers and are not included here.
