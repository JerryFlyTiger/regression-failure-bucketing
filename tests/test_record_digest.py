"""M4 perf regression guard: byte-exact FailureRecord.to_json() snapshots.

The snapshots in tests/fixtures/record_digest/ were produced by
tools/record_digest.py against the *before* state (pre M4 hot-path/
streaming rewrite) of rfb/isa.py and rfb/parse_trace.py, and re-confirmed
identical to the *after* state via ``diff -r out/m4_before out/m4_after``
(see PLAN.md). Any future change to the parsing/semantic-check hot path
that alters output must update these snapshots deliberately, not by
accident.
"""

from __future__ import annotations

import json
import os

from rfb.pipeline import parse_all

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "record_digest")
SET2_INPUT = os.path.join(
    REPO_ROOT, "testcase", "2026_08_20_samples", "problem", "benchmark_set_2", "input.csv"
)


def test_set2_digest_matches_snapshot():
    records = {r.case_id: r for r in parse_all(SET2_INPUT)}
    for case_id in (1, 2, 3):
        fixture_path = os.path.join(FIXTURE_DIR, f"case_{case_id}.json")
        with open(fixture_path) as f:
            expected = f.read().rstrip("\n")
        actual = records[case_id].to_json()
        assert actual == expected, (
            f"case {case_id} digest diverged from snapshot "
            f"{fixture_path} -- byte-exact equivalence broken"
        )
        # also sanity-check it's valid JSON with a case_id field, so a
        # trivially-truncated fixture file can't pass by accident.
        assert json.loads(expected)["case_id"] == case_id
