"""Self-verification from ARCHITECTURE.md section 4.3:

    "對公開集每個 MISMATCH case，分歧點錨點之前的區段是 RTL 與 Spike 一致的區段，
    掃描結果必須是零違規（排除規則生效後）。"

Concretely: for each MISMATCH case, the *entire* trace (not just the anchor
window) must only ever produce the case's own recurring bug signature -- the
set of ``(kind, mnemonic)`` keys observed in ``trace.violations`` must be a
subset of ``{expected_kind}:{m}`` for m in the fixture's expected_mnemonics.
This is deliberately stronger than only checking the anchor-window prefix:
a bug in the section 4.3 exclusion rule (trap entries) or in operand/target
parsing could easily produce foreign violations far outside the anchor
window, and a prefix-only check would never see them.

A single case is allowed to have *multiple* mnemonics for its expected kind
(e.g. bge/bgeu) because some MISMATCH cases (bug_107 in particular) have the
same underlying RTL bug re-manifest across more than one branch mnemonic of
the same class within a single trace -- that is still the "same" recurring
bug signature, not a foreign violation.
"""

import json
from pathlib import Path

from rfb.parse_regr import parse_regr
from rfb.parse_trace import parse_trace

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/public_expected.json").read_text())
SET_DIRS = {
    "benchmark_set_1": ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_1",
    "benchmark_set_2": ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_2",
}


def _mismatch_cases():
    for set_name, cases in FIXTURE.items():
        for case_id, info in cases.items():
            if info["family"] == "MISMATCH":
                yield set_name, case_id, info


def test_whole_trace_has_only_expected_violation_keys():
    cases = list(_mismatch_cases())
    assert cases, "fixture should contain MISMATCH cases"
    for set_name, case_id, info in cases:
        case_dir = SET_DIRS[set_name] / f"case_{case_id}"
        regr = parse_regr(str(case_dir / "regr.log"))
        trace = parse_trace(str(case_dir / "trace.log.gz"), regr)

        expected_kind = info["expected_kind"]
        expected_mnemonics = set(info["expected_mnemonics"] or [])
        allowed_keys = {f"{expected_kind}:{m}" for m in expected_mnemonics}

        observed_keys = set(trace.violations.keys())
        foreign_keys = observed_keys - allowed_keys
        if foreign_keys:
            samples = [
                s.to_dict()
                for s in trace.violation_samples
                if f"{s.kind}:{s.mnemonic}" in foreign_keys
            ]
            assert False, (
                set_name, case_id, "foreign violation keys", foreign_keys,
                "allowed", allowed_keys, "samples", samples,
            )
