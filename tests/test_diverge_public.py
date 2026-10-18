"""rfb.diverge.derive() against the public fixture (ARCHITECTURE.md section 8, M2).

Every case in tests/fixtures/public_expected.json is parsed via
rfb.pipeline.parse_all() and re-derived with rfb.diverge.derive(); for
family == "MISMATCH" the resulting Deviation.kind/mnemonic must match the
fixture. Cases with expected_kind == null only need family agreement
(non-MISMATCH families only use trace.first_violation, section 4.4).
"""

import json
import os

from rfb import pipeline
from rfb.diverge import derive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEM_ROOT = os.path.join(REPO_ROOT, "testcase", "2026_08_20_samples", "problem")
FIXTURE = json.loads(
    open(os.path.join(REPO_ROOT, "tests/fixtures/public_expected.json")).read()
)


def _records_by_case(set_name):
    input_csv = os.path.join(PROBLEM_ROOT, set_name, "input.csv")
    records = pipeline.parse_all(input_csv)
    return {r.case_id: r for r in records}


def test_mismatch_case_deviations_match_fixture():
    for set_name, cases in FIXTURE.items():
        records = _records_by_case(set_name)
        for case_id_str, info in cases.items():
            case_id = int(case_id_str)
            record = records[case_id]
            assert record.family == info["family"], (set_name, case_id)
            if info["family"] != "MISMATCH":
                continue
            dev = derive(record)
            assert dev.kind == info["expected_kind"], (
                set_name, case_id, dev.kind, dev.mnemonic, info["expected_kind"]
            )
            assert dev.mnemonic in (info["expected_mnemonics"] or []), (
                set_name, case_id, dev.mnemonic, info["expected_mnemonics"]
            )


def test_set1_case4_first_violation_is_xret_fallthrough_dret():
    records = _records_by_case("benchmark_set_1")
    record = records[4]
    fv = record.trace.first_violation
    assert fv is not None
    assert f"{fv.kind}:{fv.mnemonic}" == "XRET_FALLTHROUGH:dret"
