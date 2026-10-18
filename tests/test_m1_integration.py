"""Integration test: run all three M1 parsers across the full public set and
build FailureRecord objects, per ARCHITECTURE section 5.1 family rules."""

import json
from pathlib import Path

from rfb.parse_regr import parse_regr
from rfb.parse_sim import parse_sim
from rfb.parse_trace import parse_trace
from rfb.pipeline import classify_family
from rfb.record import FailureRecord

ROOT = Path(__file__).resolve().parents[1]
SET1 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_1"
SET2 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_2"


def _all_case_dirs():
    return sorted(SET1.glob("case_*"), key=lambda p: int(p.name.split("_")[1])) + sorted(
        SET2.glob("case_*"), key=lambda p: int(p.name.split("_")[1])
    )


def test_all_cases_parse_without_errors_and_serialize():
    families = {"MISMATCH": 0, "FATAL": 0, "ASSERT": 0, "OTHER": 0}
    for case_id, case_dir in enumerate(_all_case_dirs(), start=1):
        regr_info = parse_regr(str(case_dir / "regr.log"))
        sim_info = parse_sim(str(case_dir / "sim.log.gz"))
        trace_info = parse_trace(str(case_dir / "trace.log.gz"), regr_info)

        assert trace_info.parse_errors == [], case_dir

        family = classify_family(regr_info, sim_info)
        families[family] += 1

        record = FailureRecord(
            case_id=case_id,
            family=family,
            regr=regr_info,
            sim=sim_info,
            trace=trace_info,
            deviation=None,
            parse_errors=[],
        )
        blob = record.to_json()
        # must round-trip through json
        parsed = json.loads(blob)
        assert parsed["family"] == family

    assert families == {"MISMATCH": 18, "FATAL": 11, "ASSERT": 3, "OTHER": 0}
