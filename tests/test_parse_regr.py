from pathlib import Path

from rfb.parse_regr import parse_regr

ROOT = Path(__file__).resolve().parents[1]
SET1 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_1"
SET2 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_2"

SET1_KINDS = {
    1: "TEST_FAILED",
    2: "MISMATCH",
    3: "TEST_FAILED",
    4: "TEST_FAILED",
    5: "TEST_FAILED",
    6: "MISMATCH",
    7: "TEST_FAILED",
}


def test_set1_kinds():
    for case, kind in SET1_KINDS.items():
        info = parse_regr(str(SET1 / f"case_{case}/regr.log"))
        assert info.kind == kind, f"case_{case}"


def test_mismatch_case2_fields():
    info = parse_regr(str(SET1 / "case_2/regr.log"))
    assert info.kind == "MISMATCH"
    assert len(info.mismatches) == 1
    m = info.mismatches[0]
    assert m.n == 1
    assert m.ibex_idx == 88
    assert m.ibex_pc == 0x8000F8E0
    assert m.ibex_text == "addi\tx5,x5,-124: t0:80022da0"
    assert m.ibex_rd == "t0"
    assert m.ibex_val == 0x80022DA0
    assert m.spike_idx == 88
    assert m.spike_pc == 0x800016DA
    assert m.spike_text == "addi    t0, t0, -64: t0:80022ddc"
    assert m.spike_rd == "t0"
    assert m.spike_val == 0x80022DDC
    assert info.matched == 76
    assert info.mismatch_count == 41737


def test_mismatch_case6_fields():
    info = parse_regr(str(SET1 / "case_6/regr.log"))
    m = info.mismatches[0]
    assert m.ibex_idx == 371
    assert m.ibex_pc == 0x800005AE
    assert m.ibex_rd == "t4"
    assert m.ibex_val == 0x00000001
    assert m.spike_pc == 0x80012080
    assert m.spike_rd == "t0"
    assert m.spike_val == 0x80022DA0
    assert info.matched == 358
    assert info.mismatch_count == 63165


def test_test_failed_case1():
    info = parse_regr(str(SET1 / "case_1/regr.log"))
    assert info.kind == "TEST_FAILED"
    assert info.test_name == "riscv_interrupt_csr_test"
    assert info.mismatches == []
    assert info.matched is None
    assert info.mismatch_count is None


def test_set2_case14_test_failed():
    info = parse_regr(str(SET2 / "case_14/regr.log"))
    assert info.kind == "TEST_FAILED"
    assert info.test_name == "riscv_multiple_interrupt_test"
