"""Tests for LOAD_EXT, MEM_MISMATCH and CSR_MISMATCH (docs/ARCHITECTURE.md 4.3,
"記憶體影子與 CSR 影子"). Unit level through check_safe + observe, then
parse_trace / signature integration."""

from pathlib import Path

import pytest

from rfb import isa
from rfb.isa import ScanContext
from rfb.parse_trace import parse_trace
from rfb.record import Deviation, FailureRecord, Instr, RegrInfo, SimInfo, TraceInfo
from rfb.signature import MIN_SYS_VIOL, _hard_key, _sys_viol, signature

A = 0x80001000


def _instr(index, pc, mnemonic, operands, reads=None, writes=None, pa=None, size=4):
    return Instr(
        index=index, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
        reads=reads or {}, writes=writes or {}, pa=pa, store=None, load=None,
    )


class _Run:
    """Drives check_safe + observe the way parse_trace does."""

    def __init__(self):
        self.ctx = ScanContext()
        self.n = 0

    def step(self, mnemonic, operands, reads=None, writes=None, pa=None,
             next_pc="fall", pc=None, size=4):
        pc = 0x1000 + 4 * self.n if pc is None else pc
        if next_pc == "fall":
            next_pc = pc + size
        i = _instr(self.n, pc, mnemonic, operands, reads, writes, pa, size)
        self.n += 1
        v, err = isa.check_safe(None, i, next_pc, self.ctx)
        assert err is None
        isa.observe(self.ctx, i, next_pc)
        return v

    def sw(self, val, pa=A, **kw):
        return self.step("sw", "x7,0(x5)", reads={"x7": val}, pa=pa, **kw)

    def lw(self, val, pa=A, **kw):
        return self.step("lw", "x6,0(x5)", writes={"x6": val}, pa=pa, **kw)

    def csr_w(self, csr, val, old=0, mn="csrrw"):
        return self.step(mn, f"x5,{csr},x6", reads={"x6": val}, writes={"x5": old})

    def csr_r(self, csr, val, mn="csrrs"):
        return self.step(mn, f"x7,{csr},x0", writes={"x7": val})


# --- LOAD_EXT ---------------------------------------------------------------


def _ext(mn, val, next_pc=0x1004):
    i = _instr(0, 0x1000, mn, "x6,0(x5)", writes={"x6": val})
    v, err = isa.check_safe(None, i, next_pc, ScanContext())
    assert err is None
    return v


def test_lb_zero_extended_negative_byte_flagged():
    v = _ext("lb", 0x80)
    assert v.kind == "LOAD_EXT" and v.mnemonic == "lb"
    assert v.detail == {"expected": 0xFFFFFF80, "actual": 0x80}


def test_lb_sign_extended_not_flagged():
    assert _ext("lb", 0xFFFFFF80) is None
    assert _ext("lb", 0x7F) is None


def test_lb_dirty_upper_bits_on_positive_byte_flagged():
    v = _ext("lb", 0x100)
    assert v.kind == "LOAD_EXT" and v.detail == {"expected": 0, "actual": 0x100}


def test_lbu_sign_extended_flagged():
    v = _ext("lbu", 0xFFFFFF80)
    assert v.kind == "LOAD_EXT" and v.detail == {"expected": 0x80, "actual": 0xFFFFFF80}
    assert _ext("lbu", 0x80) is None


def test_lh_zero_extended_negative_half_flagged():
    v = _ext("lh", 0x8000)
    assert v.kind == "LOAD_EXT" and v.detail == {"expected": 0xFFFF8000, "actual": 0x8000}
    assert _ext("lh", 0xFFFF8000) is None
    assert _ext("lh", 0x7FFF) is None


def test_lhu_sign_extended_flagged():
    v = _ext("lhu", 0xFFFF8000)
    assert v.kind == "LOAD_EXT" and v.detail == {"expected": 0x8000, "actual": 0xFFFF8000}
    assert _ext("lhu", 0x8000) is None


def test_lw_never_load_ext():
    assert _ext("lw", 0xFFFFFFFF) is None
    assert _ext("lw", 0x80) is None


def test_load_ext_skips_rd_x0():
    i = _instr(0, 0x1000, "lb", "x0,0(x5)", writes={"x0": 0x80})
    v, err = isa.check_safe(None, i, 0x1004, ScanContext())
    assert err is None and v is None


def test_load_ext_skipped_when_next_pc_not_sequential():
    assert _ext("lb", 0x80, next_pc=0x2000) is None


def test_load_ext_checked_when_next_pc_unknown():
    v = _ext("lb", 0x80, next_pc=None)
    assert v is not None and v.kind == "LOAD_EXT"


# --- MEM_MISMATCH -----------------------------------------------------------


def test_sw_then_lw_different_value_flagged():
    r = _Run()
    assert r.sw(0x11223344) is None
    v = r.lw(0x11223345)
    assert v.kind == "MEM_MISMATCH" and v.mnemonic == "lw"
    assert v.detail == {"pa": A, "byte": 0, "expected": 0x44, "actual": 0x45, "nb": False}


def test_sw_then_lw_same_value_not_flagged():
    r = _Run()
    r.sw(0x11223344)
    assert r.lw(0x11223344) is None


def test_mismatch_reports_first_bad_byte_index():
    r = _Run()
    r.sw(0x11223344)
    v = r.lw(0x11993344)
    assert v.detail == {"pa": A, "byte": 2, "expected": 0x22, "actual": 0x99, "nb": False}


def test_load_seeds_shadow_then_second_load_flagged():
    r = _Run()
    assert r.lw(0xAABBCCDD) is None
    v = r.lw(0x1)
    assert v.kind == "MEM_MISMATCH"
    assert v.detail == {"pa": A, "byte": 0, "expected": 0xDD, "actual": 0x01, "nb": False}


def test_two_consecutive_wrong_reads_both_flagged():
    r = _Run()
    r.sw(0x11223344)
    assert r.lw(0x11223399).kind == "MEM_MISMATCH"
    v = r.lw(0x11223399)
    assert v is not None and v.kind == "MEM_MISMATCH"
    assert v.detail["expected"] == 0x44


def test_sb_at_offset_one_compares_only_known_byte():
    r = _Run()
    r.step("sb", "x7,1(x5)", reads={"x7": 0xAA}, pa=A + 1)
    assert r.lw(0x12ABAA34) is None
    v = r.lw(0x12ABAB34)
    assert v.detail == {"pa": A, "byte": 1, "expected": 0xAA, "actual": 0xAB, "nb": False}


def test_sh_writes_two_bytes_only():
    r = _Run()
    r.step("sh", "x7,0(x5)", reads={"x7": 0x1234}, pa=A)
    assert r.lw(0xEEEE1234) is None
    assert r.lw(0xEEEE1235).detail["byte"] == 0


def test_lh_compares_two_bytes():
    r = _Run()
    r.sw(0x11223344)
    assert r.step("lh", "x6,0(x5)", writes={"x6": 0x3344}, pa=A) is None
    v = r.step("lh", "x6,0(x5)", writes={"x6": 0x3744}, pa=A)
    assert v.kind == "MEM_MISMATCH" and v.detail["byte"] == 1


def test_unaligned_store_spans_two_words():
    r = _Run()
    r.sw(0x11223344, pa=A + 2)
    assert r.lw(0xFFFF1122, pa=A + 4) is None
    v = r.lw(0xFFFF1123, pa=A + 4)
    assert v.detail == {"pa": A + 4, "byte": 0, "expected": 0x22, "actual": 0x23, "nb": False}
    assert r.lw(0xFFFFFFFF, pa=A + 8) is None  # untouched word


def test_store_followed_by_flow_break_discards_shadow():
    r = _Run()
    r.sw(0x11223344)
    r.sw(0x55667788, next_pc=0x9000)  # trapped, store may not have happened
    assert r.lw(0x99999999) is None


def test_load_followed_by_flow_break_does_not_seed():
    r = _Run()
    r.lw(0x1, next_pc=0x9000)
    assert r.lw(0x2) is None


def test_mem_mismatch_skipped_when_load_followed_by_flow_break():
    r = _Run()
    r.sw(0x11223344)
    assert r.lw(0x99999999, next_pc=0x9000) is None


def test_sw_x0_writes_zero():
    r = _Run()
    r.sw(0x11223344)
    r.step("sw", "x0,0(x5)", pa=A)
    v = r.lw(0x11223344)
    assert v.kind == "MEM_MISMATCH" and v.detail["expected"] == 0
    r2 = _Run()
    r2.sw(0x11223344)
    r2.step("sw", "x0,0(x5)", pa=A)
    assert r2.lw(0) is None


def test_store_with_unknown_value_discards_shadow():
    r = _Run()
    r.sw(0x11223344)
    r.step("sw", "x7,0(x5)", reads={}, pa=A)
    assert r.lw(0x1) is None


def test_lw_rd_x0_skipped():
    r = _Run()
    r.sw(0x11223344)
    assert r.step("lw", "x0,0(x5)", writes={"x0": 0x1}, pa=A) is None


def _write_lines(tmp_path, lines, name):
    p = tmp_path / name
    p.write_text("Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
                 + "".join(lines))
    return p


_SW = "1\t1\t80000000\t0072a023\tsw\tx7,0(x5)\tx7:0x11223344 PA:0x80001000\n"
_LW = "3\t3\t80000004\t0002a303\tlw\tx6,0(x5)\tPA:0x80001000 x6=0x11223345\n"
_END = "4\t4\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"


def test_parse_trace_mem_mismatch_and_first_violation(tmp_path):
    lines = [_SW, "2\t2\t80000004\t0002a303\tlw\tx6,0(x5)\tPA:0x80001000 x6=0x11223345\n", _END]
    info = parse_trace(str(_write_lines(tmp_path, lines, "mem.log")))
    assert info.violations == {"MEM_MISMATCH:lw": 1}
    assert info.first_violation is not None
    assert info.first_violation.kind == "MEM_MISMATCH"


def test_malformed_line_clears_mem_shadow(tmp_path):
    lines = [_SW, "2\t2\t80000004\n", _LW, _END]
    info = parse_trace(str(_write_lines(tmp_path, lines, "mem_bad.log")))
    assert len(info.parse_errors) == 1
    assert info.violations == {}


def test_malformed_line_clears_csr_shadow(tmp_path):
    w = "1\t1\t80000000\t340312f3\tcsrrw\tx5,mscratch,x6\tx6:0x0000000f x5=0x00000000\n"
    r = "3\t3\t80000004\t340023f3\tcsrrs\tx7,mscratch,x0\tx7=0x0000001f\n"
    ok = parse_trace(str(_write_lines(tmp_path, [w, "2\t2\t80000004\n", r, _END],
                                      "csr_bad.log")))
    assert ok.violations == {}
    ctl = parse_trace(str(_write_lines(
        tmp_path, [w, r.replace("3\t3\t80000004", "2\t2\t80000004"),
                   _END.replace("4\t4", "3\t3")], "csr_ctl.log")))
    assert ctl.violations == {"CSR_MISMATCH:csrrs": 1}


def test_parse_trace_load_ext_is_first_violation(tmp_path):
    end = "2\t2\t80000004\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"
    lines = ["1\t1\t80000000\t0002a303\tlb\tx6,0(x5)\tPA:0x80001000 x6=0x00000080\n", end]
    info = parse_trace(str(_write_lines(tmp_path, lines, "ext.log")))
    assert info.violations == {"LOAD_EXT:lb": 1}
    assert info.first_violation.kind == "LOAD_EXT"


def test_parse_trace_csr_mismatch_is_first_violation(tmp_path):
    w = "1\t1\t80000000\t340312f3\tcsrrw\tx5,mscratch,x6\tx6:0x0000000f x5=0x00000000\n"
    r = "2\t2\t80000004\t340023f3\tcsrrs\tx7,mscratch,x0\tx7=0x0000001f\n"
    info = parse_trace(str(_write_lines(tmp_path, [w, r, _END.replace("4\t4", "3\t3")], "csr.log")))
    assert info.violations == {"CSR_MISMATCH:csrrs": 1}
    assert info.first_violation.kind == "CSR_MISMATCH"
    assert info.first_violation.detail["pol"] == "+"


def test_parse_trace_passes_next_pc_to_observe(tmp_path):
    # mepc written, then a trap-like pc jump after an addi: mepc is forgotten,
    # so the differing read-back later is not reported.
    w = "1\t1\t80000000\t341312f3\tcsrrw\tx5,mepc,x6\tx6:0x80000010 x5=0x00000000\n"
    a = "2\t2\t80000004\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"
    r = "3\t3\t80000100\t341023f3\tcsrrs\tx7,mepc,x0\tx7=0x80000040\n"
    end = "4\t4\t80000104\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"
    info = parse_trace(str(_write_lines(tmp_path, [w, a, r, end], "trap.log")))
    assert info.violations == {}


# --- CSR_MISMATCH -----------------------------------------------------------


def test_mscratch_write_then_read_different_flagged_with_pol():
    for actual, pol in ((0x1F, "+"), (0x07, "-"), (0x17, "~")):
        r = _Run()
        assert r.csr_w("mscratch", 0x0F) is None
        v = r.csr_r("mscratch", actual)
        assert v.kind == "CSR_MISMATCH" and v.mnemonic == "csrrs"
        assert v.detail == {"csr": "mscratch", "expected": 0x0F, "actual": actual, "pol": pol, "op": None}


def test_mscratch_same_value_not_flagged():
    r = _Run()
    r.csr_w("mscratch", 0x0F)
    assert r.csr_r("mscratch", 0x0F) is None


def test_mtvec_legalised_readback_not_flagged():
    r = _Run()
    r.csr_w("mtvec", 0xA5A5A5A5)
    assert r.csr_r("mtvec", 0xA5A5A501) is None
    assert r.csr_r("mtvec", 0xA5A5A5A5).kind == "CSR_MISMATCH"


def test_mie_mask():
    r = _Run()
    r.csr_w("mie", 0xFFFFFFFF)
    assert r.csr_r("mie", 0x7FFF0888) is None
    v = r.csr_r("mie", 0xFFFFFFFF)
    assert v.detail["expected"] == 0x7FFF0888 and v.detail["pol"] == "+"


def test_mepc_low_bit_masked():
    r = _Run()
    r.csr_w("mepc", 0x80000003)
    assert r.csr_r("mepc", 0x80000002) is None
    assert r.csr_r("mepc", 0x80000003).kind == "CSR_MISMATCH"


def test_mcause_mask():
    r = _Run()
    r.csr_w("mcause", 0xFFFFFFFF)
    assert r.csr_r("mcause", 0x8000001F) is None
    assert r.csr_r("mcause", 0xFFFFFFFF).kind == "CSR_MISMATCH"


def test_misa_is_read_only():
    r = _Run()
    r.csr_w("misa", 0x1234, old=0x40101104)  # write ignored, old read value learned
    assert r.csr_r("misa", 0x40101104) is None
    v = r.csr_r("misa", 0x1234)
    assert v.kind == "CSR_MISMATCH"
    assert v.detail["expected"] == 0x40101104


def test_mepc_forgotten_after_flow_break():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("addi", "x1,x1,1", next_pc=0x9000)  # trap / interrupt between
    assert r.csr_r("mepc", 0x80000040) is None


def test_mepc_forgotten_after_xret():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("mret", "", next_pc=0x80000010)
    assert r.csr_r("mepc", 0x80000040) is None


def test_trap_csrs_forgotten_but_mscratch_kept_after_flow_break():
    r = _Run()
    r.csr_w("mscratch", 0x55)
    r.csr_w("mcause", 0x2)
    r.csr_w("mtval", 0x3)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    assert r.csr_r("mcause", 0x99) is None
    assert r.csr_r("mtval", 0x99) is None
    assert r.csr_r("mscratch", 0x56).kind == "CSR_MISMATCH"


def test_taken_branch_in_between_keeps_mepc():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    # beq x1,x2,1008 at pc 0x1000: taken to its own target -> not a trap
    r.step("beq", "x1,x2,1008", reads={"x1": 1, "x2": 1}, pc=0x1000, next_pc=0x1008)
    v = r.csr_r("mepc", 0x80000040)
    assert v is not None and v.kind == "CSR_MISMATCH"


def test_branch_to_wrong_target_counts_as_break():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("beq", "x1,x2,1008", reads={"x1": 1, "x2": 1}, pc=0x1000, next_pc=0x9000)
    assert r.csr_r("mepc", 0x80000040) is None


def test_csrrc_clears_bits():
    r = _Run()
    r.csr_w("mscratch", 0xFF)
    r.step("csrrc", "x5,mscratch,x6", reads={"x6": 0x0F}, writes={"x5": 0xFF})
    assert r.csr_r("mscratch", 0xF0) is None
    r2 = _Run()
    r2.csr_w("mscratch", 0xFF)
    r2.step("csrrc", "x5,mscratch,x6", reads={"x6": 0x0F}, writes={"x5": 0xFF})
    assert r2.csr_r("mscratch", 0xFF).detail["pol"] == "+"


def test_csrrs_sets_bits():
    r = _Run()
    r.csr_w("mscratch", 0xF0)
    r.step("csrrs", "x5,mscratch,x6", reads={"x6": 0x0F}, writes={"x5": 0xF0})
    assert r.csr_r("mscratch", 0xFF) is None
    assert r.csr_r("mscratch", 0xF0).detail["pol"] == "-"


def test_csrrwi_immediate_operand():
    r = _Run()
    r.step("csrrwi", "x5,mscratch,5", writes={"x5": 0})
    assert r.csr_r("mscratch", 5) is None
    assert r.csr_r("mscratch", 6).kind == "CSR_MISMATCH"


def test_csrrw_with_x0_source_writes_zero():
    r = _Run()
    r.csr_w("mscratch", 0xFF)
    # rd = old value; source x0 -> writes 0 (not a pure read)
    r.step("csrrw", "x5,mscratch,x0", writes={"x5": 0xFF})
    assert r.csr_r("mscratch", 0) is None
    v = r.csr_r("mscratch", 0xFF)
    assert v.detail["expected"] == 0


def test_csrrs_x0_is_pure_read_keeps_shadow():
    r = _Run()
    r.csr_w("mscratch", 0xFF)
    assert r.csr_r("mscratch", 0xFF) is None
    assert r.csr_r("mscratch", 0xFF) is None
    assert r.csr_r("mscratch", 0xFE).kind == "CSR_MISMATCH"


def test_pure_read_learns_unknown_csr():
    r = _Run()
    assert r.csr_r("mscratch", 0x42) is None
    assert r.csr_r("mscratch", 0x43).detail["expected"] == 0x42


def test_x_operand_invalidates_shadow():
    r = _Run()
    r.csr_w("mscratch", 0x0F)
    r.step("csrrw", "x5,mscratch,x6", reads={}, writes={"x5": 0x0F})  # x6 unknown
    assert r.csr_r("mscratch", 0x99) is None


def test_csr_rd_x0_never_compared():
    r = _Run()
    r.csr_w("mscratch", 0x55)
    v = r.step("csrrs", "x0,mscratch,x0", writes={"x0": 0x99})
    assert v is None


def test_csrrw_rd_x0_still_updates_shadow():
    r = _Run()
    r.step("csrrw", "x0,mscratch,x6", reads={"x6": 0x21})
    assert r.csr_r("mscratch", 0x21) is None
    assert r.csr_r("mscratch", 0x22).kind == "CSR_MISMATCH"


def test_mstatus_is_checked_as_mstatus_mismatch_not_csr_mismatch():
    r = _Run()
    r.csr_w("mstatus", 0x1888)
    v = r.csr_r("mstatus", 0x0)
    assert v is not None
    assert v.kind == "MSTATUS_MISMATCH" and v.kind != "CSR_MISMATCH"
    assert v.detail["field"] == "MIE+MPIE+MPP"
    r2 = _Run()
    r2.csr_w("mstatus", 0x1888)
    assert r2.csr_r("mstatus", 0x1888) is None


def test_csr_mismatch_skipped_when_followed_by_flow_break():
    r = _Run()
    r.csr_w("mscratch", 0x0F)
    assert r.step("csrrs", "x7,mscratch,x0", writes={"x7": 0x1F}, next_pc=0x9000) is None


# --- viol_group -------------------------------------------------------------


def test_viol_group_load_ext():
    for mn in ("lb", "lh"):
        assert isa.viol_group("LOAD_EXT", mn) == "LD_SEXT"
    for mn in ("lbu", "lhu"):
        assert isa.viol_group("LOAD_EXT", mn) == "LD_ZEXT"


def test_viol_group_mem_mismatch():
    assert isa.viol_group("MEM_MISMATCH", "lw") == "MEMREAD"
    assert isa.viol_group("MEM_MISMATCH", "lb", {"pa": 1}) == "MEMREAD"


def test_viol_group_csr_by_polarity():
    for pol in ("+", "-", "~"):
        assert isa.viol_group("CSR_MISMATCH", "csrrs", {"pol": pol}) == "CSR" + pol
    assert isa.viol_group("CSR_MISMATCH", "csrrs", {}) == "CSR~"
    assert isa.viol_group("CSR_MISMATCH", "csrrs", None) == "CSR~"
    assert isa.viol_group("CSR_MISMATCH", "csrrs") == "CSR~"


def test_csr_pol_values():
    assert isa._csr_pol(0x0F, 0x1F) == "+"
    assert isa._csr_pol(0x0F, 0x07) == "-"
    assert isa._csr_pol(0x0F, 0x17) == "~"


# --- signature --------------------------------------------------------------


def _deviation(kind, mnemonic, detail=None):
    return Deviation(kind=kind, mnemonic=mnemonic, mclass=isa.mclass(mnemonic),
                     detail=detail or {}, source="scan", confidence=0.7)


def _record(deviation=None, violations=None):
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    trace = TraceInfo.empty()
    trace.violations = dict(violations or {})
    return FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                         trace=trace, deviation=deviation, parse_errors=[])


def test_new_kinds_scan_counts_not_in_sys_viol():
    for key in ("LOAD_EXT:lb", "MEM_MISMATCH:lw", "CSR_MISMATCH:csrrs"):
        rec = _record(violations={key: MIN_SYS_VIOL + 3})
        assert _sys_viol(rec) == (frozenset(), frozenset())
        sig = signature(rec)
        assert sig.soft["sys_viol_class"] == frozenset()
        assert sig.soft["sys_viol_keys"] == frozenset()


def test_new_kinds_deviation_enters_sys_viol_and_hard_key():
    rec = _record(deviation=_deviation("LOAD_EXT", "lbu"), violations={"LOAD_EXT:lbu": 5})
    assert _hard_key(rec) == ("MISMATCH", "LOAD_EXT", "LD_ZEXT")
    assert _sys_viol(rec) == (frozenset({("LOAD_EXT", "LD_ZEXT")}),
                              frozenset({("LOAD_EXT", "lbu")}))
    rec = _record(deviation=_deviation("MEM_MISMATCH", "lw"))
    assert _hard_key(rec) == ("MISMATCH", "MEM_MISMATCH", "MEMREAD")


def test_csr_hard_key_splits_by_polarity():
    keys = {
        pol: _hard_key(_record(deviation=_deviation("CSR_MISMATCH", "csrrs", {"pol": pol})))
        for pol in ("+", "-", "~")
    }
    assert keys["+"] == ("MISMATCH", "CSR_MISMATCH", "CSR+")
    assert keys["-"] == ("MISMATCH", "CSR_MISMATCH", "CSR-")
    assert keys["~"] == ("MISMATCH", "CSR_MISMATCH", "CSR~")
    cls, _ = _sys_viol(_record(deviation=_deviation("CSR_MISMATCH", "csrrs", {"pol": "-"})))
    assert cls == frozenset({("CSR_MISMATCH", "CSR-")})


# --- missing write values (X tokens): no exception, no verdict, no seeding ---


def _step_safe(mn, operands, ctx, reads=None, writes=None, pa=None):
    i = _instr(0, 0x1000, mn, operands, reads, writes, pa)
    return isa.check_safe(None, i, 0x1004, ctx)


def test_csr_read_without_rd_value_is_none():  # M-A
    r = _Run()
    r.csr_w("mscratch", 0x1234)
    assert _step_safe("csrrs", "x7,mscratch,x0", r.ctx, writes={}) == (None, None)


def test_lw_without_rd_value_is_none():  # M-B
    r = _Run()
    r.sw(0x11223344)
    assert _step_safe("lw", "x6,0(x5)", r.ctx, writes={}, pa=A) == (None, None)


def test_lb_without_rd_value_is_none():  # M-C
    r = _Run()
    r.sw(0x11223344)
    assert _step_safe("lb", "x6,0(x5)", r.ctx, writes={}, pa=A) == (None, None)


def test_observe_lw_without_rd_value_does_not_seed():  # M-D
    r = _Run()
    assert r.step("lw", "x6,0(x5)", writes={}, pa=A) is None
    assert r.lw(0x99999999) is None
    assert r.lw(0x11111111) is not None  # the second (real) read did seed


def test_observe_csr_rd_x0_does_not_use_written_zero():  # M-E
    r = _Run()
    r.csr_w("mscratch", 0xF0)
    r.step("csrrs", "x0,mscratch,x6", reads={"x6": 0x0F}, writes={"x0": 0})
    assert r.csr_r("mscratch", 0xFF) is None
    v = r.csr_r("mscratch", 0xF0)
    assert v is not None and v.kind == "CSR_MISMATCH"


def test_observe_lw_rd_x0_does_not_seed():  # M-F
    r = _Run()
    r.step("lw", "x0,0(x5)", writes={"x0": 0}, pa=A)
    assert r.lw(0x99) is None


@pytest.mark.parametrize("mn,ops,reg,size", [
    ("sb", "x7,0(x5)", "x7", 4),
    ("sh", "x7,0(x5)", "x7", 4),
    ("sw", "x7,0(x5)", "x7", 4),
    ("c.sw", "x9,0(x8)", "x9", 2),
    ("c.swsp", "x9,0(x2)", "x9", 2),
])
def test_store_with_unknown_pa_clears_memory_shadow(mn, ops, reg, size):  # F7
    r = _Run()
    r.sw(0x11223344)
    r.step(mn, ops, reads={reg: 0x55}, pa=None, size=size)
    assert r.lw(0x99999999) is None


def test_load_with_unknown_pa_keeps_memory_shadow():  # F7 reverse
    r = _Run()
    r.sw(0x11223344)
    assert r.step("lw", "x6,0(x5)", writes={"x6": 0x1}, pa=None) is None
    v = r.lw(0x11223345)
    assert v is not None and v.kind == "MEM_MISMATCH"
    assert v.detail["expected"] == 0x44 and v.detail["actual"] == 0x45


# --- public traces: the new kinds stay at zero --------------------------------

_PUBLIC = Path(__file__).resolve().parents[1] / "testcase/2026_08_20_samples/problem"
_NEW_KINDS = ("LOAD_EXT", "MEM_MISMATCH", "CSR_MISMATCH", "REG_ADDR_RD", "REG_ADDR_RS")


def test_public_traces_have_no_new_kind_violations():
    traces = sorted(_PUBLIC.glob("benchmark_set_*/case_*/trace.log.gz"))
    assert len(traces) == 32
    for t in traces:
        info = parse_trace(str(t))
        bad = [k for k in info.violations if k.split(":", 1)[0] in _NEW_KINDS]
        assert not bad, (str(t), bad)
