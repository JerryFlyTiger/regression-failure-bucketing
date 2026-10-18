"""CSR／MEMREAD 細分 (2026-10-14, out/csr_mem_eval/spec.txt): detail["nb"] /
detail["st"] on MEM_MISMATCH, detail["op"] on CSR_MISMATCH, and the viol_group
tokens MEMREAD_ST / CSR@W|S|C built from them.

Each test name says which behaviour it pins; the mutation each one is meant to
kill is noted as "m<n>" (the m1-m19 list in the spec)."""

import pytest

from rfb import isa
from rfb.diverge import derive
from rfb.isa import ScanContext
from rfb.parse_trace import parse_trace
from rfb.record import FailureRecord, Instr, Mismatch, RegrInfo, SimInfo
from rfb.signature import _hard_key

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

    def sb(self, val, pa, **kw):
        return self.step("sb", "x7,0(x5)", reads={"x7": val}, pa=pa, **kw)

    def lw(self, val, pa=A, **kw):
        return self.step("lw", "x6,0(x5)", writes={"x6": val}, pa=pa, **kw)

    def lbu(self, val, pa=A, **kw):
        return self.step("lbu", "x6,0(x5)", writes={"x6": val}, pa=pa, **kw)

    def csr_sync(self, mn, csr, old, sv):
        """CSR write whose rd (x5) reports the old value: a synced read."""
        return self.step(mn, f"x5,{csr},x6", reads={"x6": sv}, writes={"x5": old})

    def csr_blind(self, mn, csr, sv):
        """CSR write with rd=x0: the old value is not observed."""
        return self.step(mn, f"x0,{csr},x6", reads={"x6": sv})

    def csr_r(self, csr, val, mn="csrrs"):
        return self.step(mn, f"x7,{csr},x0", writes={"x7": val})


# --- viol_group tokens --------------------------------------------------------


def test_viol_group_csr_op_gives_csr_at_op():  # m2
    for op in ("W", "S", "C"):
        assert isa.viol_group("CSR_MISMATCH", "csrrs", {"op": op, "pol": "+"}) == "CSR@" + op
    assert isa.viol_group("CSR_MISMATCH", "csrrs", {"op": "C"}) == "CSR@C"


def test_viol_group_csr_op_none_falls_back_to_polarity():  # m2
    assert isa.viol_group("CSR_MISMATCH", "csrrs", {"op": None, "pol": "-"}) == "CSR-"
    assert isa.viol_group("CSR_MISMATCH", "csrrs", {"op": None, "pol": "+"}) == "CSR+"
    assert isa.viol_group("CSR_MISMATCH", "csrrs", {"op": None}) == "CSR~"


def test_viol_group_mem_st_gives_memread_st():  # m1
    assert isa.viol_group("MEM_MISMATCH", "lw", {"st": True}) == "MEMREAD_ST"
    assert isa.viol_group("MEM_MISMATCH", "lbu", {"pa": 1, "nb": True, "st": True}) == "MEMREAD_ST"


def test_viol_group_mem_nb_alone_does_not_split():  # m1
    assert isa.viol_group("MEM_MISMATCH", "lw", {"nb": True}) == "MEMREAD"
    assert isa.viol_group("MEM_MISMATCH", "lw", {"nb": False}) == "MEMREAD"
    assert isa.viol_group("MEM_MISMATCH", "lw", {"st": False}) == "MEMREAD"
    assert isa.viol_group("MEM_MISMATCH", "lw", None) == "MEMREAD"
    assert isa.viol_group("MEM_MISMATCH", "lw", {}) == "MEMREAD"


# --- _csr_src_op ---------------------------------------------------------------


def test_csr_src_op_csrrc_executed_as_csrrs_is_C():  # m3 (R hypothesis via C->R also fits), m5
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0, 0x0F)
    r.csr_blind("csrrc", "mscratch", 0x03)  # shadow 0x0C
    assert r.ctx.csr["mscratch"] == 0x0C
    v = r.csr_r("mscratch", 0x0F)  # clear had no effect (as csrrs)
    assert v.kind == "CSR_MISMATCH"
    assert v.detail["op"] == "C"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "CSR@C"


def test_csr_src_op_csrrs_executed_as_csrrw_is_S():
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0x0F, 0x30)  # shadow 0x3F
    v = r.csr_r("mscratch", 0x30)  # only the operand: set behaved as write
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "CSR@S"


def test_csr_src_op_csrrw_executed_as_csrrs_is_W():
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0x00, 0x30)  # chain base 0, shadow 0x30
    r.csr_sync("csrrw", "mscratch", 0x30, 0x0F)  # shadow 0x0F, chain reset
    v = r.csr_r("mscratch", 0x3F)  # write behaved as set
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "W"


def test_csr_src_op_dropped_csrrs_write_is_S():  # m3
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0x1F, 0x30)  # shadow 0x3F
    v = r.csr_r("mscratch", 0x1F)  # write dropped (S->C would give 0x0F, S->W 0x30)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_all_writes_dropped_is_judged_by_R_hypothesis():  # m3
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0x00, 0x0F)
    r.csr_blind("csrrw", "mscratch", 0xF0)
    # only W->R (every write dropped, value stays at the synced base 0) fits
    assert isa._csr_src_op(r.ctx, "mscratch", 0x00) == "W"


def test_csr_src_op_stuck_bit_is_none_with_polarity_minus():
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0, 0x0F)
    v = r.csr_r("mscratch", 0x0E)  # bit 0 stuck at 0
    assert v.kind == "CSR_MISMATCH"
    assert v.detail["op"] is None and v.detail["pol"] == "-"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "CSR-"


def test_csr_src_op_two_blind_writes_chain():
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0x0F, 0x30)
    r.csr_blind("csrrs", "mscratch", 0x300)  # shadow 0x33F
    assert r.ctx.csr["mscratch"] == 0x33F
    assert len(r.ctx.csr_chain["mscratch"][1]) == 2
    v = r.csr_r("mscratch", 0x300)  # both sets behaved as writes: last operand
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_tie_takes_last_write_op():  # m4, m5
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0, 0x0F)
    r.csr_blind("csrrs", "mscratch", 0xF0)  # shadow 0xFF
    # actual 0xF0 fits both "csrrs as csrrw" (S) and "csrrw as csrrc" (W)
    assert isa._csr_src_op(r.ctx, "mscratch", 0xF0) == "S"
    v = r.csr_r("mscratch", 0xF0)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_blind_chain_is_reset_by_synced_write():  # m6
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0x00, 0x0F)
    # synced value (0x07) disagrees with the shadow: the chain restarts from it
    r.csr_sync("csrrs", "mscratch", 0x07, 0x30)
    assert r.ctx.csr_chain["mscratch"] == (0x07, [("S", 0x30)])
    v = r.csr_r("mscratch", 0x07)  # set dropped
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_chain_of_eight_still_judged():  # m7
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0, 1)
    for k in range(1, 8):
        r.csr_blind("csrrs", "mscratch", 1 << k)
    assert len(r.ctx.csr_chain["mscratch"][1]) == isa._CSR_CHAIN_MAX == 8
    assert r.ctx.csr["mscratch"] == 0xFF
    v = r.csr_r("mscratch", 0x80)  # all sets behaved as writes: last operand
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_chain_over_eight_dropped_is_none():  # m7
    r = _Run()
    r.csr_sync("csrrs", "mscratch", 0, 1)
    for k in range(1, 9):
        r.csr_blind("csrrs", "mscratch", 1 << k)
    assert "mscratch" not in r.ctx.csr_chain
    assert r.ctx.csr["mscratch"] == 0x1FF  # shadow itself is kept
    v = r.csr_r("mscratch", 0x100)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] is None
    assert isa._csr_src_op(r.ctx, "mscratch", 0x100) is None


def test_csr_src_op_flow_break_drops_chain():  # m8
    r = _Run()
    r.csr_sync("csrrw", "mepc", 0, 0x80000010)
    assert "mepc" in r.ctx.csr_chain
    r.step("addi", "x1,x0,1", next_pc=0x9000)  # trap-like discontinuity
    assert "mepc" not in r.ctx.csr and "mepc" not in r.ctx.csr_chain
    # a blind write after the break must start a fresh chain, not extend the old
    r.csr_blind("csrrw", "mepc", 0x80000020)
    assert r.ctx.csr_chain["mepc"] == (None, [("W", 0x80000020)])
    v = r.csr_r("mepc", 0x80000030)  # = 0x...10 | 0x...20 if both W were S
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] is None


def test_csr_src_op_unknown_write_operand_drops_chain():  # m8 (non-trap CSR)
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0x0F, 0x30)
    r.step("csrrw", "x0,mscratch,x6")  # rs1 value not in the trace
    assert "mscratch" not in r.ctx.csr and "mscratch" not in r.ctx.csr_chain
    r.csr_blind("csrrw", "mscratch", 0x50)
    assert r.ctx.csr_chain["mscratch"] == (None, [("W", 0x50)])
    # 0x0F would be "both writes dropped" if the old chain were extended
    v = r.csr_r("mscratch", 0x0F)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] is None


def test_csr_src_op_flow_break_keeps_non_trap_csr_chain():
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0, 0x0F)
    r.step("addi", "x1,x0,1", next_pc=0x9000)
    assert "mscratch" in r.ctx.csr_chain


def test_csr_src_op_pure_read_seeds_chain_for_first_write():  # m9
    r = _Run()
    assert r.csr_r("mscratch", 0x0F) is None  # unknown shadow: seeded by the read
    assert r.ctx.csr_chain["mscratch"] == (0x0F, [])
    r.csr_blind("csrrs", "mscratch", 0x30)  # shadow 0x3F
    assert r.ctx.csr_chain["mscratch"] == (0x0F, [("S", 0x30)])
    v = r.csr_r("mscratch", 0x0F)  # set dropped
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_src_op_misa_read_only_is_none():
    r = _Run()
    assert r.csr_r("misa", 0x40101104) is None
    v = r.csr_r("misa", 0x40001104)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] is None
    assert isa._csr_src_op(r.ctx, "misa", 0x40001104) is None


def test_csr_src_op_unknown_csr_or_empty_chain_is_none():
    r = _Run()
    assert isa._csr_src_op(r.ctx, "mscratch", 1) is None
    r.csr_r("mscratch", 0x0F)
    assert isa._csr_src_op(r.ctx, "mscratch", 1) is None  # chain empty


def test_csr_src_op_applies_mie_mask_at_each_step():  # m19
    r = _Run()
    r.csr_sync("csrrw", "mie", 0, 0xFFFFFFFF)  # legalised to 0x7FFF0888
    r.csr_blind("csrrc", "mie", 0x00000008)  # shadow 0x7FFF0880
    assert r.ctx.csr["mie"] == 0x7FFF0880
    v = r.csr_r("mie", 0x7FFF0888)  # clear had no effect
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "C"


def test_csr_src_op_applies_mtvec_legal_to_hypothesis():  # m19
    r = _Run()
    r.csr_sync("csrrs", "mtvec", 0x1001, 0x103)  # shadow legal(0x1103) = 0x1101
    assert r.ctx.csr["mtvec"] == 0x1101
    v = r.csr_r("mtvec", 0x101)  # set behaved as write: legal(0x103)
    assert v.kind == "CSR_MISMATCH" and v.detail["op"] == "S"


def test_csr_fold_replays_with_swap_and_legal():  # m19
    legal = isa._CSR_LEGAL["mie"]
    chain = [("W", 0xFFFFFFFF), ("C", 0x8)]
    assert isa._csr_fold(0, chain, legal, ("X", "X")) == 0x7FFF0880
    assert isa._csr_fold(0, chain, legal, ("C", "R")) == 0x7FFF0888
    assert isa._csr_fold(0, chain, legal, ("C", "W")) == 0x8 & 0x7FFF0888
    assert isa._csr_fold(None, [("S", 1)], legal, ("X", "X")) is None
    assert isa._csr_fold(None, [("S", 1)], legal, ("S", "W")) == 1 & 0x7FFF0888


# --- MEM_MISMATCH detail["nb"] ----------------------------------------------------


def test_nb_true_neighbour_lane_after_later_subword_store():  # m10, m13, m14
    r = _Run()
    assert r.lw(0x11223344) is None  # seeds shadow, mem_at = lw
    assert r.sb(0xAA, A + 1) is None  # sub-word store, lane 1
    v = r.lbu(0x45)  # lane 0 reads wrong
    assert v.kind == "MEM_MISMATCH"
    assert v.detail == {"pa": A, "byte": 0, "expected": 0x44, "actual": 0x45, "nb": True}


def test_nb_false_without_later_subword_store():
    r = _Run()
    r.sw(0x11223344)
    v = r.lw(0x11223345)
    assert v.kind == "MEM_MISMATCH" and v.detail["nb"] is False
    r = _Run()
    r.lw(0x11223344)
    v = r.lbu(0x45)
    assert v.detail["nb"] is False


def test_nb_false_when_store_covers_the_byte_even_with_stale_mem_at():  # m11
    # Non-monotone indices on purpose: the byte's shadow value (set by a load
    # with index 3) is older than the covering store (index 5, value unknown so
    # it left no shadow), exercising the range test alone.
    r = _Run()
    r.n = 0
    r.lw(0x11223344)  # seeds A..A+3 at index 0
    r.n = 5
    r.step("sb", "x7,0(x5)", reads={}, pa=A + 1)  # unknown value: pops A+1, word_sub index 5
    r.n = 3
    r.lbu(0xAB, pa=A + 1)  # re-seeds A+1 with mem_at 3
    r.n = 6
    v = r.lbu(0xAC, pa=A + 1)  # mismatch on the covered byte
    assert v.kind == "MEM_MISMATCH" and v.detail["byte"] == 0
    assert v.detail["nb"] is False


def test_nb_false_when_byte_set_by_a_load_after_the_store():  # m10, m13
    r = _Run()
    r.sb(0xAA, A + 1)  # word_sub index 0, only lane 1 known
    r.lw(0x1122AA44)  # seeds lanes 0, 2, 3 at index 1 (lane 1 agrees)
    v = r.lbu(0x45)
    assert v.kind == "MEM_MISMATCH" and v.detail["expected"] == 0x44
    assert v.detail["nb"] is False


def test_nb_false_when_byte_rewritten_by_a_later_store():  # m14
    r = _Run()
    r.lw(0x11223344)
    r.sb(0xAA, A + 1)  # word_sub index 1
    r.sw(0x55667788)  # overwrites all lanes at index 2 (word_sub stays)
    v = r.lw(0x55667789)
    assert v.kind == "MEM_MISMATCH" and v.detail["expected"] == 0x88
    assert v.detail["nb"] is False


def test_sw_does_not_register_word_sub():  # m12
    r = _Run()
    r.lw(0x11223344)  # A..A+3 at index 0
    r.sw(0x55667788, pa=A + 2)  # unaligned word store: lanes A+2..A+5
    assert r.ctx.word_sub == {}
    v = r.lbu(0x45)  # lane 0, never rewritten
    assert v.kind == "MEM_MISMATCH" and v.detail["nb"] is False


def test_subword_stores_register_word_sub_aligned_word_key():
    r = _Run()
    r.sb(0x11, A + 1)
    r.step("sh", "x7,0(x5)", reads={"x7": 0x2222}, pa=A + 6)
    assert r.ctx.word_sub == {A: (0, A + 1, A + 2), A + 4: (1, A + 6, A + 8)}


def test_nb_false_after_store_with_unknown_pa():
    r = _Run()
    r.sb(0xAA, A + 1)
    r.step("sw", "x7,0(x5)", reads={"x7": 1}, pa=None)  # clears everything
    assert r.ctx.mem == {} and r.ctx.mem_at == {} and r.ctx.word_sub == {}
    r.lw(0x11223344)
    v = r.lbu(0x45)
    assert v.kind == "MEM_MISMATCH" and v.detail["nb"] is False


def test_unknown_pa_store_clears_word_sub_even_for_stale_indices():  # m18
    # Non-monotone indices on purpose (the clear is otherwise masked because
    # later seeds always carry a newer index).
    r = _Run()
    r.n = 10
    r.sb(0xAA, A + 1)  # word_sub index 10
    r.step("sw", "x7,0(x5)", reads={"x7": 1}, pa=None)
    r.n = 2
    r.lw(0x11223344)  # re-seeds with mem_at 2 (< 10)
    r.n = 3
    v = r.lbu(0x45)
    assert v.kind == "MEM_MISMATCH" and v.detail["nb"] is False


def test_load_records_mem_at_only_for_new_bytes():  # m13
    r = _Run()
    r.sb(0xAA, A + 1)  # index 0
    r.lw(0x1122AA44)  # index 1
    assert r.ctx.mem_at == {A: 1, A + 1: 0, A + 2: 1, A + 3: 1}


# --- parse_trace: trace-level detail["st"] ------------------------------------------

_HDR = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
B = 0x80001008


def _tl(i, mn, ops, regs):
    return f"{i}\t{i}\t{0x80000000 + 4 * i:x}\t0002a303\t{mn}\t{ops}\t{regs}\n"


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(_HDR + "".join(_tl(i, *row) for i, row in enumerate(rows)))
    return str(p)


_SW_A = ("sw", "x7,0(x5)", "x7:0x11223344 PA:0x80001000")
_SB_A1 = ("sb", "x8,1(x5)", "x8:0x000000aa PA:0x80001001")
_SW_B = ("sw", "x9,0(x5)", "x9:0x55667788 PA:0x80001008")
_END = ("addi", "x0,x0,0", "x0=0x00000000")


def _lw_bad_a():  # neighbour lane 0 reads wrong: nb True
    return ("lw", "x6,0(x5)", "PA:0x80001000 x6=0x1122AA45")


def _lw_bad_b():  # no sub-word store near it: nb False
    return ("lw", "x6,0(x5)", "PA:0x80001008 x6=0x55667789")


def _regr(anchor_pc):
    m = Mismatch(n=1, ibex_idx=0, ibex_pc=anchor_pc, ibex_text="lw x6,0(x5)", ibex_rd=None,
                 ibex_val=None, spike_idx=0, spike_pc=0, spike_text="", spike_rd=None,
                 spike_val=None)
    return RegrInfo(kind="FAILED", test_name=None, mismatches=[m], matched=0,
                    mismatch_count=1, raw=[])


def _record(info, regr=None):
    regr = regr or RegrInfo.empty()
    rec = FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                        trace=info, deviation=None, parse_errors=[])
    rec.deviation = derive(rec)
    return rec


def _window_mem_violations(info):
    return [v for vl in info.anchor_window_violations for v in vl
            if v is not None and v.kind == "MEM_MISMATCH"]


def test_parse_trace_all_nb_marks_first_violation_and_samples_with_st(tmp_path):  # m15, m16
    rows = [_SW_A, _SB_A1] + [_lw_bad_a()] * 3 + [_END]
    info = parse_trace(_write(tmp_path, "st.log", rows))
    assert info.violations == {"MEM_MISMATCH:lw": 3}
    assert info.first_violation.detail["nb"] is True
    assert info.first_violation.detail["st"] is True
    assert len(info.violation_samples) == 3
    assert all(v.detail.get("st") is True for v in info.violation_samples)
    rec = _record(info)
    assert rec.deviation.source == "scan"
    assert _hard_key(rec) == ("MISMATCH", "MEM_MISMATCH", "MEMREAD_ST")


def test_parse_trace_all_nb_anchor_window_deviation_carries_st(tmp_path):  # m16
    # 4 violations: the 4th is past the per-key sample cap, reachable only
    # through the anchor window.
    rows = [_SW_A, _SB_A1] + [_lw_bad_a()] * 4 + [_END]
    info = parse_trace(_write(tmp_path, "st_anchor.log", rows), _regr(0x80000000 + 4 * 5))
    assert info.violations == {"MEM_MISMATCH:lw": 4}
    assert len(info.violation_samples) == 3
    win = _window_mem_violations(info)
    assert win and all(v.detail.get("st") is True for v in win)
    last = info.anchor_window_violations[0][info.anchor_window_anchor_pos[0]]
    assert last is not None and all(last is not s for s in info.violation_samples)
    assert info.first_violation.detail["st"] is True
    rec = _record(info, _regr(0x80000000 + 4 * 5))
    assert rec.deviation.source == "anchor"
    assert rec.deviation.detail["st"] is True
    assert _hard_key(rec) == ("MISMATCH", "MEM_MISMATCH", "MEMREAD_ST")


def test_parse_trace_one_non_nb_violation_removes_st_everywhere(tmp_path):  # m15
    rows = [_SW_A, _SB_A1, _SW_B, _lw_bad_a(), _lw_bad_a(), _lw_bad_a(), _lw_bad_b(), _END]
    info = parse_trace(_write(tmp_path, "nost.log", rows), _regr(0x80000000 + 4 * 6))
    assert info.violations == {"MEM_MISMATCH:lw": 4}
    assert info.first_violation.detail["nb"] is True
    assert "st" not in info.first_violation.detail
    assert all("st" not in v.detail for v in info.violation_samples)
    win = _window_mem_violations(info)
    assert win and all("st" not in v.detail for v in win)
    rec = _record(info, _regr(0x80000000 + 4 * 6))
    assert rec.deviation.source == "anchor"
    assert _hard_key(rec) == ("MISMATCH", "MEM_MISMATCH", "MEMREAD")


def test_parse_trace_all_non_nb_violations_have_no_st(tmp_path):
    rows = [_SW_B, _lw_bad_b(), _END]
    info = parse_trace(_write(tmp_path, "nonb.log", rows))
    assert info.first_violation.detail["nb"] is False
    assert "st" not in info.first_violation.detail
    assert _hard_key(_record(info)) == ("MISMATCH", "MEM_MISMATCH", "MEMREAD")


def test_parse_trace_no_mem_violation_leaves_csr_details_untouched(tmp_path):
    w = _tl(0, "csrrw", "x5,mscratch,x6", "x6:0x0000000f x5=0x00000000")
    r = _tl(1, "csrrs", "x7,mscratch,x0", "x7=0x0000000f")
    r = r.replace("x7,mscratch,x0\tx7=0x0000000f", "x7,mscratch,x0\tx7=0x0000001f")
    p = tmp_path / "csr.log"
    p.write_text(_HDR + w + r + _tl(2, *_END))
    info = parse_trace(str(p))
    assert info.violations == {"CSR_MISMATCH:csrrs": 1}
    assert info.first_violation.detail["op"] is None
    assert "st" not in info.first_violation.detail


def _write_rows(tmp_path, name, rows):
    """rows: (mn, ops, regs) tuples, or None for a malformed line.  Valid rows
    get consecutive pcs, so a malformed line never causes a flow break."""
    out, i = [], 0
    for row in rows:
        if row is None:
            out.append("9\t9\t80000000\n")
        else:
            out.append(_tl(i, *row))
            i += 1
    p = tmp_path / name
    p.write_text(_HDR + "".join(out))
    return str(p)


_W1 = ("csrrw", "x5,mscratch,x6", "x6:0x0000000f x5=0x00000000")
_W2 = ("csrrw", "x0,mscratch,x6", "x6:0x000000f0")
_RD = ("csrrs", "x7,mscratch,x0", "x7=0x000000ff")


def test_malformed_line_clears_csr_chain_in_parse_trace(tmp_path):  # m17
    # control: without the malformed line the chain [W 0x0f, W 0xf0] explains 0xff
    ctl = parse_trace(_write_rows(tmp_path, "ctl.log", [_W1, _W2, _RD, _END]))
    assert ctl.first_violation.detail["op"] == "W"
    info = parse_trace(_write_rows(tmp_path, "bad.log", [_W1, None, _W2, _RD, _END]))
    assert len(info.parse_errors) == 1
    assert info.violations == {"CSR_MISMATCH:csrrs": 1}
    assert info.first_violation.detail["op"] is None


def test_malformed_line_clears_new_context_fields(tmp_path, monkeypatch):  # m17
    seen = []
    real = isa.observe

    def spy(ctx, instr, next_pc=None):
        seen.append(ctx)
        return real(ctx, instr, next_pc)

    monkeypatch.setattr(isa, "observe", spy)
    rows = [_SW_A, _SB_A1, _W1]
    parse_trace(_write_rows(tmp_path, "ctl.log", rows + [_END]))
    ctl = seen[-1]
    assert ctl.mem_at and ctl.word_sub and ctl.csr_chain  # control: fields do fill
    seen.clear()
    info = parse_trace(_write_rows(tmp_path, "dirty.log", rows + [None, _END]))
    assert len(info.parse_errors) == 1
    ctx = seen[-1]
    assert ctx is not ctl
    assert ctx.mem == {} and ctx.csr == {}
    assert ctx.mem_at == {} and ctx.word_sub == {} and ctx.csr_chain == {}


# --- 第一輪冷讀補強 (F2, F3, F6) -----------------------------------------------


def test_csr_src_op_unique_fit_is_not_last_write_returns_W():  # F2
    """Only W fits although the last write is S: guards both the `len(xs) > 1`
    tie-break (must not run for a unique fit) and the full-chain candidate
    scan (must not look at chain[-1] alone)."""
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0x300, 0x0F)  # chain base 0x300, W 0x0F
    r.csr_blind("csrrs", "mscratch", 0xF0)  # S 0xF0
    # W executed as S: 0x300|0x0F|0xF0 = 0x3FF (S as W would be 0xF0, W as C 0x3F0)
    assert isa._csr_src_op(r.ctx, "mscratch", 0x3FF) == "W"


def test_csr_fold_illegal_base_all_writes_dropped_returns_raw_base():  # F3
    """Base is a DUT-read value that is not legal and every write is dropped by
    the R hypothesis: the fold must return the raw base, not legal(base)."""
    legal = isa._CSR_LEGAL["mtvec"]
    assert legal(0x12345679) != 0x12345679  # precondition: base really is illegal
    assert isa._csr_fold(0x12345679, [("S", 1)], legal, ("S", "R")) == 0x12345679


def test_csr_src_op_illegal_base_dropped_write_full_path():  # F3
    """Same as above through _Run: mtvec read back as the raw illegal base
    (the S write was dropped, i.e. R) is judged S, not unexplained."""
    r = _Run()
    r.csr_sync("csrrs", "mtvec", 0x12345679, 1)
    assert isa._csr_src_op(r.ctx, "mtvec", 0x12345679) == "S"


def test_nb_wraps_around_32bit_address_space():  # F6.1
    """Neighbour byte at 0xFFFFFFFF+1 wraps to address 0: a later sub-word
    store in word 0 makes the wrapped byte a neighbour lane (nb True)."""
    r = _Run()
    r.sw(0x44332211, pa=0xFFFFFFFE)  # bytes FE FF 00 01
    r.sb(0xAA, 0x1)  # word 0, covers byte 1 only; byte 0 is its neighbour
    v = r.lw(0x44332211 ^ 0x00FF0000, pa=0xFFFFFFFE)  # byte 2 (address 0) differs
    assert v.kind == "MEM_MISMATCH" and v.detail["byte"] == 2
    assert v.detail["nb"] is True


def test_nb_wrapped_byte_without_subword_store_is_false():  # F6.1
    """Control: the wrapped byte at address 0 sees no sub-word store in word 0."""
    r = _Run()
    r.lw(0x11223344, pa=0xFFFFFFFE)  # FFFFFFFE FFFFFFFF 0 1
    r.sb(0xAA, 0xFFFFFFFD)
    v = r.lbu(0x45, pa=0x0)
    assert v.detail["nb"] is False


def test_misaligned_sh_registers_word_sub():  # F6.2
    """An odd-address sh (A+3, unaligned) still registers word_sub for its
    first word, so a differing load in that word gets nb True."""
    r = _Run()
    r.lw(0x11223344, pa=A)
    r.step("sh", "x7,0(x5)", reads={"x7": 0x5566}, pa=A + 3)  # A+3, A+4
    v = r.lbu(0x45, pa=A)
    assert v.kind == "MEM_MISMATCH" and v.detail["nb"] is True


def test_misaligned_sh_at_aligned_half_registers_word_sub():  # F6.2
    """Control for the above: a half-aligned sh at A+2 behaves the same."""
    r = _Run()
    r.lw(0x11223344, pa=A)
    r.step("sh", "x7,0(x5)", reads={"x7": 0x5566}, pa=A + 2)
    v = r.lbu(0x45, pa=A)
    assert v.detail["nb"] is True


def test_parse_trace_kind_guard_csr_violation_gets_no_st(tmp_path):  # F6.3
    """Mixed trace (CSR violation first, then all-nb MEM violations): the st
    marker is only for MEM_MISMATCH, the CSR violation's detail must not get it."""
    csr_w = ("csrrw", "x5,mscratch,x6", "x6:0x0000000f x5=0x00000000")
    csr_r = ("csrrs", "x7,mscratch,x0", "x7=0x0000001f")
    rows = [csr_w, csr_r, _SW_A, _SB_A1, _lw_bad_a(), _END]
    info = parse_trace(_write(tmp_path, "mix.log", rows))
    assert info.first_violation.kind == "CSR_MISMATCH"
    assert "st" not in info.first_violation.detail
    assert all("st" not in v.detail for v in info.violation_samples if v.kind == "CSR_MISMATCH")
    assert any(v.kind == "MEM_MISMATCH" and v.detail.get("st") for v in info.violation_samples)


# --- 第二輪冷讀補強 (R2-1, R2-2, R2-3) ------------------------------------------


def test_csr_instr_followed_by_flow_break_drops_its_own_non_trap_chain():  # R2-1
    """A non-trap CSR write whose next PC is not the fallthrough drops that
    CSR's shadow and chain (`brk` in the drop condition, not the _CSR_TRAP loop);
    another non-trap CSR's shadow and chain survive."""
    r = _Run()
    r.csr_sync("csrrw", "mie", 0, 0x8)
    r.csr_blind("csrrs", "mie", 0x80)  # control: shadow 0x88, chain [W 0x8, S 0x80]
    r.csr_sync("csrrw", "mscratch", 0, 0x0F)
    r.step("csrrw", "x5,mscratch,x6", reads={"x6": 0x30}, writes={"x5": 0x0F}, next_pc=0x9000)
    assert "mscratch" not in r.ctx.csr and "mscratch" not in r.ctx.csr_chain
    assert r.ctx.csr["mie"] == 0x88
    assert r.ctx.csr_chain["mie"] == (0, [("W", 0x8), ("S", 0x80)])


def test_csr_src_op_tie_last_op_not_in_fit_set_is_none():  # R2-2
    """Several source ops fit but the last write's op is not among them: None,
    not the last op (the tie-break intersects, it does not replace)."""
    r = _Run()
    r.csr_sync("csrrw", "mscratch", 0, 0x1)
    r.csr_blind("csrrs", "mscratch", 0x2)
    r.csr_blind("csrrc", "mscratch", 0x4)  # shadow 0x3
    # actual 0x2 fits W (as C or R) and S (as W) but not the last write's op C
    assert isa._csr_src_op(r.ctx, "mscratch", 0x2) is None


def test_csr_chain_restart_after_overflow_uses_shadow_as_base():  # R2-3 (F8)
    """After the 8th blind write (9th write) drops a full chain, the next blind write
    starts a new chain whose base is the shadow's predicted value; another
    CSR's chain is not dropped with it."""
    r = _Run()
    r.csr_sync("csrrw", "mie", 0, 0x8)  # control: chain (0, [W 0x8])
    r.csr_sync("csrrs", "mscratch", 0, 1)
    for k in range(1, 9):
        r.csr_blind("csrrs", "mscratch", 1 << k)  # 9th write overflows: chain dropped
    assert "mscratch" not in r.ctx.csr_chain and r.ctx.csr["mscratch"] == 0x1FF
    assert r.ctx.csr_chain["mie"] == (0, [("W", 0x8)])
    r.csr_blind("csrrs", "mscratch", 1 << 9)
    assert r.ctx.csr_chain["mscratch"] == (0x1FF, [("S", 0x200)])


# --- 第四輪冷讀補強 (R4-3, R4-4) ------------------------------------------------


def test_csr_set_without_shadow_drops_only_its_own_entry():  # R4-3
    """csrrs with rd=x0 and no shadow cannot compute the new value: it leaves
    no shadow/chain for that CSR and does not touch another CSR's entry."""
    r = _Run()
    r.csr_sync("csrrw", "mie", 0, 0x8)  # control
    r.csr_blind("csrrs", "mscratch", 0x4)
    assert "mscratch" not in r.ctx.csr and "mscratch" not in r.ctx.csr_chain
    assert r.ctx.csr["mie"] == 0x8 and r.ctx.csr_chain["mie"] == (0, [("W", 0x8)])


def test_csr_imm_write_rd_x0_followed_by_flow_break_drops_its_chain():  # R4-4
    """`csrrwi x0,...` (old value not observed, immediate source) followed by a
    flow break is dropped like any other non-trap CSR write."""
    r = _Run()
    r.step("csrrwi", "x0,mscratch,5")
    assert r.ctx.csr["mscratch"] == 5  # fallthrough: kept
    r = _Run()
    r.step("csrrwi", "x0,mscratch,5", next_pc=0x9000)
    assert "mscratch" not in r.ctx.csr and "mscratch" not in r.ctx.csr_chain
