"""Path coverage for the LOAD_EXT / MEM_MISMATCH / CSR_MISMATCH shadows
(cold-read mutations M-G..M-Q): every store/load width, flow-break handling in
``_flow_break``, CSR legalisation and set/clear immediates, check ordering and
32-bit address wrap."""

import pytest

from rfb import isa
from rfb.isa import ScanContext
from rfb.record import Instr

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
             next_pc="fall", pc=None, size=4, allow_err=False):
        pc = 0x1000 + 4 * self.n if pc is None else pc
        if next_pc == "fall":
            next_pc = pc + size
        i = _instr(self.n, pc, mnemonic, operands, reads, writes, pa, size)
        self.n += 1
        v, err = isa.check_safe(None, i, next_pc, self.ctx)
        assert allow_err or err is None
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


# --- M-G: compressed stores, width 4 ----------------------------------------


@pytest.mark.parametrize("mn,ops,reg", [("c.sw", "x9,0(x8)", "x9"), ("c.swsp", "x9,0(x2)", "x9")])
def test_compressed_store_then_lw_mismatch_width4(mn, ops, reg):
    r = _Run()
    assert r.step(mn, ops, reads={reg: 0x11223344}, pa=A, size=2) is None
    # only the top byte differs: needs all four bytes stored
    v = r.lw(0x12223344)
    assert v.kind == "MEM_MISMATCH" and v.mnemonic == "lw"
    assert v.detail == {"pa": A, "byte": 3, "expected": 0x11, "actual": 0x12, "nb": False}
    assert r.lw(0x11223344) is None


# --- M-H: every load width ---------------------------------------------------

_LOADS = [
    ("lb", "x6,0(x5)", 1, 4),
    ("lbu", "x6,0(x5)", 1, 4),
    ("lh", "x6,0(x5)", 2, 4),
    ("lhu", "x6,0(x5)", 2, 4),
    ("lw", "x6,0(x5)", 4, 4),
    ("c.lw", "x9,0(x8)", 4, 2),
    ("c.lwsp", "x6,0(x2)", 4, 2),
]


def _stored_run():
    r = _Run()
    r.sw(0x11223344)
    return r


@pytest.mark.parametrize("mn,ops,width,size", _LOADS)
def test_load_mismatch_on_last_byte_of_width(mn, ops, width, size):
    r = _stored_run()
    rd = ops.split(",")[0]
    good = 0x11223344 & ((1 << (8 * width)) - 1)
    bad = good + (1 << (8 * (width - 1)))  # last byte +1, stays positive
    v = r.step(mn, ops, writes={rd: bad}, pa=A, size=size)
    exp = (0x11223344 >> (8 * (width - 1))) & 0xFF
    assert v.kind == "MEM_MISMATCH" and v.mnemonic == mn
    assert v.detail == {"pa": A, "byte": width - 1, "expected": exp, "actual": exp + 1, "nb": False}


@pytest.mark.parametrize("mn,ops,width,size", _LOADS)
def test_load_matching_value_not_flagged_beyond_width_ignored(mn, ops, width, size):
    r = _stored_run()
    rd = ops.split(",")[0]
    good = 0x11223344 & ((1 << (8 * width)) - 1)
    # bytes beyond the load width (zero here, 0x22/0x11 in memory) are not compared
    assert r.step(mn, ops, writes={rd: good}, pa=A, size=size) is None


# --- M-I / M-J / M-K: flow-break handling ------------------------------------

T = 0x2000


def _jumps():
    return [
        ("jal", "x1,2000", {}, {"x1": 0x1004}, 4),
        ("jalr", "x1,0(x5)", {"x5": T + 1}, {"x1": 0x1004}, 4),
        ("c.jr", "x5", {"x5": T + 1}, {}, 2),
        ("c.j", "2000", {}, {}, 2),
    ]


@pytest.mark.parametrize("mn,ops,reads,writes,size", _jumps())
def test_jump_to_target_keeps_csr_shadow(mn, ops, reads, writes, size):
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step(mn, ops, reads=reads, writes=writes, next_pc=T, size=size)
    v = r.csr_r("mepc", 0x80000040)
    assert v.kind == "CSR_MISMATCH"
    assert v.detail["csr"] == "mepc" and v.detail["expected"] == 0x80000010
    assert v.detail["actual"] == 0x80000040


@pytest.mark.parametrize("mn,ops,reads,writes,size", _jumps())
def test_jump_to_wrong_target_is_break(mn, ops, reads, writes, size):
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step(mn, ops, reads=reads, writes=writes, next_pc=T + 0x10, size=size)
    assert r.csr_r("mepc", 0x80000040) is None


def test_taken_branch_keeps_csr_shadow_wrong_target_breaks():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("beq", "x1,x2,2000", reads={"x1": 1, "x2": 1}, next_pc=T)
    assert r.csr_r("mepc", 0x80000040).kind == "CSR_MISMATCH"
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("beq", "x1,x2,2000", reads={"x1": 1, "x2": 1}, next_pc=T + 8)
    assert r.csr_r("mepc", 0x80000040) is None


# (mnemonic, operands, reads that make the condition true)
_TAKEN_BRANCHES = [
    ("bne", "x1,x2,2000", {"x1": 1, "x2": 2}),
    ("blt", "x1,x2,2000", {"x1": 0xFFFFFFFF, "x2": 1}),  # -1 < 1 signed
    ("bltu", "x1,x2,2000", {"x1": 1, "x2": 0xFFFFFFFF}),
    ("bge", "x1,x2,2000", {"x1": 1, "x2": 0xFFFFFFFF}),  # 1 >= -1 signed
    ("bgeu", "x1,x2,2000", {"x1": 0xFFFFFFFF, "x2": 1}),
    ("c.beqz", "x1,2000", {"x1": 0}),
    ("c.bnez", "x1,2000", {"x1": 5}),
]


@pytest.mark.parametrize("mn,ops,reads", _TAKEN_BRANCHES)
def test_taken_branch_variants_keep_csr_shadow(mn, ops, reads):
    size = 2 if mn.startswith("c.") else 4
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    assert r.step(mn, ops, reads=reads, next_pc=T, size=size) is None
    v = r.csr_r("mepc", 0x80000040)
    assert v is not None and v.kind == "CSR_MISMATCH"
    assert v.detail["expected"] == 0x80000010
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step(mn, ops, reads=reads, next_pc=T + 8, size=size)
    assert r.csr_r("mepc", 0x80000040) is None


# --- pc wrap: the fallthrough of an instruction at the top of the 32-bit space

_TOP = 0xFFFFFFFC


def test_pc_wrap_fallthrough_is_not_a_break_in_observe():
    # _flow_break: unmasked, 0x100000000 != 0 would count as a trap
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("addi", "x1,x1,0", pc=_TOP, next_pc=0)
    v = r.csr_r("mepc", 0x80000040)
    assert v is not None and v.kind == "CSR_MISMATCH"


def test_pc_wrap_fallthrough_csr_read_is_checked():
    # _lsu_csr_mismatch: unmasked, the wrap would look like a break and skip the check
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    v = r.step("csrrs", "x7,mepc,x0", writes={"x7": 0x80000040}, pc=_TOP, next_pc=0)
    assert v is not None and v.kind == "CSR_MISMATCH"


def test_pc_wrap_fallthrough_load_is_checked():
    r = _stored_run()
    v = r.step("lw", "x6,0(x5)", writes={"x6": 0x11223345}, pa=A, pc=_TOP, next_pc=0)
    assert v is not None and v.kind == "MEM_MISMATCH"


def test_pc_wrap_fallthrough_load_ext_is_checked():
    # check(): fallthrough must be masked, else next_pc=0 != 0x100000000 skips LOAD_EXT
    i = _instr(0, _TOP, "lb", "x6,0(x5)", writes={"x6": 0x80})
    v, err = isa.check_safe(None, i, 0, ScanContext())
    assert err is None
    assert v is not None and v.kind == "LOAD_EXT"
    assert v.detail == {"expected": 0xFFFFFF80, "actual": 0x80}


@pytest.mark.parametrize("ops", ["x1,x2,zz", "x1,x2", "x1"])
def test_branch_with_garbage_or_short_operands_discontinuity_is_break(ops):
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("beq", ops, reads={"x1": 1, "x2": 1}, next_pc=0x3000, allow_err=True)
    assert r.csr_r("mepc", 0x80000040) is None


def test_jal_with_garbage_target_discontinuity_is_break():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("jal", "x1,zz", writes={"x1": 0x1004}, next_pc=0x3000, allow_err=True)
    assert r.csr_r("mepc", 0x80000040) is None


def test_branch_garbage_operands_fallthrough_not_break():
    r = _Run()
    r.csr_w("mepc", 0x80000010)
    r.step("beq", "x1,x2,zz", reads={"x1": 1, "x2": 1}, allow_err=True)
    assert r.csr_r("mepc", 0x80000040).kind == "CSR_MISMATCH"


def test_observe_without_next_pc_is_not_a_break():
    ctx = ScanContext()
    st = _instr(0, 0x1000, "sw", "x7,0(x5)", reads={"x7": 0x11223344}, pa=A)
    isa.observe(ctx, st)  # next_pc defaults to None
    ld = _instr(1, 0x1004, "lw", "x6,0(x5)", writes={"x6": 0x11223345}, pa=A)
    v, err = isa.check_safe(None, ld, 0x1008, ctx)
    assert err is None and v.kind == "MEM_MISMATCH"
    assert v.detail == {"pa": A, "byte": 0, "expected": 0x44, "actual": 0x45, "nb": False}


def test_observe_load_without_next_pc_seeds_shadow():
    ctx = ScanContext()
    ld = _instr(0, 0x1000, "lw", "x6,0(x5)", writes={"x6": 0xAABBCCDD}, pa=A)
    isa.observe(ctx, ld)
    ld2 = _instr(1, 0x1004, "lw", "x6,0(x5)", writes={"x6": 0x1}, pa=A)
    v, err = isa.check_safe(None, ld2, None, ctx)
    assert err is None and v.kind == "MEM_MISMATCH" and v.detail["expected"] == 0xDD


def test_observe_csr_without_next_pc_keeps_shadow():
    ctx = ScanContext()
    w = _instr(0, 0x1000, "csrrw", "x5,mepc,x6", reads={"x6": 0x80000010}, writes={"x5": 0})
    isa.observe(ctx, w)
    assert ctx.csr["mepc"] == 0x80000010


# --- M-L: mtval --------------------------------------------------------------


def test_mtval_read_back_unmasked():
    r = _Run()
    r.csr_w("mtval", 0x3)
    assert r.csr_r("mtval", 0x3) is None
    v = r.csr_r("mtval", 0x4)
    assert v.kind == "CSR_MISMATCH"
    assert v.detail["csr"] == "mtval" and v.detail["expected"] == 0x3 and v.detail["actual"] == 0x4


# --- M-M: csrrsi / csrrci ----------------------------------------------------


def test_csrrsi_sets_bits():
    r = _Run()
    r.csr_w("mscratch", 0x10)
    assert r.step("csrrsi", "x5,mscratch,3", writes={"x5": 0x10}) is None
    assert r.csr_r("mscratch", 0x13) is None
    v = r.csr_r("mscratch", 0x10)
    assert v.kind == "CSR_MISMATCH" and v.detail["expected"] == 0x13 and v.detail["actual"] == 0x10


def test_csrrci_clears_bits():
    r = _Run()
    r.csr_w("mscratch", 0xF)
    assert r.step("csrrci", "x5,mscratch,5", writes={"x5": 0xF}) is None
    assert r.csr_r("mscratch", 0xA) is None
    v = r.csr_r("mscratch", 0xF)
    assert v.kind == "CSR_MISMATCH" and v.detail["expected"] == 0xA and v.detail["actual"] == 0xF


# --- M-N: set/clear with unknown old value -----------------------------------


@pytest.mark.parametrize("mn", ["csrrs", "csrrc"])
def test_set_clear_with_unknown_old_value_does_not_invent_shadow(mn):
    r = _Run()
    r.step(mn, "x0,mscratch,x6", reads={"x6": 0x5})
    assert "mscratch" not in r.ctx.csr
    assert r.csr_r("mscratch", 0x77) is None


# --- M-O: check order --------------------------------------------------------


def test_read_mismatch_wins_over_mem_mismatch():
    r = _Run()
    r.ctx.regs["x5"] = 0x100  # known value differs from what the load reads
    r.sw(0x11223344)
    ld = _instr(1, 0x1004, "lw", "x6,0(x5)", reads={"x5": A},
                writes={"x6": 0x11223345}, pa=A)
    # both checks fire on their own
    assert isa._read_mismatch(ld, 0x1008, r.ctx).kind == "READ_MISMATCH"
    assert isa._lsu_csr_mismatch(ld, 0x1008, r.ctx).kind == "MEM_MISMATCH"
    v, err = isa.check_safe(None, ld, 0x1008, r.ctx)
    assert err is None and v.kind == "READ_MISMATCH"
    assert v.detail["reg"] == "x5" and v.detail["expected"] == 0x100 and v.detail["actual"] == A


# --- M-Q: 32-bit address wrap -----------------------------------------------


def test_store_wraps_past_4g_then_lb_at_zero():
    r = _Run()
    r.sw(0x44332211, pa=0xFFFFFFFE)
    v = r.step("lb", "x6,0(x5)", writes={"x6": 0x34}, pa=0x0)
    assert v.kind == "MEM_MISMATCH" and v.mnemonic == "lb"
    assert v.detail == {"pa": 0, "byte": 0, "expected": 0x33, "actual": 0x34, "nb": False}
    assert r.step("lb", "x6,0(x5)", writes={"x6": 0x33}, pa=0x0) is None


def test_load_wraps_past_4g_seeds_shadow_at_zero():
    r = _Run()
    assert r.lw(0x44332211, pa=0xFFFFFFFE) is None
    v = r.step("lb", "x6,0(x5)", writes={"x6": 0x34}, pa=0x0)
    assert v.kind == "MEM_MISMATCH"
    assert v.detail == {"pa": 0, "byte": 0, "expected": 0x33, "actual": 0x34, "nb": False}


def test_lw_across_wrap_compares_wrapped_bytes():
    r = _Run()
    r.sw(0x44332211, pa=0xFFFFFFFE)
    v = r.lw(0x45332211, pa=0xFFFFFFFE)
    assert v.kind == "MEM_MISMATCH"
    assert v.detail == {"pa": 0xFFFFFFFE, "byte": 3, "expected": 0x44, "actual": 0x45, "nb": False}
    v = r.lw(0x44342211, pa=0xFFFFFFFE)
    assert v.detail["byte"] == 2 and v.detail["expected"] == 0x33
