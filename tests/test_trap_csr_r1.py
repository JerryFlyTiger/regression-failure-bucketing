"""R1 fixes for the trap CSR prediction: load/store fault sets, xret straight
into a trap entry, unknown mtvec base clears state."""

import pytest

from rfb.isa import ScanContext
from tests.test_trap_csr import BASE, PC, _Run, _trap

FT = PC + 4


def _mepc_read(r, val):
    return r.read("mepc", val)


def _cause_read(r, val):
    return r.read("mcause", val)


# --- R1-1 ---------------------------------------------------------------------


@pytest.mark.parametrize("mn,ops", [("lw", "x0,-12(x2)"), ("c.lw", "x8,0(x9)"), ("c.lwsp", "x5,4(x2)")])
@pytest.mark.parametrize("cause,bad", [(1, False), (2, False), (4, False), (5, False), (7, True)])
def test_ldfault_cause_set(mn, ops, cause, bad):
    r = _Run()
    _trap(r, mn, ops)
    assert (_cause_read(r, cause) is not None) == bad


@pytest.mark.parametrize("epc,bad", [(PC, False), (FT, False), (PC + 8, True)])
def test_ld_x0_mepc_set(epc, bad):
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "lw", "x0,-12(x2)")
    assert r.ctx.xmepc is None
    assert (_mepc_read(r, epc) is not None) == bad


@pytest.mark.parametrize("cause,bad", [(1, False), (2, False), (5, True), (4, True)])
def test_ld_rd_written_is_other(cause, bad):
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "lw", "x7,-12(x2)")
    assert r.pred()["mcause"][0] == "other"
    assert (_cause_read(r, cause) is not None) == bad
    assert "mepc" not in r.pred()
    assert r.ctx.xmepc is None


@pytest.mark.parametrize("cause,bad", [(1, False), (2, False), (6, False), (7, False), (5, True)])
def test_store_cause_set(cause, bad):
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "sw", "x5,0(x2)")
    assert (_cause_read(r, cause) is not None) == bad
    assert "mepc" not in r.pred()
    assert r.ctx.xmepc is None


# --- R1-2 ---------------------------------------------------------------------


@pytest.mark.parametrize("mpp,pmpp", [(3, 3), (0, 0)])
def test_mret_to_irq_slot(mpp, pmpp):
    r = _Run()
    r.ctx.mpp = mpp
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=BASE + 4 * 7)
    assert r.pred() == {"mcause": ("irq", frozenset((0x80000007,)))}
    assert r.ctx.priv == "M"
    assert r.ctx.mpp == pmpp
    assert r.ctx.xmepc is None
    assert _cause_read(r, 0x80000007) is None
    r = _Run()  # fresh run: the prediction above was popped by the first read
    r.ctx.mpp = mpp
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=BASE + 4 * 7)
    assert _cause_read(r, 11).kind == "TRAP_MCAUSE"
    r = _Run()
    r.ctx.mpp = mpp
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=BASE + 4 * 7)
    # then an ecall in M-mode predicts {11}
    r.step("ecall", next_pc=BASE)
    assert r.pred()["mcause"][1] == frozenset((11,))


def test_mret_to_irq_slot_unknown_priv_mpp_none():
    r = _Run()
    r.ctx.mpp = None
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=BASE + 4 * 3)
    assert r.ctx.mpp is None and r.ctx.priv == "M"


@pytest.mark.parametrize("mn", ["mret", "dret"])
def test_xret_to_base_ambiguous(mn):
    r = _Run()
    r.ctx.mpp = 0
    r.ctx.xmepc = 0x3000
    r.step(mn, next_pc=BASE)
    assert r.pred() == {}
    assert r.ctx.priv == "M" and r.ctx.mpp is None and r.ctx.xmepc is None
    r.step("ecall", next_pc=BASE)
    assert r.pred()["mcause"][1] == frozenset((11,))
    # same state, but a following mret first: mpp None -> priv None -> {8, 11}
    r = _Run()
    r.ctx.mpp = 0
    r.ctx.xmepc = 0x3000
    r.step(mn, next_pc=BASE)
    r.step("mret", next_pc=0x3000)
    assert r.ctx.priv is None
    r.step("ecall", next_pc=BASE)
    assert r.pred()["mcause"][1] == frozenset((8, 11))


def test_mret_normal_return_to_base_is_not_trap():
    r = _Run()
    r.ctx.mpp = 0
    r.ctx.xmepc = BASE
    r.step("mret", next_pc=BASE)
    assert r.ctx.priv == "U" and r.ctx.mpp == 0
    assert r.ctx.xmepc == BASE
    assert r.pred() == {}


# --- R1-3 ---------------------------------------------------------------------


def test_unknown_base_clears_state():
    r = _Run(base=None)
    r.ctx.xmepc = 0x3000
    r.ctx.mpp = 3
    r.ctx.priv = "M"
    r.step("ecall", next_pc=0x5000)
    assert r.ctx.xmepc is None and r.ctx.priv is None and r.ctx.mpp is None


def test_unknown_base_xret_keeps_state():
    r = _Run(base=None)
    r.ctx.xmepc = 0x3000
    r.ctx.mpp = 0
    r.step("mret", next_pc=0x3000)
    assert r.ctx.xmepc == 0x3000 and r.ctx.priv == "U"


# --- x0 destination guards (R1-4) ---------------------------------------------


def test_faulting_load_with_x0_write_entry_still_ldfault():
    # the tracer may list an x0 write for the faulting load; rd is still x0
    r = _Run()
    _trap(r, "lw", "x0,-12(x2)", writes={"x0": 0})
    assert r.pred()["mcause"][0] == "ldfault"
    assert r.pred()["mepc"][1] == frozenset({PC, FT})


def test_retired_load_to_x7_is_not_ldfault_even_with_x0_noise():
    r = _Run()
    _trap(r, "lw", "x7,-12(x2)", writes={"x0": 0, "x7": 5})
    assert r.pred()["mcause"][0] == "other"


# --- R2-5: compressed loads at base always take the fault set -------------------


@pytest.mark.parametrize("mn,ops", [("c.lw", "x8,0(x9)"), ("c.lwsp", "x5,4(x2)")])
def test_compressed_load_at_base_is_ldfault(mn, ops):
    r = _Run()
    _trap(r, mn, ops, size=2)
    assert r.pred()["mcause"] == ("ldfault", frozenset({1, 2, 4, 5}))
    assert r.pred()["mepc"] == ("ldfault", frozenset({PC, PC + 2}))


def test_lw_nonzero_rd_stays_other():
    r = _Run()
    _trap(r, "lw", "x7,0(x2)")
    assert r.pred()["mcause"] == ("other", frozenset({1, 2}))
    assert "mepc" not in r.pred()


# --- R3 F1 (replaces R2-2) / SV-3: dret into a slot -> priv M, mpp/xmepc unknown


@pytest.mark.parametrize("priv", ["U", "M", None])
def test_dret_to_irq_slot_sets_m_unknown_mpp(priv):
    r = _Run(priv=priv)
    r.ctx.mpp = 0
    r.ctx.xmepc = 0x3000
    r.step("dret", next_pc=BASE + 4 * 7)
    assert r.pred() == {"mcause": ("irq", frozenset((0x80000007,)))}
    assert r.ctx.priv == "M"
    assert r.ctx.mpp is None
    assert r.ctx.xmepc is None


def test_dret_to_irq_slot_then_ecall_predicts_m_cause():
    # F1: from U, the handler's ecall is taken in M (11), not U (8)
    r = _Run(priv="U")
    r.ctx.mpp = 0
    r.ctx.xmepc = 0x3000
    r.step("dret", next_pc=BASE + 4 * 7)
    r.step("ecall", next_pc=BASE)
    assert r.pred()["mcause"][1] == frozenset((11,))
    assert _cause_read(r, 11) is None


def test_mret_to_irq_slot_still_updates_state():
    r = _Run(priv="U")
    r.ctx.mpp = 3
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=BASE + 4 * 7)
    assert r.ctx.priv == "M" and r.ctx.mpp == 3 and r.ctx.xmepc is None


# --- SV-1: writing 0 to mepc is a known value -----------------------------------


def test_csrrw_mepc_zero_sets_xmepc_zero():
    r = _Run()
    r.ctx.xmepc = 0x3000
    r.step("csrrw", "x0,mepc,x0")
    assert r.ctx.xmepc == 0


# --- SV-2: dret is never an XRET_TARGET candidate -------------------------------


def test_dret_to_unknown_target_not_xret_target():
    r = _Run()
    r.ctx.xmepc = 0x3000
    v = r.step("dret", next_pc=0x7000)
    assert v is None or v.kind != "XRET_TARGET"
