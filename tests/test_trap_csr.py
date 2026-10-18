"""Trap CSR prediction: TRAP_MCAUSE / TRAP_MEPC / TRAP_MTVAL (handler reads a
hardware-written CSR whose value does not match the predicted trap) and
XRET_TARGET (mret lands somewhere other than the known mepc). Also privilege
(priv / MPP) tracking and the signature exclusion of the TRAP_ kinds."""

import pytest

from rfb import isa
from rfb.isa import ScanContext
from rfb.record import Deviation, FailureRecord, Instr, RegrInfo, SimInfo, TraceInfo
from rfb.signature import _sys_viol

BASE = 0x2000
PC = 0x1000


def _instr(index, pc, mnemonic, operands, reads=None, writes=None, size=4):
    return Instr(
        index=index, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
        reads=reads or {}, writes=writes or {}, pa=None, store=None, load=None,
    )


class _Run:
    """Drives check_safe + observe the way parse_trace does."""

    def __init__(self, base=BASE, priv="M"):
        self.ctx = ScanContext()
        self.ctx.mtvec_base = base
        self.ctx.priv = priv
        self.n = 0
        self.pc = PC

    def step(self, mnemonic, operands="", reads=None, writes=None, next_pc="fall",
             pc=None, size=4):
        pc = self.pc if pc is None else pc
        if next_pc == "fall":
            next_pc = pc + size
        i = _instr(self.n, pc, mnemonic, operands, reads, writes, size)
        self.n += 1
        v, err = isa.check_safe(None, i, next_pc, self.ctx)
        assert err is None
        isa.observe(self.ctx, i, next_pc)
        self.pc = next_pc if next_pc is not None else pc + size
        return v

    def read(self, csr, val, rd="x5"):
        """Handler-side CSR read (csrrs rd,csr,x0) at a fall-through pc."""
        return self.step("csrrs", f"{rd},{csr},x0", writes={rd: val})

    def pred(self):
        return self.ctx.trap_pred


def _trap(run, mn, ops="", landing=BASE, **kw):
    return run.step(mn, ops, next_pc=landing, **kw)


# --- sync exception prediction ------------------------------------------------


@pytest.mark.parametrize("priv,cause", [("U", {8}), ("M", {11}), (None, {8, 11})])
def test_ecall_mcause_by_priv(priv, cause):  # (8,)->(11,), (11,)->(8,), (8,11)
    r = _Run(priv=priv)
    _trap(r, "ecall", pc=PC)
    assert r.pred()["mcause"] == ("ecall", frozenset(cause))
    assert r.pred()["mepc"] == ("ecall", frozenset({PC}))
    assert r.pred()["mtval"] == ("ecall", frozenset({0}))


def test_ecall_reads_matching_report_nothing():
    r = _Run(priv="U")
    _trap(r, "ecall", pc=PC)
    assert r.read("mcause", 8) is None
    assert r.read("mepc", PC) is None
    assert r.read("mtval", 0) is None


def test_ecall_mcause_mismatch_reported():
    r = _Run(priv="U")
    _trap(r, "ecall", pc=PC)
    v = r.read("mcause", 11)
    assert v.kind == "TRAP_MCAUSE"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "TRAP_MCAUSE:ecall"
    assert v.detail["expected"] == [8] and v.detail["actual"] == 11


def test_ecall_priv_unknown_accepts_either_but_not_other():
    r = _Run(priv=None)
    _trap(r, "ecall")
    assert r.read("mcause", 11) is None
    r = _Run(priv=None)
    _trap(r, "ecall")
    assert r.read("mcause", 3).kind == "TRAP_MCAUSE"


def test_ecall_mepc_mismatch_reported_as_sync():
    r = _Run()
    _trap(r, "ecall", pc=PC)
    v = r.read("mepc", PC + 4)
    assert v.kind == "TRAP_MEPC"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "TRAP_MEPC:sync"


def test_ecall_mtval_mismatch_reported():
    r = _Run()
    _trap(r, "ecall")
    v = r.read("mtval", 0x1234)
    assert v.kind == "TRAP_MTVAL"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "TRAP_MTVAL:ecall"


@pytest.mark.parametrize("mn", ["ebreak", "c.ebreak"])
def test_ebreak_predicts_cause_3(mn):
    r = _Run()
    _trap(r, mn, size=2 if mn == "c.ebreak" else 4)
    assert r.pred()["mcause"] == ("ebreak", frozenset({3}))
    assert r.pred()["mepc"][1] == frozenset({PC})
    assert "mtval" not in r.pred()
    assert r.read("mcause", 3) is None
    assert "mcause" not in r.pred()  # popped by the first read
    r = _Run()
    _trap(r, mn)
    assert r.read("mcause", 11).kind == "TRAP_MCAUSE"


def test_load_fault_predicts_1_2_4_5():
    r = _Run()
    _trap(r, "lw", "x0,0(x5)", writes={})
    assert r.pred()["mcause"] == ("ldfault", frozenset({1, 2, 4, 5}))
    assert r.read("mcause", 5) is None
    r = _Run()
    _trap(r, "lw", "x0,0(x5)")
    assert r.read("mcause", 6).kind == "TRAP_MCAUSE"


def test_store_fault_predicts_1_2_6_7():
    r = _Run()
    _trap(r, "sw", "x7,0(x5)")
    assert r.pred()["mcause"] == ("stfault", frozenset({1, 2, 6, 7}))
    assert r.read("mcause", 7) is None
    r = _Run()
    _trap(r, "sw", "x7,0(x5)")
    assert r.read("mcause", 5).kind == "TRAP_MCAUSE"


def test_other_instruction_landing_on_base_predicts_1_2_without_mepc():
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "addi", "x1,x1,1")
    assert r.pred() == {"mcause": ("other", frozenset({1, 2}))}
    assert r.ctx.xmepc is None
    assert r.read("mepc", 0x9999) is None  # no mepc prediction
    assert r.read("mcause", 2) is None
    r = _Run()
    _trap(r, "addi", "x1,x1,1")
    assert r.read("mcause", 7).kind == "TRAP_MCAUSE"


def test_sync_trap_sets_xmepc_to_trapping_pc():
    r = _Run()
    _trap(r, "ecall", pc=PC)
    assert r.ctx.xmepc == PC


# --- interrupt prediction -----------------------------------------------------


@pytest.mark.parametrize("k", [3, 7, 11, 16])
def test_irq_slot_predicts_mcause_and_fallthrough_mepc(k):
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4 * k, pc=PC)
    assert r.pred()["mcause"] == ("irq", frozenset({0x80000000 | k}))
    assert r.pred()["mepc"] == ("irq", frozenset({PC + 4}))
    assert r.ctx.xmepc == PC + 4
    assert r.read("mcause", 0x80000000 | k) is None
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4 * k)
    assert r.read("mcause", 0x80000000 | (k + 1)).kind == "TRAP_MCAUSE"


def test_irq_mepc_mismatch_group_is_irq():
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 12, pc=PC)
    assert r.read("mepc", PC).kind == "TRAP_MEPC"
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 12, pc=PC)
    v = r.read("mepc", PC)
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "TRAP_MEPC:irq"


def test_irq_on_control_flow_instruction_has_no_mepc_pred():
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "jal", "x1,0x40", landing=BASE + 4 * 7)
    assert "mepc" not in r.pred()
    assert r.pred()["mcause"] == ("irq", frozenset({0x80000007}))
    assert r.ctx.xmepc is None


def test_irq_cause_matches_taken_branch_not_confused():
    # a branch taken to its real target is not a flow break: no prediction
    r = _Run()
    r.ctx.xmepc = 0x3000
    r.step("jal", f"x1,0x{PC + 0x40:x}", next_pc=PC + 0x40)  # operand is the absolute target
    assert r.pred() == {} and r.ctx.xmepc == 0x3000


def test_non_irq_slot_not_predicted():
    r = _Run()
    r.ctx.xmepc = 0x3000  # seeded: the assertion below must see it cleared
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4)
    assert r.pred() == {}
    assert r.ctx.xmepc is None


def test_misaligned_offset_not_predicted():
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4 * 3 + 2)
    assert r.pred() == {}


def test_mtvec_base_unknown_no_prediction():
    r = _Run(base=None)
    _trap(r, "ecall", landing=BASE)
    assert r.pred() == {}
    assert r.read("mcause", 1) is None


# --- reader-side guards -------------------------------------------------------


def test_rd_x0_not_reported():
    r = _Run()
    _trap(r, "ecall")
    assert r.step("csrrs", "x0,mcause,x0", writes={"x0": 99}) is None


@pytest.mark.parametrize("name,num", [("mcause", "0x342"), ("mepc", "0x341"), ("mtval", "0x343")])
def test_numeric_csr_names_work(name, num):
    r = _Run()
    _trap(r, "ecall")
    v = r.read(num, 0x7777)
    assert v is not None and v.detail["csr"] == name
    r = _Run()
    _trap(r, "ecall")
    ok = {"mcause": 11, "mepc": PC, "mtval": 0}[name]
    assert r.read(num, ok) is None


def test_reading_instruction_that_breaks_flow_not_reported():
    r = _Run()
    _trap(r, "ecall")
    v = r.step("csrrs", "x5,mcause,x0", writes={"x5": 0x55}, next_pc=0x9000)
    assert v is None


def test_unrelated_csr_read_not_reported():
    r = _Run()
    _trap(r, "ecall")
    assert r.read("mscratch", 0x55) is None


def test_missing_read_value_not_reported():
    r = _Run()
    _trap(r, "ecall")
    assert r.step("csrrs", "x5,mcause,x0", writes={}) is None


@pytest.mark.parametrize("name", ["mcause", "mepc", "mtval"])
def test_software_write_clears_prediction(name):
    r = _Run()
    _trap(r, "ecall")
    r.step("csrrw", f"x0,{name},x6", reads={"x6": 0x1234})
    assert name not in r.pred()
    v = r.read(name, 0x1234)
    assert v is None or not v.kind.startswith("TRAP_")
    # the other predictions survive
    others = {"mcause", "mepc", "mtval"} - {name}
    assert others <= set(r.pred())


def test_pred_cleared_on_next_flow_break():
    r = _Run()
    _trap(r, "ecall")
    assert r.pred()
    r.step("addi", "x1,x1,1", next_pc=0x7000)
    assert r.pred() == {}


# --- privilege tracking -------------------------------------------------------


@pytest.mark.parametrize("val,mpp", [(0x0000, 0), (0x1800, 3), (0x0800, None), (0x1000, None)])
def test_csrrw_mstatus_sets_mpp(val, mpp):
    r = _Run()
    r.ctx.mpp = 3 if mpp != 3 else 0  # start from a different value
    r.step("csrrw", "x0,mstatus,x6", reads={"x6": val})
    assert r.ctx.mpp == mpp


def test_csrrw_mstatus_numeric_name():
    r = _Run()
    r.step("csrrw", "x0,0x300,x6", reads={"x6": 0x1800})
    assert r.ctx.mpp == 3


@pytest.mark.parametrize("mn", ["csrrs", "csrrc"])
def test_set_clear_touching_mpp_bits_unknown(mn):
    r = _Run()
    r.ctx.mpp = 3
    r.step(mn, "x0,mstatus,x6", reads={"x6": 0x800})
    assert r.ctx.mpp is None


def test_set_without_mpp_bits_keeps_mpp_when_rd_x0():
    r = _Run()
    r.ctx.mpp = 3
    r.step("csrrs", "x0,mstatus,x6", reads={"x6": 0x8})
    assert r.ctx.mpp == 3


def test_set_without_mpp_bits_rd_old_value_gives_mpp():
    r = _Run()
    r.ctx.mpp = None
    r.step("csrrs", "x5,mstatus,x6", reads={"x6": 0x8}, writes={"x5": 0x1808})
    assert r.ctx.mpp == 3
    r.step("csrrc", "x5,mstatus,x6", reads={"x6": 0x8}, writes={"x5": 0x0008})
    assert r.ctx.mpp == 0
    r.step("csrrs", "x5,mstatus,x6", reads={"x6": 0x8}, writes={"x5": 0x0808})
    assert r.ctx.mpp is None


def test_set_with_unknown_operand_makes_mpp_unknown():
    r = _Run()
    r.ctx.mpp = 3
    r.step("csrrs", "x0,mstatus,x6", reads={})
    assert r.ctx.mpp is None


@pytest.mark.parametrize("mpp,priv", [(0, "U"), (3, "M"), (None, None)])
def test_mret_restores_priv_from_mpp_and_clears_mpp(mpp, priv):
    r = _Run(priv="M" if priv != "M" else "U")
    r.ctx.mpp = mpp
    r.step("mret", next_pc=0x3000)
    assert r.ctx.priv == priv
    assert r.ctx.mpp == 0


def test_mret_with_mpp_01_gives_unknown_priv():
    r = _Run()
    r.ctx.mpp = 1
    r.step("mret", next_pc=0x3000)
    assert r.ctx.priv is None


@pytest.mark.parametrize("prev,mpp", [("M", 3), ("U", 0), (None, None)])
def test_trap_entry_saves_mpp_and_enters_m(prev, mpp):
    r = _Run(priv=prev)
    _trap(r, "ecall")
    assert r.ctx.mpp == mpp
    assert r.ctx.priv == "M"


@pytest.mark.parametrize("prev,mpp", [("M", 3), ("U", 0)])
def test_irq_entry_saves_mpp_and_enters_m(prev, mpp):
    r = _Run(priv=prev)
    _trap(r, "addi", "x1,x1,1", landing=BASE + 12)
    assert r.ctx.mpp == mpp
    assert r.ctx.priv == "M"


def test_ecall_in_u_after_mret_roundtrip():
    r = _Run(priv="M")
    r.ctx.mpp = 0
    r.step("mret", next_pc=0x3000)
    _trap(r, "ecall", pc=0x3000)
    assert r.pred()["mcause"] == ("ecall", frozenset({8}))


# --- XRET_TARGET --------------------------------------------------------------


def _mret(r, target, pc=0x5000):
    return r.step("mret", next_pc=target, pc=pc)


def test_mret_to_known_mepc_not_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, 0x3000) is None


def test_mret_to_other_target_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    v = _mret(r, 0x3008)
    assert v.kind == "XRET_TARGET"
    assert v.detail == {"expected": 0x3000, "actual": 0x3008}


def test_mret_fallthrough_not_reported_by_xret_target():
    r = _Run()
    r.ctx.xmepc = 0x3000
    v = _mret(r, 0x5004)
    assert v is not None and v.kind == "XRET_FALLTHROUGH"


def test_mret_to_trap_entry_not_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    r.ctx.trap_entries.add(0x6000)
    assert _mret(r, 0x6000) is None


def test_mret_to_irq_slot_not_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, BASE + 4 * 7) is None


def test_mret_to_non_irq_slot_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, BASE + 4).kind == "XRET_TARGET"


def test_mret_to_mtvec_base_not_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, BASE) is None


def test_mret_unknown_xmepc_not_reported():
    r = _Run()
    r.ctx.xmepc = None
    assert _mret(r, 0x3008) is None


def test_mret_next_pc_none_not_reported():
    r = _Run()
    r.ctx.xmepc = 0x3000
    i = _instr(0, 0x5000, "mret", "")
    v, err = isa.check_safe(None, i, None, r.ctx)
    assert err is None and v is None


def test_csrw_mepc_sets_known_value_low_bit_cleared():
    r = _Run()
    r.step("csrrw", "x0,mepc,x6", reads={"x6": 0x3001})
    assert r.ctx.xmepc == 0x3000
    assert _mret(r, 0x3000) is None
    r.step("csrrw", "x0,0x341,x6", reads={"x6": 0x4003})
    assert r.ctx.xmepc == 0x4002


def test_csrr_mepc_pure_read_adopts_read_value():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrs", "x5,mepc,x0", writes={"x5": 0x4000})
    assert r.ctx.xmepc == 0x4000
    r.step("csrrc", "x6,mepc,x0", writes={"x6": 0x4400})
    assert r.ctx.xmepc == 0x4400


def test_pure_read_with_rd_x0_keeps_xmepc():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrs", "x0,mepc,x0")
    assert r.ctx.xmepc == 0x1111


def test_csrs_mepc_with_nonzero_operand_makes_unknown():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrs", "x0,mepc,x6", reads={"x6": 4})
    assert r.ctx.xmepc is None


def test_csrw_mepc_unknown_value_makes_unknown():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrw", "x0,mepc,x6", reads={})
    assert r.ctx.xmepc is None


def test_unrecognised_landing_clears_xmepc():
    r = _Run()
    r.ctx.xmepc = 0x3000
    r.step("addi", "x1,x1,1", next_pc=BASE + 4)
    assert r.ctx.xmepc is None


# --- signature exclusion ------------------------------------------------------


def _record(violations):
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    trace = TraceInfo.empty()
    trace.violations = dict(violations)
    return FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                         trace=trace, deviation=None, parse_errors=[])


@pytest.mark.parametrize("kind", ["TRAP_MCAUSE", "TRAP_MEPC", "TRAP_MTVAL"])
def test_trap_kinds_excluded_from_sys_viol(kind):
    cls, keys = _sys_viol(_record({f"{kind}:csrrs": 5}))
    assert cls == frozenset() and keys == frozenset()


def test_non_excluded_kind_control_still_counts():
    cls, keys = _sys_viol(_record({"XRET_TARGET:mret": 5}))
    assert cls
    assert ("XRET_TARGET", "mret") in keys


# --- xret is not a trap entry ---------------------------------------------------


@pytest.mark.parametrize("landing", [BASE, BASE + 4 * 7])
def test_mret_landing_on_trap_vector_makes_no_prediction(landing):
    r = _Run(priv="M")
    r.ctx.mpp = 0
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=landing)
    # R1-2: the landing is itself a trap entry right after the mret
    assert r.ctx.xmepc is None
    assert r.ctx.priv == "M"
    if landing == BASE:
        assert r.pred() == {}
        assert r.ctx.mpp is None
    else:
        assert r.pred() == {"mcause": ("irq", frozenset((0x80000007,)))}
        assert r.ctx.mpp == 0  # the mret dropped to U (mpp was 0)


# --- x0 destination guards (R1-4): a trace may carry a bogus x0 write ----------


@pytest.mark.parametrize("mn,start,old,expect", [
    ("csrrs", 3, 0x0008, 3),   # would turn into 0 if the x0 "old value" were trusted
    ("csrrc", 0, 0x1808, 0),   # would turn into 3
])
def test_mstatus_set_clear_rd_x0_ignores_x0_write(mn, start, old, expect):
    r = _Run()
    r.ctx.mpp = start
    r.step(mn, "x0,mstatus,x6", reads={"x6": 0x8}, writes={"x0": old})
    assert r.ctx.mpp == expect


def test_pure_mepc_read_rd_x0_ignores_x0_write():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrs", "x0,mepc,x0", writes={"x0": 0x4000})
    assert r.ctx.xmepc == 0x1111


def test_pure_mepc_read_rd_x5_with_x0_write_noise_still_adopts_x5():
    r = _Run()
    r.ctx.xmepc = 0x1111
    r.step("csrrs", "x5,mepc,x0", writes={"x0": 0x9999, "x5": 0x4000})
    assert r.ctx.xmepc == 0x4000


# --- mutation-driven additions (batch B) --------------------------------------


def test_mret_to_ordinary_target_clears_pending_prediction():
    # M-A: the handler's mret ends the trap; a stale prediction must not be
    # checked against a CSR read of the code that follows.
    r = _Run()
    _trap(r, "ecall", pc=PC)
    assert r.pred()
    r.ctx.xmepc = 0x3000
    r.step("mret", next_pc=0x3000)
    assert r.pred() == {}
    assert r.read("mcause", 0x1234) is None


def test_taken_jal_to_true_target_leaves_state_alone():
    # M-B: not a flow break, so xmepc/priv survive and nothing is predicted.
    r = _Run(priv="U")
    r.ctx.xmepc = 0x3000
    r.ctx.mpp = 0
    r.step("jal", f"x1,0x{PC + 0x40:x}", next_pc=PC + 0x40)
    assert r.ctx.xmepc == 0x3000 and r.ctx.priv == "U" and r.ctx.mpp == 0
    assert r.pred() == {}


@pytest.mark.parametrize("target", [BASE, BASE + 4 * 7])
def test_jal_whose_true_target_is_vector_slot_is_not_a_trap(target):
    # M-B: a jump that really lands on base / an irq slot is correct control
    # flow; reading the target as a trap entry would predict mcause.
    r = _Run()
    r.step("jal", f"x0,0x{target:x}", pc=PC, next_pc=target)
    assert r.pred() == {}


def test_taken_branch_to_true_target_is_not_a_flow_break():
    # M-B with a conditional branch: the taken target equals next_pc.
    r = _Run()
    r.ctx.xmepc = 0x3000
    r.step("beq", "x1,x2,0x2000", reads={"x1": 5, "x2": 5}, pc=PC, next_pc=0x2000)
    assert r.ctx.xmepc == 0x3000
    assert r.pred() == {}


@pytest.mark.parametrize("mn", ["csrrs", "csrrc"])
def test_set_clear_with_mpp_bit12_only_makes_mpp_unknown(mn):
    # M-G: 0x1000 touches only the high MPP bit; the 0x1800 mask must catch it.
    r = _Run()
    r.ctx.mpp = 3
    r.step(mn, "x0,mstatus,x6", reads={"x6": 0x1000})
    assert r.ctx.mpp is None


# interrupt slot boundaries: Ibex only raises 3, 7, 11, 16..31
_IRQ_YES = [3, 7, 11, 16, 30, 31]


@pytest.mark.parametrize("k", _IRQ_YES)
def test_irq_slot_boundary_entry_predicted(k):
    r = _Run()
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4 * k)
    assert r.pred()["mcause"][1] == frozenset({0x80000000 | k})


@pytest.mark.parametrize("k", [4, 12, 15, 32, 33])
def test_non_irq_slot_boundary_entry_not_predicted(k):
    r = _Run()
    r.ctx.xmepc = 0x3000
    _trap(r, "addi", "x1,x1,1", landing=BASE + 4 * k)
    assert r.pred() == {} and r.ctx.xmepc is None


@pytest.mark.parametrize("k", _IRQ_YES)
def test_mret_to_irq_slot_boundary_not_reported_and_predicted(k):
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, BASE + 4 * k) is None
    assert r.pred()["mcause"][1] == frozenset({0x80000000 | k})


@pytest.mark.parametrize("k", [4, 12, 15, 32, 33])
def test_mret_to_non_irq_slot_boundary_reported(k):
    r = _Run()
    r.ctx.xmepc = 0x3000
    assert _mret(r, BASE + 4 * k).kind == "XRET_TARGET"
    assert r.pred() == {}
