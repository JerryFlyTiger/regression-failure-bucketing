"""MSTATUS_MISMATCH: model of the readable Ibex mstatus bits (MIE, MPIE, MPP,
MPRV, TW), checked against every mstatus read (isa._observe_mstatus /
isa._mstatus_check). Unit level through check_safe + observe, then
parse_trace, grouping and cluster integration."""

from pathlib import Path

import pytest

from rfb import isa
from rfb.cluster import _Atom, _atom_distance
from rfb.isa import ScanContext
from rfb.parse_trace import parse_trace
from rfb.record import FailureRecord, Instr, RegrInfo, SimInfo, TraceInfo
from rfb.signature import _SCAN_EXCLUDED, Signature, _ratio_vector, _sys_viol

BASE = 0x2000
PC = 0x1000
MIE, MPIE, MPP, MPRV, TW = 0x8, 0x80, 0x1800, 0x20000, 0x200000
READABLE = MIE | MPIE | MPP | MPRV | TW
MPP_M = 0x1800
# (mask, field name) for every readable field
FIELDS = [(MIE, "MIE"), (MPIE, "MPIE"), (0x800, "MPP"), (MPRV, "MPRV"), (TW, "TW")]


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
        self.init = (base, priv)
        self.log = []  # replayable: a read re-syncs the model, so flips need a fresh run

    def set_xmepc(self, val):
        self.ctx.xmepc = val
        self.log.append(("xmepc", val))

    def step(self, mnemonic, operands="", reads=None, writes=None, next_pc="fall",
             pc=None, size=4):
        self.log.append(("step", (mnemonic, operands, reads, writes, next_pc, pc, size)))
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

    def wr(self, val, mn="csrrw", **kw):
        """Pure mstatus write (rd = x0): the model takes the written value."""
        return self.step(mn, "x0,mstatus,x6", reads={"x6": val}, **kw)

    def rd_sync(self, val, **kw):
        """A real read: the printed value re-syncs the model."""
        return self.step("csrrs", "x7,mstatus,x0", writes={"x7": val}, **kw)

    def rd(self, val, **kw):
        """Probe: the read runs on a fresh replay, so ``self`` is unchanged
        and several probes can follow one setup."""
        return _replay(self).rd_sync(val, **kw)

    def trap(self, landing=BASE, mn="ecall"):
        return self.step(mn, "", next_pc=landing)

    def irq(self, k):
        return self.step("addi", "x1,x1,1", next_pc=BASE + 4 * k)


def _field(v):
    assert v is not None and v.kind == "MSTATUS_MISMATCH"
    return v.detail["field"]


def _replay(r):
    n = _Run(*r.init)
    for kind, a in r.log:
        if kind == "xmepc":
            n.set_xmepc(a)
        else:
            n.step(*a)
    return n


def _assert_read(r, expected, *, free=0):
    """Reading ``expected`` is accepted; flipping any known readable bit
    (not in ``free``) is reported with the right field name.  Reads are probes (see _Run.rd)."""
    assert r.rd(expected) is None
    for m, name in FIELDS:
        if m & free:
            continue
        v = r.rd(expected ^ m)
        assert _field(v) == name, (name, hex(expected))
        assert v.detail["actual"] == expected ^ m


# --- csr write then read ------------------------------------------------------


@pytest.mark.parametrize("mask,name", FIELDS)
def test_write_read_each_field_mismatch(mask, name):
    r = _Run()
    r.wr(0)
    v = r.rd(mask)
    assert _field(v) == name
    assert v.detail["expected"] == 0 and v.detail["actual"] == mask
    assert v.mnemonic == "csrrs"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "MST:" + name


@pytest.mark.parametrize("mask,name", FIELDS)
def test_write_read_each_field_match(mask, name):
    r = _Run()
    r.wr(mask if mask != 0x800 else MPP_M)
    expected = mask if mask != 0x800 else MPP_M
    assert r.rd(expected) is None
    v = r.rd(0)
    assert _field(v) == name
    assert v.detail["expected"] == expected


def test_write_read_all_fields_exact_for_every_bit():
    r = _Run()
    r.wr(READABLE)
    _assert_read(r, READABLE)


def test_multiple_differing_fields_are_joined():
    r = _Run()
    r.wr(MIE | MPIE)
    v = r.rd(0)
    assert _field(v) == "MIE+MPIE"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "MST:MIE+MPIE"
    assert v.detail["src"] is None


def test_src_is_csr_for_a_single_field():
    r = _Run()
    r.wr(0)
    assert r.rd(MIE).detail["src"] == "csr"


def test_unreadable_bits_read_zero():
    r = _Run()
    r.wr(0xFFFFFFFF)
    assert r.rd(READABLE) is None
    v = r.rd(READABLE | 0x1)
    assert _field(v) == "OTHER"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "MST:OTHER"
    assert _field(r.rd(READABLE | 0x80000000)) == "OTHER"


def test_rd_x0_read_is_not_checked():
    r = _Run()
    r.wr(0)
    assert r.step("csrrs", "x0,mstatus,x0", writes={"x0": MIE}) is None


def test_csr_hex_name_is_recognised():
    r = _Run()
    r.step("csrrw", "x0,0x300,x6", reads={"x6": 0})
    assert _field(r.step("csrrs", "x7,0x300,x0", writes={"x7": MIE})) == "MIE"


@pytest.mark.parametrize("val", [0x0800, 0x1000])
def test_mpp_write_01_10_makes_mpp_unknown(val):
    r = _Run()
    r.wr(val)
    for mpp in (0, 0x800, 0x1000, 0x1800):
        assert r.rd(mpp) is None
    assert _field(r.rd(MIE)) == "MIE"  # the other fields stay known
    # control: a legal MPP stays checked
    r2 = _Run()
    r2.wr(0x1800)
    assert _field(r2.rd(0)) == "MPP"


def test_mpp_unknown_survives_csrrs_of_other_bits():
    r = _Run()
    r.wr(0x0800)
    r.step("csrrsi", f"x0,mstatus,{MIE}")
    assert r.rd(MIE | 0x1800) is None
    assert r.rd(MIE) is None


def test_csrrs_with_unknown_original_knows_only_set_bits():
    r = _Run()
    r.step("csrrsi", f"x0,mstatus,{MIE}")
    assert r.rd(MIE) is None
    assert r.rd(MIE | MPIE | MPRV | TW) is None  # unknown: not reported
    assert r.rd(MIE | 0x1000) is None  # MPP unknown (not in the known mask)
    assert _field(r.rd(0)) == "MIE"
    assert _field(r.rd(0x1 | MIE)) == "OTHER"  # non-readable bits always known 0


def test_csrrc_with_unknown_original_knows_only_cleared_bits():
    r = _Run()
    r.step("csrrci", f"x0,mstatus,{MIE}")
    assert r.rd(0) is None
    v = r.rd(MIE)
    assert _field(v) == "MIE" and v.detail["expected"] & MIE == 0
    assert r.rd(MPIE | TW) is None


def test_csrrc_register_operand_clears_bits():
    r = _Run()
    r.wr(MIE | MPIE)
    r.step("csrrc", "x0,mstatus,x6", reads={"x6": MIE})
    assert r.rd(MPIE) is None
    assert _field(r.rd(MIE | MPIE)) == "MIE"


def test_csrrs_register_operand_sets_bits():
    r = _Run()
    r.wr(MPIE)
    r.step("csrrs", "x0,mstatus,x6", reads={"x6": MIE})
    assert r.rd(MIE | MPIE) is None
    assert _field(r.rd(MPIE)) == "MIE"


def test_csrrsi_csrrci_on_known_state():
    r = _Run()
    r.wr(MPIE)
    r.step("csrrsi", f"x0,mstatus,{MIE}")
    assert r.rd(MIE | MPIE) is None
    r.step("csrrci", f"x0,mstatus,{MPIE}")
    assert r.rd(MIE) is None
    assert _field(r.rd(MIE | MPIE)) == "MPIE"


def test_csr_op_with_zero_operand_keeps_state():
    r = _Run()
    r.wr(MIE)
    r.step("csrrs", "x0,mstatus,x0")
    assert _field(r.rd(0)) == "MIE"


def test_csr_write_with_unknown_operand_forgets_everything():
    r = _Run()
    r.wr(MIE)
    r.step("csrrw", "x0,mstatus,x6", reads={})  # x6 value not printed
    assert r.rd(0) is None
    assert r.rd(READABLE) is None


def test_read_value_resyncs_the_model():
    r = _Run()
    assert r.rd_sync(MIE | MPIE) is None  # nothing known yet: accepted, resynced
    v = r.rd(MIE)
    assert _field(v) == "MPIE"
    assert v.detail["expected"] == (MIE | MPIE)


def test_read_with_rd_resync_then_write_applies():
    r = _Run()
    r.step("csrrs", "x5,mstatus,x6", reads={"x6": MIE}, writes={"x5": MPIE | MPP_M})
    # old value MPIE|MPP resynced, then MIE set
    assert r.rd(MIE | MPIE | MPP_M) is None
    assert _field(r.rd(MPIE | MPP_M)) == "MIE"


def test_other_csr_does_not_touch_the_model():
    r = _Run()
    r.wr(MIE)
    r.step("csrrw", "x0,mscratch,x6", reads={"x6": 0})
    assert _field(r.rd(0)) == "MIE"


def test_mismatching_read_is_not_a_csr_mismatch():
    r = _Run()
    r.wr(MIE)
    assert r.rd(0).kind == "MSTATUS_MISMATCH"


# --- read followed by a flow break --------------------------------------------


def test_read_followed_by_flow_break_is_not_checked():
    r = _Run()
    r.wr(0)
    assert r.rd(MIE, next_pc=BASE) is None
    r2 = _Run()
    r2.wr(0)
    assert r2.rd(MIE, next_pc=BASE + 4 * 7) is None
    # control: fall-through is checked
    r3 = _Run()
    r3.wr(0)
    assert _field(r3.rd(MIE)) == "MIE"


# --- trap entry -----------------------------------------------------------------


@pytest.mark.parametrize("priv,mpp", [("M", MPP_M), ("U", 0)])
def test_trap_entry_effects_mpp_by_priv(priv, mpp):
    r = _Run(priv=priv)
    r.wr(MIE)  # MIE=1, MPIE=0, MPP=0
    r.trap()
    # MPIE <- MIE (1), MIE <- 0, MPP <- priv
    _assert_read(r, MPIE | mpp)


def test_trap_entry_with_mie_clear_gives_mpie_zero():
    r = _Run(priv="U")
    r.wr(MPIE | MPRV)
    r.trap()
    _assert_read(r, MPRV)


def test_trap_entry_src_is_trap():
    r = _Run(priv="U")
    r.wr(MIE)
    r.trap()
    assert r.rd(0).detail["src"] == "trap"  # MPIE expected 1


def test_trap_entry_priv_unknown_leaves_mpp_unknown():
    r = _Run(priv=None)
    r.wr(MIE)
    r.trap()
    for mpp in (0, 0x1800):
        assert r.rd(MPIE | mpp) is None
    assert _field(r.rd(MPIE | MIE)) == "MIE"


def test_trap_entry_with_unknown_mie_makes_mpie_unknown():
    r = _Run(priv="U")
    r.step("csrrw", "x0,mstatus,x6", reads={})  # everything unknown
    r.trap()
    assert r.rd(MPIE) is None
    assert r.rd(0) is None
    assert _field(r.rd(MIE)) == "MIE"  # MIE is certainly 0


def test_interrupt_slot_applies_trap_entry():
    r = _Run(priv="U")
    r.wr(MIE)
    r.irq(7)
    _assert_read(r, MPIE)


def test_every_irq_slot_is_a_trap_entry():
    for k in (3, 7, 11, 16, 31):
        r = _Run(priv="M")
        r.wr(MIE)
        r.irq(k)
        _assert_read(r, MPIE | MPP_M)


def test_unrecognised_landing_forgets_everything():
    r = _Run()
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    assert r.rd(0) is None
    assert r.rd(READABLE) is None


def test_non_irq_slot_is_an_unrecognised_landing():
    r = _Run()
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=BASE + 4)  # slot 1: not an irq
    assert r.rd(0) is None


def test_mtvec_unknown_forgets_everything():
    r = _Run(base=None)
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    assert r.rd(0) is None


def test_csr_op_landing_on_base_ignores_the_write():
    r = _Run(priv="U")
    r.wr(0)
    r.wr(MPIE, next_pc=BASE)  # may itself have trapped: write not counted
    # MIE <- 0 certain, MPP <- U certain; MPIE (old MIE unknown) unknown.
    assert r.rd(0) is None
    assert r.rd(MPIE) is None
    assert _field(r.rd(MIE)) == "MIE"
    assert _field(r.rd(0x1800)) == "MPP"


def test_csr_op_landing_on_base_forgets_unrelated_known_bits():
    r = _Run(priv="U")
    r.wr(MPRV | TW)
    r.wr(0, next_pc=BASE)
    assert r.rd(0) is None  # MPRV/TW no longer known (write may not count)
    assert r.rd(MPRV | TW) is None


def test_csr_op_landing_on_irq_slot_applies_the_write():
    r = _Run(priv="U")
    r.wr(0)
    r.wr(MIE, next_pc=BASE + 4 * 7)  # retired, then the interrupt
    # MPIE <- MIE (1 from the write), MIE <- 0, MPP <- U
    _assert_read(r, MPIE)


def test_csr_op_landing_on_irq_slot_sets_mpie_from_written_zero():
    r = _Run(priv="U")
    r.wr(MIE)
    r.wr(0, next_pc=BASE + 4 * 7)
    _assert_read(r, 0)


# --- mret -------------------------------------------------------------------------

# (value before mret, value after mret), priv M, MPP=U/M as written.
MRET_CASES = [
    (MPIE | MPRV, MIE | MPIE | 0),        # MIE<-1, MPIE<-1, MPP<-U, MPRV<-0 (MPP was U)
    (0, MPIE),                              # MIE<-0 (MPIE was 0), MPIE<-1
    (MPIE | MPP_M | MPRV, MIE | MPIE | MPRV),  # MPP was M: MPRV kept
    (MPIE | MPP_M | TW, MIE | MPIE | TW),  # MPRV 0, TW untouched
    (MIE | MPIE, MIE | MPIE),              # MIE stays 1
]


@pytest.mark.parametrize("before,after", MRET_CASES)
def test_mret_effects(before, after):
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(before)
    r.step("mret", "", next_pc=0x3000)
    _assert_read(r, after)


def test_mret_src_is_mret():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE)
    r.step("mret", "", next_pc=0x3000)
    v = r.rd(MIE | MPIE | 0x800)  # MPP expected U
    assert v.detail["field"] == "MPP" and v.detail["src"] == "mret"


def test_mret_with_unknown_mpie_makes_mie_unknown():
    r = _Run()
    r.set_xmepc(0x3000)
    r.step("csrrsi", f"x0,mstatus,{MIE}")  # MPIE unknown
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(MPIE) is None
    assert r.rd(MPIE | MIE) is None
    assert _field(r.rd(0)) == "MPIE"


def test_mret_with_unknown_mpp_makes_mprv_unknown():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(0x0800 | MPIE)  # MPP unknown (legalised), MPIE 1
    r.wr(MPRV, mn="csrrs")  # set MPRV (MPP stays unknown)
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(MIE | MPIE | MPRV) is None
    assert r.rd(MIE | MPIE) is None  # MPRV unknown
    assert _field(r.rd(MPIE)) == "MIE"


def test_mret_to_irq_slot_then_trap_entry():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE)  # MPP = U written, MPIE = 1
    r.step("mret", "", next_pc=BASE + 4 * 7)
    # mret: MIE=1, MPIE=1, MPP=U; then the irq: MPIE <- 1, MIE <- 0, MPP <- U
    _assert_read(r, MPIE)


def test_mret_to_irq_slot_returning_to_machine_mode():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE | MPP_M)
    r.step("mret", "", next_pc=BASE + 4 * 7)
    # returns to M (MPP=3 -> priv M), then the irq saves MPP <- M
    _assert_read(r, MPIE | MPP_M)


def test_mret_to_base_only_knows_mie_zero():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE | MPP_M | MPRV | TW)
    r.step("mret", "", next_pc=BASE)
    assert r.rd(0) is None
    assert r.rd(READABLE & ~MIE) is None
    assert _field(r.rd(MIE)) == "MIE"


def test_mret_to_base_when_target_equals_mepc_is_a_normal_return():
    r = _Run()
    r.set_xmepc(BASE)
    r.wr(MPIE)
    r.step("mret", "", next_pc=BASE)
    assert r.rd(MIE | MPIE) is None
    assert _field(r.rd(MPIE)) == "MIE"


def _nmi_sequence(slot):
    r = _Run(priv="U")
    r.wr(MIE)
    r.irq(slot)
    r.step("mret", "", next_pc=r.ctx.xmepc)
    return r


def test_mret_after_nmi_makes_mpie_and_mpp_unknown():
    r = _nmi_sequence(31)
    for mpie in (0, MPIE):
        for mpp in (0, 0x800, 0x1000, MPP_M):
            assert r.rd(MIE | mpie | mpp) is None
    assert _field(r.rd(0)) == "MIE"  # MIE <- MPIE (1) is still certain
    assert _field(r.rd(MIE | MPRV)) == "MPRV"


def test_mret_after_ordinary_irq_keeps_mpie_and_mpp_known():
    r = _nmi_sequence(7)
    _assert_read(r, MIE | MPIE)


def test_nmi_flag_clears_after_its_mret():
    r = _Run(priv="U")
    r.wr(MIE)
    r.irq(31)
    r.step("mret", "", next_pc=r.ctx.xmepc)
    # next ordinary irq + mret: MPIE/MPP known again
    r.irq(7)
    r.step("mret", "", next_pc=r.ctx.xmepc)
    _assert_read(r, MIE | MPIE)


def test_mret_after_unrecognised_landing_is_all_unknown():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    assert r.rd(0) is None  # state forgotten at the landing
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(0) is None
    assert r.rd(MPIE) is None
    assert r.rd(READABLE) is None
    # the unrecognised flag is consumed: the next mret models normally
    r.wr(MPIE)
    r.step("mret", "", next_pc=0x3000)
    _assert_read(r, MIE | MPIE)


def test_mret_after_unrecognised_landing_to_irq_slot_knows_mie_zero():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    r.step("mret", "", next_pc=BASE + 4 * 7)
    assert r.rd(READABLE & ~MIE) is None
    assert _field(r.rd(MIE)) == "MIE"


# --- dret -------------------------------------------------------------------------


def test_dret_makes_mprv_unknown_only():
    r = _Run()
    r.wr(MIE | MPIE | MPP_M | MPRV | TW)
    r.step("dret", "", next_pc=0x3000)
    for mprv in (0, MPRV):
        assert r.rd(READABLE & ~MPRV | mprv) is None
    assert _field(r.rd(READABLE & ~MPRV & ~MIE)) == "MIE"
    assert _field(r.rd(READABLE & ~MPRV & ~TW)) == "TW"


def test_dret_to_irq_slot_knows_mie_zero_and_tw():
    r = _Run()
    r.wr(MIE | MPIE | MPP_M | MPRV | TW)
    r.step("dret", "", next_pc=BASE + 4 * 7)
    for free in (0, MPIE, 0x1800, MPRV, MPIE | 0x1800 | MPRV):
        assert r.rd(TW | free) is None
    assert _field(r.rd(TW | MIE)) == "MIE"
    assert _field(r.rd(0)) == "TW"


def test_dret_to_base_only_knows_mie_zero():
    r = _Run()
    r.wr(MIE | MPIE | MPP_M | MPRV | TW)
    r.step("dret", "", next_pc=BASE)
    assert r.rd(0) is None
    assert r.rd(READABLE & ~MIE) is None
    assert _field(r.rd(MIE)) == "MIE"


def test_dret_return_to_dpc_equal_to_mepc_is_not_a_trap_landing():
    r = _Run()
    r.set_xmepc(BASE)
    r.wr(MIE | TW)
    r.step("dret", "", next_pc=BASE)
    assert r.rd(MIE | TW) is None  # MIE kept: only MPRV forgotten
    assert _field(r.rd(TW)) == "MIE"


# --- parse_trace integration -------------------------------------------------------


def _write_lines(tmp_path, lines, name):
    p = tmp_path / name
    p.write_text("Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
                 + "".join(lines))
    return p


_W = "1\t1\t80000000\t30031073\tcsrrw\tx0,mstatus,x6\tx6:0x00000008\n"
_R = "{n}\t{n}\t80000004\t300023f3\tcsrrs\tx7,mstatus,x0\tx7=0x{v:08x}\n"
_E = "{n}\t{n}\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"


def test_parse_trace_key_carries_the_field(tmp_path):
    lines = [_W, _R.format(n=2, v=0), _E.format(n=3)]
    info = parse_trace(str(_write_lines(tmp_path, lines, "mst.log")))
    assert info.violations == {"MSTATUS_MISMATCH:csrrs@MIE": 1}


def test_parse_trace_key_for_different_field(tmp_path):
    lines = [_W, _R.format(n=2, v=MIE | MPIE), _E.format(n=3)]
    info = parse_trace(str(_write_lines(tmp_path, lines, "mst2.log")))
    assert info.violations == {"MSTATUS_MISMATCH:csrrs@MPIE": 1}


def test_parse_trace_matching_read_reports_nothing(tmp_path):
    lines = [_W, _R.format(n=2, v=MIE), _E.format(n=3)]
    info = parse_trace(str(_write_lines(tmp_path, lines, "mst_ok.log")))
    assert info.violations == {}


def test_malformed_line_clears_mstatus_model(tmp_path):
    bad = "2\t2\t80000004\n"
    ok = parse_trace(str(_write_lines(
        tmp_path, [_W, bad, _R.format(n=3, v=0), _E.format(n=4)], "mst_bad.log")))
    assert len(ok.parse_errors) == 1
    assert ok.violations == {}
    ctl = parse_trace(str(_write_lines(
        tmp_path, [_W, _R.format(n=2, v=0), _E.format(n=3)], "mst_ctl.log")))
    assert ctl.violations == {"MSTATUS_MISMATCH:csrrs@MIE": 1}


# --- grouping / signature --------------------------------------------------------------


def test_viol_group_per_field():
    assert isa.viol_group("MSTATUS_MISMATCH", "csrrs", {"field": "MIE"}) == "MST:MIE"
    assert isa.viol_group("MSTATUS_MISMATCH", "csrrs", {"field": "MPIE"}) == "MST:MPIE"
    assert isa.viol_group("MSTATUS_MISMATCH", "csrrw", {"field": "MPP"}) == "MST:MPP"
    assert isa.viol_group("MSTATUS_MISMATCH", "csrrs", {}) == "MST:?"
    assert isa.viol_group("MSTATUS_MISMATCH", "csrrs") == "MST:?"


def _record(violations):
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    trace = TraceInfo.empty()
    trace.violations = dict(violations)
    return FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                         trace=trace, deviation=None, parse_errors=[])


def test_mstatus_kind_excluded_from_sys_viol():
    assert "MSTATUS_MISMATCH" in _SCAN_EXCLUDED
    cls, keys = _sys_viol(_record({"MSTATUS_MISMATCH:csrrs@MIE": 5}))
    assert cls == frozenset() and keys == frozenset()


def test_non_excluded_kind_control_still_counts():
    cls, keys = _sys_viol(_record({"XRET_TARGET:mret": 5}))
    assert cls


def _sig(case_id, viol_counts):
    soft = {
        "viol_vec": _ratio_vector(viol_counts),
        "viol_class": None,
        "sys_viol_class": frozenset(),
        "sys_viol_keys": frozenset(),
        "tail_ngrams": {},
        "implicit": 0,
        "assert_set": frozenset(),
        "fatal_tmpl": None,
        "sig_last": None,
    }
    return Signature(case_id=case_id, family="MISMATCH",
                     hard_key=("MISMATCH", "WRONG_RESULT", "K"), soft=soft)


def test_r4_distance_separates_different_mstatus_fields(tmp_path):
    def counts(field):
        lines = [_W, _R.format(n=2, v={"MIE": 0, "MPIE": MIE | MPIE}[field]), _E.format(n=3)]
        return parse_trace(str(_write_lines(tmp_path, lines, field + ".log"))).violations

    a, b, c = counts("MIE"), counts("MPIE"), counts("MIE")
    d_diff = _atom_distance(_Atom(("MISMATCH", "WRONG_RESULT", "K"), [_sig(1, a)]),
                            _Atom(("MISMATCH", "WRONG_RESULT", "K"), [_sig(2, b)]))
    d_same = _atom_distance(_Atom(("MISMATCH", "WRONG_RESULT", "K"), [_sig(1, a)]),
                            _Atom(("MISMATCH", "WRONG_RESULT", "K"), [_sig(3, c)]))
    assert d_same == pytest.approx(0.0)
    assert d_diff > 0.0


# --- public traces: MSTATUS_MISMATCH stays at zero ---------------------------------------

_PUBLIC = Path(__file__).resolve().parents[1] / "testcase/2026_08_20_samples/problem"


def test_public_traces_have_no_mstatus_mismatch():
    traces = sorted(_PUBLIC.glob("benchmark_set_*/case_*/trace.log.gz"))
    assert len(traces) == 32
    for t in traces:
        info = parse_trace(str(t))
        bad = [k for k in info.violations if k.split(":", 1)[0] == "MSTATUS_MISMATCH"]
        assert not bad, (str(t), bad)


# --- round 2 (F1 mret off-landing, F2 stale priv, F3 NMI slot, N2-N5) -------------

ALL_COMBOS = [(a, b, c) for a in (0, MIE) for b in (0, MPIE) for c in (0, MPP_M)]


def _all_unknown_but_tw(r):
    """MIE, MPIE, MPP and MPRV are all unknown; TW (written 0) is still known."""
    for mie, mpie, mpp in ALL_COMBOS:
        for mprv in (0, MPRV):
            assert r.rd(mie | mpie | mpp | mprv) is None
    assert _field(r.rd(TW)) == "TW"


@pytest.mark.parametrize("next_pc", ["fall", None])
@pytest.mark.parametrize("before,after", MRET_CASES)
def test_mret_not_landing_on_base_or_slot_is_an_ordinary_mret(before, after, next_pc):
    # F1: next_pc = pc + 4 (or unknown) is not a control-flow break, but the
    # mret still pops MIE <- MPIE, MPIE <- 1, MPP <- U, MPRV <- 0 (MPP != M).
    r = _Run()
    r.wr(before)
    r.step("mret", "", next_pc=next_pc)
    _assert_read(r, after)


def test_mret_fallthrough_src_is_mret():
    r = _Run()
    r.wr(MPIE)
    r.step("mret", "")
    v = r.rd(MIE | MPIE | 0x800)
    assert v.detail["field"] == "MPP" and v.detail["src"] == "mret"


def test_mret_fallthrough_with_unknown_state_knows_only_the_pushed_values():
    r = _Run()
    r.step("csrrw", "x0,mstatus,x6", reads={})  # unknown operand: forgets everything
    r.step("mret", "")
    assert r.rd(MIE | MPIE) is None  # MIE, MPRV, TW unknown; MPIE <- 1, MPP <- U known
    assert _field(r.rd(MIE)) == "MPIE"
    assert _field(r.rd(MIE | MPIE | MPP_M)) == "MPP"


def test_dret_fallthrough_makes_mprv_unknown_only():
    r = _Run()
    r.wr(MIE | MPIE | MPP_M | MPRV | TW)
    r.step("dret", "")
    for mprv in (0, MPRV):
        assert r.rd(READABLE & ~MPRV | mprv) is None
    assert _field(r.rd(READABLE & ~MPRV & ~MIE)) == "MIE"


def test_mpp_high_bit_alone_is_the_mpp_field():
    # N2: flipping only bit 12 reports MPP, not OTHER.
    r = _Run()
    r.wr(MPP_M)
    assert r.rd(MPP_M) is None
    v = r.rd(MPP_M ^ 0x1000)
    assert _field(v) == "MPP" and v.detail["actual"] == 0x800
    assert _field(r.rd(MPP_M ^ 0x800)) == "MPP"


# F2: stale priv after a landing the model does not recognise.


def _stale_u_then_trap(landing=0x9000):
    r = _Run(priv="U")
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=landing)  # reset-like jump out of U mode
    r.trap()
    return r


def test_stale_priv_trap_entry_leaves_mpp_unknown():
    r = _stale_u_then_trap()
    assert r.ctx.mst_priv_unk
    for mpie in (0, MPIE):
        for mpp in (0, MPP_M):
            assert r.rd(mpie | mpp) is None


def test_unrecognised_landing_control_known_priv_checks_mpp():
    # control for the test above: without the unrecognised landing MPP is checked
    r = _Run(priv="U")
    r.wr(MIE)
    r.trap()
    assert not r.ctx.mst_priv_unk
    assert r.rd(MPIE) is None
    assert _field(r.rd(MPIE | MPP_M)) == "MPP"


def test_stale_priv_mret_with_known_mpp_clears_the_flag():
    r = _stale_u_then_trap()
    r.set_xmepc(0x3000)
    r.wr(MPIE | MPP_M)  # handler writes MPP = M: now known
    r.step("mret", "", next_pc=0x3000)
    assert not r.ctx.mst_priv_unk
    r.wr(MIE)
    r.trap()
    assert r.rd(MPIE | MPP_M) is None
    assert _field(r.rd(MPIE)) == "MPP"


def test_stale_priv_mret_with_unknown_mpp_keeps_the_flag():
    r = _stale_u_then_trap()
    r.set_xmepc(0x3000)
    r.step("mret", "", next_pc=0x3000)  # MPP never became known
    assert r.ctx.mst_priv_unk
    r.wr(MIE)
    r.trap()
    for mpp in (0, MPP_M):
        assert r.rd(MPIE | mpp) is None


def test_unknown_mtvec_landing_sets_the_stale_priv_flag():
    r = _Run(base=None, priv="U")
    r.wr(MIE)
    r.trap(landing=0x9000)
    assert r.ctx.mst_priv_unk and r.ctx.mst_unk


# F3: dret / unrecognised-state mret landing in the NMI slot.


def test_dret_to_nmi_slot_then_mret_makes_mpie_and_mpp_unknown():
    r = _Run()
    r.wr(MIE | MPIE)
    r.step("dret", "", next_pc=BASE + 4 * 31)
    r.step("mret", "", next_pc=0x3000)
    _all_unknown_but_tw(r)


def test_dret_to_ordinary_slot_then_mret_makes_mpie_and_mpp_known():
    # control for the NMI test: slot 7 is no NMI, MPIE <- 1 and MPP <- U stay known
    r = _Run()
    r.wr(MIE | MPIE)
    r.step("dret", "", next_pc=BASE + 4 * 7)
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(MIE | MPIE) is None
    assert _field(r.rd(MIE | MPIE | MPP_M)) == "MPP"
    assert _field(r.rd(MIE)) == "MPIE"


def test_unk_state_mret_to_nmi_slot_then_mret_makes_mpie_and_mpp_unknown():
    r = _Run()
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)  # unrecognised landing: mst_unk
    r.step("mret", "", next_pc=BASE + 4 * 31)
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(0) is None
    for mie, mpie, mpp in ALL_COMBOS:
        assert r.rd(mie | mpie | mpp) is None


def test_unk_state_mret_to_ordinary_slot_then_mret_models_mpie_known():
    r = _Run()
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    r.step("mret", "", next_pc=BASE + 4 * 7)
    r.step("mret", "", next_pc=0x3000)
    assert _field(r.rd(0)) == "MPIE"


# N3: mret into the NMI slot, then the NMI handler's mret.


def test_mret_to_nmi_slot_then_mret_makes_mpie_and_mpp_unknown():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE)
    r.step("mret", "", next_pc=BASE + 4 * 31)
    r.step("mret", "", next_pc=0x3000)
    for mie, mpie, mpp in ALL_COMBOS:
        if mie:
            assert r.rd(mie | mpie | mpp) is None
    assert _field(r.rd(0)) == "MIE"  # MIE <- MPIE (1, from the slot entry) is certain


# N4: an unrecognised landing followed by a dret into a slot ends the unk state.


def test_unrecognised_landing_then_dret_to_slot_then_mret_models_normally():
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    r.step("dret", "", next_pc=BASE + 4 * 7)
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(MPIE) is None
    assert _field(r.rd(0)) == "MPIE"
    assert _field(r.rd(MPIE | MPP_M)) == "MPP"


# N5: mtvec unknown at the landing, known afterwards.


def test_unknown_mtvec_landing_then_known_mtvec_then_mret_is_all_unknown():
    r = _Run(base=None)
    r.wr(MIE)
    r.trap(landing=0x9000)
    r.step("csrrw", "x0,mtvec,x6", reads={"x6": BASE | 1})
    assert r.ctx.mtvec_base == BASE
    r.set_xmepc(0x3000)
    r.step("mret", "", next_pc=0x3000)
    assert r.rd(0) is None
    assert r.rd(MPIE) is None
    assert r.rd(MIE | MPIE) is None
    assert r.rd(READABLE) is None


# parse_trace on real line formats: trap entry, handler read, mret, read.


def _ln(pc, mn, ops="", regs=""):
    return f"1\t1\t{pc:08x}\t00000013\t{mn}\t{ops}\t{regs}\n"


def _trap_mret_lines(handler_val, after_val):
    ecall = 0x80001008
    return [
        _ln(0x80001000, "csrrw", "x0,mtvec,x6", "x6:0x80000001"),
        _ln(0x80001004, "csrrw", "x0,mstatus,x5", "x5:0x00000008"),
        _ln(ecall, "ecall"),
        _ln(0x80000000, "csrrs", "x7,mstatus,x0", f"x7=0x{handler_val:08x}"),
        _ln(0x80000004, "mret"),
        _ln(ecall, "csrrs", "x7,mstatus,x0", f"x7=0x{after_val:08x}"),
        _ln(ecall + 4, "addi", "x0,x0,0", "x0=0x0"),
    ]


def test_parse_trace_trap_entry_mret_and_read_match(tmp_path):
    # trap: MPIE <- 1, MIE <- 0, MPP <- M;  mret: MIE <- 1, MPIE <- 1, MPP <- U
    lines = _trap_mret_lines(MPIE | MPP_M, MIE | MPIE)
    info = parse_trace(str(_write_lines(tmp_path, lines, "tm_ok.log")))
    assert info.violations == {}


def test_parse_trace_trap_entry_mret_wrong_mpp_after_mret(tmp_path):
    lines = _trap_mret_lines(MPIE | MPP_M, MIE | MPIE | 0x800)
    info = parse_trace(str(_write_lines(tmp_path, lines, "tm_bad.log")))
    assert info.violations == {"MSTATUS_MISMATCH:csrrs@MPP": 1}


def test_parse_trace_trap_entry_wrong_mpie_in_handler(tmp_path):
    lines = _trap_mret_lines(MPP_M, MIE | MPIE)
    info = parse_trace(str(_write_lines(tmp_path, lines, "tm_bad2.log")))
    # the handler read re-syncs the model to the wrong MPIE, so the mret's
    # MIE <- MPIE then disagrees with the (correct) final read as well
    assert info.violations == {"MSTATUS_MISMATCH:csrrs@MPIE": 1,
                               "MSTATUS_MISMATCH:csrrs@MIE": 1}


# --- second cold read (R2-1..R2-6) -------------------------------------------


def test_unk_branch_mret_keeps_priv_unk_flag_behaviour():
    # R2-1: an mret on the mst_unk branch must not clear mst_priv_unk, else the
    # stale ctx.mpp is trusted for the next trap's MPP
    r = _Run()
    r.set_xmepc(0x3000)
    r.wr(MPIE | MPP_M)                          # ctx.mpp = M (stale after the reset below)
    r.step("addi", "x1,x1,1", next_pc=0x9000)   # reset-like landing: flag + mst_unk
    r.step("mret", "", next_pc=0x3000)          # unk branch; real MPP is the reset value
    r.wr(MIE)
    r.trap()
    for mpp in (0, MPP_M):
        assert r.rd(MPIE | mpp) is None


def test_unk_branch_mret_keeps_priv_unk_flag_state():
    # R2-1 (state): same sequence, the flag itself survives the mret
    r = _Run(priv="U")
    r.set_xmepc(0x3000)
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    r.wr(MPIE | MPP_M)
    r.step("mret", "", next_pc=0x3000)
    assert r.ctx.mst_priv_unk


def test_nmi_mret_keeps_priv_unk_flag():
    # R2-2: the `not ctx.mst_nmi` in the flag-clearing condition; an NMI's mret
    # restores from mstack, so a handler MPP write must not make priv trusted
    r = _Run(priv="U")
    r.wr(MIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)   # unrecognised landing -> flag
    r.irq(31)                                    # NMI entry
    assert r.ctx.mst_nmi and r.ctx.mst_priv_unk
    r.wr(MPIE | MPP_M)                           # handler writes MPP: model + ctx.mpp known
    r.step("mret", "", next_pc=0x3000)           # NMI mret restores from mstack
    assert r.ctx.mst_priv_unk


def test_fallthrough_mret_with_unknown_mtvec_is_ordinary():
    # R2-3: the `brk and` in `if brk and base is None`; a fallthrough mret with
    # no mtvec base is still an ordinary mret
    r = _Run(base=None)
    r.wr(MPIE)
    r.step("mret", "")
    _assert_read(r, MIE | MPIE)
    assert not r.ctx.mst_unk


def test_fallthrough_mret_after_unrecognised_landing_forgets_everything():
    # R2-4: a fallthrough mret in the mst_unk state takes the unk branch
    r = _Run()
    r.wr(MPIE)
    r.step("addi", "x1,x1,1", next_pc=0x9000)
    r.step("mret", "")
    assert r.rd(0) is None
    assert r.rd(READABLE) is None
    assert not r.ctx.mst_unk


def _nested_handler_mret_run(slot):
    """irq in M mode, its mret, then the (outer) handler's own mret, then an
    ecall entry: the second mret decides priv for the trap's MPP.  Returns
    the run before the ecall and the priv the second mret left behind."""
    r = _Run(priv="M")
    r.wr(MIE)
    r.irq(slot)
    r.step("mret", "", next_pc=0x3000)
    r.step("mret", "", next_pc=0x3000)
    priv = r.ctx.priv
    r.trap()
    return r, priv


def test_nmi_mret_restores_mpp_from_mstack_so_priv_is_unknown():
    # R2-5: an NMI's mret must leave ctx.mpp unknown, so the handler's own mret
    # does not turn priv into U and the next trap's MPP is not reported
    r, priv = _nested_handler_mret_run(31)
    assert priv is None
    for mpp in (0, MPP_M):
        assert r.rd(MPIE | mpp) is None


def test_ordinary_irq_mret_chain_still_predicts_mpp():
    # R2-5 control: same sequence with an ordinary irq: mret zeroes ctx.mpp,
    # the second mret gives U, and a wrong MPP is reported
    r, priv = _nested_handler_mret_run(7)
    assert priv == "U"
    assert _field(r.rd(MPIE | MPP_M)) == "MPP"


def _ecall_cause_run(*, stale):
    r = _Run(priv="U")
    r.wr(MIE)
    if stale:
        r.step("addi", "x1,x1,1", next_pc=0x9000)  # reset out of U: priv stale, flag set
    r.trap()
    return r.step("csrrs", "x7,mcause,x0", writes={"x7": 11})


def test_ecall_cause_with_stale_priv_accepts_both_causes():
    # R2-6: with mst_priv_unk set, ecall mcause 11 is not reported even if the
    # remembered priv is U
    assert _ecall_cause_run(stale=True) is None


def test_ecall_cause_from_known_user_priv_reports_machine_cause():
    # R2-6 control: flag clear, priv U, mcause 11 is reported
    v = _ecall_cause_run(stale=False)
    assert v is not None and v.kind == "TRAP_MCAUSE"


# --- third cold read (R3-1, R3-6) --------------------------------------------

def _ecall_cause_run_m(*, stale, cause):
    r = _Run(priv="M")
    r.wr(MIE)
    if stale:
        r.step("addi", "x1,x1,1", next_pc=0x9000)  # unrecognised landing: flag set
    r.trap()
    return r.step("csrrs", "x7,mcause,x0", writes={"x7": cause})


def test_ecall_cause_8_with_stale_m_priv_is_accepted():
    # R3-1: remembered priv M but mst_priv_unk set: mcause 8 (really from U) is not reported
    assert _ecall_cause_run_m(stale=True, cause=8) is None


def test_ecall_cause_8_from_known_machine_priv_is_reported():
    # R3-1 control: flag clear, priv M: mcause 8 is reported
    v = _ecall_cause_run_m(stale=False, cause=8)
    assert v is not None and v.kind == "TRAP_MCAUSE"


def _ecall_after_nmi_run(cause):
    r = _Run(priv="U")
    r.wr(MIE)
    r.irq(31)
    r.step("mret", "", next_pc=r.ctx.xmepc)
    assert r.ctx.priv == "U"
    r.trap()
    return r.step("csrrs", "x7,mcause,x0", writes={"x7": cause})


def test_ecall_after_nmi_mret_from_user_reads_cause_8_only():
    # R3-6: NMI interrupts U, its mret leaves priv U, so the next ecall accepts only 8
    assert _ecall_after_nmi_run(8) is None
    v = _ecall_after_nmi_run(11)
    assert v is not None and v.kind == "TRAP_MCAUSE"
