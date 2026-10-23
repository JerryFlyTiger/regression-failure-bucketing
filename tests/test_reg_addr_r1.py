"""REG_ADDR round-1 cold-read fixes (PLAN.md 4c F1/F2/F3/F4/F6)."""
from __future__ import annotations

import pytest

from rfb import isa
from rfb import parse_trace as pt
from rfb.cluster import D_VIOL, _Atom, _atom_distance
from rfb.record import FailureRecord, Instr, RegrInfo, SimInfo
from rfb.signature import _sys_viol, signature
from tests import test_parse_trace_legacy_equiv as leq

PC = 0x80000000


def _ins(mn, insn, ops, reads, writes, pc=PC):
    return Instr(index=0, pc=pc, size=2 if len(insn) <= 4 else 4, mnemonic=mn,
                 operands=ops, reads=reads, writes=writes, pa=None, store=None, load=None,
                 insn=insn)


def _chk(ins, next_pc, ctx=None):
    v, err = isa.check_safe(None, ins, next_pc, ctx or isa.ScanContext())
    assert err is None
    return v


@pytest.fixture(autouse=True)
def fresh_caches():
    for c in (isa._ra_cache, isa._ra_fields, isa._ra_pool):
        c.clear()
    yield
    for c in (isa._ra_cache, isa._ra_fields, isa._ra_pool):
        c.clear()


# --- F1 ---------------------------------------------------------------------

def _atom(tmp_path, name, rs_ops, reads, n=3, insn="00b50533", mn="add"):
    lines = []
    pc = PC
    for _ in range(n):
        lines.append(leq.mk_line(pc, insn, mn, rs_ops, reads))
        pc += 4
    path = leq.write_trace(tmp_path, name, lines, gz=False)
    trace = pt.parse_trace(path, None)
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    rec = FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                        trace=trace, deviation=None, parse_errors=[])
    s = signature(rec)
    return rec, _Atom(s.hard_key, [s])


RS1_BUG = ("x10,x5,x11", "x5:0x1 x11:0x2 x10=0x3")
RS2_BUG = ("x10,x10,x7", "x10:0x1 x7:0x2 x10=0x3")


def test_f1_scan_key_carries_field(tmp_path):
    rec, _ = _atom(tmp_path, "a", *RS1_BUG)
    assert any(k.startswith("REG_ADDR_RS:add") for k in rec.trace.violations)
    cls, keys = _sys_viol(rec)
    assert ("REG_ADDR_RS", "RA_RS1:I") in cls
    assert ("REG_ADDR_RS", "add") in keys  # keys stay field-free


def test_f1_different_fields_not_linked(tmp_path):
    ra, a = _atom(tmp_path, "a", *RS1_BUG)
    rb, b = _atom(tmp_path, "b", *RS2_BUG)
    assert _sys_viol(ra)[0] != _sys_viol(rb)[0]
    assert not (a.vcls & b.vcls)
    assert _atom_distance(a, b) != D_VIOL


def test_f1_same_field_linked(tmp_path):
    _, a = _atom(tmp_path, "a", *RS2_BUG)
    _, b = _atom(tmp_path, "b", "x10,x10,x9", "x10:0x1 x9:0x2 x10=0x3")
    assert _atom_distance(a, b) == D_VIOL


def test_f1_scan_class_equals_hard_key_group(tmp_path):
    rec, _ = _atom(tmp_path, "a", *RS2_BUG)
    v = rec.trace.first_violation
    assert (v.kind, isa.viol_group(v.kind, v.mnemonic, v.detail)) in _sys_viol(rec)[0]


# --- F2 ---------------------------------------------------------------------

X0_READ = {"x0": 0, "x11": 2}   # rs1 (x10) printed as x0: a "cleared" rs1_addr


def test_f2_x0_read_before_trap_entry_not_reported():
    assert _chk(_ins("add", "00b50533", "x10,x0,x11", X0_READ, {"x10": 2}), 0x80001000) is None


def test_f2_x0_read_with_fallthrough_reported():
    v = _chk(_ins("add", "00b50533", "x10,x0,x11", X0_READ, {"x10": 2}), PC + 4)
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_f2_unknown_next_pc_still_checked():
    assert _chk(_ins("add", "00b50533", "x10,x0,x11", X0_READ, {"x10": 2}), None) is not None


def test_f2_taken_branch_field_bug_still_reported():
    # beq x10,x11 with rs2 printed as x5; equal values so it is taken to pc+8.
    ins = _ins("beq", "00b50463", "x10,x5,0x80000008", {"x10": 1, "x5": 1}, {"x0": 0})
    v = _chk(ins, PC + 8)
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_f2_branch_to_other_target_is_flow_break():
    ins = _ins("beq", "00b50463", "x10,x5,0x80000008", {"x10": 1, "x5": 1}, {"x0": 0})
    ctx = isa.ScanContext()
    ctx.trap_entries = {0x80001000}  # keeps check() quiet; REG_ADDR must skip it too
    assert _chk(ins, 0x80001000, ctx) is None
    assert _chk(ins, PC + 8) is not None


# --- F3 ---------------------------------------------------------------------

def test_f3_equal_entries_are_shared():
    a = isa._ra_entry("lw", "ff412383")
    b = isa._ra_entry("lw", "00412383")
    assert a == b and a is b


def test_f3_pool_bounded(monkeypatch):
    monkeypatch.setattr(isa, "_RA_CAP", 4)
    dicts = {"pool": isa._ra_pool, "cache": isa._ra_cache, "fields": isa._ra_fields}
    peak = dict.fromkeys(dicts, 0)
    for i in range(32):
        isa._ra_entry("add", f"{(i << 20) | 0x33:08x}")
        for name, d in dicts.items():
            peak[name] = max(peak[name], len(d))
    # Peak over every insertion, not just the last length; and each dict on its
    # own actually reaches the cap (32 distinct entries > 4), so no bound is vacuous.
    assert peak == dict.fromkeys(dicts, 4)


def test_f3_ra_cap_value():
    # An earlier round claimed to lower this but the change never reached the code.
    assert isa._RA_CAP == 20000


# --- F4 ---------------------------------------------------------------------

def test_f4_rs1_equals_rs2_unnamed_field_not_reported():
    # add x10,x10,x10 (rs1 = rs2 = x10); printed reads x10 and x7.
    assert _chk(_ins("add", "00a50533", "x10,x10,x7", {"x10": 1, "x7": 2}, {"x10": 3}), PC + 4) is None


def test_f4_both_fields_wrong_still_reported_as_rs():
    v = _chk(_ins("add", "00b50533", "x10,x5,x7", {"x5": 1, "x7": 2}, {"x10": 3}), PC + 4)
    assert v is not None and v.detail["field"] == "RS"


def test_f4_compressed_unaffected():
    v = _chk(_ins("c.and", "8cf5", "x9,x5", {"x9": 1, "x5": 3}, {"x9": 1}), PC + 2)
    assert v is not None


# --- F6 ---------------------------------------------------------------------

def test_f6_compressed_and_32bit_do_not_share_key():
    assert isa._ra_entry("add", "0033")[1] is None       # 16-bit "add": nothing to check
    assert isa._ra_entry("add", "00000033")[1] is not None


# --- Round 2 (R2-1 .. R2-4) ---------------------------------------------------

def _rec_of(tmp_path, name, lines):
    path = leq.write_trace(tmp_path, name, lines, gz=False)
    trace = pt.parse_trace(path, None)
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    return FailureRecord(case_id=1, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                         trace=trace, deviation=None, parse_errors=[])


def test_r2_1_rd_scan_path_key_and_class(tmp_path):
    # add x10,x10,x11 encoded (00b50533) but rd printed as x12.
    lines = [leq.mk_line(PC + 4 * i, "00b50533", "add", "x12,x10,x11", "x10:0x1 x11:0x2 x12=0x3")
             for i in range(3)]
    rec = _rec_of(tmp_path, "rd", lines)
    assert rec.trace.violations == {"REG_ADDR_RD:add@RD": 3}
    v = rec.trace.first_violation
    assert v.kind == "REG_ADDR_RD"
    grp = isa.viol_group(v.kind, v.mnemonic, v.detail)
    assert grp == "RA_RD:I"
    cls, keys = _sys_viol(rec)
    assert ("REG_ADDR_RD", "RA_RD:I") in cls
    assert ("REG_ADDR_RD", "add") in keys
    assert (v.kind, grp) in cls


def test_r2_1_compressed_key_has_no_field_suffix(tmp_path):
    lines = [leq.mk_line(PC + 2 * i, "8cf5", "c.and", "x9,x5", "x9:0x1 x5:0x3 x9=0x1")
             for i in range(3)]
    rec = _rec_of(tmp_path, "cand", lines)
    assert "REG_ADDR_RS:c.and" in rec.trace.violations
    assert all("@" not in k for k in rec.trace.violations if k.startswith("REG_ADDR"))
    assert ("REG_ADDR_RS", "RA_RS:c.and") in _sys_viol(rec)[0]


# --- R2-2: _flow_break gate for jumps / branches -----------------------------

TGT = 0x8000165C


@pytest.mark.parametrize("label,ins,npc,kind,field", [
    # jalr x2,118(x26) = 076d0167, rs1 renamed to x29 / rd renamed to x29
    ("jalr_rs1", _ins("jalr", "076d0167", "x2,118(x29)", {"x29": TGT}, {"x2": PC + 4}),
     (TGT + 118) & ~1, "REG_ADDR_RS", "RS1"),
    ("jalr_rd", _ins("jalr", "076d0167", "x29,118(x26)", {"x26": TGT}, {"x29": PC + 4}),
     (TGT + 118) & ~1, "REG_ADDR_RD", "RD"),
    # jal x0,0x8000fc5a (30f0f06f at 0x8000f000), rd renamed to x29.
    # Not a real line: the immediate is not compared, and the real target of
    # 30f0f06f at 0x8000f000 is 0x8001EB0E, not 0x8000FC5A.  The test only
    # needs next_pc to equal the printed target text.
    ("jal_rd", _ins("jal", "30f0f06f", "x29,8000fc5a", {}, {"x29": 0x8000F004}, pc=0x8000F000),
     0x8000FC5A, "REG_ADDR_RD", "RD"),
    ("c.jr", _ins("c.jr", "8e02", "x29", {"x29": 0x8000F30A}, {}), 0x8000F30A & ~1,
     "REG_ADDR_RS", "RS"),
    ("c.jalr", _ins("c.jalr", "9d02", "x29", {"x29": 0x8000DAA1}, {"x1": PC + 2}),
     0x8000DAA1 & ~1, "REG_ADDR_RS", "RS"),
    # c.beqz rows: not real lines.  d169 encodes offset -62, the printed target
    # uses +6 and the immediate is not compared; the check uses the printed text.
    ("c.beqz_taken", _ins("c.beqz", "d169", "x29,8000019a", {"x29": 0}, {}, pc=0x80000194),
     0x8000019A, "REG_ADDR_RS", "RS"),
    ("c.beqz_not_taken", _ins("c.beqz", "d169", "x29,8000019a", {"x29": 1}, {}, pc=0x80000194),
     0x80000196, "REG_ADDR_RS", "RS"),
    ("bne_taken", _ins("bne", "01631763", "x29,x22,80004f0e", {"x29": 3, "x22": 1}, {"x0": 0},
                       pc=0x80004F00), 0x80004F0E, "REG_ADDR_RS", "RS1"),
])
def test_r2_2_renamed_jump_or_branch_to_target_still_reported(label, ins, npc, kind, field):
    v = _chk(ins, npc)
    assert v is not None, label
    assert v.kind == kind, label
    assert v.detail["field"] == field, label


def test_r2_2_jump_to_trap_entry_not_reported():
    # jalr x2,118(x0): rs1 printed as x0 (a cleared rs1_addr), then a trap entry.
    ins = _ins("jalr", "076d0167", "x2,118(x0)", {"x0": 0}, {"x2": PC + 4})
    ctx = isa.ScanContext()
    ctx.trap_entries = {0x80000100}
    assert _chk(ins, 0x80000100, ctx) is None
    # Same instruction landing on the computed target is judged.
    v = _chk(ins, 118)
    assert v is not None and v.kind == "REG_ADDR_RS"


# --- R2-4: viol_group default field ------------------------------------------

def test_r2_4_viol_group_defaults_without_detail():
    assert isa.viol_group("REG_ADDR_RD", "add", None) == "RA_RD:I"
    assert isa.viol_group("REG_ADDR_RS", "add", None) == "RA_RS:I"
    assert isa.viol_group("REG_ADDR_RD", "c.and", None) == "RA_RD:c.and"
