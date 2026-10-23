"""REG_ADDR: printed registers vs. the register fields of the Insn bits."""
from __future__ import annotations

import random
import re

import pytest

from rfb import isa
from rfb import parse_trace as pt
from rfb.record import Instr
from tests import test_parse_trace_legacy_equiv as leq


def _ins(mn, insn, ops, reads, writes):
    return Instr(index=0, pc=0x80000000, size=2 if len(insn) <= 4 else 4, mnemonic=mn,
                 operands=ops, reads=reads, writes=writes, pa=None, store=None, load=None,
                 insn=insn)


def _v(ins):
    v, err = isa.check_safe(None, ins, None, isa.ScanContext())
    assert err is None
    return v


@pytest.mark.parametrize("mn,insn,ops,reads,writes", [
    ("c.and", "8cf5", "x9,x13", {"x9": 1, "x13": 3}, {"x9": 1}),
    ("c.mv", "8522", "x10,x8", {"x0": 0, "x8": 5}, {"x10": 5}),        # expansion's x0 read
    ("c.addi4spn", "0848", "x10,x2,20", {}, {"x10": 20}),               # sp not printed
    ("beq", "00b50463", "x10,x11,8", {"x10": 1, "x11": 2}, {"x0": 0}),  # branch prints x0=
    ("lw", "ff412383", "x0,-12(x2)", {"x2": 0x100}, {"x0": 0}),        # faulting load
    ("add", "00b50533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3}),
])
def test_tracer_conventions_are_clean(mn, insn, ops, reads, writes):
    assert _v(_ins(mn, insn, ops, reads, writes)) is None


def test_compressed_rs2_field_bug():
    v = _v(_ins("c.and", "8cf5", "x9,x5", {"x9": 1, "x5": 3}, {"x9": 1}))
    assert v.kind == "REG_ADDR_RS"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "RA_RS:c.and"


def test_compressed_rs2_mapped_to_x0_is_flagged():
    v = _v(_ins("c.and", "8ce1", "x9,x0", {"x9": 1, "x0": 0}, {"x9": 0}))
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_32bit_rd_and_rs2_groups_name_the_field():
    rd = _v(_ins("add", "00b50533", "x11,x10,x11", {"x10": 1, "x11": 2}, {"x11": 3}))
    assert isa.viol_group(rd.kind, rd.mnemonic, rd.detail) == "RA_RD:I"
    rs = _v(_ins("add", "00b50533", "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 3}))
    assert isa.viol_group(rs.kind, rs.mnemonic, rs.detail) == "RA_RS2:I"


def test_bits_not_matching_mnemonic_skip():
    assert _v(_ins("c.and", "0001", "x9,x5", {"x9": 1, "x5": 3}, {"x9": 1})) is None


def test_insn_not_serialized():
    assert "insn" not in _ins("add", "00b50533", "", {}, {}).to_dict()


def _seq_lines(rng, n):
    """Fallthrough add lines (pc += 4) whose printed rs2 is random.  Measured on
    a single 3-line trace: r=0 and r=10 give WRONG_RESULT, r=11 gives
    READ_MISMATCH, the other 29 values give REG_ADDR_RS@RS2.
    r=0: check() treats x0 as 0, so the result does not match (expected 1,
    actual 3) and WRONG_RESULT is reported first.
    r=10: ``x10:0x1`` and ``x10:0x2`` collapse into one key of the ``reads``
    dict (2+2 != 3), so WRONG_RESULT is reported; even if the values matched,
    F5 would not report it.
    r=11: READ_MISMATCH starts from the 2nd line; the 1st line has no earlier
    write to compare against, so it has no violation.
    The ``v is None and _RA_ON`` guard keeps REG_ADDR out when another
    violation is already reported."""
    out = []
    pc = 0x80000000
    for _ in range(n):
        r = rng.randrange(0, 32)
        out.append(leq.mk_line(pc, "00b50533", "add", f"x10,x10,x{r}",
                               f"x10:0x1 x{r}:0x2 x10=0x3"))
        pc += 4
    return out


@pytest.mark.parametrize("trial", range(40))
def test_stream_equiv_with_reg_addr(tmp_path, monkeypatch, trial):
    rng = random.Random(1000 + trial)
    lines, _apc, _atext = leq._gen_random_trace(rng, rng.randrange(1, 40))
    lines += _seq_lines(rng, 20)
    path = leq.write_trace(tmp_path, f"ra_{trial}", lines, gz=trial % 2 == 0)
    mem = leq._run(pt, path, None)
    monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", 0)
    assert leq._run(pt, path, None) == mem


def test_fuzz_does_trigger_reg_addr(tmp_path):
    """REG_ADDR does fire through this pipeline on the trace built here.  The hits
    come from the appended ``_seq_lines`` (fallthrough pcs); the legacy fuzz
    part alone yields none for the fixed seeds used here: its random next_pc
    looks like a trap after every line and REG_ADDR skips those, but the last
    line (next_pc None) is still checked and about 0.47% of fuzz traces (14 of
    3000) fire there; a 200-trace test changing seed fails about two thirds of
    the time."""
    hits = 0
    for trial in range(40):
        rng = random.Random(1000 + trial)
        lines, _a, _b = leq._gen_random_trace(rng, rng.randrange(1, 40))
        lines += _seq_lines(rng, 20)
        r = leq._run(pt, leq.write_trace(tmp_path, f"t{trial}", lines, gz=False), None)
        if r[0] == "ok":
            hits += sum(n for k, n in r[1]["violations"].items() if k.startswith("REG_ADDR"))
    assert hits > 0


# ---------------------------------------------------------------------------
# Gates, caches and field handling (each pins one mutation).
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_caches():
    isa._ra_cache.clear()
    isa._ra_fields.clear()
    yield
    isa._ra_cache.clear()
    isa._ra_fields.clear()


def test_32bit_opcode_gate_mnemonic_vs_bits(fresh_caches):
    # Bits are a jal (opcode 0x6f) but the mnemonic column says add: the
    # decode does not match the printed mnemonic, so nothing is checked.
    assert _v(_ins("add", "30f0f06f", "x9,x5,x7", {"x5": 1, "x7": 2}, {"x9": 3})) is None
    # Same registers on a bit pattern that really is an add are reported.
    v = _v(_ins("add", "007184b3", "x9,x5,x7", {"x5": 1, "x7": 2}, {"x9": 3}))
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_cache_redecodes_when_mnemonic_changes(fresh_caches):
    insn = "00b50533"
    # First printed as a jal: bits do not match, entry has no register sets.
    assert _v(_ins("lui", insn, "x10,0x1", {}, {"x10": 0x1000})) is None
    # Same Insn string now printed as add with a wrong rs2: must be re-decoded.
    v = _v(_ins("add", insn, "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 3}))
    assert v is not None and v.kind == "REG_ADDR_RS"
    # And back again to the mismatching mnemonic: still skipped.
    assert _v(_ins("lui", insn, "x10,0x1", {}, {"x10": 0x1000})) is None


def test_32bit_second_level_key_keeps_rs2(fresh_caches):
    # Two adds identical except rs2 (x11 vs x12).  The second must not reuse
    # the first one's decode through the masked-immediate key.
    assert _v(_ins("add", "00b50533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3})) is None
    v = _v(_ins("add", "00c50533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3}))
    assert v is not None and v.kind == "REG_ADDR_RS"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "RA_RS2:I"
    # Reverse order, fresh caches: the correct one is still clean.
    isa._ra_cache.clear()
    isa._ra_fields.clear()
    assert _v(_ins("add", "00c50533", "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 3})) is None


def test_32bit_second_level_key_keeps_rs1_and_rd(fresh_caches):
    assert _v(_ins("add", "00b50533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3})) is None
    # rs1 = x12, rs2 = x11 (00b60533) with x10 printed as rs1: only rs1 is wrong
    v = _v(_ins("add", "00b60533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3}))
    assert v is not None and v.kind == "REG_ADDR_RS"
    # rd = x12 (00b50633) printed as x10
    v = _v(_ins("add", "00b50633", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3}))
    assert v is not None and v.kind == "REG_ADDR_RD"


def test_csr_imm_form_has_no_rs1():
    # csrrwi x23,mscratch,18 (34095bf3): bits 19:15 are the immediate 18.
    assert _v(_ins("csrrwi", "34095bf3", "x23,mscratch,18", {}, {"x23": 0})) is None
    # A printed read of x18 would be the zimm mistaken for rs1: flagged.
    v = _v(_ins("csrrwi", "34095bf3", "x23,mscratch,18", {"x18": 0}, {"x23": 0}))
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_csr_reg_form_checks_rs1():
    # csrrw x0,misa,x22 (301b1073)
    assert _v(_ins("csrrw", "301b1073", "x0,misa,x22", {"x22": 1}, {"x0": 0})) is None
    v = _v(_ins("csrrw", "301b1073", "x0,misa,x23", {"x23": 1}, {"x0": 0}))
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_six_column_line_carries_insn(tmp_path):
    ok = leq.mk_line(0x80000000, "00b50533", "add", regs=" x10:0x1 x11:0x2 x10=0x3",
                     drop_operands_field=True)
    bad = leq.mk_line(0x80000004, "00b50533", "add", regs=" x10:0x1 x12:0x2 x10=0x3",
                      drop_operands_field=True)
    ti = pt.parse_trace(leq.write_trace(tmp_path, "six", [ok, bad], gz=False), None)
    assert [i.insn for i in ti.head] == ["00b50533", "00b50533"]
    assert ti.first_violation is not None and ti.first_violation.kind == "REG_ADDR_RS"
    assert ti.first_violation.index == 1


def test_seven_column_line_carries_insn(tmp_path):
    ok = leq.mk_line(0x80000000, "00b50533", "add", "x10,x10,x11", " x10:0x1 x11:0x2 x10=0x3")
    ti = pt.parse_trace(leq.write_trace(tmp_path, "seven", [ok], gz=False), None)
    assert [i.insn for i in ti.head] == ["00b50533"]


# ---------------------------------------------------------------------------
# Table-driven: one real public-set line per mnemonic in the decode tables
# (tab columns Insn / mnemonic / operands / registers, copied from
# testcase/*/trace.log.gz).  c.j has neither a read nor a write register, so
# its clean line is checked but there is nothing to rename (that case is
# skipped); a printed x1= write on a c.j is covered by a dedicated test below.
# ---------------------------------------------------------------------------

_REAL = [
    ('add', '007184b3', 'x9,x3,x7', '  x3:0x00000000  x7:0xffe24190  x9=0xffe24190'),
    ('addi', '00000013', 'x0,x0,0', '  x0:0x00000000  x0=0x00000000'),
    ('and', '0018fc33', 'x24,x17,x1', ' x17:0x80000000  x1:0x00000000 x24=0x00000000'),
    ('andi', 'f8827c13', 'x24,x4,-120', '  x4:0x7fffffff x24=0x7fffff88'),
    ('auipc', '00027997', 'x19,0x27', ' x19=0x80027128'),
    ('beq', '060b0c63', 'x22,x0,8000f9f4', ' x22:0x00000000  x0:0x00000000  x0=0x00000000'),
    ('bge', '036cd163', 'x25,x22,80000a1a', ' x25:0x8001c0b3 x22:0x00013000  x0=0x00000000'),
    ('bgeu', '008af263', 'x21,x8,80000c3e', ' x21:0x0000015b  x8:0x001fffff  x0=0x00000000'),
    ('blt', 'ffeec6e3', 'x29,x30,800001b0', ' x29:0x00000000 x30:0x0000000e  x0=0x00000000'),
    ('bltu', 'ffbf6ee3', 'x30,x27,8000097e', ' x30:0x00000000 x27:0x0000000b  x0=0x00000000'),
    ('bne', '01631763', 'x6,x22,80004f0e', '  x6:0x00000003 x22:0x00000001  x0=0x00000000'),
    ('c.add', '9ad6', 'x21,x21', ' x21:0x49648224 x21:0x49648224 x21=0x92c90448'),
    ('c.addi', '0b0d', 'x22,3', ' x22:0x00030000 x22=0x00030003'),
    ('c.addi16sp', '6119', 'x2,384', '  x2:0x00000000  x2=0x00000180'),
    ('c.addi4spn', '0d10', 'x12,x2,656', ' x12=0x5296a150'),
    ('c.and', '8e6d', 'x12,x11', ' x12:0x8ffffca9 x11:0x221c5e07 x12=0x021c5c01'),
    ('c.andi', '98fd', 'x9,-1', '  x9:0x80000000  x9=0x80000000'),
    ('c.beqz', 'd169', 'x10,8000019a', ' x10:0x00000000'),
    ('c.bnez', 'e609', 'x12,800009c2', ' x12:0x00000068'),
    ('c.j', 'be1d', '80004dec', ''),
    ('c.jal', '2021', '800001f8', '  x1=0x800001f2'),
    ('c.jalr', '9d02', 'x26', ' x26:0x8000daa1  x1=0x8000f8a0'),
    ('c.jr', '8e02', 'x28', ' x28:0x8000f30a'),
    ('c.li', '4081', 'x1,0', '  x1=0x00000000'),
    ('c.lui', '6091', 'x1,0x4', '  x1=0x00004000'),
    ('c.lw', '42c0', 'x8,4(x13)', ' x13:0x80015784  x8=0x88898a8b PA:0x80015788 store:0x74757677'),
    ('c.lwsp', '4a86', 'x21,64(x2)', '  x2:0x80018c54 x21=0x94959697 PA:0x80018c94 store:0x00000000'),
    ('c.mv', '80d2', 'x1,x20', '  x0:0x00000000 x20:0xffffffff  x1=0xffffffff'),
    ('c.or', '8e55', 'x12,x13', ' x12:0x5296a150 x13:0x8ffffc34 x12=0xdffffd74'),
    ('c.slli', '0b22', 'x22,0x8', ' x22:0x00000300 x22=0x00030000'),
    ('c.srai', '8595', 'x11,0x5', ' x11:0x221c5e07 x11=0x0110e2f0'),
    ('c.srli', '83bd', 'x15,0xf', ' x15:0xfa11812e x15=0x0001f423'),
    ('c.sub', '8c1d', 'x8,x15', '  x8:0x40bf0808 x15:0xfa11812e  x8=0x46ad86da'),
    ('c.sw', 'dd40', 'x8,60(x10)', ' x10:0x8001bb18  x8:0x0000001c PA:0x8001bb54 store:0x0000001c load:0x0000001c'),
    ('c.swsp', 'dad6', 'x21,116(x2)', '  x2:0x80018c54 x21:0x800166f4 PA:0x80018cc8 store:0x800166f4 load:0x800166f4'),
    ('c.xor', '8fb5', 'x15,x13', ' x15:0x00006000 x13:0xff000000 x15=0xff006000'),
    ('csrrc', '340f3773', 'x14,mscratch,x30', ' x30:0x0000000e x14=0x00000012'),
    ('csrrci', '3400fff3', 'x31,mscratch,1', ' x31=0x00000010'),
    ('csrrs', '30002b73', 'x22,mstatus,x0', '  x0:0x00000000 x22=0x00001888'),
    ('csrrsi', '3012e173', 'x2,misa,5', '  x2=0x40101104'),
    ('csrrw', '301b1073', 'x0,misa,x22', ' x22:0x40001104  x0=0x00000000'),
    ('csrrwi', '34095bf3', 'x23,mscratch,18', ' x23=0x00000000'),
    ('div', '03a9cd33', 'x26,x19,x26', ' x19:0x80026e7c x26:0x8000165c x26=0x00000000'),
    ('divu', '035c5733', 'x14,x24,x21', ' x24:0x00000000 x21:0x49648224 x14=0x00000000'),
    ('jal', '30f0f06f', 'x0,8000fc5a', '  x0=0x00000000'),
    ('jalr', '076d0167', 'x2,118(x26)', ' x26:0x8000165c  x2=0x8000018c'),
    ('lb', 'ffd10183', 'x3,-3(x2)', '  x2:0x80015ede  x3=0xffffffd8 PA:0x80015edb store:0x00000012'),
    ('lbu', 'ff314583', 'x11,-13(x2)', '  x2:0x80015ede x11=0x000000d2 PA:0x80015ed1 store:0x80026e7c'),
    ('lh', '00011783', 'x15,0(x2)', '  x2:0x80015ede x15=0xffffdcdd PA:0x80015ede store:0x00000000'),
    ('lhu', '0e2dd203', 'x4,226(x27)', ' x27:0x80015b52  x4=0x00003637 PA:0x80015c34 store:0x8001a38a'),
    ('lui', '5296a137', 'x2,0x5296a', '  x2=0x5296a000'),
    ('lw', '0042a083', 'x1,4(x5)', '  x5:0x80022da0  x1=0x00000000 PA:0x80022da4 store:0x06f7d1a6'),
    ('mul', '02d58b33', 'x22,x11,x13', ' x11:0x221c5e07 x13:0x80000000 x22=0x80000000'),
    ('mulh', '03099933', 'x18,x19,x16', ' x19:0x80026e7c x16:0x19fae0a7 x18=0xf302ced8'),
    ('mulhsu', '02d5ab33', 'x22,x11,x13', ' x11:0x221c5e07 x13:0x80000000 x22=0x110e2f03'),
    ('mulhu', '03cdb233', 'x4,x27,x28', ' x27:0x80000000 x28:0xffffffff  x4=0x7fffffff'),
    ('or', '01886433', 'x8,x16,x24', ' x16:0x19fae0a7 x24:0x00000000  x8=0x19fae0a7'),
    ('ori', '001b6b13', 'x22,x22,1', ' x22:0x80012000 x22=0x80012001'),
    ('rem', '03cae8b3', 'x17,x21,x28', ' x21:0x0ffffffc x28:0x00000000 x17=0x0ffffffc'),
    ('remu', '03667fb3', 'x31,x12,x22', ' x12:0xfcb0de05 x22:0x00000000 x31=0xfcb0de05'),
    ('sb', 'ff410ba3', 'x20,-9(x2)', '  x2:0x80015ede x20:0xffffffff PA:0x80015ed5 store:0xffffffff load:0x000000ff'),
    ('sh', 'ffc11a23', 'x28,-12(x2)', '  x2:0x80015ede x28:0x00000000 PA:0x80015ed2 store:0x00000000 load:0x00000000'),
    ('sll', '00e911b3', 'x3,x18,x14', ' x18:0x46485956 x14:0x00000000  x3=0x46485956'),
    ('slli', '018b1693', 'x13,x22,0x18', ' x22:0xffffffff x13=0xff000000'),
    ('slt', '0097a233', 'x4,x15,x9', ' x15:0xfa11812e  x9:0x00000000  x4=0x00000001'),
    ('slti', 'e2c2ab93', 'x23,x5,-468', '  x5:0x80022e1c x23=0x00000001'),
    ('sltiu', 'ebaeb813', 'x16,x29,-326', ' x29:0x00000000 x16=0x00000001'),
    ('sltu', '015fbcb3', 'x25,x31,x21', ' x31:0x80000000 x21:0x0ffffffc x25=0x00000000'),
    ('sra', '4034dcb3', 'x25,x9,x3', '  x9:0x80000000  x3:0x7ea5e000 x25=0x80000000'),
    ('srai', '409ad093', 'x1,x21,0x9', ' x21:0x0ffffffc  x1=0x0007ffff'),
    ('srl', '017e54b3', 'x9,x28,x23', ' x28:0x00000000 x23:0x00000000  x9=0x00000000'),
    ('srli', '01a1d493', 'x9,x3,0x1a', '  x3:0x00000000  x9=0x00000000'),
    ('sub', '415e0fb3', 'x31,x28,x21', ' x28:0x00000000 x21:0x49648224 x31=0xb69b7ddc'),
    ('sw', '01642023', 'x22,0(x8)', '  x8:0x8ffffffc x22:0x00030003 PA:0x8ffffffc store:0x00030003 load:0x00030003'),
    ('xor', '008ac633', 'x12,x21,x8', ' x21:0x0ffffffc  x8:0x8ffffffc x12=0x80000000'),
    ('xori', '09dc4613', 'x12,x24,157', ' x24:0x8ffffc34 x12=0x8ffffca9'),
]

_NO_RD_OPERAND = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "sb", "sh", "sw", "c.sw", "c.swsp",
                  "c.beqz", "c.bnez", "c.jr", "c.jalr", "c.jal", "c.j"}
_FREE = [29, 30, 28, 27, 26, 25]


def _rename(mn, ops, regs, kind):
    """Rename every read (kind RS) or the write (kind RD) register, in the
    operand text and in the xN:/xN= tokens together."""
    used = {int(n) for n in re.findall(r"x(\d+)", ops + regs)}
    free = [r for r in _FREE if r not in used]
    mapping: dict[str, str] = {}
    sep = ":" if kind == "RS" else "="

    def tok(m):
        old = m.group(1)
        if old not in mapping:
            mapping[old] = str(free[len(mapping)])
        return "x" + mapping[old] + sep

    new_regs = re.sub(r"x(\d+)" + re.escape(sep), tok, regs)
    skip = 0 if mn in _NO_RD_OPERAND else 1
    if kind == "RD" and not skip:
        return ops, new_regs, mapping
    seen = [0]

    def opt(m):
        i = seen[0]
        seen[0] += 1
        if kind == "RS" and i < skip:
            return m.group(0)
        if kind == "RD" and i >= 1:
            return m.group(0)
        old = m.group(1)
        return "x" + mapping[old] if old in mapping else m.group(0)

    return re.sub(r"x(\d+)", opt, ops), new_regs, mapping


def _pc_for(mn, insn, ops, regs):
    """Link / auipc results are checked against the pc, so give the single
    line the pc its printed result implies."""
    m = re.search(r"x\d+=0x([0-9a-f]+)", regs)
    if m is None:
        return 0x80000000
    val = int(m.group(1), 16)
    if mn in ("jal", "jalr", "c.jal", "c.jalr"):
        return val - (2 if mn.startswith("c.") else 4)
    if mn == "auipc":
        return val - (int(ops.split(",")[1], 16) << 12)
    return 0x80000000


def _first_kind(tmp_path, name, insn, mn, ops, regs, pc=0x80000000):
    line = leq.mk_line(pc, insn, mn, ops, regs)
    ti = pt.parse_trace(leq.write_trace(tmp_path, name, [line], gz=False), None)
    return ti.first_violation


def test_real_table_covers_every_decode_entry():
    got = {r[0] for r in _REAL}
    assert got == set(isa._RA_C) | set(isa._RA_32)


@pytest.mark.parametrize("mn,insn,ops,regs", _REAL, ids=[r[0] for r in _REAL])
def test_real_line_clean_and_renamed_flagged(tmp_path, mn, insn, ops, regs):
    pc = _pc_for(mn, insn, ops, regs)
    assert _first_kind(tmp_path, "clean", insn, mn, ops, regs, pc) is None
    has_read = re.search(r"x\d+:", regs) is not None
    has_write = re.search(r"x\d+=", regs) is not None
    if has_read:
        kind, want = "RS", "REG_ADDR_RS"
    elif has_write:
        kind, want = "RD", "REG_ADDR_RD"
    else:
        pytest.skip("no register token to rename")
    nops, nregs, mapping = _rename(mn, ops, regs, kind)
    assert mapping, (mn, regs)
    v = _first_kind(tmp_path, "renamed", insn, mn, nops, nregs, pc)
    assert v is not None and v.kind == want, (nops, nregs, v)


def test_32bit_printed_x0_read_not_encoded_is_flagged():
    # add x10,x10,x11: only x10/x11 are encoded, a printed x0 read is not.
    v = _v(_ins("add", "00b50533", "x10,x10,x0", {"x10": 1, "x0": 0}, {"x10": 1}))
    assert v is not None and v.kind == "REG_ADDR_RS"


def test_compressed_printed_x0_write_is_never_checked():
    # Faulting c.lw (rd' = x8) printing an x0 write: clean, like the 32-bit case.
    assert _v(_ins("c.lw", "42c0", "x0,4(x13)", {"x13": 0x100}, {"x0": 0})) is None
    # A different non-x0 destination is still reported.
    v = _v(_ins("c.lw", "42c0", "x9,4(x13)", {"x13": 0x100}, {"x9": 0}))
    assert v is not None and v.kind == "REG_ADDR_RD"


def test_compressed_quadrant_gate_mnemonic_vs_bits(fresh_caches):
    # 0b22 is c.slli (quadrant 2, funct3 0); printed as c.addi (quadrant 1,
    # funct3 0) the bits do not match, so nothing is checked.
    assert _v(_ins("c.addi", "0b22", "x5,3", {"x5": 1}, {"x5": 4})) is None


# ---------------------------------------------------------------------------
# Survived-mutation pins (cold-read round 1).
# ---------------------------------------------------------------------------

def test_c_j_printing_a_link_write_is_flagged(fresh_caches):
    # c.j (be1d) writes nothing; a printed x1= is the c.jal expansion's rd.
    assert _v(_ins("c.j", "be1d", "80004dec", {}, {})) is None
    v = _v(_ins("c.j", "be1d", "80004dec", {}, {"x1": 0x800052b8}))
    assert v is not None and v.kind == "REG_ADDR_RD"
    # c.jal's funct3 is 1, c.j's is 5: bits of a c.jal printed as c.j are skipped.
    assert _v(_ins("c.j", "2021", "800001f8", {}, {"x1": 0x800001f2})) is None


def test_check_violation_is_not_overwritten_by_reg_addr(fresh_caches):
    # add x10,x10,x11 (00b50533) printed with rs2 x12 AND a wrong result:
    # check()'s WRONG_RESULT must survive, REG_ADDR only fills a clean slot.
    bad = _ins("add", "00b50533", "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 99})
    v = _v(bad)
    assert v is not None and not v.kind.startswith("REG_ADDR"), v
    # Control: with the right result the same register fault is REG_ADDR_RS.
    ok = _ins("add", "00b50533", "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 3})
    assert _v(ok).kind == "REG_ADDR_RS"


def test_branch_printing_non_x0_write_is_flagged(fresh_caches):
    # beq (00b50463) never writes; x0= is the tracer convention, x5= is a fault.
    v = _v(_ins("beq", "00b50463", "x10,x11,8", {"x10": 1, "x11": 2}, {"x5": 0}))
    assert v is not None and v.kind == "REG_ADDR_RD"
    # Bits 11:7 of a branch are immediate bits (here 8): they must not be
    # taken for an rd field, so a printed x8= is a fault too.
    v = _v(_ins("beq", "00b50463", "x10,x11,8", {"x10": 1, "x11": 2}, {"x8": 0}))
    assert v is not None and v.kind == "REG_ADDR_RD"


def test_both_source_fields_wrong_groups_as_generic_rs(fresh_caches):
    # add x10,x10,x11 printed as reading x12 and x13: neither field is found.
    v = _v(_ins("add", "00b50533", "x10,x12,x13", {"x12": 1, "x13": 2}, {"x10": 3}))
    assert v is not None and v.kind == "REG_ADDR_RS"
    assert v.detail["field"] == "RS"
    assert isa.viol_group(v.kind, v.mnemonic, v.detail) == "RA_RS:I"


def test_csr_gate_needs_nonzero_funct3(fresh_caches):
    # ecall (00000073) has funct3 0: it is not a CSR instruction, so a csrrw
    # mnemonic over those bits is "bits do not match", nothing is checked.
    assert _v(_ins("csrrw", "00000073", "x5,mscratch,x6", {"x6": 1}, {"x5": 0})) is None
    # A real csrrw over the same opcode (funct3 1) is checked.
    v = _v(_ins("csrrw", "301b1073", "x0,misa,x23", {"x23": 1}, {"x0": 0}))
    assert v is not None and v.kind == "REG_ADDR_RS"


@pytest.mark.parametrize("insn", ["zzzz", "zzzzzzzz", "00b5053g"])
def test_non_hex_insn_is_skipped_without_error(fresh_caches, insn):
    ins = _ins("add" if len(insn) > 4 else "c.and", insn, "x9,x5", {"x9": 1, "x5": 3}, {"x9": 1})
    v, err = isa.check_safe(None, ins, None, isa.ScanContext())
    assert err is None
    assert v is None


def test_reg_addr_precedes_read_mismatch(fresh_caches):
    ctx = isa.ScanContext()
    ctx.regs["x10"] = 7                      # the printed x10:0x1 disagrees
    ins = _ins("add", "00b50533", "x10,x10,x12", {"x10": 1, "x12": 2}, {"x10": 3})
    v, err = isa.check_safe(None, ins, 0x80000004, ctx)
    assert err is None and v is not None and v.kind == "REG_ADDR_RS"
    # Control: with the right register the same read does raise READ_MISMATCH.
    ok = _ins("add", "00b50533", "x10,x10,x11", {"x10": 1, "x11": 2}, {"x10": 3})
    v, err = isa.check_safe(None, ok, 0x80000004, ctx)
    assert err is None and v is not None and v.kind == "READ_MISMATCH"


def test_reg_addr_precedes_trap_csr(fresh_caches):
    # csrrs x5,mcause,x0 (342022f3) printed with rd x6 while a trap is
    # predicted and the printed mcause is wrong: REG_ADDR_RD wins.
    pred = {"mcause": ("irq", frozenset((0x80000007,)))}
    ctx = isa.ScanContext()
    ctx.trap_pred = pred
    ins = _ins("csrrs", "342022f3", "x6,mcause,x0", {}, {"x6": 0x80000003})
    v, err = isa.check_safe(None, ins, 0x80000004, ctx)
    assert err is None and v is not None and v.kind == "REG_ADDR_RD"
    # Control: the right rd with the wrong mcause is the trap check.
    ctx.trap_pred = pred
    ok = _ins("csrrs", "342022f3", "x5,mcause,x0", {}, {"x5": 0x80000003})
    v, err = isa.check_safe(None, ok, 0x80000004, ctx)
    assert err is None and v is not None and v.kind == "TRAP_MCAUSE"


# add x10,x16,x5: no single-bit flip of a field makes two encoded sources equal.
_ADD = 0x00580533


@pytest.mark.parametrize("bit", list(range(7, 12)) + list(range(15, 20)) + list(range(20, 25)))
def test_32bit_second_level_key_keeps_every_register_bit(fresh_caches, bit):
    ok = _ins("add", f"{_ADD:08x}", "x10,x16,x5", {"x16": 1, "x5": 2}, {"x10": 3})
    assert _v(ok) is None
    # Same printed registers, one register-field bit flipped in the Insn.
    flipped = _ins("add", f"{_ADD ^ (1 << bit):08x}", "x10,x16,x5",
                   {"x16": 1, "x5": 2}, {"x10": 3})
    v = _v(flipped)
    assert v is not None and v.kind.startswith("REG_ADDR"), (bit, v)
