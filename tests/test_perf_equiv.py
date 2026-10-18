"""Equivalence tests for the behaviour-preserving speedups in rfb/isa.py and
rfb/parse_trace.py (base commit 5f03b0f). The reference functions below are
condensed copies of the 5f03b0f originals (rfb/isa.py: parse_regs,
_split_operands, _parse_imm, _mem_operands and their regexes). Only names that
already existed in 5f03b0f are used, so internals may change freely."""

import random
import re

from rfb import isa
from rfb.isa import ScanContext
from rfb.parse_trace import parse_trace
from rfb.record import Instr

# --- reference (copied semantics from 5f03b0f) ------------------------------

_R_READ = re.compile(r"^x(\d+):0x([0-9a-fA-F]+)$")
_R_WRITE = re.compile(r"^x(\d+)=0x([0-9a-fA-F]+)$")
_R_PA = re.compile(r"^PA:0x([0-9a-fA-F]+)$")
_R_STORE = re.compile(r"^store:0x([0-9a-fA-F]+)$")
_R_LOAD = re.compile(r"^load:0x([0-9a-fA-F]+)$")
_R_MEM = re.compile(r"^(-?(?:0x[0-9a-fA-F]+|\d+))\((x\d+)\)$")


def ref_parse_regs(regs_field):
    reads, writes = {}, {}
    pa = store = load = None
    for tok in regs_field.split():
        m = _R_READ.match(tok)
        if m:
            reads[f"x{m.group(1)}"] = int(m.group(2), 16)
            continue
        m = _R_WRITE.match(tok)
        if m:
            writes[f"x{m.group(1)}"] = int(m.group(2), 16)
            continue
        m = _R_PA.match(tok)
        if m:
            pa = int(m.group(1), 16)
            continue
        m = _R_STORE.match(tok)
        if m:
            store = int(m.group(1), 16)
            continue
        m = _R_LOAD.match(tok)
        if m:
            load = int(m.group(1), 16)
            continue
    return reads, writes, pa, store, load


def ref_split_operands(operands):
    if not operands:
        return []
    return [p.strip() for p in operands.split(",")]


def ref_parse_imm(tok):
    tok = tok.strip()
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:]
    if tok.lower().startswith("0x"):
        val = int(tok, 16)
    else:
        val = int(tok, 10)
    return -val if neg else val


def ref_mem_operands(mnemonic, ops):
    if len(ops) < 2:
        return None
    m = _R_MEM.match(ops[1])
    if not m:
        return None
    return ops[0], ref_parse_imm(m.group(1)), m.group(2)


def _outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as e:  # noqa: BLE001 - compare exception type and text
        return ("exc", type(e), str(e))


# --- random input generators -------------------------------------------------

_HEX = "0123456789abcdefABCDEF"


def _rhex(r, n=None):
    return "".join(r.choice(_HEX) for _ in range(n or r.randint(1, 9)))


def _gen_token(r):
    k = r.randint(0, 43)
    n = r.randint(0, 40)
    if k == 0:
        return f"x{n}:0x{_rhex(r)}"
    if k == 1:
        return f"x{n}=0x{_rhex(r)}"
    if k == 2:
        return f"PA:0x{_rhex(r)}"
    if k == 3:
        return f"store:0x{_rhex(r)}"
    if k == 4:
        return f"load:0x{_rhex(r)}"
    if k == 5:
        return f"X{n}:0x{_rhex(r)}"
    if k == 6:
        return f"x{n}:0x{_rhex(r)}zz"
    if k == 7:
        return f"x{n}=0xg"
    if k == 8:
        return f"PA:{r.randint(0, 99)}"
    if k == 9:
        return "store:0xg"
    if k == 10:
        return r.choice(["xz", "x0", "x", "0x", "abc", "", "PA:", "pa:0x1", "load:0x"])
    if k == 11:
        return f"x{n}:0X{_rhex(r)}"
    if k == 12:
        return f"x{n}:0x{_rhex(r)}\n"
    if k == 13:
        return "x１:0x5"  # fullwidth digit
    if k == 14:
        return f"x{n}:0x" + _rhex(r, 300)
    if k == 15:
        return f"x{n}:0x{_rhex(r)}:0x{_rhex(r)}"
    if k == 16:
        return f"x{n} :0x{_rhex(r)}"
    if k == 17:
        return f"x{n}::0x{_rhex(r)}"
    if k == 18:
        return f"store=0x{_rhex(r)}"
    if k == 19:
        return f"x{n}:-0x{_rhex(r)}"
    if k == 20:
        return f"load:0x{_rhex(r)}x"
    if k == 21:
        return f"PA:0x{_rhex(r)} "
    if k == 22:
        return f"x{n}=0x{_rhex(r)}\t"
    if k == 23:
        return f"x{n}:0x_{_rhex(r)}"
    if k == 24:
        return "x" + "9" * r.randint(1, 60) + ":0x1"
    # malformed shapes next to the valid ones: all must be ignored
    if k == 25:
        return f"x:0x{_rhex(r)}"  # no register number
    if k == 26:
        return f"x{n}:0x"  # no hex digits (read)
    if k == 27:
        return f"x{n}=0x"  # no hex digits (write)
    if k == 28:
        return f"x{n}0x{_rhex(r)}"  # no separator
    if k == 29:
        # wrong separator; a space would split into two tokens, so it becomes "_"
        return f"x{n}{r.choice(';,.|-+/ ')}0x{_rhex(r)}".replace(" ", "_")
    if k == 30:
        return f"PA={r.choice(['0x', '0x' + _rhex(r)])}"
    if k == 31:
        return f"load{r.choice(':=')}0x"  # no hex digits
    if k == 32:
        return f"store{r.choice(':=')}0x"  # no hex digits
    if k == 33:
        return "PA:0x"
    if k == 34:
        return f"load=0x{_rhex(r)}"
    if k == 35:
        return f"store=0x{_rhex(r)}"
    if k == 36:
        return f"PA={_rhex(r)}"
    if k == 37:
        return f"x{n}{r.choice(':=')}{_rhex(r)}"  # no 0x prefix
    if k == 38:
        return f"x{n}{r.choice(':=')}x{_rhex(r)}"  # x without the 0
    if k == 39:
        return f"x0{n}{r.choice(':=')}0x{_rhex(r)}"  # leading zero in register number
    pfx = r.choice(["PA", "store", "load"])
    if k == 40:
        return f"{pfx}:0X{_rhex(r)}"
    if k == 41:
        return f"{pfx}:{r.randint(0, 99)}"  # no 0x prefix
    if k == 42:
        return f"{pfx}:0x" + _rhex(r, 40)
    return f"{pfx}:0x{_rhex(r)}:0x{_rhex(r)}"


def _gen_regs_field(r):
    return r.choice([" ", "  ", "\t"]).join(_gen_token(r) for _ in range(r.randint(0, 5)))


def _gen_imm(r):
    k = r.randint(0, 17)
    if k == 0:
        return str(r.randint(0, 99999))
    if k == 1:
        return str(-r.randint(0, 99999))
    if k == 2:
        return f"0x{_rhex(r)}"
    if k == 3:
        return f"-0x{_rhex(r)}"
    if k == 4:
        return f"0X{_rhex(r)}"
    if k == 5:
        return f" {r.randint(0, 99)} "
    return r.choice([
        "", "0x", "-", "-0x", "abc", "0xg", "1_0", "0x1_0", "１２",
        "--5", "+5", "- 5", "5" * 400, "0x" + "f" * 400, "1.5", "0b11", " ",
        "--0x5", "-+1", "0x-1", "\t7\n",
    ][: r.randint(1, 21)])


def _gen_mem(r):
    k = r.randint(0, 11)
    n = r.randint(0, 40)
    if k == 0:
        return f"{r.randint(0, 4095)}(x{n})"
    if k == 1:
        return f"-{r.randint(0, 4095)}(x{n})"
    if k == 2:
        return f"0x{_rhex(r)}(x{n})"
    if k == 3:
        return f"-0x{_rhex(r)}(x{n})"
    return r.choice([
        "0(x5", "0x5)", "(x5)", "0()", "0(X5)", "0(xz)", "0(x5) ", " 0(x5)",
        "abc", "", "0x(x5)", "0xg(x5)", "1_0(x5)", "１(x5)", "0(x5)0",
        "5" * 400 + "(x5)", "--5(x5)", "0(x１)", "0(x5)\n", "0 (x5)",
    ])


def _gen_operands(r):
    parts = [r.choice(["x1", "X1", "x0", "y", "mstatus", "MSTATUS", "", " a ", "0x10", "-4(x2)"])
             for _ in range(r.randint(0, 4))]
    return r.choice([",", ", ", ","]).join(parts) if r.random() < 0.9 else r.choice(
        ["", " ", ",", ",,", "a,,b", "A,B", "x1, y", "x1 ,y", "\t,\t", "é,ü"])


# --- oracle property tests ----------------------------------------------------

N = 2500


def test_oracle_parse_regs_matches_reference():
    """Oracle: parse_regs vs the 5f03b0f regex implementation, incl. reads/writes insertion order."""
    r = random.Random(1001)
    inputs = [_gen_regs_field(r) for _ in range(N)]
    for s in inputs:
        got, want = _outcome(isa.parse_regs, s), _outcome(ref_parse_regs, s)
        assert got == want, repr(s)
        if got[0] == "ok":
            assert list(got[1][0].items()) == list(want[1][0].items()), repr(s)
            assert list(got[1][1].items()) == list(want[1][1].items()), repr(s)


def test_oracle_split_operands_matches_reference():
    """Oracle: _split_operands vs reference over random operand strings."""
    r = random.Random(1002)
    inputs = [_gen_operands(r) for _ in range(N)]
    for s in inputs:
        assert _outcome(isa._split_operands, s) == _outcome(ref_split_operands, s), repr(s)


def test_oracle_parse_imm_matches_reference():
    """Oracle: _parse_imm vs reference, incl. same-type/same-text exceptions."""
    r = random.Random(1003)
    inputs = [_gen_imm(r) for _ in range(N)]
    for s in inputs:
        assert _outcome(isa._parse_imm, s) == _outcome(ref_parse_imm, s), repr(s)


def test_oracle_mem_operands_matches_reference():
    """Oracle: _mem_operands vs reference over valid and broken imm(rs) shapes."""
    r = random.Random(1004)
    inputs = []
    for _ in range(N):
        rd = r.choice(["x1", "x6", "x31", "X2", "", "y"])
        k = r.random()
        if k < 0.1:
            ops = [rd]
        elif k < 0.15:
            ops = []
        else:
            ops = [rd, _gen_mem(r)]
        inputs.append(ops)
    for ops in inputs:
        assert _outcome(isa._mem_operands, "lw", ops) == _outcome(
            ref_mem_operands, "lw", ops), repr(ops)


# --- concrete cases -----------------------------------------------------------


def _instr(index, pc, mnemonic, operands, reads=None, writes=None, pa=None, size=4):
    return Instr(
        index=index, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
        reads=reads or {}, writes=writes or {}, pa=pa, store=None, load=None,
    )


def _check(mn, ops, reads, writes):
    i = _instr(0, 0x1000, mn, ops, reads=reads, writes=writes)
    return isa.check(None, i, 0x1004, ScanContext())


def test_s1_x0_as_rs1_counts_as_zero():
    """S1: _arith_operands treats x0 as rs1 value 0 (add x5,x0,x6)."""
    v = _check("add", "x5,x0,x6", {"x6": 7}, {"x5": 99})
    assert v is not None and v.kind == "WRONG_RESULT"
    assert v.detail["expected"] == 7
    assert _check("add", "x5,x0,x6", {"x6": 7}, {"x5": 7}) is None


def test_s2_x0_as_rs2_counts_as_zero():
    """S2: _arith_operands treats x0 as rs2 value 0 (add/or x5,x6,x0)."""
    for mn in ("add", "or"):
        v = _check(mn, "x5,x6,x0", {"x6": 9}, {"x5": 123})
        assert v is not None and v.kind == "WRONG_RESULT", mn
        assert _check(mn, "x5,x6,x0", {"x6": 9}, {"x5": 9}) is None, mn


def test_s3_x0_in_reg_imm_counts_as_zero():
    """S3: _arith_operands reg-imm form with x0 source (addi x5,x0,5)."""
    v = _check("addi", "x5,x0,5", {}, {"x5": 6})
    assert v is not None and v.kind == "WRONG_RESULT"
    assert _check("addi", "x5,x0,5", {}, {"x5": 5}) is None


def test_s1_s3_expected_value_reported():
    """S1-S3: the violation carries the correct expected value computed with x0 = 0."""
    for mn, ops, reads, want in (
        ("add", "x5,x0,x6", {"x6": 7}, 7),
        ("add", "x5,x6,x0", {"x6": 9}, 9),
        ("addi", "x5,x0,5", {}, 5),
    ):
        v = _check(mn, ops, reads, {"x5": 0x12345})
        assert v is not None and v.kind == "WRONG_RESULT"
        assert v.detail["expected"] == want, (mn, v)


def _write_trace(path, extra_cols):
    hdr = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
    rows = [
        ("1", "1", "80000000", "00000013", "addi", "x0,x0,0", "x0=0x00000000"),
        ("2", "2", "80000004", "00500093", "addi", "x1,x0,5", "x1=0x00000005"),
        ("3", "3", "80000008", "00108133", "add", "x2,x1,x1", "x1:0x00000005 x2=0x0000000a"),
        ("4", "4", "8000000c", "0062a023", "sw", "x6,0(x5)", "PA:0x80001000 store:0x000000aa"),
    ]
    with open(path, "w") as f:
        f.write(hdr)
        for row in rows:
            f.write("\t".join(row + extra_cols) + "\n")


def test_s4_extra_tab_columns_ignored(tmp_path):
    """S4: parse_trace fast split ignores columns beyond the 7th."""
    p7 = tmp_path / "t7.log"
    p9 = tmp_path / "t9.log"
    _write_trace(p7, ())
    _write_trace(p9, ("junk1", "x9=0xdead"))
    a = parse_trace(str(p7)).to_dict()
    b = parse_trace(str(p9)).to_dict()
    assert a["n_instr"] == 4
    assert a == b


_IGNORED_TOKENS = [
    "x:0x5", "x5:0x", "x5=0x", "x50x5", "x5;0x5", "x5,0x5", "x5.0x5",
    "PA=0x10", "PA=0x", "PA:0x", "PA:", "load=0x10", "load:0x", "load=0x",
    "store=0x10", "store:0x", "store=0x",
]


def test_malformed_tokens_ignored_deterministic():
    """Each malformed token alone, and mixed with a valid one, yields no entry (as in 5f03b0f)."""
    empty = ({}, {}, None, None, None)
    for tok in _IGNORED_TOKENS:
        assert isa.parse_regs(tok) == empty, tok
        assert ref_parse_regs(tok) == empty, tok
        mixed = f"x1:0x7 {tok} x2=0x8"
        assert isa.parse_regs(mixed) == ({"x1": 7}, {"x2": 8}, None, None, None), tok


def test_parse_regs_insertion_order_and_overwrite():
    """reads/writes keep first-insertion order; a repeated register overwrites the value in place."""
    reads, writes, *_ = isa.parse_regs("x9:0x1 x2:0x2 x9:0x3 x7=0x4 x1=0x5 x7=0x6")
    assert list(reads.items()) == [("x9", 3), ("x2", 2)]
    assert list(writes.items()) == [("x7", 6), ("x1", 5)]
    ref_r, ref_w, *_ = ref_parse_regs("x9:0x1 x2:0x2 x9:0x3 x7=0x4 x1=0x5 x7=0x6")
    assert list(ref_r.items()) == list(reads.items())
    assert list(ref_w.items()) == list(writes.items())


def test_parse_regs_pa_store_load_last_wins():
    """A repeated PA/store/load token: the last one wins (as in 5f03b0f)."""
    field = "PA:0x1 PA:0x2 store:0x3 store:0x4 load:0x5 load:0x6"
    assert isa.parse_regs(field)[2:] == (2, 4, 6)
    assert ref_parse_regs(field)[2:] == (2, 4, 6)


_BAD_REGS = "x1:0x5zz x2=0xg X3:0x1 PA:5 store:0xg xz PA:0x10zz store:0x20zz load:0x30zz"


def test_s5_s6_noncompliant_tokens_ignored():
    """S5/S6: parse_regs ignores malformed tokens exactly like the reference."""
    got = isa.parse_regs(_BAD_REGS)
    assert got == ref_parse_regs(_BAD_REGS)
    assert got == ({}, {}, None, None, None)
    mixed = _BAD_REGS + " x4:0x10 x5=0x20 PA:0x30 store:0x40 load:0x50"
    assert isa.parse_regs(mixed) == ref_parse_regs(mixed) == (
        {"x4": 0x10}, {"x5": 0x20}, 0x30, 0x40, 0x50)


def test_s7_split_operands_preserves_case():
    """S7: _split_operands preserves operand case (lower-case input must not alias upper-case)."""
    assert isa._split_operands("x1,y") == ["x1", "y"]
    assert isa._split_operands("X1,Y") == ["X1", "Y"]
    assert isa._split_operands("x1,y") == ["x1", "y"]
    assert isa._split_operands("mstatus,x1") == ["mstatus", "x1"]
    assert isa._split_operands("MSTATUS,x1") == ["MSTATUS", "x1"]


class _Run:
    def __init__(self):
        self.ctx = ScanContext()
        self.n = 0

    def step(self, mnemonic, operands, reads=None, writes=None, pa=None):
        pc = 0x1000 + 4 * self.n
        i = _instr(self.n, pc, mnemonic, operands, reads, writes, pa)
        self.n += 1
        v, err = isa.check_safe(None, i, pc + 4, self.ctx)
        assert err is None
        isa.observe(self.ctx, i, pc + 4)
        return v


def test_s7_csr_name_case():
    """S7: CSR shadow keyed by CSR name stays case-correct for mixed-case operand text."""
    r = _Run()
    assert r.step("csrrw", "x5,mscratch,x6", reads={"x6": 0x11}, writes={"x5": 0}) is None
    v = r.step("csrrs", "x7,mscratch,x0", writes={"x7": 0x22})
    assert v is not None and v.kind == "CSR_MISMATCH"
    assert r.step("csrrs", "x7,mscratch,x0", writes={"x7": 0x11}) is None
    assert isa._split_operands("x7,MSCRATCH,x0") == ["x7", "MSCRATCH", "x0"]


def test_s8_mem_operands_use_own_rd():
    """S8: _mem_operands returns each call's own rd for identical offset(rs) text."""
    assert isa._mem_operands("lw", ["x6", "0(x5)"]) == ("x6", 0, "x5")
    assert isa._mem_operands("lw", ["x9", "0(x5)"]) == ("x9", 0, "x5")
    assert isa._mem_operands("lw", ["x6", "0(x5)"]) == ("x6", 0, "x5")


def test_s8_load_ext_uses_own_rd():
    """S8: LOAD_EXT sees each lb's own rd value although the offset operand text is identical."""
    ctx = ScanContext()
    bad = _instr(0, 0x1000, "lb", "x6,0(x5)", writes={"x6": 0x80})  # not sign-extended
    ok = _instr(1, 0x1004, "lb", "x9,0(x5)", writes={"x9": 0xFFFFFF80})
    v1, e1 = isa.check_safe(None, bad, 0x1004, ctx)
    v2, e2 = isa.check_safe(None, ok, 0x1008, ctx)
    assert e1 is None and e2 is None
    assert v1 is not None and v1.kind == "LOAD_EXT"
    assert v2 is None
    # reverse order, fresh ctx: still own rd
    ctx = ScanContext()
    assert isa.check_safe(None, ok, 0x1008, ctx)[0] is None
    assert isa.check_safe(None, bad, 0x1004, ctx)[0].kind == "LOAD_EXT"


def test_s8_mem_shadow_uses_own_rd():
    """S8: MEM_MISMATCH shadow compares each lw's own rd even with identical offset text."""
    r = _Run()
    A = 0x80001000
    r.step("sw", "x7,0(x5)", reads={"x7": 0x11223344}, pa=A)
    v = r.step("lw", "x6,0(x5)", writes={"x6": 0x11223345}, pa=A)
    assert v is not None and v.kind == "MEM_MISMATCH"
    assert r.step("lw", "x9,0(x5)", writes={"x9": 0x11223344}, pa=A) is None
