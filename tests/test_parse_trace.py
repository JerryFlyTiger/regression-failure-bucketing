import gzip
from pathlib import Path

from rfb.parse_regr import parse_regr
from rfb.parse_trace import parse_trace
from rfb import isa

ROOT = Path(__file__).resolve().parents[1]
SET1 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_1"
SET2 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_2"

def _gz_line_count(path: Path) -> int:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def test_n_instr_matches_line_count_minus_header():
    for base in (SET1, SET2):
        for case_dir in sorted(base.glob("case_*")):
            trace_path = case_dir / "trace.log.gz"
            expected = _gz_line_count(trace_path) - 1
            info = parse_trace(str(trace_path))
            assert info.n_instr == expected, case_dir
            assert info.parse_errors == [], case_dir


def test_head_first_instr_case1():
    info = parse_trace(str(SET1 / "case_1/trace.log.gz"))
    h0 = info.head[0]
    assert h0.pc == 0x80000080
    assert h0.size == 4
    assert h0.mnemonic == "addi"
    assert h0.operands == "x0,x0,0"
    assert h0.reads == {"x0": 0}
    assert h0.writes == {"x0": 0}
    assert info.first_pc == 0x80000080


def test_anchor_hits_mismatch_case():
    regr_info = parse_regr(str(SET1 / "case_2/regr.log"))
    trace_info = parse_trace(str(SET1 / "case_2/trace.log.gz"), regr_info)
    assert trace_info.anchor_hits >= 1
    anchor = regr_info.mismatches[0]
    assert trace_info.anchor_windows
    for window in trace_info.anchor_windows:
        pcs = [i.pc for i in window]
        assert anchor.ibex_pc in pcs


def test_parse_regs_unit():
    reads, writes, pa, store, load = isa.parse_regs(
        "  x2:0x5296a000  x2=0x52969ec0"
    )
    assert reads == {"x2": 0x5296A000}
    assert writes == {"x2": 0x52969EC0}
    assert pa is None and store is None and load is None

    reads, writes, pa, store, load = isa.parse_regs(
        "  x8:0x8ffffffc x22:0x00030003 PA:0x8ffffffc store:0x00030003 load:0x00030003"
    )
    assert reads == {"x8": 0x8FFFFFFC, "x22": 0x00030003}
    assert writes == {}
    assert pa == 0x8FFFFFFC
    assert store == 0x00030003
    assert load == 0x00030003

    reads, writes, pa, store, load = isa.parse_regs("")
    assert reads == {} and writes == {} and pa is None and store is None and load is None


def test_mclass_one_per_row():
    expected = {
        "BR_BEQ": "beq",
        "BR_BNE": "bne",
        "BR_LT": "blt",
        "BR_GE": "bge",
        "JUMP": "jal",
        "TRAP_INSTR": "ecall",
        "XRET": "mret",
        "ADDSUB": "addi",
        "LOGIC": "and",
        "SHIFT": "sll",
        "SLT": "slt",
        "MUL": "mul",
        "DIV": "div",
        "LOAD": "lw",
        "STORE": "sw",
        "CSR": "csrrw",
        "FENCE": "fence",
        "OTHER": "wfi",
    }
    for cls, mnemonic in expected.items():
        assert isa.mclass(mnemonic) == cls


def test_all_public_mnemonics_have_mclass():
    seen = set()
    for base in (SET1, SET2):
        for case_dir in sorted(base.glob("case_*")):
            info = parse_trace(str(case_dir / "trace.log.gz"))
            seen.update(info.mnemonic_hist.keys())
    assert seen, "no mnemonics collected"
    for mnemonic in seen:
        # ARCHITECTURE 4.2 claims every mnemonic in the public set has an
        # explicit mclass entry (OTHER is only for things not in the trace,
        # e.g. wfi). Assert the strong claim rather than the trivially-true
        # "resolves to something" check.
        assert mnemonic in isa.MCLASS, mnemonic
