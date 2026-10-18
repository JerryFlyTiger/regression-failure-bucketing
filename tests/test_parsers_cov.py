"""Coverage-driven tests for parse_regr / parse_sim / parse_trace / record /
signature edge paths, using tiny hand-written inputs (tmp_path)."""

import gzip

import pytest

from rfb import parse_trace as pt
from rfb.parse_regr import _spike_pc, _split_rd_val, parse_regr
from rfb.parse_sim import parse_sim
from rfb.parse_trace import _decompressed_size, _split_trace_line, parse_trace
from rfb.record import (
    Deviation,
    FailureRecord,
    Mismatch,
    RegrInfo,
    SimInfo,
    TraceInfo,
    Violation,
    _to_jsonable,
)
from rfb.signature import _sys_viol, signature


# =============================================================================
# parse_regr
# =============================================================================


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_split_rd_val_without_trailing_rd_is_none_pair():
    # regr.py 20-21
    assert _split_rd_val("jal x0, 80001000") == (None, None)
    assert _split_rd_val("") == (None, None)
    assert _split_rd_val("addi t0, t0, -64: t0:80022ddc") == ("t0", 0x80022DDC)


def test_spike_pc_strips_ffffffff_prefix_only_when_present():
    # parse_regr.py 26->28
    assert _spike_pc("ffffffff800016da") == 0x800016DA
    assert _spike_pc("800016da") == 0x800016DA
    assert _spike_pc("ffffffffffffffff") == 0xFFFFFFFF  # only one prefix stripped


_MISMATCH_LOG = (
    "Mismatch[1]:\n"
    "ibex[88] : pc[8000f8e0] jal\tx0,80001000\n"
    "spike[88] : pc[8000f8e0] jal x0, 80001000\n"
    "this line matches nothing\n"
    "\n"
    "[FAILED]: 76 matched, 3 mismatch\n"
)


def test_regr_mismatch_without_rd_and_unmatched_lines_ignored(tmp_path):
    # parse_regr.py 21 (via parse), 26->28, 96->43 (unmatched line)
    info = parse_regr(_write(tmp_path, "regr.log", _MISMATCH_LOG))
    assert info.kind == "MISMATCH"
    assert info.test_name is None
    assert (info.matched, info.mismatch_count) == (76, 3)
    assert len(info.mismatches) == 1
    m = info.mismatches[0]
    assert (m.ibex_rd, m.ibex_val, m.spike_rd, m.spike_val) == (None, None, None, None)
    assert (m.ibex_pc, m.spike_pc) == (0x8000F8E0, 0x8000F8E0)
    assert (m.ibex_text, m.spike_text) == ("jal\tx0,80001000", "jal x0, 80001000")
    # every line is kept in raw, including the unmatched ones
    assert info.raw[3:5] == ["this line matches nothing", ""]
    assert len(info.raw) == 6


def test_regr_test_name_after_mismatch_keeps_mismatch_kind(tmp_path):
    # parse_regr.py 98->100: kind only becomes TEST_FAILED from UNKNOWN
    text = _MISMATCH_LOG + "riscv_x_test.0 : [FAILED]\n"
    info = parse_regr(_write(tmp_path, "regr.log", text))
    assert info.kind == "MISMATCH"
    assert info.test_name == "riscv_x_test"


def test_regr_test_name_only_gives_test_failed(tmp_path):
    info = parse_regr(_write(tmp_path, "regr.log", "riscv_x_test.0 : [FAILED]\n"))
    assert info.kind == "TEST_FAILED"
    assert info.test_name == "riscv_x_test"
    assert info.mismatches == []


# =============================================================================
# parse_sim
# =============================================================================

def _assert_line(message, time="4165000"):
    return (
        f"UVM_ERROR /w/prim_assert.sv(21) @ {time}: reporter [ASSERT FAILED] {message}\n"
    )


def test_assert_failed_dut_prefix_stripped_from_name(tmp_path):
    info = parse_sim(_write(tmp_path, "sim.log", _assert_line(
        "[core_ibex_tb_top.dut.u_core.Foo] Foo: x (/a/b.sv:1)")))
    assert info.error_templates == {"u_core.Foo": 1}
    assert info.first_errors[0].template == "u_core.Foo"


def test_assert_failed_name_without_dut_prefix_kept_verbatim(tmp_path):
    # parse_sim.py 34->36
    info = parse_sim(_write(tmp_path, "sim.log", _assert_line(
        "[some.other.Name] Name: x (/a/b.sv:1)")))
    assert info.error_templates == {"some.other.Name": 1}


def test_assert_failed_without_bracket_falls_back_to_normalize(tmp_path):
    # parse_sim.py 37-38
    info = parse_sim(_write(tmp_path, "sim.log", _assert_line(
        "plain failure 0x1F at 12")))
    assert info.error_templates == {"plain failure <HEX> at <N>": 1}
    assert info.first_errors[0].id == "ASSERT FAILED"


def test_missing_testname_leaves_none_and_first_testname_wins(tmp_path):
    # parse_sim.py 72->75
    none = parse_sim(_write(tmp_path, "a.log", "hello\nworld\n"))
    assert none.test_name is None
    assert none.line_count == 2
    two = parse_sim(_write(
        tmp_path, "b.log", "x\nrun +UVM_TESTNAME=first_test +y\n+UVM_TESTNAME=second\n"))
    assert two.test_name == "first_test"


def test_id_section_ends_at_first_non_row_line(tmp_path):
    # parse_sim.py 93: after a non-matching line the id section is closed,
    # so a later row-shaped line must not be collected.
    text = (
        "** Report counts by id\n"
        "[RNTST]     1\n"
        "some non row line\n"
        "[LATE]     7\n"
    )
    info = parse_sim(_write(tmp_path, "sim.log", text))
    assert info.report_ids == {"RNTST": 1}


def test_severity_section_ends_at_first_non_row_line(tmp_path):
    text = (
        "** Report counts by severity\n"
        "UVM_INFO :    4\n"
        "some non row line\n"
        "UVM_ERROR :    9\n"
    )
    info = parse_sim(_write(tmp_path, "sim.log", text))
    assert info.counts == {"UVM_INFO": 4}


def test_uvm_warning_lines_are_counted(tmp_path):
    # parse_sim.py 102-104: no report section -> counts come from the scan
    text = (
        "UVM_WARNING /w/a.sv(1) @ 10: reporter [X] first\n"
        "UVM_WARNING /w/a.sv(2) @ 20: reporter [X] second\n"
        + _assert_line("[a.b] c") 
    )
    info = parse_sim(_write(tmp_path, "sim.log", text))
    assert info.counts == {"UVM_WARNING": 2, "UVM_ERROR": 1}
    assert info.fatals == []


def test_report_section_counts_override_scanned_counts(tmp_path):
    text = (
        "UVM_WARNING /w/a.sv(1) @ 10: reporter [X] first\n"
        "** Report counts by severity\n"
        "UVM_WARNING :    5\n"
    )
    info = parse_sim(_write(tmp_path, "sim.log", text))
    assert info.counts == {"UVM_WARNING": 5}


# =============================================================================
# parse_trace
# =============================================================================

_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"


def test_split_trace_line_five_columns_has_empty_operands_and_regs():
    # parse_trace.py 69-72
    assert _split_trace_line(["1", "1", "80000000", "00000013", "nop"]) == (
        "80000000", "00000013", "nop", "", "")
    assert _split_trace_line(["1", "1", "80000000", "00000013", "nop", "x1=0x1"]) == (
        "80000000", "00000013", "nop", "", "x1=0x1")
    assert _split_trace_line(["1", "1", "80000000", "00000013", "nop", "x1,x2", "x1=0x1"]) == (
        "80000000", "00000013", "nop", "x1,x2", "x1=0x1")
    assert _split_trace_line(["1", "1", "80000000", "00000013"]) is None


def test_parse_trace_five_and_six_column_lines(tmp_path):
    text = (
        _HEADER
        + "1\t1\t80000000\t00000013\tnop\n"
        + "2\t2\t80000004\t00000013\tnop\tx1=0x00000001\n"
    )
    info = parse_trace(_write(tmp_path, "trace.log", text))
    assert info.parse_errors == []
    assert info.n_instr == 2
    first, second = info.head
    assert (first.mnemonic, first.operands, first.reads, first.writes) == ("nop", "", {}, {})
    assert (second.operands, second.writes) == ("", {"x1": 1})


def test_parse_trace_blank_lines_are_skipped_silently(tmp_path):
    # 空行（含 "\r\n" 那行）被略過。注意 open_text 用 universal newlines，
    # "\r\n" 在 parse_trace 看到前已變 "\n"，所以這裡不是在測 CR 處理。
    # parse_trace.py 131-132 (pass 1) and 326-327 (pass 2)
    text = (
        _HEADER
        + "1\t1\t80000000\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"
        + "\n"
        + "\r\n"
        + "2\t2\t80000004\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n"
        + "\n"
    )
    info = parse_trace(_write(tmp_path, "trace.log", text))
    assert info.parse_errors == []
    assert info.n_instr == 2
    assert info.implicit_jumps == 0
    assert (info.first_pc, info.last_pc) == (0x80000000, 0x80000004)


def test_pass1_blank_lines_do_not_reset_previous_instruction():
    # blank line between two non-contiguous instructions must not hide the
    # implicit jump (prev_pc survives the skipped line).
    lines = iter([
        "hdr",
        "1\t1\t80000000\t00000013\taddi\tx0,x0,0\tx0=0x0",
        "",
        "2\t2\t80000100\t00000013\taddi\tx0,x0,0\tx0=0x0",
    ])
    assert pt._pass1_trap_entries(lines) == (1, {0x80000100: 1}, {})


def test_decompressed_size_plain_file_is_byte_size(tmp_path):
    p = _write(tmp_path, "t.log", "abc")
    assert _decompressed_size(p) == 3


def test_decompressed_size_tiny_gz_is_untrustworthy(tmp_path):
    # parse_trace.py 170-171: below the 18-byte minimum gzip size
    p = tmp_path / "tiny.gz"
    # trailer would claim 100 bytes (>= file size), yet 17 bytes is too short
    p.write_bytes(b"\x00" * 13 + (100).to_bytes(4, "little"))
    assert _decompressed_size(str(p)) is None


def test_decompressed_size_gz_boundary_and_isize_checks(tmp_path):
    # exactly 18 bytes is *not* tiny: ISIZE (last 4 bytes, little endian) is read
    p = tmp_path / "edge.gz"
    p.write_bytes(b"\x00" * 14 + (18).to_bytes(4, "little"))
    assert _decompressed_size(str(p)) == 18
    # ISIZE smaller than the file itself (wrapped / bogus) -> None
    q = tmp_path / "wrap.gz"
    q.write_bytes(b"\x00" * 14 + (17).to_bytes(4, "little"))
    assert _decompressed_size(str(q)) is None
    # a real gzip stream reports the true uncompressed size
    r = tmp_path / "real.gz"
    r.write_bytes(gzip.compress(b"x" * 1000))
    assert _decompressed_size(str(r)) == 1000


# =============================================================================
# record._to_jsonable (record.py 25-26)
# =============================================================================


def test_to_jsonable_set_is_sorted_not_iteration_order():
    if list(frozenset({16, 8, 1})) == [1, 8, 16]:
        pytest.skip("此直譯器的 frozenset 迭代順序恰好已排序，無法區分「有排序」與「沒排序」")
    assert _to_jsonable(frozenset({16, 8, 1})) == [1, 8, 16]
    assert _to_jsonable({(2, "b"), (1, "a")}) == [[1, "a"], [2, "b"]]


def test_violation_to_dict_serialises_set_detail_sorted():
    v = Violation(index=1, kind="K", mnemonic="m", detail={"s": frozenset({16, 8, 1})},
                  pc=4, next_pc=None)
    assert v.to_dict()["detail"] == {"s": [1, 8, 16]}


def test_to_dict_stringifies_int_keys():
    t = TraceInfo.empty()
    t.implicit_targets = {5: 1}
    assert t.to_dict()["implicit_targets"] == {"5": 1}


# =============================================================================
# signature
# =============================================================================


def _mm(ibex_text):
    return Mismatch(
        n=1, ibex_idx=1, ibex_pc=0x1000, ibex_text=ibex_text, ibex_rd=None,
        ibex_val=None, spike_idx=1, spike_pc=0x1000, spike_text="x",
        spike_rd=None, spike_val=None,
    )


def _rec(family="MISMATCH", mismatches=(), deviation=None, violations=None):
    regr = RegrInfo.empty()
    regr.mismatches = list(mismatches)
    sim = SimInfo.empty()
    trace = TraceInfo.empty()
    trace.violations = violations or {}
    return FailureRecord(
        case_id=1, family=family, regr=regr, sim=sim, trace=trace,
        deviation=deviation, parse_errors=[],
    )


def _dev(kind, mnemonic):
    return Deviation(kind=kind, mnemonic=mnemonic, mclass=None, detail={},
                     source="t", confidence=1.0)


@pytest.mark.parametrize("dev", [None, _dev("NONE", "addi")])
def test_none_kind_mismatch_uses_first_token_of_first_ibex_text(dev):
    # signature.py 26-30, 45-48
    r = _rec(mismatches=[_mm("c.addi4spn\tx8,x2,16: s0:80000000"), _mm("zzz")], deviation=dev)
    # first token only (not the whole text, not the 2nd mismatch), normalized
    # "c.addi<N>spn" 是刻意釘住 normalize(mnemonic) 的輸出（ARCHITECTURE 5.3）
    assert signature(r).hard_key == ("MISMATCH", "NONE", "c.addi<N>spn")


def test_none_kind_mismatch_with_blank_or_missing_text_gives_empty_token():
    assert signature(_rec(mismatches=[_mm("")])).hard_key == ("MISMATCH", "NONE", "")
    assert signature(_rec(mismatches=[_mm("   ")])).hard_key == ("MISMATCH", "NONE", "")
    assert signature(_rec(mismatches=[])).hard_key == ("MISMATCH", "NONE", "")


def test_deviation_without_mnemonic_adds_class_but_no_key():
    # signature.py 106->109
    for mn in (None, ""):
        r = _rec(deviation=_dev("WRONG_RESULT", mn))
        assert _sys_viol(r) == (frozenset({("WRONG_RESULT", "OTHER")}), frozenset())
        assert signature(r).hard_key == ("MISMATCH", "WRONG_RESULT", "OTHER")


def test_deviation_with_mnemonic_adds_class_and_key():
    r = _rec(deviation=_dev("WRONG_RESULT", "sra"))
    cls = frozenset({("WRONG_RESULT", "sra")})
    assert _sys_viol(r) == (cls, cls)
    r2 = _rec(deviation=_dev("BRANCH_NOT_TAKEN", "beq"))
    assert _sys_viol(r2) == (
        frozenset({("BRANCH_NOT_TAKEN", "BR_EQ-")}),
        frozenset({("BRANCH_NOT_TAKEN", "beq")}),
    )


def test_deviation_ignored_for_non_mismatch_family_and_none_kind():
    assert _sys_viol(_rec(family="FATAL", deviation=_dev("WRONG_RESULT", "sra"))) == (
        frozenset(), frozenset())
    assert _sys_viol(_rec(deviation=_dev("NONE", "sra"))) == (frozenset(), frozenset())


def test_sys_viol_threshold_is_two_occurrences():
    r = _rec(violations={"WRONG_RESULT:add": 1, "WRONG_RESULT:sub": 2, "TRAP_NOT_TAKEN:ecall": 5})
    assert _sys_viol(r) == (
        frozenset({("WRONG_RESULT", "sub"), ("TRAP_NOT_TAKEN", "TRAP_INSTR")}),
        frozenset({("WRONG_RESULT", "sub"), ("TRAP_NOT_TAKEN", "ecall")}),
    )
