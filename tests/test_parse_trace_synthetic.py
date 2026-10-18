"""Hand-designed synthetic trace.log used to pin down exact expected values
that the public-set tests can't (public traces are too large to hand-verify).

Layout (10 valid instructions, indices 0-9, plus one malformed trailer line):

idx  pc          mnemonic  notes
0    80000000    addi      non-control-flow, next pc matches pc+size (no jump)
1    80000004    addi      non-control-flow, next pc jumps to 0x80000100  -> implicit jump #1
2    80000100    addi      next pc matches pc+size (no jump)
3    80000104    jal       control-flow, next pc jumps to 0x80000200 (not counted, control flow)
4    80000200    addi      non-control-flow, next pc jumps to 0x80000300  -> implicit jump #2
5    80000300    sw        PA == sig addr, store recorded
6    80000304    sw        PA != sig addr, store NOT recorded
7    80000308    addi      first occurrence of anchor pc, writes x7=0x7
8    8000030c    jal       control-flow, next pc jumps back to 0x80000308 (not counted)
9    80000308    addi      second occurrence of anchor pc, writes x7=0x4d (matches ibex_val)

Then one trailing line with too few columns (parse error, not counted in n_instr).
"""

import rfb.parse_trace as parse_trace_module
from rfb.parse_trace import _MAX_CHECK_ERRORS, parse_trace
from rfb.record import Mismatch, RegrInfo

_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"

_LINES = [
    "1\t1\t80000000\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    "2\t2\t80000004\t00000093\taddi\tx1,x0,1\tx1=0x00000001\n",
    "3\t3\t80000100\t00000113\taddi\tx2,x0,2\tx2=0x00000002\n",
    "4\t4\t80000104\t006000ef\tjal\tx1,0x0\tx1=0x80000108\n",
    "5\t5\t80000200\t00000193\taddi\tx3,x0,3\tx3=0x00000003\n",
    "6\t6\t80000300\t0062a023\tsw\tx6,0(x5)\tPA:0x8ffffffc store:0x000000aa\n",
    "7\t7\t80000304\t0072a023\tsw\tx7,0(x5)\tPA:0x12345678 store:0x000000bb\n",
    "8\t8\t80000308\t00000393\taddi\tx7,x0,7\tx7=0x00000007\n",
    "9\t9\t8000030c\t008000ef\tjal\tx9,0x0\tx9=0x80000310\n",
    "10\t10\t80000308\t04d00393\taddi\tx7,x0,77\tx7=0x0000004d\n",
    "11\t11\t80000400\n",  # malformed: not enough columns
]


def _write_trace(tmp_path):
    path = tmp_path / "trace.log"
    with open(path, "w") as f:
        f.write(_HEADER)
        f.writelines(_LINES)
    return path


def test_implicit_jumps_exact_count_and_targets(tmp_path):
    path = _write_trace(tmp_path)
    info = parse_trace(str(path))

    assert info.n_instr == 10
    assert info.implicit_jumps == 2
    assert info.implicit_targets == {0x80000100: 1, 0x80000300: 1}


def test_sig_writes_only_records_sig_addr(tmp_path):
    path = _write_trace(tmp_path)
    info = parse_trace(str(path))

    assert info.sig_writes == [0xAA]


def test_malformed_line_reported_and_excluded(tmp_path):
    path = _write_trace(tmp_path)
    info = parse_trace(str(path))

    assert info.n_instr == 10
    assert len(info.parse_errors) == 1
    assert "line 12" in info.parse_errors[0]
    assert "not enough columns" in info.parse_errors[0]


def test_anchor_hits_second_occurrence_only(tmp_path):
    path = _write_trace(tmp_path)
    anchor = Mismatch(
        n=1,
        ibex_idx=0,
        ibex_pc=0x80000308,
        ibex_text="addi x7,x0,77: x7:0000004d",
        ibex_rd="x7",
        ibex_val=0x4D,
        spike_idx=0,
        spike_pc=0x80000308,
        spike_text="addi x7,x0,77: x7:0000004d",
        spike_rd="x7",
        spike_val=0x4D,
    )
    regr_info = RegrInfo(
        kind="MISMATCH",
        test_name=None,
        mismatches=[anchor],
        matched=None,
        mismatch_count=None,
        raw=[],
    )
    info = parse_trace(str(path), regr_info)

    assert info.anchor_hits == 1
    assert len(info.anchor_windows) == 1
    window = info.anchor_windows[0]
    # lo = max(0, 9-16) = 0, hi = min(10, 9+4+1) = 10 -> window covers all instrs
    assert len(window) == 10
    # index 7 (first occurrence, pc matches but write value doesn't) must NOT
    # be the one credited as the hit.
    assert window[7].pc == 0x80000308
    assert window[7].writes.get("x7") == 0x7
    # index 9 (second occurrence) is the actual anchor hit.
    assert window[9].pc == 0x80000308
    assert window[9].writes.get("x7") == 0x4D


# --- malformed single-instruction check(): isolated, not fatal to the trace --
#
# 10 lines, one of them (index 4, a "beq" with a non-hex target token) makes
# isa.check() raise ValueError. Before and after it there are two known
# WRONG_RESULT violations that must still be detected -- a single malformed
# instruction must not blank out the whole trace (see parse_errors handling
# via isa.check_safe).

_CHECK_FAIL_LINES = [
    "1\t1\t80000000\t00000013\taddi\tx1,x0,5\tx1=0x00000005\n",
    "2\t2\t80000004\t0020a1b3\tadd\tx3,x1,x2\tx1:0x00000005 x2:0x00000003 x3=0x000003e7\n",
    "3\t3\t80000008\t00000213\taddi\tx4,x0,1\tx4=0x00000001\n",
    "4\t4\t8000000c\t00000293\taddi\tx5,x0,1\tx5=0x00000001\n",
    "5\t5\t80000010\t00208463\tbeq\tx1,x2,zz\tx1:0x00000001 x2:0x00000001\n",
    "6\t6\t80000014\t00000313\taddi\tx6,x0,1\tx6=0x00000001\n",
    "7\t7\t80000018\t0020a3b3\tadd\tx7,x1,x2\tx1:0x00000005 x2:0x00000003 x7=0x0000006f\n",
    "8\t8\t8000001c\t00000413\taddi\tx8,x0,1\tx8=0x00000001\n",
    "9\t9\t80000020\t00000493\taddi\tx9,x0,1\tx9=0x00000001\n",
    "10\t10\t80000024\t00000513\taddi\tx10,x0,1\tx10=0x00000001\n",
]


def _write_lines(tmp_path, lines, name="trace.log"):
    path = tmp_path / name
    with open(path, "w") as f:
        f.write(_HEADER)
        f.writelines(lines)
    return path


def test_malformed_check_isolated_to_one_instruction(tmp_path):
    path = _write_lines(tmp_path, _CHECK_FAIL_LINES)
    info = parse_trace(str(path))

    assert info.n_instr == 10
    # The bad "beq" (index 4) must not blow up the whole trace: the
    # WRONG_RESULT violation before it (index 1) is still the first one...
    assert info.first_violation is not None
    assert info.first_violation.kind == "WRONG_RESULT"
    assert info.first_violation.mnemonic == "add"
    assert info.first_violation.index == 1
    # ...and the one after it (index 6) is still detected, proving the scan
    # kept going past the malformed instruction.
    assert info.violations.get("WRONG_RESULT:add") == 2
    assert any(v.index == 6 for v in info.violation_samples)
    assert any("check failed" in e for e in info.parse_errors)


def test_malformed_check_cap_at_max_errors(tmp_path):
    # 10 instructions, all with a malformed immediate -> check() raises for
    # every one of them (addi's WRONG_RESULT check runs regardless of
    # next_pc, unlike branch targets, so even the last instruction -- which
    # is checked with next_pc=None -- still raises). parse_errors should log
    # at most _MAX_CHECK_ERRORS individual entries, then a single summary
    # line covering all 10.
    lines = []
    for n in range(10):
        pc = 0x80000000 + n * 4
        lines.append(
            f"{n + 1}\t{n + 1}\t{pc:x}\t00000013\taddi\tx1,x2,zz\t"
            "x2:0x00000001 x1=0x00000002\n"
        )
    path = _write_lines(tmp_path, lines, name="trace2.log")
    info = parse_trace(str(path))

    assert info.n_instr == 10
    check_failed_lines = [e for e in info.parse_errors if e.startswith("instr ")]
    assert len(check_failed_lines) == _MAX_CHECK_ERRORS
    assert "check failed on 10 instructions total" in info.parse_errors


def _malformed_addi_lines(n: int) -> list[str]:
    lines = []
    for i in range(n):
        pc = 0x80000000 + i * 4
        lines.append(
            f"{i + 1}\t{i + 1}\t{pc:x}\t00000013\taddi\tx1,x2,zz\t"
            "x2:0x00000001 x1=0x00000002\n"
        )
    return lines


def test_malformed_check_exactly_at_max_errors(tmp_path):
    # Exactly _MAX_CHECK_ERRORS (8) malformed instructions: every one gets
    # an individual "instr N (...): check failed: ..." entry, and since the
    # count never exceeds the cap, no summary line is emitted.
    lines = _malformed_addi_lines(_MAX_CHECK_ERRORS)
    path = _write_lines(tmp_path, lines, name="trace3.log")
    info = parse_trace(str(path))

    assert info.n_instr == _MAX_CHECK_ERRORS
    check_failed_lines = [e for e in info.parse_errors if e.startswith("instr ")]
    assert len(check_failed_lines) == _MAX_CHECK_ERRORS
    assert not any("instructions total" in e for e in info.parse_errors)


def test_malformed_check_one_over_max_errors(tmp_path):
    # _MAX_CHECK_ERRORS + 1 (9) malformed instructions: still only
    # _MAX_CHECK_ERRORS individual entries logged, plus a summary line
    # covering all 9 once the cap is exceeded.
    lines = _malformed_addi_lines(_MAX_CHECK_ERRORS + 1)
    path = _write_lines(tmp_path, lines, name="trace4.log")
    info = parse_trace(str(path))

    assert info.n_instr == _MAX_CHECK_ERRORS + 1
    check_failed_lines = [e for e in info.parse_errors if e.startswith("instr ")]
    assert len(check_failed_lines) == _MAX_CHECK_ERRORS
    assert (
        f"check failed on {_MAX_CHECK_ERRORS + 1} instructions total"
        in info.parse_errors
    )


# --- 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: shadow x2 --------
# observe() ordering & staleness, at the full parse_trace level -----------


def test_c_addi4spn_own_write_does_not_affect_own_check(tmp_path):
    # instr0 sets shadow to 0x100 via an ordinary addi write to x2.
    # instr1 is a (synthetically unrealistic, but valid for pinning down
    # observe() ordering) c.addi4spn whose Regs field ALSO carries an x2=
    # write token alongside its real rd (x8) write: real ISA c.addi4spn can
    # never write x2 (rd is one of x8-x15), so this token exists purely to
    # prove that isa.observe() is applied *after* check() -- if it were
    # applied before, this instruction's own bogus x2=0x999 write would be
    # used as the shadow for its own recompute, producing a false
    # WRONG_RESULT (0x999+4=0x9a3 != actual write 0x104). With the correct
    # ordering, the recompute still uses the *previous* shadow (0x100), so
    # 0x100+4=0x104 matches the real x8 write and no violation is raised.
    lines = [
        "1\t1\t80000000\t00000013\taddi\tx2,x0,0x100\tx2=0x00000100\n",
        "2\t2\t80000004\t0000e013\tc.addi4spn\tx8,x2,4\tx8=0x00000104 x2=0x00000999\n",
        "3\t3\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_own_write.log")
    info = parse_trace(str(path))

    assert info.violations == {}


def test_c_addi4spn_stale_shadow_flagged_wrong_result(tmp_path):
    # instr0 sets shadow to 0x100. instr1 is a c.addi4spn with no explicit
    # x2 read whose real write (0x999) does not match what the (correctly
    # tracked, not stale in this particular example) shadow predicts
    # (0x100+4=0x104): the checker trusts the shadow and flags
    # WRONG_RESULT, exactly as it would for a genuine forwarding/regfile
    # bug that silently changed x2 without the tracer ever printing a
    # fresh read of it (docs/ARCHITECTURE.md 4.3 known-risk note).
    lines = [
        "1\t1\t80000000\t00000013\taddi\tx2,x0,0x100\tx2=0x00000100\n",
        "2\t2\t80000004\t0000e013\tc.addi4spn\tx8,x2,4\tx8=0x00000999\n",
        "3\t3\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_stale_shadow.log")
    info = parse_trace(str(path))

    assert info.violations == {"WRONG_RESULT:c.addi4spn": 1}
    assert info.first_violation is not None
    assert info.first_violation.kind == "WRONG_RESULT"
    assert info.first_violation.mnemonic == "c.addi4spn"


def test_c_addi4spn_shadow_synced_from_failed_check_instruction(tmp_path):
    # instr0 is a "beq" whose target token is not valid hex ("zz"): its own
    # isa.check_safe() call fails (recorded as a "check failed" parse_error)
    # -- but it also writes x2 = 0x200. isa.observe() must still run on it
    # (parse_trace.py calls observe() unconditionally after check_safe,
    # regardless of whether check_safe errored), so the shadow is synced to
    # 0x200 anyway. instr1 is a c.addi4spn with no explicit x2 read whose
    # write (0x210) matches shadow + imm (0x200+16=0x210): no violation.
    lines = [
        "1\t1\t80000000\t00000013\tbeq\tx1,x2,zz\tx1:0x00000001 x2=0x00000200\n",
        "2\t2\t80000004\t0000e013\tc.addi4spn\tx8,x2,16\tx8=0x00000210\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_check_fail_shadow_ok.log")
    info = parse_trace(str(path))

    assert any("check failed" in e for e in info.parse_errors)
    assert info.violations == {}


def test_c_addi4spn_shadow_from_failed_check_instruction_wrong_result(tmp_path):
    # Same setup, but instr1's c.addi4spn write (0x999) does NOT match
    # shadow + imm (0x200+16=0x210 != 0x999): the shadow synced from the
    # failed-check instruction still drives the recompute, so WRONG_RESULT
    # fires.
    lines = [
        "1\t1\t80000000\t00000013\tbeq\tx1,x2,zz\tx1:0x00000001 x2=0x00000200\n",
        "2\t2\t80000004\t0000e013\tc.addi4spn\tx8,x2,16\tx8=0x00000999\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_check_fail_shadow_bad.log")
    info = parse_trace(str(path))

    assert any("check failed" in e for e in info.parse_errors)
    assert info.violations == {"WRONG_RESULT:c.addi4spn": 1}
    assert info.first_violation is not None
    assert info.first_violation.kind == "WRONG_RESULT"
    assert info.first_violation.mnemonic == "c.addi4spn"


# --- READ_MISMATCH at the full parse_trace level ----------------------------


def _anchor_regr(pc, text="add x6,x5,x0: x6:0000000b", val=0xB):
    m = Mismatch(
        n=1, ibex_idx=0, ibex_pc=pc, ibex_text=text, ibex_rd="x6", ibex_val=val,
        spike_idx=0, spike_pc=pc, spike_text=text, spike_rd="x6", spike_val=val,
    )
    return RegrInfo(kind="MISMATCH", test_name=None, mismatches=[m], matched=None,
                    mismatch_count=None, raw=[])


_RM_LINES = [
    "1\t1\t80000000\t00500293\taddi\tx5,x0,10\tx5=0x0000000a\n",
    "2\t2\t80000004\t00028333\tadd\tx6,x5,x0\tx5:0x0000000b x6=0x0000000b\n",
    "3\t3\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
]


def test_read_mismatch_counted_but_not_first_violation(tmp_path):
    path = _write_lines(tmp_path, _RM_LINES, name="trace_rm.log")
    info = parse_trace(str(path))

    assert info.violations == {"READ_MISMATCH:add": 1}
    assert info.first_violation is None
    assert [s.kind for s in info.violation_samples] == ["READ_MISMATCH"]


def test_read_mismatch_does_not_shadow_later_first_violation(tmp_path):
    lines = _RM_LINES[:2] + [
        # jal at 0x80000008 whose next pc is not the target -> non-READ violation
        "3\t3\t80000008\t008000ef\tjal\tx9,0x10\tx9=0x8000000c\n",
        "4\t4\t80000100\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_rm2.log")
    info = parse_trace(str(path))

    assert info.violations.get("READ_MISMATCH:add") == 1
    assert info.first_violation is not None
    assert info.first_violation.kind != "READ_MISMATCH"


def test_read_mismatch_only_violation_in_anchor_window_is_deviation(tmp_path):
    from rfb.diverge import derive
    from rfb.record import FailureRecord, SimInfo

    path = _write_lines(tmp_path, _RM_LINES, name="trace_rm3.log")
    regr = _anchor_regr(0x80000004)
    info = parse_trace(str(path), regr)
    assert info.first_violation is None
    assert info.anchor_hits == 1

    rec = FailureRecord(case_id=0, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
                        trace=info, deviation=None, parse_errors=[])
    dev = derive(rec)
    assert dev.kind == "READ_MISMATCH"
    assert dev.mnemonic == "add"
    assert dev.source == "anchor"


def test_read_mismatch_not_reported_after_bad_pc_hex_line(tmp_path):
    lines = [
        "1\t1\t80000000\t00500293\taddi\tx5,x0,10\tx5=0x0000000a\n",
        "2\t2\tZZZZ\t00500293\taddi\tx5,x0,11\tx5=0x0000000b\n",  # pc not hex
        "3\t3\t80000004\t00028333\tadd\tx6,x5,x0\tx5:0x0000000c x6=0x0000000c\n",
        "4\t4\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_rm5.log")
    info = parse_trace(str(path))

    assert any("bad pc/insn hex" in e for e in info.parse_errors)
    assert info.violations == {}


def test_read_mismatch_not_reported_after_malformed_line(tmp_path):
    lines = [
        "1\t1\t80000000\t00500293\taddi\tx5,x0,10\tx5=0x0000000a\n",
        "2\t2\t80000004\n",  # malformed: may have rewritten x5, unseen
        "3\t3\t80000004\t00028333\tadd\tx6,x5,x0\tx5:0x0000000b x6=0x0000000b\n",
        "4\t4\t80000008\t00000013\taddi\tx0,x0,0\tx0=0x00000000\n",
    ]
    path = _write_lines(tmp_path, lines, name="trace_rm4.log")
    info = parse_trace(str(path))

    assert info.violations == {}
    assert len(info.parse_errors) == 1


# --- CSR X-origin (hknegu, PLAN.md 4c 公開合併 0.96 根因) ----------------------


def _x_origins(tmp_path, rows):
    """rows: (mnemonic, operands, regs_field); returns TraceInfo.x_origins."""
    path = tmp_path / "trace_x.log"
    with open(path, "w") as f:
        f.write(_HEADER)
        for i, (mn, ops, regs) in enumerate(rows):
            pc = 0x80000000 + 4 * i
            f.write(f"{i + 1}\t{i + 1}\t{pc:08x}\t00000013\t{mn}\t{ops}\t{regs}\n")
    return parse_trace(str(path)).x_origins


def test_x_origin_counts_csr_read_with_x_writeback(tmp_path):
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5a5a5  x1=0x00000X00")] * 3
    assert _x_origins(tmp_path, rows) == {"mip:.....X..": 3}


def test_x_origin_skipped_when_source_read_has_x(tmp_path):
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5X5a5  x1=0x00000X00")] * 3
    assert _x_origins(tmp_path, rows) == {}


def test_x_origin_lowercase_x_mask_normalized(tmp_path):
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5a5a5  x1=0x0000x000")]
    assert _x_origins(tmp_path, rows) == {"mip:....X...": 1}


def test_x_origin_mask_padded_to_eight_digits(tmp_path):
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5a5a5  x1=0xX")]
    assert _x_origins(tmp_path, rows) == {"mip:.......X": 1}


def test_x_origin_csr_name_is_second_operand(tmp_path):
    rows = [("csrrs", "x5,mstatus,x0", "x5=0xXXXXXXXX")]
    assert list(_x_origins(tmp_path, rows)) == ["mstatus:XXXXXXXX"]


def test_x_origin_ignores_non_csr_instruction(tmp_path):
    rows = [("lw", "x1,0(x2)", "x2:0x80001000  x1=0x00000X00")] * 3
    assert _x_origins(tmp_path, rows) == {}


def test_x_origin_register_names_alone_do_not_count(tmp_path):
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5a5a5  x1=0x00000000")]
    assert _x_origins(tmp_path, rows) == {}


def test_x_origin_counting_stops_at_max_instr(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_trace_module, "_MAX_INSTR", 3)
    rows = [("csrrw", "x1,mip,x13", "x13:0xa5a5a5a5  x1=0x00000X00")] * 5
    assert _x_origins(tmp_path, rows) == {"mip:.....X..": 3}
