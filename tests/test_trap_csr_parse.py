"""Trap CSR prediction through the real ``parse_trace`` path (R1-4): the
TRAP_* / XRET_TARGET kinds show up in ``TraceInfo.violations`` and the
check_safe -> observe order is right (a CSR read is checked against the
prediction made by the *previous* line, before its own observe pops it)."""

from rfb.parse_trace import parse_trace

_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
_BASE = 0x80000000
_ECALL_PC = 0x80001004
_MRET_PC = 0x80000010


def _ln(pc, mn, ops="", regs=""):
    insn = "0001" if mn.startswith("c.") else "00000013"
    return f"1\t1\t{pc:08x}\t{insn}\t{mn}\t{ops}\t{regs}\n"


def _run(tmp_path, lines):
    p = tmp_path / "trace.log"
    p.write_text(_HEADER + "".join(lines))
    return parse_trace(str(p))


def _prologue():
    return [
        _ln(0x80001000, "csrrw", "x0,mtvec,x6", "x6:0x80000001"),
        _ln(_ECALL_PC, "ecall"),
    ]


def _handler(mcause=0xB, mepc=_ECALL_PC, mtval=0, ret=_ECALL_PC):
    return [
        _ln(_BASE, "csrrs", "x5,mcause,x0", f"x5=0x{mcause:08x}"),
        _ln(_BASE + 4, "csrrs", "x6,mepc,x0", f"x6=0x{mepc:08x}"),
        _ln(_BASE + 8, "csrrs", "x7,mtval,x0", f"x7=0x{mtval:08x}"),
        _ln(_MRET_PC - 4, "addi", "x0,x0,0", "x0=0x0"),
        _ln(_MRET_PC, "mret"),
        _ln(ret, "addi", "x0,x0,0", "x0=0x0"),
    ]


def test_clean_ecall_handler_no_violation(tmp_path):
    info = _run(tmp_path, _prologue() + _handler())
    assert info.violations == {}
    assert info.first_violation is None


def test_trap_mcause_reported_in_real_parse(tmp_path):
    info = _run(tmp_path, _prologue() + _handler(mcause=0x7))
    assert info.violations == {"TRAP_MCAUSE:csrrs": 1}
    v = info.first_violation
    assert v.kind == "TRAP_MCAUSE" and v.pc == _BASE
    assert v.detail["expected"] == [11] and v.detail["actual"] == 7


def test_trap_mepc_reported_in_real_parse(tmp_path):
    # the handler read adopts the (wrong) value as the mret target, so a
    # matching return address shows up as XRET_TARGET too only when it differs
    info = _run(tmp_path, _prologue() + _handler(mepc=_ECALL_PC + 8, ret=_ECALL_PC + 8))
    assert info.violations == {"TRAP_MEPC:csrrs": 1}
    assert info.first_violation.kind == "TRAP_MEPC"


def test_trap_mtval_reported_in_real_parse(tmp_path):
    info = _run(tmp_path, _prologue() + _handler(mtval=0x1234))
    assert info.violations == {"TRAP_MTVAL:csrrs": 1}


def test_xret_target_reported_in_real_parse(tmp_path):
    info = _run(tmp_path, _prologue() + _handler(ret=0x80002000))
    assert info.violations == {"XRET_TARGET:mret": 1}
    v = info.first_violation
    assert v.kind == "XRET_TARGET"
    assert v.detail == {"expected": _ECALL_PC, "actual": 0x80002000}


def test_all_kinds_together_and_order(tmp_path):
    info = _run(tmp_path, _prologue() + _handler(mcause=7, mtval=5, ret=0x80002000))
    assert info.violations == {
        "TRAP_MCAUSE:csrrs": 1, "TRAP_MTVAL:csrrs": 1, "XRET_TARGET:mret": 1,
    }
    assert info.first_violation.kind == "TRAP_MCAUSE"


def test_read_on_trap_entry_line_itself_checked_after_prediction(tmp_path):
    # Order: ecall line (predicts) -> csr read line (checked) -> its own observe
    # pops the prediction, so a second read of the same CSR is not checked again.
    lines = _prologue() + [
        _ln(_BASE, "csrrs", "x5,mcause,x0", "x5=0x00000007"),
        _ln(_BASE + 4, "csrrs", "x5,mcause,x0", "x5=0x00000007"),
    ]
    info = _run(tmp_path, lines)
    assert info.violations == {"TRAP_MCAUSE:csrrs": 1}
