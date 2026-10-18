"""Unit tests for rfb.diverge._pcdelta_deviation() edge cases (whitespace-only
``ibex_text``, ARCHITECTURE.md section 4.4 pcdelta fallback)."""

from rfb.diverge import derive
from rfb.record import FailureRecord, Mismatch, RegrInfo, SimInfo, TraceInfo


def _mismatch(spike_pc, ibex_pc, ibex_text, spike_val=None, ibex_val=None):
    return Mismatch(
        n=1, ibex_idx=0, ibex_pc=ibex_pc, ibex_text=ibex_text, ibex_rd=None,
        ibex_val=ibex_val, spike_idx=0, spike_pc=spike_pc, spike_text="",
        spike_rd=None, spike_val=spike_val,
    )


def _record(mismatch):
    regr = RegrInfo(kind="MISMATCH", test_name=None, mismatches=[mismatch],
                     matched=0, mismatch_count=1, raw=[])
    return FailureRecord(
        case_id=0, family="MISMATCH", regr=regr, sim=SimInfo.empty(),
        trace=TraceInfo.empty(), deviation=None, parse_errors=[],
    )


def test_pcdelta_whitespace_ibex_text_no_raise():
    # spike_pc/ibex_pc far apart (>= 0x1000) -> "ibex continuous" branch,
    # no anchor window instructions recorded, so the code falls back to
    # anchor.ibex_text.split()[0]. A whitespace-only ibex_text used to
    # IndexError on that .split()[0]; it must now degrade to None cleanly.
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="   ")
    record = _record(m)
    dev = derive(record)
    assert dev.kind == "NONE"


def test_pcdelta_whitespace_ibex_text_wrong_result_branch_no_raise():
    # spike_pc == ibex_pc but values differ -> WRONG_RESULT branch, same
    # whitespace-only ibex_text fallback.
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x1000, ibex_text="   ",
                   spike_val=1, ibex_val=2)
    record = _record(m)
    dev = derive(record)
    assert dev.kind == "NONE"


def test_pcdelta_empty_ibex_text_no_raise():
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="")
    record = _record(m)
    dev = derive(record)
    assert dev.kind == "NONE"


def test_pcdelta_normal_ibex_text_still_works():
    # non-whitespace ibex_text: mnemonic still extracted normally.
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="addi x1,x0,1")
    record = _record(m)
    dev = derive(record)
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.mnemonic == "addi"
