"""rfb.diverge.derive() 分支覆蓋：anchor 視窗掃描、scan 後備、pcdelta 後備。"""

from rfb.diverge import derive
from rfb.record import (
    FailureRecord, Instr, Mismatch, RegrInfo, SimInfo, TraceInfo, Violation,
)


def _instr(mnemonic, pc=0x80000000):
    return Instr(index=0, pc=pc, size=4, mnemonic=mnemonic, operands="",
                 reads={}, writes={}, pa=None, store=None, load=None)


def _viol(kind, mnemonic, **detail):
    return Violation(index=0, kind=kind, mnemonic=mnemonic, detail=detail,
                     pc=0x80000000, next_pc=None)


def _mismatch(spike_pc=0x1000, ibex_pc=0x1000, ibex_text="", spike_val=None, ibex_val=None):
    return Mismatch(
        n=1, ibex_idx=0, ibex_pc=ibex_pc, ibex_text=ibex_text, ibex_rd=None,
        ibex_val=ibex_val, spike_idx=0, spike_pc=spike_pc, spike_text="",
        spike_rd=None, spike_val=spike_val,
    )


def _record(family="MISMATCH", mismatches=(), trace=None):
    regr = RegrInfo(kind=family, test_name=None, mismatches=list(mismatches),
                    matched=0, mismatch_count=len(mismatches), raw=[])
    return FailureRecord(
        case_id=0, family=family, regr=regr, sim=SimInfo.empty(),
        trace=trace if trace is not None else TraceInfo.empty(),
        deviation=None, parse_errors=[],
    )


def test_non_mismatch_family_uses_first_violation_scan():
    # 覆蓋 diverge.py:104（非 MISMATCH 且有 first_violation -> scan / 0.7）
    trace = TraceInfo.empty()
    trace.first_violation = _viol("TRAP_NOT_TAKEN", "ecall", pc_seen=1)
    dev = derive(_record(family="FATAL", trace=trace))
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.mnemonic == "ecall"
    assert dev.mclass == "TRAP_INSTR"
    assert dev.detail == {"pc_seen": 1}
    assert dev.source == "scan"
    assert dev.confidence == 0.7


def test_mismatch_falls_back_to_first_violation_when_no_windows():
    # 覆蓋 diverge.py:113（MISMATCH、無 anchor 視窗、有 first_violation）
    trace = TraceInfo.empty()
    trace.first_violation = _viol("WRONG_RESULT", "add", expected=1)
    dev = derive(_record(trace=trace))
    assert (dev.kind, dev.mnemonic, dev.mclass) == ("WRONG_RESULT", "add", "ADDSUB")
    assert dev.source == "scan"
    assert dev.confidence == 0.7


def test_anchor_window_finds_nearest_violation_backward():
    # 覆蓋 diverge.py:39-42；j 由 pos 往回掃，取最近（pos 上者優先於更早者）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add"), _instr("sub"), _instr("beq")]]
    trace.anchor_window_violations = [[_viol("A", "add"), _viol("B", "sub"), None]]
    trace.anchor_window_anchor_pos = [2]
    trace.first_violation = _viol("SCAN", "or")
    dev = derive(_record(trace=trace))
    assert dev.kind == "B"
    assert dev.mnemonic == "sub"
    assert dev.source == "anchor"
    assert dev.confidence == 0.9


def test_anchor_window_skips_window_with_out_of_range_pos():
    # 覆蓋 diverge.py:38（pos 越界 -> continue 到下一個視窗）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add")], [_instr("sub")], [_instr("and")]]
    trace.anchor_window_violations = [[_viol("BAD", "add")], [_viol("BAD2", "sub")],
                                      [_viol("GOOD", "and")]]
    # 第一個 pos 為負：此案例與 range(-1, -1, -1) 為空等價，即使拿掉 pos < 0
    # 也不會出錯，故不是這個測項真正守的東西；真正守的是第二個的 `pos >= len(vlist)`
    # （否則 vlist[1] 會 IndexError）。第三個有效。
    trace.anchor_window_anchor_pos = [-1, 1, 0]
    dev = derive(_record(trace=trace))
    assert dev.kind == "GOOD"
    assert dev.mnemonic == "and"
    assert dev.source == "anchor"


def test_anchor_window_all_none_moves_to_next_window():
    # 覆蓋 diverge.py:39->34（整個視窗掃完皆 None，換下一視窗）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add"), _instr("sub")], [_instr("and")]]
    trace.anchor_window_violations = [[None, None], [_viol("SECOND", "and")]]
    trace.anchor_window_anchor_pos = [1, 0]
    dev = derive(_record(trace=trace))
    assert dev.kind == "SECOND"
    assert dev.source == "anchor"


def test_anchor_window_ignores_violations_after_anchor_pos():
    # 只往回掃：pos 之後的 violation 不算 -> 無 anchor 結果，落到 scan 後備
    # 覆蓋 diverge.py:43、109->112
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add"), _instr("sub")]]
    trace.anchor_window_violations = [[None, _viol("AFTER", "sub")]]
    trace.anchor_window_anchor_pos = [0]
    trace.first_violation = _viol("SCAN", "or")
    dev = derive(_record(trace=trace))
    assert dev.kind == "SCAN"
    assert dev.source == "scan"
    assert dev.confidence == 0.7


def test_anchor_none_and_no_violation_and_no_mismatch_is_none():
    # 覆蓋 diverge.py:43、49：anchor 無結果、無 first_violation、無 mismatches
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add")]]
    trace.anchor_window_violations = [[None]]
    trace.anchor_window_anchor_pos = [0]
    dev = derive(_record(trace=trace))
    assert dev.kind == "NONE"
    assert dev.source == "none"
    assert dev.mnemonic is None
    assert dev.confidence == 0.0


def test_pcdelta_no_mismatches_returns_none_deviation():
    # 覆蓋 diverge.py:49
    dev = derive(_record(mismatches=[]))
    assert dev.kind == "NONE" and dev.source == "none"


def test_pcdelta_wrong_result_uses_anchor_window_instr():
    # 覆蓋 diverge.py:59-62（取 anchor instr）與 88-96（WRONG_RESULT）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("lw"), _instr("mul")]]
    trace.anchor_window_anchor_pos = [1]
    m = _mismatch(spike_pc=0x2000, ibex_pc=0x2000, ibex_text="add x1, x2",
                  spike_val=5, ibex_val=7)
    dev = derive(_record(mismatches=[m], trace=trace))
    assert dev.kind == "WRONG_RESULT"
    assert dev.mnemonic == "mul"          # 來自視窗 instr，不是 ibex_text 的 add
    assert dev.mclass == "MUL"
    assert dev.detail == {"expected": 5, "actual": 7}
    assert dev.source == "pcdelta"
    assert dev.confidence == 0.5


def test_pcdelta_wrong_result_falls_back_to_ibex_text():
    # 覆蓋 diverge.py:82-95 的 ibex_text 後備
    m = _mismatch(ibex_text="sub x1, x2, x3", spike_val=1, ibex_val=2)
    dev = derive(_record(mismatches=[m]))
    assert dev.kind == "WRONG_RESULT"
    assert dev.mnemonic == "sub"
    assert dev.mclass == "ADDSUB"
    assert dev.detail == {"expected": 1, "actual": 2}


def test_pcdelta_wrong_result_no_mnemonic_returns_none():
    # ibex_text 空且無視窗 -> 86-87 回 None -> NONE
    m = _mismatch(ibex_text="", spike_val=1, ibex_val=2)
    assert derive(_record(mismatches=[m])).kind == "NONE"


def test_pcdelta_equal_values_or_missing_values_no_deviation():
    # pc 相同但值相同 / 缺值 -> 不是 WRONG_RESULT
    same = _mismatch(ibex_text="add", spike_val=3, ibex_val=3)
    assert derive(_record(mismatches=[same])).kind == "NONE"
    missing = _mismatch(ibex_text="add", spike_val=None, ibex_val=3)
    assert derive(_record(mismatches=[missing])).kind == "NONE"
    # 另一側缺值也不是 WRONG_RESULT
    missing2 = _mismatch(ibex_text="add", spike_val=3, ibex_val=None)
    assert derive(_record(mismatches=[missing2])).kind == "NONE"


def test_pcdelta_trap_not_taken_uses_prev_instr_of_anchor_window():
    # 覆蓋 diverge.py:59-62 中 pos > 0 的 prev_instr 分支與 TRAP_NOT_TAKEN
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("beq"), _instr("add")]]
    trace.anchor_window_anchor_pos = [1]
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="sub x1")
    dev = derive(_record(mismatches=[m], trace=trace))
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.mnemonic == "beq"          # prev_instr，不是 anchor(add) 或 ibex_text(sub)
    assert dev.mclass == "BR_BEQ"
    assert dev.detail == {"spike_pc": 0x1000, "ibex_pc": 0x3000}
    assert dev.source == "pcdelta"
    assert dev.confidence == 0.5


def test_pcdelta_anchor_at_window_start_has_no_prev_falls_back_to_ibex_text():
    # 覆蓋 diverge.py:61 的 pos == 0 -> prev_instr None
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add")]]
    trace.anchor_window_anchor_pos = [0]
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="jal x1, 8")
    dev = derive(_record(mismatches=[m], trace=trace))
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.mnemonic == "jal"
    assert dev.mclass == "JUMP"


def test_pcdelta_window_scan_skips_invalid_pos_then_uses_valid_one():
    # 覆蓋 diverge.py:59 條件（pos 越界的視窗不用）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add")], [_instr("lw"), _instr("sub")]]
    trace.anchor_window_anchor_pos = [5, 1]
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="")
    dev = derive(_record(mismatches=[m], trace=trace))
    assert dev.mnemonic == "lw"


def test_pcdelta_small_pc_delta_different_pc_is_none():
    # pc 差 < 0x1000 且 pc 不同 -> 96: return None
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x1004, ibex_text="add",
                  spike_val=1, ibex_val=2)
    assert derive(_record(mismatches=[m])).kind == "NONE"


def test_pcdelta_threshold_is_exactly_0x1000():
    # 差值恰為 0x1000 -> TRAP_NOT_TAKEN；0x0FFF -> 不是（pc 不同、無值）-> NONE
    at = _mismatch(spike_pc=0x1000, ibex_pc=0x2000, ibex_text="jal x1, 8")
    dev = derive(_record(mismatches=[at]))
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.confidence == 0.5
    below = _mismatch(spike_pc=0x1000, ibex_pc=0x1FFF, ibex_text="jal x1, 8")
    assert derive(_record(mismatches=[below])).kind == "NONE"


def test_pcdelta_window_scan_skips_negative_pos_and_uses_next_window():
    # 視窗 [[add],[lw,sub]]、pos [-1,1]：負 pos 不可用（否則 window[-1]=add 被取到）
    trace = TraceInfo.empty()
    trace.anchor_windows = [[_instr("add")], [_instr("lw"), _instr("sub")]]
    trace.anchor_window_anchor_pos = [-1, 1]
    m = _mismatch(spike_pc=0x1000, ibex_pc=0x3000, ibex_text="")
    dev = derive(_record(mismatches=[m], trace=trace))
    assert dev.mnemonic == "lw"
    assert dev.confidence == 0.5


def test_pcdelta_threshold_uses_absolute_value_both_directions():
    # spike_pc 大於 ibex_pc：abs 才會命中；若寫成 ibex_pc - spike_pc 會是負數而漏掉。
    above = _mismatch(spike_pc=0x2000, ibex_pc=0x1000, ibex_text="jal x1, 8")
    dev = derive(_record(mismatches=[above]))
    assert dev.kind == "TRAP_NOT_TAKEN"
    assert dev.detail == {"spike_pc": 0x2000, "ibex_pc": 0x1000}
    # 鏡像的下方案例：差 0x0FFF -> NONE
    below = _mismatch(spike_pc=0x1FFF, ibex_pc=0x1000, ibex_text="jal x1, 8")
    assert derive(_record(mismatches=[below])).kind == "NONE"
