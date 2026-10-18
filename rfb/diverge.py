"""Divergence-point candidate -> Deviation. See docs/ARCHITECTURE.md 4.4.

Public interface: ``derive(record: FailureRecord) -> Deviation``.
"""

from __future__ import annotations

from rfb import isa
from rfb.record import Deviation, FailureRecord


def _none() -> Deviation:
    return Deviation(kind="NONE", mnemonic=None, mclass=None, detail={}, source="none", confidence=0.0)


def _from_violation(v, source: str, confidence: float) -> Deviation:
    return Deviation(
        kind=v.kind,
        mnemonic=v.mnemonic,
        mclass=isa.mclass(v.mnemonic),
        detail=dict(v.detail),
        source=source,
        confidence=confidence,
    )


def _anchor_deviation(trace) -> Deviation | None:
    """Scan each anchor window backward from the anchor for the nearest
    Violation (section 4.4, source="anchor", confidence 0.9)."""
    windows = trace.anchor_windows
    win_violations = trace.anchor_window_violations
    anchor_pos = trace.anchor_window_anchor_pos
    n = min(len(windows), len(win_violations), len(anchor_pos))
    for w in range(n):
        vlist = win_violations[w]
        pos = anchor_pos[w]
        if pos < 0 or pos >= len(vlist):
            continue
        for j in range(pos, -1, -1):
            v = vlist[j]
            if v is not None:
                return _from_violation(v, source="anchor", confidence=0.9)
    return None


def _pcdelta_deviation(record: FailureRecord) -> Deviation | None:
    mismatches = record.regr.mismatches
    if not mismatches:
        return None
    anchor = mismatches[0]
    spike_pc = anchor.spike_pc
    ibex_pc = anchor.ibex_pc
    # Find the anchor instruction (and its predecessor) among the recorded
    # anchor window instructions, since that's the only place we have the
    # actual Instr objects for the divergence neighbourhood.
    anchor_instr = None
    prev_instr = None
    for window, pos in zip(record.trace.anchor_windows, record.trace.anchor_window_anchor_pos):
        if 0 <= pos < len(window):
            anchor_instr = window[pos]
            prev_instr = window[pos - 1] if pos > 0 else None
            break
    if abs(spike_pc - ibex_pc) >= 0x1000:
        # "ibex continuous" -- best-effort proxy: prev instruction (if any)
        # falls straight through to the anchor pc.
        ibex_toks = anchor.ibex_text.split() if anchor.ibex_text else []
        mnemonic = prev_instr.mnemonic if prev_instr is not None else (
            ibex_toks[0] if ibex_toks else None
        )
        if mnemonic is None:
            return None
        return Deviation(
            kind="TRAP_NOT_TAKEN",
            mnemonic=mnemonic,
            mclass=isa.mclass(mnemonic),
            detail={"spike_pc": spike_pc, "ibex_pc": ibex_pc},
            source="pcdelta",
            confidence=0.5,
        )
    if spike_pc == ibex_pc and anchor.spike_val is not None and anchor.ibex_val is not None \
            and anchor.spike_val != anchor.ibex_val:
        ibex_toks = anchor.ibex_text.split() if anchor.ibex_text else []
        mnemonic = anchor_instr.mnemonic if anchor_instr is not None else (
            ibex_toks[0] if ibex_toks else None
        )
        if mnemonic is None:
            return None
        return Deviation(
            kind="WRONG_RESULT",
            mnemonic=mnemonic,
            mclass=isa.mclass(mnemonic),
            detail={"expected": anchor.spike_val, "actual": anchor.ibex_val},
            source="pcdelta",
            confidence=0.5,
        )
    return None


def derive(record: FailureRecord) -> Deviation:
    trace = record.trace
    if record.family != "MISMATCH":
        v = trace.first_violation
        if v is not None:
            return _from_violation(v, source="scan", confidence=0.7)
        return _none()

    if trace.anchor_windows:
        dev = _anchor_deviation(trace)
        if dev is not None:
            return dev

    if trace.first_violation is not None:
        return _from_violation(trace.first_violation, source="scan", confidence=0.7)

    dev = _pcdelta_deviation(record)
    if dev is not None:
        return dev

    return _none()
