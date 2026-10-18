"""trace.log(.gz) parser -- pre-M4-streaming reference implementation.

Frozen copy of ``rfb/parse_trace.py`` as of commit 6556e04 (the single-pass,
full-``list[Instr]``-in-memory version, before the M4 two-pass streaming
rewrite). This file is a test oracle ONLY: used by
``tests/test_parse_trace_legacy_equiv.py`` to check the streaming rewrite
in ``rfb/parse_trace.py`` produces byte-identical ``TraceInfo`` for the
same inputs, including truncation (``_MAX_INSTR``) and anchor-window
(``_MAX_ANCHOR_WINDOWS``) edge cases. Not imported by any production code
path; do not modify its logic, only its imports if ``rfb`` internals move.

Exception (2026-10-01): an independent implementation of ``xret_returns`` was
added here, because the fuzz equivalence test compares the TraceInfo
(violations) built with that table against ``rfb/parse_trace.py``, whose
streaming pass 1 computes the same table. Only that table and the
``ScanContext(...)`` line that passes it changed; the rest of the original
single-pass logic is untouched.

Note (2026-10-04): ``ScanContext.mtvec_base`` (interrupt vector slot exclusion)
is NOT implemented here. This oracle never calls ``isa.observe()`` (same as
``sp_shadow`` stays None), so mtvec_base stays None and the exclusion never
fires. The fuzz equivalence test does not need an independent mtvec tracker
because its mnemonic list has no CSR instructions, hence no mtvec writes: both
implementations see mtvec_base None on every fuzz trace.
"""

from __future__ import annotations

from rfb import isa
from rfb.io_utils import open_text
from rfb.record import Instr, TraceInfo, Violation

_MAX_INSTR = 2_000_000
_HEAD_N = 8
_TAIL_N = 64
_MAX_IMPLICIT_TARGETS = 32
_MAX_ANCHOR_WINDOWS = 8
_ANCHOR_BEFORE = 16
_ANCHOR_AFTER = 4
_MAX_SIG_WRITES = 16
_SIG_ADDR = 0x8FFFFFFC
_MAX_SAMPLES_PER_KEY = 3


def _mnemonic_of(text: str) -> str:
    """First whitespace-delimited token of a regr.log instruction text."""
    return text.split()[0] if text.split() else ""


def _split_trace_line(parts: list[str]):
    """Given a tab-split trace line (excluding header), return
    (pc_hex, insn_hex, mnemonic, operands, regs_field) or None if the line
    doesn't have enough columns."""
    if len(parts) < 5:
        return None
    pc_hex = parts[2]
    insn_hex = parts[3]
    mnemonic = parts[4]
    if len(parts) >= 7:
        operands = parts[5]
        regs_field = parts[6]
    elif len(parts) == 6:
        operands = ""
        regs_field = parts[5]
    else:
        operands = ""
        regs_field = ""
    return pc_hex, insn_hex, mnemonic, operands, regs_field


def parse_trace(path: str, regr_info=None) -> TraceInfo:
    parse_errors: list[str] = []
    instrs: list[Instr] = []
    mnemonic_hist: dict[str, int] = {}
    sig_writes: list[int] = []

    with open_text(path) as f:
        header_skipped = False
        for lineno, line in enumerate(f, start=1):
            if not header_skipped:
                header_skipped = True
                continue
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            fields_ = _split_trace_line(parts)
            if fields_ is None:
                parse_errors.append(f"line {lineno}: not enough columns")
                continue
            pc_hex, insn_hex, mnemonic, operands, regs_field = fields_
            try:
                pc = int(pc_hex, 16)
                insn_val_len = len(insn_hex)
            except ValueError:
                parse_errors.append(f"line {lineno}: bad pc/insn hex")
                continue
            if insn_val_len <= 4:
                size = 2
            else:
                size = 4
            reads, writes, pa, store, load = isa.parse_regs(regs_field)

            if len(instrs) >= _MAX_INSTR:
                if not parse_errors or "truncated" not in parse_errors[-1]:
                    parse_errors.append(
                        f"line {lineno}: truncated at {_MAX_INSTR} instructions"
                    )
                continue

            index = len(instrs)
            instr = Instr(
                index=index,
                pc=pc,
                size=size,
                mnemonic=mnemonic,
                operands=operands,
                reads=reads,
                writes=writes,
                pa=pa,
                store=store,
                load=load,
            )
            instrs.append(instr)
            mnemonic_hist[mnemonic] = mnemonic_hist.get(mnemonic, 0) + 1
            if isa.mclass(mnemonic) == "STORE" and pa == _SIG_ADDR and store is not None:
                sig_writes.append(store)
                if len(sig_writes) > _MAX_SIG_WRITES:
                    sig_writes = sig_writes[-_MAX_SIG_WRITES:]

    n_instr = len(instrs)
    first_pc = instrs[0].pc if instrs else None
    last_pc = instrs[-1].pc if instrs else None
    head = instrs[:_HEAD_N]
    tail = instrs[-_TAIL_N:] if n_instr else []

    # Pass 2a: implicit jumps (non-control-flow instr whose next_pc breaks
    # the pc+size sequence). This must be known before the semantic scan.
    implicit_jumps = 0
    implicit_targets_full: dict[int, int] = {}
    for i, instr in enumerate(instrs):
        next_pc = instrs[i + 1].pc if i + 1 < n_instr else None
        if next_pc is None:
            continue
        if not isa.is_control_flow(instr.mnemonic):
            expected = (instr.pc + instr.size) & 0xFFFFFFFF
            if next_pc != expected:
                implicit_jumps += 1
                implicit_targets_full[next_pc] = implicit_targets_full.get(next_pc, 0) + 1

    implicit_targets = dict(
        sorted(implicit_targets_full.items(), key=lambda kv: (-kv[1], kv[0]))[
            :_MAX_IMPLICIT_TARGETS
        ]
    )

    # Pass 2b: semantic scan using isa.check() (ARCHITECTURE.md section 4.3).
    # ctx.trap_entries uses the *full* implicit-target set (not the capped
    # implicit_targets), per the section 4.3 exclusion rule.
    # xret_returns (async-entry exclusion, ARCHITECTURE.md 4.3): independent
    # list-based computation, latest xret index per return PC.
    xret_returns: dict[int, int] = {}
    for i, instr in enumerate(instrs[:-1]):
        if isa.mclass(instr.mnemonic) == "XRET":
            xret_returns[instrs[i + 1].pc] = instr.index
    ctx = isa.ScanContext(
        trap_entries=set(implicit_targets_full), xret_returns=xret_returns
    )
    violations: dict[str, int] = {}
    first_violation: Violation | None = None
    violation_samples: list[Violation] = []
    violations_by_index: list[Violation | None] = [None] * n_instr
    for i, instr in enumerate(instrs):
        prev = instrs[i - 1] if i > 0 else None
        next_pc = instrs[i + 1].pc if i + 1 < n_instr else None
        v = isa.check(prev, instr, next_pc, ctx)
        violations_by_index[i] = v
        if v is not None:
            key = f"{v.kind}:{v.mnemonic}"
            violations[key] = violations.get(key, 0) + 1
            if first_violation is None:
                first_violation = v
            same_key_count = sum(1 for s in violation_samples if f"{s.kind}:{s.mnemonic}" == key)
            if same_key_count < _MAX_SAMPLES_PER_KEY:
                violation_samples.append(v)

    # Anchor windows: locate the divergence anchor from regr_info.mismatches[0]
    anchor_windows: list[list[Instr]] = []
    anchor_window_violations: list[list[Violation | None]] = []
    anchor_window_anchor_pos: list[int] = []
    anchor_hits = 0
    if regr_info is not None and getattr(regr_info, "mismatches", None):
        anchor = regr_info.mismatches[0]
        anchor_mnemonic = _mnemonic_of(anchor.ibex_text)
        for i, instr in enumerate(instrs):
            if instr.pc != anchor.ibex_pc:
                continue
            if anchor.ibex_val is not None:
                hit = anchor.ibex_val in instr.writes.values()
            else:
                hit = instr.mnemonic == anchor_mnemonic
            if not hit:
                continue
            anchor_hits += 1
            if len(anchor_windows) < _MAX_ANCHOR_WINDOWS:
                lo = max(0, i - _ANCHOR_BEFORE)
                hi = min(n_instr, i + _ANCHOR_AFTER + 1)
                anchor_windows.append(instrs[lo:hi])
                anchor_window_violations.append(violations_by_index[lo:hi])
                anchor_window_anchor_pos.append(i - lo)

    return TraceInfo(
        n_instr=n_instr,
        first_pc=first_pc,
        last_pc=last_pc,
        head=head,
        tail=tail,
        mnemonic_hist=mnemonic_hist,
        implicit_jumps=implicit_jumps,
        implicit_targets=implicit_targets,
        violations=violations,
        first_violation=first_violation,
        violation_samples=violation_samples,
        anchor_windows=anchor_windows,
        anchor_hits=anchor_hits,
        sig_writes=sig_writes,
        parse_errors=parse_errors,
        anchor_window_violations=anchor_window_violations,
        anchor_window_anchor_pos=anchor_window_anchor_pos,
    )
