"""trace.log(.gz) parser. See docs/ARCHITECTURE.md sections 4.4 and 4.6.

M4 perf note: rewritten to two streaming passes instead of holding the
full ``list[Instr]`` in memory. Pass 1 only needs (pc, size,
is_control_flow, is-XRET) per instruction to compute ``ctx.trap_entries``
(``implicit_targets_full``) and the xret return table
(``xret_returns``: PC -> latest index of an XRET whose next line has that
PC) before the semantic scan can run; it does not
build ``Instr`` objects or call ``isa.parse_regs``. Pass 2 builds one
``Instr`` at a time with a single instruction of lookahead (needed to get
``next_pc`` for ``isa.check``), and only retains: ``head`` (first 8), a
bounded ``tail`` deque (last 64), a bounded lookback deque (last 16, for
anchor windows), and any in-flight anchor windows (at most 8, each at
most 16+1+4 instructions). Truncation semantics (``_MAX_INSTR``) are
unchanged.

Files are read with a *bounded* read (at most ``_IN_MEMORY_MAX_BYTES + 1``
characters, see ``_line_source``) into a single in-memory ``str`` that both
passes iterate via ``_iter_lines``, after a cheap ISIZE-based pre-filter
(``_decompressed_size``) rejects files that are obviously too large. Files
whose bounded read hits the cap (or whose ISIZE pre-filter already flags
them as too large / untrustworthy, e.g. wrapped past 4 GiB) are instead
re-opened and streamed for each pass, so per-worker memory stays bounded
regardless of what the gzip ISIZE trailer claims.
"""

from __future__ import annotations

import os
from collections import deque
from operator import methodcaller
from typing import Callable, Iterator

from rfb import isa
from rfb.io_utils import open_text
from rfb.record import Instr, TraceInfo, Violation

_MAX_INSTR = 2_000_000
_IN_MEMORY_MAX_BYTES = 256 * 1024 * 1024
_HEAD_N = 8
_TAIL_N = 64
_MAX_IMPLICIT_TARGETS = 32
_MAX_ANCHOR_WINDOWS = 8
_ANCHOR_BEFORE = 16
_ANCHOR_AFTER = 4
_MAX_SIG_WRITES = 16
_SIG_ADDR = 0x8FFFFFFC
_MAX_SAMPLES_PER_KEY = 3
_MAX_CHECK_ERRORS = 8


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


def _read_all_text(path: str) -> str:
    """Decompress (if ``.gz``) and read the whole file into one ``str``.
    Used by ``tests/test_parse_trace_line_source.py`` as a reference
    implementation (``_line_source`` itself no longer unconditionally
    reads the whole file; see its docstring)."""
    with open_text(path) as f:
        return f.read()


def _iter_lines(text: str) -> Iterator[str]:
    """Yield lines of ``text`` (no trailing ``\\n``), without building a
    ``list[str]`` of all lines (unlike ``str.splitlines()``)."""
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            yield text[pos:]
            return
        yield text[pos:nl]
        pos = nl + 1


_mclass_get = isa.MCLASS.get
_CF_CLASSES = isa._CONTROL_FLOW_CLASSES


def _pass1_trap_entries(
    lines: Iterator[str],
) -> tuple[int, dict[int, int], dict[int, int]]:
    """First streaming pass: only pc/size/control-flow-ness/is-XRET, to
    compute the *full* implicit-jump target set (used as ctx.trap_entries by
    pass 2) and the xret return table before the semantic scan runs. Mirrors
    the pair-comparison in the original single-pass "pass 2a" exactly, but
    over a lightweight (pc, size, is_control_flow, is-XRET) stream instead of
    built Instr objects.

    Returns ``(implicit_jumps, implicit_targets_full, xret_returns)``.
    ``xret_returns`` maps a PC R to the *latest* ``Instr.index`` of an XRET
    instruction (``isa.mclass == "XRET"``) whose next line has pc R
    (an interrupt return lands on the pre-empted instruction). The index
    is the pass-2 ``Instr.index``: both passes skip exactly the same
    malformed lines (``len(parts) < 5`` or non-hex pc) and both stop at
    ``_MAX_INSTR`` valid lines, so the running count of valid lines
    (``n_seen`` here, ``n_instr`` in pass 2) is identical. Used by
    ``isa.check`` to tell an asynchronous interrupt entry after a
    control-flow instruction from a real BRANCH_WRONG_TARGET.

    Stops early once _MAX_INSTR instructions have been seen: instructions
    beyond the cap never contribute to implicit_jumps/implicit_targets in
    the original code either (they were never appended to ``instrs``).

    M4 perf: only pc/insn/mnemonic (columns 2-4) are needed here, so the
    line is split with ``maxsplit=5`` instead of unbounded ``split("\t")``:
    this stops scanning once the 5th tab is found instead of walking the
    (sometimes very long, e.g. big register dumps) operands/regs suffix.
    ``parts[0:5]`` are identical to a full ``split("\t")`` either way; only
    the trailing "rest" element (unused here) differs, and the same
    ``len(parts) < 5`` malformed-line check applies (A/B: +22% pass1 time
    on the largest T3 stress case, see out/m4b_pass1_ab.txt).
    """
    implicit_jumps = 0
    implicit_targets_full: dict[int, int] = {}
    xret_returns: dict[int, int] = {}
    n_seen = 0
    prev_is_xret = False
    prev_pc = None
    prev_size = None
    prev_is_cf = None

    header_skipped = False
    for line in lines:
        if not header_skipped:
            header_skipped = True
            continue
        line = line.rstrip("\r")
        if not line:
            continue
        parts = line.split("\t", 5)
        if len(parts) < 5:
            continue
        pc_hex, insn_hex, mnemonic = parts[2], parts[3], parts[4]
        try:
            pc = int(pc_hex, 16)
        except ValueError:
            continue
        size = 2 if len(insn_hex) <= 4 else 4

        if prev_pc is not None and not prev_is_cf:
            expected = (prev_pc + prev_size) & 0xFFFFFFFF
            if pc != expected:
                implicit_jumps += 1
                implicit_targets_full[pc] = implicit_targets_full.get(pc, 0) + 1

        if prev_is_xret:
            xret_returns[pc] = n_seen - 1  # latest wins (indices increase)

        cls = _mclass_get(mnemonic, "OTHER")
        prev_pc, prev_size, prev_is_cf = pc, size, cls in _CF_CLASSES
        prev_is_xret = cls == "XRET"
        n_seen += 1
        if n_seen >= _MAX_INSTR:
            break

    return implicit_jumps, implicit_targets_full, xret_returns


def _decompressed_size(path: str) -> int | None:
    """Uncompressed size in bytes, or None when it cannot be trusted.
    For .gz this is the ISIZE trailer (size mod 2**32 of the last member of
    the gzip stream), so it is rejected when smaller than the compressed
    file itself -- but note ISIZE can also *wrap* (mod 2**32) for
    decompressed sizes above 4 GiB, silently reporting a small size for a
    huge file. Callers MUST treat a non-None return here as a fast initial
    filter only (an *upper bound* is not guaranteed), never as ground
    truth: see ``_line_source``, which re-checks with a bounded read
    before committing to the in-memory path."""
    size = os.path.getsize(path)
    if not str(path).endswith(".gz"):
        return size
    if size < 18:
        return None
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        isize = int.from_bytes(f.read(4), "little")
    return isize if isize >= size else None


_RSTRIP_NL = methodcaller("rstrip", "\n")


def _stream_lines(path: str) -> Iterator[str]:
    """Same lines as ``_iter_lines(_read_all_text(path))``, streamed."""
    with open_text(path) as f:
        # A line from file iteration has at most one "\n", at its end, so
        # rstrip("\n") drops exactly what ``line[:-1]`` did.
        yield from map(_RSTRIP_NL, f)


def _line_source(path: str) -> Callable[[], Iterator[str]]:
    """Return a factory giving a fresh line iterator per pass: one in-memory
    copy for small files, a re-opened stream per pass for large ones.

    The ISIZE-based ``_decompressed_size`` is only a quick pre-filter: it
    can never be trusted as an upper bound on its own (see its docstring),
    so it is used only to reject *obviously* large files without reading
    them at all. Whenever it reports a size within budget (or is
    untrustworthy, i.e. None), the file is opened and read with a bound of
    ``_IN_MEMORY_MAX_BYTES + 1`` *characters* (text mode: ``read(n)``
    counts decoded characters, not bytes, so the byte-oriented
    ``_IN_MEMORY_MAX_BYTES`` constant is used directly as a character
    count here -- multi-byte UTF-8 sequences only make the actual byte
    count larger than what was read, never smaller, so this remains a safe
    bound). If that bounded read returns more than ``_IN_MEMORY_MAX_BYTES``
    characters, the partial read is discarded and the file falls back to
    the streaming path; otherwise the (complete, since it read fewer
    characters than the cap) in-memory text is reused directly instead of
    re-reading the file. This guarantees the in-memory path never holds
    more than ``_IN_MEMORY_MAX_BYTES + 1`` characters in memory at once,
    regardless of what ISIZE claims."""
    size = _decompressed_size(path)
    if size is not None and size > _IN_MEMORY_MAX_BYTES:
        return lambda: _stream_lines(path)
    with open_text(path) as f:
        text = f.read(_IN_MEMORY_MAX_BYTES + 1)
    if len(text) > _IN_MEMORY_MAX_BYTES:
        return lambda: _stream_lines(path)
    return lambda: _iter_lines(text)


_HEXX = set("Xx")


def _csr_x_origin(mnemonic, operands, regs_field):
    """CSR read whose destination value carries X/x nibbles while no
    source register read does: key 'csr:mask' (mask = per-nibble X or .)."""
    if not mnemonic.startswith("csrr"):
        return None
    wmask = None
    for tok in regs_field.split():
        if ":0x" in tok and any(ch in _HEXX for ch in tok.split(":0x", 1)[1]):
            return None
        if "=0x" in tok:
            val = tok.split("=0x", 1)[1]
            if any(ch in _HEXX for ch in val):
                wmask = "".join("X" if ch in _HEXX else "." for ch in val.rjust(8, "0"))
    if wmask is None:
        return None
    ops = operands.split(",")
    csr = ops[1].strip() if len(ops) > 1 else "?"
    return csr + ":" + wmask


def parse_trace(path: str, regr_info=None) -> TraceInfo:
    parse_errors: list[str] = []
    mnemonic_hist: dict[str, int] = {}
    sig_writes: list[int] = []
    x_origins: dict[str, int] = {}

    lines = _line_source(path)
    implicit_jumps, implicit_targets_full, xret_returns = _pass1_trap_entries(
        lines()
    )
    implicit_targets = dict(
        sorted(implicit_targets_full.items(), key=lambda kv: (-kv[1], kv[0]))[
            :_MAX_IMPLICIT_TARGETS
        ]
    )
    ctx = isa.ScanContext(
        trap_entries=set(implicit_targets_full), xret_returns=xret_returns
    )

    # --- anchor setup (see ARCHITECTURE.md 4.4) -----------------------------
    anchor_pc = None
    anchor_val = None
    anchor_mnemonic = None
    if regr_info is not None and getattr(regr_info, "mismatches", None):
        anchor = regr_info.mismatches[0]
        anchor_pc = anchor.ibex_pc
        anchor_val = anchor.ibex_val
        anchor_mnemonic = _mnemonic_of(anchor.ibex_text)
    anchor_active = anchor_pc is not None

    anchor_windows: list[list[Instr]] = []
    anchor_window_violations: list[list[Violation | None]] = []
    anchor_window_anchor_pos: list[int] = []
    anchor_hits = 0
    pending_windows: list[dict] = []

    lookback: deque = deque(maxlen=_ANCHOR_BEFORE)
    lookback_v: deque = deque(maxlen=_ANCHOR_BEFORE)

    head: list[Instr] = []
    tail: deque = deque(maxlen=_TAIL_N)

    violations: dict[str, int] = {}
    first_violation: Violation | None = None
    violation_samples: list[Violation] = []
    sample_counts: dict[str, int] = {}

    def _close_window(w: dict) -> None:
        instrs_list = w["before"] + [w["anchor"]] + w["after"]
        v_list = w["before_v"] + [w["anchor_v"]] + w["after_v"]
        anchor_windows.append(instrs_list)
        anchor_window_violations.append(v_list)
        anchor_window_anchor_pos.append(len(w["before"]))

    # 2026-10-14 CSR／MEMREAD 細分: MEM_MISMATCH count and whether all of
    # them had detail["nb"]; see the "st" marking before the return.
    mem_n = 0
    mem_all_nb = True

    def _record_violation(instr: Instr, v: Violation | None) -> None:
        nonlocal first_violation, mem_n, mem_all_nb
        if v is None:
            return
        if v.kind == "MEM_MISMATCH":
            mem_n += 1
            if not v.detail.get("nb"):
                mem_all_nb = False
        key = f"{v.kind}:{v.mnemonic}"
        violations[key] = violations.get(key, 0) + 1
        # READ_MISMATCH only becomes a deviation through the anchor window;
        # it must never be the first_violation fallback.
        if first_violation is None and v.kind != "READ_MISMATCH":
            first_violation = v
        same_key_count = sample_counts.get(key, 0)
        if same_key_count < _MAX_SAMPLES_PER_KEY:
            violation_samples.append(v)
            sample_counts[key] = same_key_count + 1

    def _feed(instr: Instr, v: Violation | None) -> None:
        nonlocal anchor_hits
        tail.append(instr)

        if pending_windows:
            still_open = []
            for w in pending_windows:
                w["after"].append(instr)
                w["after_v"].append(v)
                if len(w["after"]) >= _ANCHOR_AFTER:
                    _close_window(w)
                else:
                    still_open.append(w)
            pending_windows[:] = still_open

        if anchor_active and instr.pc == anchor_pc:
            if anchor_val is not None:
                hit = anchor_val in instr.writes.values()
            else:
                hit = instr.mnemonic == anchor_mnemonic
            if hit:
                anchor_hits += 1
                if len(anchor_windows) + len(pending_windows) < _MAX_ANCHOR_WINDOWS:
                    pending_windows.append(
                        {
                            "before": list(lookback),
                            "before_v": list(lookback_v),
                            "anchor": instr,
                            "anchor_v": v,
                            "after": [],
                            "after_v": [],
                        }
                    )

        lookback.append(instr)
        lookback_v.append(v)

    n_instr = 0
    first_pc: int | None = None
    last_pc: int | None = None
    pending_instr: Instr | None = None
    n_check_errors = 0

    header_skipped = False
    # A skipped malformed line may have written registers we cannot see:
    # drop ``ctx.regs`` once the pending instruction has been observed
    # (the clear must come after that observe, else its writes survive).
    regs_dirty = False
    for lineno, line in enumerate(lines(), start=1):
        if not header_skipped:
            header_skipped = True
            continue
        line = line.rstrip("\r")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            pc_hex, insn_hex, mnemonic, operands, regs_field = parts[2:7]
        else:
            fields_ = _split_trace_line(parts)
            if fields_ is None:
                parse_errors.append(f"line {lineno}: not enough columns")
                regs_dirty = True
                continue
            pc_hex, insn_hex, mnemonic, operands, regs_field = fields_
        try:
            pc = int(pc_hex, 16)
            insn_val_len = len(insn_hex)
        except ValueError:
            parse_errors.append(f"line {lineno}: bad pc/insn hex")
            regs_dirty = True
            continue
        if insn_val_len <= 4:
            size = 2
        else:
            size = 4
        # Past the cap the line is skipped, so its Regs field is never parsed.
        if n_instr >= _MAX_INSTR:
            if not parse_errors or "truncated" not in parse_errors[-1]:
                parse_errors.append(
                    f"line {lineno}: truncated at {_MAX_INSTR} instructions"
                )
            continue

        reads, writes, pa, store, load = isa.parse_regs(regs_field)
        if mnemonic.startswith("csrr"):
            xo = _csr_x_origin(mnemonic, operands, regs_field)
            if xo is not None:
                x_origins[xo] = x_origins.get(xo, 0) + 1

        index = n_instr
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

        if pending_instr is not None:
            v, err = isa.check_safe(None, pending_instr, instr.pc, ctx)
            if err is not None:
                n_check_errors += 1
                if n_check_errors <= _MAX_CHECK_ERRORS:
                    parse_errors.append(
                        f"instr {pending_instr.index} ({pending_instr.mnemonic}): "
                        f"check failed: {err}"
                    )
            # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗: observe() must run
            # after check_safe evaluated pending_instr, so its own writes
            # never influence its own check (docs/ARCHITECTURE.md 4.3).
            isa.observe(ctx, pending_instr, instr.pc)
            if regs_dirty:
                ctx.mem.clear()
                ctx.csr.clear()
                ctx.csr_chain.clear()
                ctx.mem_at.clear()
                ctx.word_sub.clear()
                ctx.regs.clear()
                ctx.regs_prev.clear()
                ctx.regs_wr_at.clear()
                regs_dirty = False
            if v is not None:
                _record_violation(pending_instr, v)
            _feed(pending_instr, v)
        pending_instr = instr

        if first_pc is None:
            first_pc = pc
        last_pc = pc
        if len(head) < _HEAD_N:
            head.append(instr)
        mnemonic_hist[mnemonic] = mnemonic_hist.get(mnemonic, 0) + 1
        if _mclass_get(mnemonic, "OTHER") == "STORE" and pa == _SIG_ADDR and store is not None:
            sig_writes.append(store)
            if len(sig_writes) > _MAX_SIG_WRITES:
                sig_writes = sig_writes[-_MAX_SIG_WRITES:]
        n_instr += 1

    if pending_instr is not None:
        v, err = isa.check_safe(None, pending_instr, None, ctx)
        if err is not None:
            n_check_errors += 1
            if n_check_errors <= _MAX_CHECK_ERRORS:
                parse_errors.append(
                    f"instr {pending_instr.index} ({pending_instr.mnemonic}): "
                    f"check failed: {err}"
                )
        isa.observe(ctx, pending_instr)
        _record_violation(pending_instr, v)
        _feed(pending_instr, v)

    if n_check_errors > _MAX_CHECK_ERRORS:
        parse_errors.append(f"check failed on {n_check_errors} instructions total")

    for w in pending_windows:
        _close_window(w)
    pending_windows = []

    if mem_n and mem_all_nb:
        # Every MEM_MISMATCH of the trace is a store-side spill (isa.viol_group
        # "MEMREAD_ST"); mark the violations a deviation can be built from.
        for v in [first_violation, *violation_samples, *(x for vl in anchor_window_violations for x in vl)]:
            if v is not None and v.kind == "MEM_MISMATCH":
                v.detail["st"] = True

    return TraceInfo(
        n_instr=n_instr,
        first_pc=first_pc,
        last_pc=last_pc,
        head=head,
        tail=list(tail),
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
        x_origins=x_origins,
    )
