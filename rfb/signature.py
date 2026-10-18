"""FailureRecord -> Signature. See docs/ARCHITECTURE.md section 5.3.

``normalize()`` is implemented in rfb.io_utils (shared with parse_sim.py's
UVM message template construction, per section 5.2); this module imports
and reuses it rather than redefining it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rfb.io_utils import normalize
from rfb.isa import alu_op as isa_alu_op
from rfb.isa import mclass as isa_mclass
from rfb.isa import viol_group as isa_viol_group
from rfb.record import FailureRecord


@dataclass(slots=True)
class Signature:
    case_id: int
    family: str
    hard_key: tuple
    soft: dict = field(default_factory=dict)


def _mismatch_mnemonic(ibex_text: str) -> str:
    """First whitespace-separated token of a Mismatch[n] ibex text line."""
    parts = ibex_text.split()
    return parts[0] if parts else ""


def _hard_key(record: FailureRecord) -> tuple:
    family = record.family

    if family == "MISMATCH":
        dev = record.deviation
        kind = dev.kind if dev is not None else "NONE"
        if kind and kind != "NONE":
            # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: ALU
            # WRONG_RESULT groups at mnemonic granularity via viol_group,
            # not the coarser dev.mclass, so e.g. sra vs mulhsu never share
            # a hard_key. See docs/ARCHITECTURE.md 5.3.
            return (family, kind, isa_viol_group(kind, dev.mnemonic or "", dev.detail))
        mnemonic = ""
        if record.regr.mismatches:
            mnemonic = _mismatch_mnemonic(record.regr.mismatches[0].ibex_text)
        return (family, "NONE", normalize(mnemonic))

    if family == "FATAL":
        tmpl = record.sim.fatals[0].template if record.sim.fatals else None
        return (family, tmpl, frozenset(record.sim.error_templates), _xsig(record))

    if family == "ASSERT":
        return (family, frozenset(record.sim.error_templates), _xsig(record))

    # OTHER
    raw_joined = " ".join(record.regr.raw)
    return (family, normalize(raw_joined), _xsig(record))


def _ratio_vector(counts: dict) -> dict:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _tail_ngrams(tail: list) -> dict:
    mnemonics = [instr.mnemonic for instr in tail]
    grams: dict[tuple, int] = {}
    for i in range(len(mnemonics) - 2):
        key = (mnemonics[i], mnemonics[i + 1], mnemonics[i + 2])
        grams[key] = grams.get(key, 0) + 1
    return grams


def _viol_class(trace) -> tuple | None:
    v = trace.first_violation
    if v is None:
        return None
    return (v.kind, isa_mclass(v.mnemonic))


# 2026-09-28 批次二修正，依據 Q&A A11: systematic-violation soft fields, see
# docs/ARCHITECTURE.md section 5.3/5.4.
MIN_SYS_VIOL = 2
# Kinds whose scan counts stay out of sys_viol_class/keys (only a deviation gets in).
_SCAN_EXCLUDED = ("READ_MISMATCH", "LOAD_EXT", "MEM_MISMATCH", "CSR_MISMATCH",
                  "TRAP_MCAUSE", "TRAP_MEPC", "TRAP_MTVAL")


def _sys_viol(record: FailureRecord) -> tuple[frozenset, frozenset]:
    """Systematic violation class/key sets: violations.kind:mnemonic that
    occur >= MIN_SYS_VIOL times in the trace, plus (for MISMATCH cases with a
    non-NONE deviation) the deviation itself."""
    trace = record.trace
    cls: set = set()
    keys: set = set()
    for key, n in trace.violations.items():
        # READ_MISMATCH scan counts stay out of the systematic sets; only a
        # READ_MISMATCH deviation (below) gets in.
        if n >= MIN_SYS_VIOL and key.split(":", 1)[0] not in _SCAN_EXCLUDED:
            kind, mnemonic = key.split(":", 1)
            # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: see
            # docs/ARCHITECTURE.md 5.3.
            cls.add((kind, isa_viol_group(kind, mnemonic)))
            keys.add((kind, mnemonic))
    dev = record.deviation
    if record.family == "MISMATCH" and dev is not None and dev.kind != "NONE":
        cls.add((dev.kind, isa_viol_group(dev.kind, dev.mnemonic or "", dev.detail)))
        if dev.mnemonic:
            keys.add((dev.kind, dev.mnemonic))
    return frozenset(cls), frozenset(keys)


def _wr_sys(record: FailureRecord) -> frozenset:
    """2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: per-atom
    footprint of ALU mnemonics whose WRONG_RESULT count crossed
    MIN_SYS_VIOL from the scan alone (never the deviation), used by
    cluster._alu_op_link (R1b) to tell a well-evidenced same-op bug apart
    from a thin one. See docs/ARCHITECTURE.md 5.3."""
    trace = record.trace
    out: set = set()
    for key, n in trace.violations.items():
        if n < MIN_SYS_VIOL:
            continue
        kind, mnemonic = key.split(":", 1)
        if kind == "WRONG_RESULT" and isa_alu_op(mnemonic) is not None:
            out.add(mnemonic)
    return frozenset(out)


def _uvm_ev_sys(record: FailureRecord) -> str:
    """``_uvm_ev(record)`` if the deviation's kind is one of ``_SCAN_EXCLUDED``
    (the kinds R1 cannot see) and the trace has at least ``MIN_SYS_VIOL``
    violations of that kind (any mnemonic), else ``""``. Other kinds always
    give ``""``: R1m must not be looser than R1 there. Lets cluster.py's R1m
    tell systematic evidence from a single noisy first_violation. See
    docs/ARCHITECTURE.md 5.3 (``uvm_ev_sys``) and 5.4."""
    ev = _uvm_ev(record)
    if not ev:
        return ""
    kind = record.deviation.kind
    if kind not in _SCAN_EXCLUDED:
        return ""
    n = sum(c for k, c in record.trace.violations.items() if k.split(":", 1)[0] == kind)
    return ev if n >= MIN_SYS_VIOL else ""


def _uvm_ev(record: FailureRecord) -> str:
    """Trace evidence class of a UVM-family case (FATAL/ASSERT/OTHER),
    derived from its deviation (= ``trace.first_violation``), or ``""``
    without one. MISMATCH cases always get ``""`` (their deviation is already
    in hard_key). Used only by ``cluster._split_conflicting``: a shared UVM
    message is split by this class only when its cases disagree, so it never
    enters hard_key. See docs/ARCHITECTURE.md 5.3 (``uvm_ev``) and 5.4.

    Granularity is coarse in only two cases, so one bug is not mistaken for
    conflicting evidence there (F1):
    - WRONG_RESULT on an ALU mnemonic -> ``"ALU:" + alu_op`` (same grain as
      R1b: sra and c.srai are both ``ALU:sra``, add and addi ``ALU:add``);
    - BRANCH_NOT_TAKEN / BRANCH_WRONGLY_TAKEN -> bare ``viol_group`` without
      a kind prefix, so beq NOT_TAKEN and bne WRONGLY_TAKEN share ``BR_EQ-``
      (EQ polarity), and blt NOT_TAKEN and blt WRONGLY_TAKEN share ``BR_LT``;
    - everything else -> ``"<kind>:<viol_group>"``.
    Other cases can still split one bug into different classes (e.g.
    BRANCH_WRONG_TARGET is still split by mclass, and the shifter's sll, srl
    and sra are different ops); see docs/ARCHITECTURE.md 5.5 residual risk."""
    if record.family == "MISMATCH":
        return ""
    dev = record.deviation
    if dev is None or dev.kind == "NONE":
        return ""
    mn = dev.mnemonic or ""
    if dev.kind == "WRONG_RESULT":
        op = isa_alu_op(mn)
        if op is not None:
            return "ALU:" + op
    if dev.kind in ("BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"):
        return isa_viol_group(dev.kind, mn, dev.detail)
    return dev.kind + ":" + isa_viol_group(dev.kind, mn, dev.detail)


def _xsig(record: FailureRecord) -> frozenset:
    return frozenset(k for k, n in record.trace.x_origins.items() if n >= MIN_SYS_VIOL)


def _soft(record: FailureRecord) -> dict:
    trace = record.trace
    sim = record.sim
    sys_viol_class, sys_viol_keys = _sys_viol(record)
    return {
        "viol_vec": _ratio_vector(trace.violations),
        "viol_class": _viol_class(trace),
        "sys_viol_class": sys_viol_class,
        "sys_viol_keys": sys_viol_keys,
        "wr_sys": _wr_sys(record),
        "xsig": _xsig(record),
        "uvm_ev": _uvm_ev(record),
        "uvm_ev_sys": _uvm_ev_sys(record),
        "tail_ngrams": _tail_ngrams(trace.tail),
        "implicit": min(trace.implicit_jumps, 8),
        "assert_set": frozenset(sim.error_templates),
        "fatal_tmpl": sim.fatals[0].template if sim.fatals else None,
        "sig_last": trace.sig_writes[-1] if trace.sig_writes else None,
    }


def signature(record: FailureRecord) -> Signature:
    return Signature(
        case_id=record.case_id,
        family=record.family,
        hard_key=_hard_key(record),
        soft=_soft(record),
    )


def build_signatures(records: list[FailureRecord]) -> list[Signature]:
    return [signature(r) for r in records]
