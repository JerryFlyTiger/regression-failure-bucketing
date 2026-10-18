"""Data model shared by the parsers and downstream stages.

See docs/ARCHITECTURE.md section 3 for the authoritative field list.
All dataclasses use ``slots=True`` and provide ``to_dict()`` for JSON
serialization (used by tools/dump_records.py and the M2/M3 test fixtures).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json


def _to_jsonable(value):
    """Recursively convert dataclasses / dicts / sets into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return value.to_dict()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, int):
                k = str(k)
            out[k] = _to_jsonable(v)
        return out
    if isinstance(value, (frozenset, set)):
        return sorted(_to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _dataclass_to_dict(obj) -> dict:
    out = {}
    for f in fields(obj):
        out[f.name] = _to_jsonable(getattr(obj, f.name))
    return out


@dataclass(slots=True)
class Instr:
    """One line of trace.log."""

    index: int
    pc: int
    size: int
    mnemonic: str
    operands: str
    reads: dict[str, int]
    writes: dict[str, int]
    pa: int | None
    store: int | None
    load: int | None

    to_dict = _dataclass_to_dict


@dataclass(slots=True)
class Mismatch:
    """One ``Mismatch[n]:`` block in regr.log."""

    n: int
    ibex_idx: int
    ibex_pc: int
    ibex_text: str
    ibex_rd: str | None
    ibex_val: int | None
    spike_idx: int
    spike_pc: int
    spike_text: str
    spike_rd: str | None
    spike_val: int | None

    to_dict = _dataclass_to_dict


@dataclass(slots=True)
class RegrInfo:
    kind: str
    test_name: str | None
    mismatches: list[Mismatch]
    matched: int | None
    mismatch_count: int | None
    raw: list[str]

    to_dict = _dataclass_to_dict

    @classmethod
    def empty(cls) -> RegrInfo:
        return cls(
            kind="UNKNOWN",
            test_name=None,
            mismatches=[],
            matched=None,
            mismatch_count=None,
            raw=[],
        )


@dataclass(slots=True)
class UvmMsg:
    severity: str
    file: str
    line: int | None
    scope: str
    id: str
    template: str
    raw: str

    to_dict = _dataclass_to_dict


@dataclass(slots=True)
class SimInfo:
    test_name: str | None
    verdict: str | None
    fatals: list[UvmMsg]
    first_errors: list[UvmMsg]
    error_templates: dict[str, int]
    counts: dict[str, int]
    report_ids: dict[str, int]
    line_count: int

    to_dict = _dataclass_to_dict

    @classmethod
    def empty(cls) -> SimInfo:
        return cls(
            test_name=None,
            verdict=None,
            fatals=[],
            first_errors=[],
            error_templates={},
            counts={},
            report_ids={},
            line_count=0,
        )


@dataclass(slots=True)
class Violation:
    index: int
    kind: str
    mnemonic: str
    detail: dict
    pc: int
    next_pc: int | None

    to_dict = _dataclass_to_dict


@dataclass(slots=True)
class TraceInfo:
    n_instr: int
    first_pc: int | None
    last_pc: int | None
    head: list[Instr]
    tail: list[Instr]
    mnemonic_hist: dict[str, int]
    implicit_jumps: int
    implicit_targets: dict[int, int]
    violations: dict[str, int]
    first_violation: Violation | None
    violation_samples: list[Violation]
    anchor_windows: list[list[Instr]]
    anchor_hits: int
    sig_writes: list[int]
    # Necessary addition beyond ARCHITECTURE section 3: parse_trace needs a
    # way to surface per-line parse failures back to the caller without
    # raising, matching the "don't interrupt the stream" requirement in
    # section 4.6. Flagged in the M1 report.
    parse_errors: list[str] = field(default_factory=list)
    # M2 addition (see DISPATCH.md exception clause): diverge.py's anchor
    # search (section 4.4) needs the per-instruction isa.check() result for
    # every instruction inside each anchor window, computed with the full
    # (uncapped) trap_entries set that only parse_trace has access to.
    # anchor_window_violations[w][j] parallels anchor_windows[w][j];
    # anchor_window_anchor_pos[w] is the index of the anchor instruction
    # itself within anchor_windows[w].
    anchor_window_violations: list[list["Violation | None"]] = field(default_factory=list)
    anchor_window_anchor_pos: list[int] = field(default_factory=list)
    x_origins: dict[str, int] = field(default_factory=dict)

    to_dict = _dataclass_to_dict

    @classmethod
    def empty(cls) -> TraceInfo:
        return cls(
            n_instr=0,
            first_pc=None,
            last_pc=None,
            head=[],
            tail=[],
            mnemonic_hist={},
            implicit_jumps=0,
            implicit_targets={},
            violations={},
            first_violation=None,
            violation_samples=[],
            anchor_windows=[],
            anchor_hits=0,
            sig_writes=[],
            parse_errors=[],
            anchor_window_violations=[],
            anchor_window_anchor_pos=[],
        )


@dataclass(slots=True)
class Deviation:
    kind: str
    mnemonic: str | None
    mclass: str | None
    detail: dict
    source: str
    confidence: float

    to_dict = _dataclass_to_dict


@dataclass(slots=True)
class FailureRecord:
    case_id: int
    family: str
    regr: RegrInfo
    sim: SimInfo
    trace: TraceInfo
    deviation: Deviation | None
    parse_errors: list[str]

    to_dict = _dataclass_to_dict

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
