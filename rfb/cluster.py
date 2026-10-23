"""Signature clustering. See docs/ARCHITECTURE.md sections 5.4 and 5.5.

2026-09-28 批次二修正，依據 Q&A A11: rewritten to the M3 evidence-ladder
distance (R1-R7) + UPGMA merge, replacing the M2 per-family distance rules.
All distance weights and thresholds are named constants at module scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rfb import isa
from rfb.signature import Signature

# --- section 5.4: evidence-ladder distance constants ------------------------

D_VIOL = 0.10
D_VIOLVEC = 0.30
VIOLVEC_COS_MIN = 0.5

ASSERT_BASE = 0.20
ASSERT_SPAN = 0.60
ASSERT_XFAM_MIN_OVL = 0.5

D_TRAP_LINK = 0.40
D_UVM_NONE = 0.55
D_UVM_ONESIDED_ASSERT = 0.70

TMPL_BONUS = 0.10
TAIL_W = 0.05
D_CROSS = 0.90

TRAP_LINK = {
    "ebreak": ("DEBUG", "EBREAK"),
    "c.ebreak": ("DEBUG", "EBREAK"),
    "dret": ("DRET",),
    "ecall": ("ECALL",),
    "mret": ("MRET",),
}

# families treated as "UVM-side" (no inherent instruction-trace evidence)
_UVM_FAMILIES = frozenset({"FATAL", "ASSERT", "OTHER"})

# retained from M2 for the MISMATCH x MISMATCH rule (R4)
MISMATCH_VIOL_VEC_WEIGHT = 0.5
MISMATCH_MCLASS_WEIGHT = 0.5


def _cos(a: dict, b: dict) -> float:
    """Cosine similarity of two sparse vectors given as dicts. 0 if either
    vector is empty/all-zero (no information -> treated as maximally
    dissimilar, i.e. distance contribution of 1)."""
    if not a or not b:
        return 0.0
    # Sum in sorted-key order so the float accumulation (not associative) is
    # independent of dict insertion order (which in turn traces back to
    # member/case iteration order upstream).
    dot = 0.0
    for k in sorted(a.keys() & b.keys()):
        dot += a[k] * b[k]
    na = math.sqrt(sum(a[k] * a[k] for k in sorted(a.keys())))
    nb = math.sqrt(sum(b[k] * b[k] for k in sorted(b.keys())))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _indicator_diff(a: frozenset, b: frozenset) -> float:
    """0.0 if the sets share any element (or both are empty: no info, treat
    as same), else 1.0."""
    if not a and not b:
        return 0.0
    return 0.0 if (a & b) else 1.0


def _avg_vector(members: list[Signature], key: str) -> dict:
    total: dict = {}
    n = len(members)
    for m in members:
        for k, v in (m.soft.get(key) or {}).items():
            total[k] = total.get(k, 0.0) + v
    return {k: v / n for k, v in total.items()}


def canon(x):
    """Deterministic sort key for hard_key tuples that may contain
    frozensets. repr()-based sorting is unsafe across processes because
    frozenset iteration order depends on PYTHONHASHSEED; this instead
    recurses into a fully-ordered, hash-seed-independent representation."""
    if isinstance(x, frozenset):
        return ("<fs>",) + tuple(sorted(canon(e) for e in x))
    if isinstance(x, tuple):
        return tuple(canon(e) for e in x)
    if x is None:
        return ("<none>",)
    return (type(x).__name__, str(x))


@dataclass
class _Cluster:
    members: list[Signature]

    @property
    def families(self) -> frozenset:
        return frozenset(m.family for m in self.members)

    @property
    def family(self) -> str | None:
        fams = self.families
        return next(iter(fams)) if len(fams) == 1 else None

    @property
    def viol_vec_avg(self) -> dict:
        return _avg_vector(self.members, "viol_vec")

    @property
    def tail_ngrams_avg(self) -> dict:
        return _avg_vector(self.members, "tail_ngrams")

    @property
    def assert_set_union(self) -> frozenset:
        out: set = set()
        for m in self.members:
            out |= m.soft.get("assert_set") or frozenset()
        return frozenset(out)

    @property
    def fatal_tmpl_set(self) -> frozenset:
        return frozenset(
            m.soft.get("fatal_tmpl")
            for m in self.members
            if m.soft.get("fatal_tmpl") is not None
        )

    @property
    def mclass_token_set(self) -> frozenset:
        # ARCHITECTURE 5.4's "deviation.mclass 不同" indicator: for MISMATCH
        # members, hard_key[2] is deviation.mclass when kind != "NONE", or a
        # normalized mnemonic fallback when kind == "NONE" (section 5.3);
        # either way it is the class-discriminating token for the group.
        return frozenset(
            m.hard_key[2]
            for m in self.members
            if m.family == "MISMATCH" and len(m.hard_key) > 2
        )


class _Atom:
    """One hard_key group with the soft evidence needed by the distance
    ladder, per docs/ARCHITECTURE.md 5.4."""

    def __init__(self, hard_key: tuple, members: list[Signature]):
        self.hard_key = hard_key
        # Sort members by case_id before any floating-point aggregation
        # (_avg_vector's summation order affects the result, since float
        # addition is not associative) so that vvec/tail are independent of
        # the input CSV row order. See ARCHITECTURE.md 5.5.
        members = sorted(members, key=lambda m: m.case_id)
        self.ids = [m.case_id for m in members]
        self.n = len(self.ids)
        cluster = _Cluster(members=members)
        self.family = cluster.family
        self.asserts = cluster.assert_set_union
        self.tmpls = cluster.fatal_tmpl_set
        self.vcls = frozenset().union(
            *[m.soft.get("sys_viol_class") or frozenset() for m in members]
        )
        self.vkeys = frozenset().union(
            *[m.soft.get("sys_viol_keys") or frozenset() for m in members]
        )
        # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: see
        # docs/ARCHITECTURE.md 5.3/5.4.
        self.wr_sys = frozenset().union(
            *[m.soft.get("wr_sys") or frozenset() for m in members]
        )
        self.xsig = frozenset().union(
            *[m.soft.get("xsig") or frozenset() for m in members]
        )
        self.vvec = cluster.viol_vec_avg
        self.tail = cluster.tail_ngrams_avg
        self.mtok = cluster.mclass_token_set
        # Trace evidence classes of UVM members, and the original hard_key
        # if this atom is a sub-group made by _split_conflicting (else None).
        self.ev = frozenset(m.soft.get("uvm_ev") or "" for m in members) - {""}
        # The subset of ev backed by >= MIN_SYS_VIOL violations of its kind (R1m).
        self.ev_sys = frozenset(m.soft.get("uvm_ev_sys") or "" for m in members) - {""}
        last = hard_key[-1] if hard_key else None
        self.split_of = (
            hard_key[:-1] if isinstance(last, str) and last.startswith("<ev>") else None
        )


def _link(x: _Atom, y: _Atom) -> bool:
    """True if x's systematic trap/xret-not-taken violation links to y's
    FATAL template (R3)."""
    if y.family == "MISMATCH" or y.vcls or not y.tmpls:
        return False
    toks: set = set()
    for kind, mnemonic in x.vkeys:
        if kind in ("TRAP_NOT_TAKEN", "XRET_FALLTHROUGH"):
            toks |= set(TRAP_LINK.get(mnemonic, ()))
    up = " ".join(sorted(y.tmpls)).upper()
    return any(t in up for t in toks)


def _alu_op_link(a: _Atom, b: _Atom) -> bool:
    """2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: R1b. True if
    both atoms have systematic WRONG_RESULT evidence on the *same base ALU
    operation* (``isa.alu_op``, e.g. sra and c.srai both map to "sra"),
    linking different encodings of what is plausibly the same bug --
    unless both sides' ``wr_sys`` footprints (mnemonics that individually
    crossed MIN_SYS_VIOL) are already established for that op and don't
    overlap, which signals form-specific (decoder-level) bugs instead. See
    docs/ARCHITECTURE.md 5.4."""
    ops_a = {isa.alu_op(m) for kind, m in a.vkeys if kind == "WRONG_RESULT"}
    ops_a.discard(None)
    ops_b = {isa.alu_op(m) for kind, m in b.vkeys if kind == "WRONG_RESULT"}
    ops_b.discard(None)
    for op in ops_a & ops_b:
        sa = {m for m in a.wr_sys if isa.alu_op(m) == op}
        sb = {m for m in b.wr_sys if isa.alu_op(m) == op}
        if sa and sb and not (sa & sb):
            continue
        return True
    return False


def _ev_mismatch_link(x: _Atom, y: _Atom) -> bool:
    """R1m: x is a UVM atom with systematic trace evidence (``ev_sys``, split
    out by _split_conflicting or not), and one of its evidence classes
    (``"<kind>:<viol_group>"``) is a systematic class of the MISMATCH atom y.
    R1 already links an unsplit UVM atom to MISMATCH through ``vcls``; only the
    kinds kept out of ``sys_viol_class`` (``signature._SCAN_EXCLUDED``:
    READ_MISMATCH, LOAD_EXT, MEM_MISMATCH, CSR_MISMATCH, TRAP_MCAUSE,
    TRAP_MEPC, TRAP_MTVAL, MSTATUS_MISMATCH) were blocked, and only those
    ever get an ``ev_sys``. See docs/ARCHITECTURE.md 5.4."""
    if not x.ev_sys or x.family not in _UVM_FAMILIES:
        return False
    if y.family != "MISMATCH":
        return False
    return bool(x.ev_sys & {f"{kind}:{grp}" for kind, grp in y.vcls})


def _atom_distance(a: _Atom, b: _Atom) -> float:
    # R1: shared systematic violation class.
    if a.vcls & b.vcls:
        return D_VIOL

    # R1m: UVM atom's systematic evidence class is a systematic
    # class of a MISMATCH atom -> same strength as R1.
    if _ev_mismatch_link(a, b) or _ev_mismatch_link(b, a):
        return D_VIOL

    # R1b: same base ALU operation, systematic WRONG_RESULT evidence on
    # both sides (2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗，方案 G).
    if _alu_op_link(a, b):
        return D_VIOLVEC

    # R1x-: both sides carry systematic CSR X-origins and share none:
    # different undriven CSR fields -> different defects. UVM families only.
    if (
        a.xsig
        and b.xsig
        and not (a.xsig & b.xsig)
        and a.family in _UVM_FAMILIES
        and b.family in _UVM_FAMILIES
    ):
        return D_CROSS

    # R1e-: two sub-atoms split out of the same UVM hard_key group by
    # _split_conflicting, both with trace evidence classes and sharing none:
    # the shared message was shown to be non-specific, so keep R2/R7 from
    # re-merging them on that message or its assertion set. A sub-atom
    # without evidence is not affected (thin evidence can still re-merge).
    if (
        a.ev
        and b.ev
        and not (a.ev & b.ev)
        and a.family in _UVM_FAMILIES
        and b.family in _UVM_FAMILIES
        and a.split_of is not None
        and a.split_of == b.split_of
    ):
        return D_CROSS

    # R2: both sides have assertion evidence.
    if a.asserts and b.asserts:
        ovl = len(a.asserts & b.asserts) / min(len(a.asserts), len(b.asserts))
        both_uvm_no_vcls = (
            a.family in _UVM_FAMILIES
            and b.family in _UVM_FAMILIES
            and not (a.vcls or b.vcls)
        )
        if ovl >= ASSERT_XFAM_MIN_OVL or both_uvm_no_vcls:
            d = ASSERT_BASE + ASSERT_SPAN * (1 - ovl)
            if a.tmpls & b.tmpls:
                d -= TMPL_BONUS
            d += TAIL_W * (1 - _cos(a.tail, b.tail))
            return max(D_VIOL, d)

    # R3: trap-instruction violation <-> FATAL template link.
    if _link(a, b) or _link(b, a):
        return D_TRAP_LINK

    # R4: both MISMATCH.
    if a.family == "MISMATCH" and b.family == "MISMATCH":
        return MISMATCH_VIOL_VEC_WEIGHT * (1 - _cos(a.vvec, b.vvec)) + (
            MISMATCH_MCLASS_WEIGHT * _indicator_diff(a.mtok, b.mtok)
        )

    # R5: either side has systematic-violation evidence.
    if a.vcls or b.vcls:
        if _cos(a.vvec, b.vvec) >= VIOLVEC_COS_MIN:
            return D_VIOLVEC
        return D_CROSS

    # R6: either side is MISMATCH (without shared violation evidence).
    if a.family == "MISMATCH" or b.family == "MISMATCH":
        return D_CROSS

    # R7: both UVM (FATAL/ASSERT/OTHER), no violation evidence on either
    # side (both-sided-asserts case already handled by R2).
    if a.asserts or b.asserts:
        d = D_UVM_ONESIDED_ASSERT
    else:
        d = D_UVM_NONE
    if a.tmpls & b.tmpls:
        d -= TMPL_BONUS
    return d + TAIL_W * (1 - _cos(a.tail, b.tail))


def _upgma_merge(atoms: list[_Atom], k: int) -> list[list[int]]:
    """UPGMA agglomerative merge with Lance-Williams case-count weighting.
    Returns, for each surviving cluster, the sorted list of member atom
    indices that were merged into it (not case_ids)."""
    n = len(atoms)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _atom_distance(atoms[i], atoms[j])
            dist[i, j] = d
            dist[j, i] = d

    sizes = {i: atoms[i].n for i in range(n)}
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    alive = list(range(n))
    target = max(k, 1)

    while len(alive) > target:
        # Vectorized argmin over the alive x alive submatrix, restricted to
        # the strict upper triangle (i < j). `alive` is always kept sorted
        # ascending (only ever shrinks by removing `j`, `i` stays in place),
        # so submatrix positions map monotonically to original atom indices
        # and np.argmin's row-major (C-order) first-occurrence tie-break is
        # exactly the (round(d, 9), i, j) lexicographic order of the
        # original nested-loop reference implementation.
        idx = np.array(alive)
        m = len(alive)
        sub = np.round(dist[np.ix_(idx, idx)], 9)
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        masked = np.where(mask, sub, np.inf)
        flat = int(np.argmin(masked))
        pi, pj = divmod(flat, m)
        i, j = int(idx[pi]), int(idx[pj])

        ni, nj = sizes[i], sizes[j]
        others = idx[(idx != i) & (idx != j)]
        if len(others) > 0:
            new_vals = (ni * dist[i, others] + nj * dist[j, others]) / (ni + nj)
            dist[i, others] = new_vals
            dist[others, i] = new_vals
        sizes[i] = ni + nj
        members[i] = members[i] + members[j]
        alive.remove(j)

    return [members[i] for i in alive]


def _split_conflicting(groups: dict[tuple, list[Signature]]) -> dict[tuple, list[Signature]]:
    """Split a non-MISMATCH hard_key group whose members carry two or more
    different trace evidence classes (soft ``uvm_ev``): the shared sim.log
    message is then evidently not specific to one bug. Members are regrouped
    by their class; members without evidence (``""``) form their own group.
    A group with zero or one class is left whole, so a case with no or thin
    evidence stays with its message siblings. The split key extends the
    hard_key so canon() ordering stays deterministic. See
    docs/ARCHITECTURE.md 5.4."""
    out: dict[tuple, list[Signature]] = {}
    for hk, sigs in groups.items():
        evs = {s.soft.get("uvm_ev") or "" for s in sigs} - {""}
        if hk[0] == "MISMATCH" or len(evs) < 2:
            out[hk] = sigs
            continue
        for s in sigs:
            out.setdefault(hk + ("<ev>" + (s.soft.get("uvm_ev") or ""),), []).append(s)
    return out


def cluster(signatures: list[Signature], k: int) -> dict[int, str]:
    """Cluster signatures into k buckets (k is a hard target, per Q&A A11).
    Returns case_id -> label ("bucket_<n>", n numbered by ascending minimum
    case_id per bucket).

    Deterministic: does not read case order, only hard_key/soft content and
    case_id (used solely as a tie-breaking/labeling key, never as a
    clustering feature). Groups by hard_key first (atoms), sorted by
    canon(hard_key) so the atom index assignment is stable regardless of
    PYTHONHASHSEED or input ordering.
    """
    if not signatures:
        return {}

    groups: dict[tuple, list[Signature]] = {}
    for sig in signatures:
        groups.setdefault(sig.hard_key, []).append(sig)
    groups = _split_conflicting(groups)

    hard_keys_sorted = sorted(groups.keys(), key=canon)
    atoms = [_Atom(hk, groups[hk]) for hk in hard_keys_sorted]

    if len(atoms) <= max(k, 1):
        final_groups = [[i] for i in range(len(atoms))]
    else:
        final_groups = _upgma_merge(atoms, k)

    bucket_ids: list[list[int]] = [
        sorted(cid for idx in group for cid in atoms[idx].ids) for group in final_groups
    ]
    order = sorted(range(len(bucket_ids)), key=lambda i: min(bucket_ids[i]))

    labels: dict[int, str] = {}
    for rank, idx in enumerate(order, start=1):
        label = f"bucket_{rank}"
        for cid in bucket_ids[idx]:
            labels[cid] = label
    return labels
