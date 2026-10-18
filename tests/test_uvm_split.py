"""Tests for the UVM trace-evidence split (signature._uvm_ev,
cluster._split_conflicting, distance rule R1e-). See docs/ARCHITECTURE.md 5.3 (uvm_ev) and 5.4 (split, R1e-);
5.5 is residual risk."""

import random

import pytest

from rfb.cluster import D_CROSS, D_VIOL, D_VIOLVEC, _Atom, _atom_distance, _split_conflicting, cluster
from rfb.record import Deviation, FailureRecord, RegrInfo, SimInfo, TraceInfo
from rfb.signature import _uvm_ev, signature
from tests.test_cluster import _sig


def _dev(kind, mnemonic="add"):
    return Deviation(
        kind=kind, mnemonic=mnemonic, mclass=None, detail={}, source="scan", confidence=0.7
    )


def _rec(family, deviation=None, case_id=1):
    regr = RegrInfo.empty()
    regr.raw = ["x"]
    return FailureRecord(
        case_id=case_id,
        family=family,
        regr=regr,
        sim=SimInfo.empty(),
        trace=TraceInfo.empty(),
        deviation=deviation,
        parse_errors=[],
    )


# --- _uvm_ev -----------------------------------------------------------------


def _dev_d(kind, mnemonic, detail):
    return Deviation(
        kind=kind, mnemonic=mnemonic, mclass=None, detail=detail, source="scan", confidence=0.7
    )


def _ev(kind, mnemonic, detail=None, family="FATAL"):
    return _uvm_ev(_rec(family, _dev_d(kind, mnemonic, detail or {})))


def test_uvm_ev_non_mismatch_with_violation():
    assert _ev("WRONG_RESULT", "add") == "ALU:add"


@pytest.mark.parametrize("family", ["FATAL", "ASSERT", "OTHER"])
def test_uvm_ev_families_use_deviation_group(family):
    assert _ev("BRANCH_NOT_TAKEN", "beq", family=family) == "BR_EQ-"
    assert _ev("TRAP_NOT_TAKEN", "ecall", family=family).startswith("TRAP_NOT_TAKEN:")


def test_uvm_ev_detail_kinds_pinned():
    assert _ev("CSR_MISMATCH", "csrrw", {"pol": "+"}) == "CSR_MISMATCH:CSR+"
    assert _ev("READ_MISMATCH", "add", {"fwd": True}) == "READ_MISMATCH:REGREAD_FWD"
    assert _ev("READ_MISMATCH", "add") == "READ_MISMATCH:REGREAD"
    assert _ev("LOAD_EXT", "lb") == "LOAD_EXT:LD_SEXT"
    assert _ev("LOAD_EXT", "lbu") == "LOAD_EXT:LD_ZEXT"


def test_uvm_ev_kind_prefix_pinned():
    # BRANCH_WRONG_TARGET is not in the bare-viol_group branch: keeps its prefix.
    assert _ev("BRANCH_WRONG_TARGET", "beq") == "BRANCH_WRONG_TARGET:BR_BEQ"
    assert _ev("BRANCH_WRONG_TARGET", "beq") != _ev("BRANCH_NOT_TAKEN", "beq")
    assert _ev("XRET_FALLTHROUGH", "mret") == "XRET_FALLTHROUGH:XRET"
    assert _ev("MEM_MISMATCH", "lw") == "MEM_MISMATCH:MEMREAD"
    # Same viol_group, told apart only by the kind prefix.
    assert _ev("WRONG_LINK", "jal") == "WRONG_LINK:JUMP"
    assert _ev("BRANCH_WRONG_TARGET", "jal") == "BRANCH_WRONG_TARGET:JUMP"
    assert _ev("WRONG_ADDR", "sw") == "WRONG_ADDR:STORE"
    assert _ev("WRONG_STORE_DATA", "sw") == "WRONG_STORE_DATA:STORE"


def test_uvm_ev_alu_same_op_same_string():
    assert _ev("WRONG_RESULT", "sra") == _ev("WRONG_RESULT", "c.srai") == "ALU:sra"
    assert _ev("WRONG_RESULT", "add") == _ev("WRONG_RESULT", "addi") == "ALU:add"
    assert _ev("WRONG_RESULT", "sra") != _ev("WRONG_RESULT", "add")


def test_uvm_ev_branch_direction_merged():
    assert _ev("BRANCH_NOT_TAKEN", "beq") == _ev("BRANCH_WRONGLY_TAKEN", "bne") == "BR_EQ-"
    assert _ev("BRANCH_WRONGLY_TAKEN", "beq") == _ev("BRANCH_NOT_TAKEN", "bne") == "BR_EQ+"
    assert _ev("BRANCH_NOT_TAKEN", "blt") == _ev("BRANCH_WRONGLY_TAKEN", "blt") == "BR_LT"
    assert _ev("BRANCH_NOT_TAKEN", "bge") == _ev("BRANCH_WRONGLY_TAKEN", "bge") == "BR_GE"
    assert _ev("BRANCH_NOT_TAKEN", "blt") != _ev("BRANCH_NOT_TAKEN", "bge")


def test_uvm_ev_cluster_same_bug_different_mnemonic_same_bucket():
    ev_a = _ev("WRONG_RESULT", "sra")
    ev_b = _ev("WRONG_RESULT", "c.srai")
    ev_c = _ev("BRANCH_NOT_TAKEN", "blt")
    ev_d = _ev("BRANCH_WRONGLY_TAKEN", "blt")
    sigs = [
        _sig(1, "FATAL", ("FATAL", "m"), uvm_ev=ev_a),
        _sig(2, "FATAL", ("FATAL", "m"), uvm_ev=ev_b),
        _sig(3, "FATAL", ("FATAL", "n"), uvm_ev=ev_c),
        _sig(4, "FATAL", ("FATAL", "n"), uvm_ev=ev_d),
    ]
    labels = cluster(sigs, k=2)
    assert labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[1] != labels[3]


def test_uvm_ev_no_violation_is_empty():
    assert _uvm_ev(_rec("FATAL", None)) == ""
    assert _uvm_ev(_rec("ASSERT", _dev("NONE", None))) == ""


def test_uvm_ev_mismatch_always_empty():
    assert _uvm_ev(_rec("MISMATCH", _dev("WRONG_RESULT", "add"))) == ""
    assert _uvm_ev(_rec("MISMATCH", None)) == ""


# --- _split_conflicting ------------------------------------------------------


def _usig(cid, ev, family="FATAL", hk=("FATAL", "m")):
    return _sig(cid, family, hk, uvm_ev=ev)


def _ids(groups):
    return {k: sorted(s.case_id for s in v) for k, v in groups.items()}


def test_split_zero_evidence_untouched():
    g = {("FATAL", "m"): [_usig(1, ""), _usig(2, "")]}
    assert _ids(_split_conflicting(g)) == {("FATAL", "m"): [1, 2]}


def test_split_one_evidence_untouched():
    g = {("FATAL", "m"): [_usig(1, "A:x"), _usig(2, "A:x"), _usig(3, "")]}
    assert _ids(_split_conflicting(g)) == {("FATAL", "m"): [1, 2, 3]}


def test_split_two_evidence_splits_and_empty_own_group():
    g = {("FATAL", "m"): [_usig(1, "A:x"), _usig(2, "B:y"), _usig(3, ""), _usig(4, "A:x")]}
    out = _ids(_split_conflicting(g))
    assert out == {
        ("FATAL", "m", "<ev>A:x"): [1, 4],
        ("FATAL", "m", "<ev>B:y"): [2],
        ("FATAL", "m", "<ev>"): [3],
    }


def test_split_three_evidence_splits():
    g = {("FATAL", "m"): [_usig(1, "A"), _usig(2, "B"), _usig(3, "C")]}
    assert len(_split_conflicting(g)) == 3


def test_split_mismatch_never_split():
    hk = ("MISMATCH", "WRONG_RESULT", "add")
    g = {hk: [_usig(1, "A", "MISMATCH", hk), _usig(2, "B", "MISMATCH", hk)]}
    assert _ids(_split_conflicting(g)) == {hk: [1, 2]}


def test_split_other_groups_independent():
    g = {
        ("FATAL", "m"): [_usig(1, "A"), _usig(2, "B")],
        ("FATAL", "n"): [_usig(3, "A"), _usig(4, "A")],
    }
    out = _ids(_split_conflicting(g))
    assert ("FATAL", "n") in out
    assert ("FATAL", "m") not in out
    assert len(out) == 3


def test_split_independent_of_input_order():
    sigs = [_usig(i, ev) for i, ev in enumerate(["A", "B", "", "A", "B", ""], start=1)]
    base = _ids(_split_conflicting({("FATAL", "m"): list(sigs)}))
    rng = random.Random(0)
    for _ in range(5):
        sh = list(sigs)
        rng.shuffle(sh)
        out = _split_conflicting({("FATAL", "m"): sh})
        assert _ids(out) == base
        assert sorted(out) == sorted(base)


# --- R1e- --------------------------------------------------------------------


def _atom(cid, ev, family="FATAL", split="m", asserts=frozenset({"T"})):
    hk = (family, split, "<ev>" + ev) if split is not None else (family, "m", ev)
    return _Atom(hk, [_sig(cid, family, hk, uvm_ev=ev, assert_set=asserts)])


def test_r1e_minus_fires_for_disjoint_evidence_same_split():
    assert _atom_distance(_atom(1, "A:x"), _atom(2, "B:y")) == D_CROSS


def test_r1e_minus_not_when_split_of_differs():
    a, b = _atom(1, "A:x", split="m"), _atom(2, "B:y", split="n")
    assert a.split_of != b.split_of
    assert _atom_distance(a, b) < D_CROSS


def test_r1e_minus_not_when_split_of_none():
    a, b = _atom(1, "A:x", split=None), _atom(2, "B:y", split=None)
    assert a.split_of is None and b.split_of is None
    assert _atom_distance(a, b) < D_CROSS


def test_r1e_minus_not_when_one_ev_empty():
    assert _atom_distance(_atom(1, "A:x"), _atom(2, "")) < D_CROSS
    assert _atom_distance(_atom(1, ""), _atom(2, "B:y")) < D_CROSS


def test_r1e_minus_not_when_ev_overlaps():
    a = _Atom(("FATAL", "m", "<ev>A"), [
        _sig(1, "FATAL", ("FATAL", "m", "<ev>A"), uvm_ev="A", assert_set=frozenset({"T"})),
        _sig(2, "FATAL", ("FATAL", "m", "<ev>A"), uvm_ev="B", assert_set=frozenset({"T"})),
    ])
    b = _atom(3, "B")
    assert a.ev & b.ev
    assert _atom_distance(a, b) < D_CROSS


def test_r1e_minus_not_when_not_uvm():
    a = _atom(1, "A:x", family="MISMATCH")
    b = _atom(2, "B:y", family="MISMATCH")
    assert a.split_of == b.split_of is not None
    assert _atom_distance(a, b) < D_CROSS


def _sib(cid, ev, **soft):
    """Sibling atom (same split_of, disjoint ev) carrying extra soft fields."""
    hk = ("FATAL", "m", "<ev>" + ev)
    return _Atom(hk, [_sig(cid, "FATAL", hk, uvm_ev=ev, assert_set=frozenset({"T"}), **soft)])


def test_r1_takes_priority_over_r1e_minus():
    a = _sib(1, "A:x", sys_viol_class=frozenset({"V"}))
    b = _sib(2, "B:y", sys_viol_class=frozenset({"V"}))
    assert a.split_of == b.split_of is not None and not (a.ev & b.ev)
    assert _atom_distance(a, b) == D_VIOL


def test_r1b_takes_priority_over_r1e_minus():
    # Same base ALU op WRONG_RESULT on both sides, wr_sys empty -> R1b holds;
    # vcls empty so R1 does not.
    keys = frozenset({("WRONG_RESULT", "add")})
    a = _sib(1, "A:x", sys_viol_keys=keys)
    b = _sib(2, "B:y", sys_viol_keys=keys)
    assert a.split_of == b.split_of is not None and not (a.ev & b.ev)
    assert not (a.vcls & b.vcls)
    assert _atom_distance(a, b) == D_VIOLVEC


# --- end to end --------------------------------------------------------------


def _e2e_sig(cid, kind):
    rec = _rec("FATAL", _dev(kind, "add" if kind == "WRONG_RESULT" else "beq"), case_id=cid)
    return signature(rec)


def test_end_to_end_conflicting_evidence_splits_into_two_buckets():
    sigs = [_e2e_sig(1, "WRONG_RESULT"), _e2e_sig(2, "BRANCH_NOT_TAKEN")]
    assert sigs[0].hard_key == sigs[1].hard_key
    labels = cluster(sigs, k=2)
    assert labels[1] != labels[2]
    assert len(set(labels.values())) == 2


def test_end_to_end_same_evidence_same_bucket():
    sigs = [_e2e_sig(1, "WRONG_RESULT"), _e2e_sig(2, "WRONG_RESULT")]
    labels = cluster(sigs, k=2)
    assert labels[1] == labels[2]


def test_end_to_end_upgma_does_not_remerge_split_siblings():
    # Same ASSERT message {"T"} for 4 cases; evidence LOAD_EXT (1,2) vs
    # CSR_MISMATCH (3,4) -> _split_conflicting yields 2 sibling atoms. A third
    # atom (5,6) has an unrelated message {"U"}. 3 atoms > k=2, so UPGMA runs.
    # Without R1e-, R2 (both UVM, no vcls) gives the siblings d~0.2 (+tail), the
    # closest pair, so they would merge. With R1e- they are D_CROSS apart and
    # the unrelated atom (d~0.8) merges with one of them instead.
    t, u = frozenset({"T"}), frozenset({"U"})
    hk, hu = ("ASSERT", "m"), ("ASSERT", "n")
    sigs = [
        _sig(1, "ASSERT", hk, uvm_ev="LOAD_EXT:a", assert_set=t),
        _sig(2, "ASSERT", hk, uvm_ev="LOAD_EXT:a", assert_set=t),
        _sig(3, "ASSERT", hk, uvm_ev="CSR_MISMATCH:b", assert_set=t),
        _sig(4, "ASSERT", hk, uvm_ev="CSR_MISMATCH:b", assert_set=t),
        _sig(5, "ASSERT", hu, assert_set=u),
        _sig(6, "ASSERT", hu, assert_set=u),
    ]
    labels = cluster(sigs, k=2)
    assert len(set(labels.values())) == 2
    assert labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[1] != labels[3]
