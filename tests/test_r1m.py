"""Tests for R1m (cluster._ev_mismatch_link, signature._uvm_ev_sys). See
docs/ARCHITECTURE.md 5.3 (uvm_ev_sys) and 5.4 (R1m)."""

import pytest

from rfb.cluster import D_CROSS, D_VIOL, D_VIOLVEC, _Atom, _atom_distance, cluster
from rfb.signature import _uvm_ev, _uvm_ev_sys, signature
from tests.test_cluster import _sig
from tests.test_uvm_split import _dev_d, _rec

SEXT = ("LOAD_EXT", "LD_SEXT")
EV = "LOAD_EXT:LD_SEXT"


# --- _uvm_ev_sys -------------------------------------------------------------


def _ev_sys(violations, kind="LOAD_EXT", mnemonic="lb", family="FATAL", detail=None):
    rec = _rec(family, _dev_d(kind, mnemonic, detail or {}))
    rec.trace.violations = violations
    return _uvm_ev_sys(rec)


@pytest.mark.parametrize(
    "kind,mnemonic,detail,ev",
    [
        ("LOAD_EXT", "lb", None, "LOAD_EXT:LD_SEXT"),
        ("MEM_MISMATCH", "lw", None, "MEM_MISMATCH:MEMREAD"),
        ("CSR_MISMATCH", "csrrw", {"pol": "+"}, "CSR_MISMATCH:CSR+"),
    ],
)
def test_uvm_ev_sys_two_of_kind_returns_ev(kind, mnemonic, detail, ev):
    # Every kind in _SCAN_EXCLUDED (except unreachable READ_MISMATCH) qualifies.
    assert _ev_sys({f"{kind}:{mnemonic}": 2}, kind, mnemonic, detail=detail) == ev


def test_uvm_ev_sys_one_of_kind_returns_empty():
    assert _ev_sys({"LOAD_EXT:lb": 1}) == ""


def test_uvm_ev_sys_counts_across_mnemonics_of_kind():
    # lb and lh once each: per-kind total 2 -> evidence kept.
    assert _ev_sys({"LOAD_EXT:lb": 1, "LOAD_EXT:lh": 1}) == EV


def test_uvm_ev_sys_other_kinds_do_not_count():
    assert _ev_sys({"LOAD_EXT:lb": 1, "WRONG_RESULT:add": 5}) == ""


def test_uvm_ev_sys_mismatch_family_empty():
    assert _ev_sys({"LOAD_EXT:lb": 3}, family="MISMATCH") == ""


def test_uvm_ev_sys_no_deviation_empty():
    rec = _rec("FATAL", None)
    rec.trace.violations = {"LOAD_EXT:lb": 3}
    assert _uvm_ev_sys(rec) == ""


@pytest.mark.parametrize(
    "kind,mnemonic,other",
    [
        ("TRAP_NOT_TAKEN", "ecall", "mret"),
        ("BRANCH_WRONG_TARGET", "beq", "bne"),
        ("WRONG_RESULT", "lw", "lh"),  # non-ALU WRONG_RESULT -> "<kind>:<group>"
    ],
)
def test_uvm_ev_sys_non_excluded_kind_empty(kind, mnemonic, other):
    # R1 needs the same kind:mnemonic >= 2 times; R1m must not be looser for
    # kinds R1 already sees (not in _SCAN_EXCLUDED), even at kind total >= 2.
    # Control: _uvm_ev is non-empty here, so only the guard yields "".
    assert _uvm_ev(_rec("FATAL", _dev_d(kind, mnemonic, {}))).startswith(kind + ":")
    v = {f"{kind}:{mnemonic}": 1, f"{kind}:{other}": 1}
    assert _ev_sys(v, kind=kind, mnemonic=mnemonic) == ""
    v = {f"{kind}:{mnemonic}": 3}
    assert _ev_sys(v, kind=kind, mnemonic=mnemonic) == ""


def test_uvm_ev_sys_counts_kind_not_group():
    # Pins current behaviour (docs 5.5 residual risk): the threshold is per
    # kind, so one lb (LD_SEXT) first_violation plus five lbu (LD_ZEXT) passes.
    assert _ev_sys({"LOAD_EXT:lb": 1, "LOAD_EXT:lbu": 5}) == EV


def test_signature_soft_carries_uvm_ev_sys():
    rec = _rec("FATAL", _dev_d("LOAD_EXT", "lb", {}))
    rec.trace.violations = {"LOAD_EXT:lb": 2}
    assert signature(rec).soft["uvm_ev_sys"] == EV
    # thin evidence: uvm_ev set but uvm_ev_sys empty (not just uvm_ev copied).
    rec = _rec("FATAL", _dev_d("LOAD_EXT", "lb", {}))
    rec.trace.violations = {"LOAD_EXT:lb": 1}
    soft = signature(rec).soft
    assert soft["uvm_ev"] == EV
    assert soft["uvm_ev_sys"] == ""


# --- R1m distance ------------------------------------------------------------


def _uvm(cid, ev, ev_sys, split=True, family="FATAL", **soft):
    hk = (family, "m", "<ev>" + ev) if split else (family, "m", ev)
    return _Atom(hk, [_sig(cid, family, hk, uvm_ev=ev, uvm_ev_sys=ev_sys, **soft)])


def _mm(cid, vcls=frozenset({SEXT})):
    hk = ("MISMATCH", "LOAD_EXT", "LD_SEXT")
    return _Atom(hk, [_sig(cid, "MISMATCH", hk, sys_viol_class=vcls)])


def test_r1m_hits_split_uvm_vs_mismatch():
    x, y = _uvm(1, EV, EV), _mm(2)
    assert x.ev_sys == frozenset({EV}) and x.split_of == ("FATAL", "m")
    assert _atom_distance(x, y) == D_VIOL


def test_r1m_not_hit_when_unsplit():
    x, y = _uvm(1, EV, EV, split=False), _mm(2)
    assert x.split_of is None
    assert _atom_distance(x, y) == D_CROSS


def test_r1m_not_hit_when_ev_sys_empty():
    x, y = _uvm(1, EV, ""), _mm(2)
    assert x.ev == frozenset({EV}) and x.ev_sys == frozenset()
    assert _atom_distance(x, y) == D_CROSS


def test_r1m_not_hit_when_other_is_uvm():
    x = _uvm(1, EV, EV)
    y = _uvm(2, "B:y", "", family="FATAL", sys_viol_class=frozenset({SEXT}))
    assert y.family == "FATAL" and y.vcls == frozenset({SEXT})
    assert _atom_distance(x, y) == D_CROSS


def test_r1m_symmetric_argument_order():
    x, y = _uvm(1, EV, EV), _mm(2)
    assert _atom_distance(y, x) == D_VIOL
    assert _atom_distance(x, y) == _atom_distance(y, x)


def test_r1m_requires_kind_prefix():
    # A bare viol_group must not match "<kind>:<viol_group>".
    x, y = _uvm(1, "LD_SEXT", "LD_SEXT"), _mm(2)
    assert _atom_distance(x, y) == D_CROSS


def test_r1m_different_class_not_hit():
    x = _uvm(1, "CSR_MISMATCH:CSR+", "CSR_MISMATCH:CSR+")
    assert _atom_distance(x, _mm(2)) == D_CROSS


@pytest.mark.parametrize("thick_first", [False, True])
def test_r1m_ev_sys_is_union_over_members(thick_first):
    # _Atom orders members by case_id: cover the valued member being last and first.
    hk = ("FATAL", "m", "<ev>" + EV)
    thin_id, thick_id = (2, 1) if thick_first else (1, 2)
    thin = _sig(thin_id, "FATAL", hk, uvm_ev=EV, uvm_ev_sys="")
    thick = _sig(thick_id, "FATAL", hk, uvm_ev=EV, uvm_ev_sys=EV)
    x = _Atom(hk, [thin, thick])
    assert x.ev_sys == frozenset({EV})
    assert _atom_distance(x, _mm(3)) == D_VIOL
    assert _atom_distance(_mm(3), x) == D_VIOL


def test_r1m_beats_r2_assert_overlap():
    t, u = frozenset({"T"}), frozenset({"T", "U"})
    x = _uvm(1, EV, EV, family="ASSERT", assert_set=u)
    hk = ("MISMATCH", "LOAD_EXT", "LD_SEXT")
    y = _Atom(hk, [_sig(2, "MISMATCH", hk, sys_viol_class=frozenset({SEXT}), assert_set=t)])
    assert _atom_distance(x, y) == D_VIOL
    # Control: without ev_sys the ladder falls through to R2, strictly farther.
    x0 = _uvm(1, EV, "", family="ASSERT", assert_set=u)
    assert _atom_distance(x0, y) > D_VIOL


def test_r1m_beats_r1b_alu_op_link():
    wr = frozenset({("WRONG_RESULT", "add")})
    x = _uvm(1, EV, EV, sys_viol_keys=wr)
    hk = ("MISMATCH", "LOAD_EXT", "LD_SEXT")
    y = _Atom(
        hk, [_sig(2, "MISMATCH", hk, sys_viol_class=frozenset({SEXT}), sys_viol_keys=wr)]
    )
    assert _atom_distance(x, y) == D_VIOL
    # Control: without ev_sys R1b fires with a different distance, so the
    # assertion above discriminates the R1m-before-R1b order.
    x0 = _uvm(1, EV, "", sys_viol_keys=wr)
    assert _atom_distance(x0, y) == D_VIOLVEC != D_VIOL


def test_r1m_other_family_hits():
    x = _uvm(1, EV, EV, family="OTHER")
    assert x.family == "OTHER" and x.split_of == ("OTHER", "m")
    assert _atom_distance(x, _mm(2)) == D_VIOL


# --- end to end --------------------------------------------------------------


def test_end_to_end_split_assert_returns_to_own_mismatch_bucket():
    # Two bugs (LOAD_EXT, CSR_MISMATCH) share ASSERT message "T" -> split into
    # two sub-atoms by evidence. Each bug also has MISMATCH cases whose
    # sys_viol_class holds the bug's class; both MISMATCH atoms are otherwise
    # identical (same vvec, mclass), so only R1m separates them. 4 atoms, k=2.
    t = frozenset({"T"})
    hk = ("ASSERT", "m")
    csr = ("CSR_MISMATCH", "CSR+")
    ev_c = "CSR_MISMATCH:CSR+"
    mk = ("MISMATCH", "X")
    sigs = [
        _sig(1, "ASSERT", hk, uvm_ev=EV, uvm_ev_sys=EV, assert_set=t),
        _sig(2, "ASSERT", hk, uvm_ev=EV, uvm_ev_sys=EV, assert_set=t),
        _sig(3, "ASSERT", hk, uvm_ev=ev_c, uvm_ev_sys=ev_c, assert_set=t),
        _sig(4, "ASSERT", hk, uvm_ev=ev_c, uvm_ev_sys=ev_c, assert_set=t),
        _sig(5, "MISMATCH", mk + ("a",), sys_viol_class=frozenset({SEXT})),
        _sig(6, "MISMATCH", mk + ("a",), sys_viol_class=frozenset({SEXT})),
        _sig(7, "MISMATCH", mk + ("b",), sys_viol_class=frozenset({csr})),
        _sig(8, "MISMATCH", mk + ("b",), sys_viol_class=frozenset({csr})),
    ]
    labels = cluster(sigs, k=2)
    assert len(set(labels.values())) == 2
    assert labels[1] == labels[2] == labels[5] == labels[6]
    assert labels[3] == labels[4] == labels[7] == labels[8]
    assert labels[1] != labels[3]


# --- TRAP_* kinds (in _SCAN_EXCLUDED, so R1m applies) -----------------------------

TRAP_CASES = [
    ("TRAP_MCAUSE", {"trap": "ecall"}, "TRAP_MCAUSE:TRAP_MCAUSE:ecall"),
    ("TRAP_MEPC", {"trap": "irq"}, "TRAP_MEPC:TRAP_MEPC:irq"),
    ("TRAP_MTVAL", {"trap": "ecall"}, "TRAP_MTVAL:TRAP_MTVAL:ecall"),
]


@pytest.mark.parametrize("kind,detail,ev", TRAP_CASES)
def test_uvm_ev_sys_trap_kinds_two_returns_ev(kind, detail, ev):
    assert _ev_sys({f"{kind}:csrrs": 2}, kind, "csrrs", detail=detail) == ev


@pytest.mark.parametrize("kind,detail,ev", TRAP_CASES)
def test_uvm_ev_sys_trap_kinds_one_returns_empty(kind, detail, ev):
    assert _ev_sys({f"{kind}:csrrs": 1}, kind, "csrrs", detail=detail) == ""


def test_uvm_ev_sys_trap_kind_counts_per_kind_not_other_trap_kinds():
    # one TRAP_MCAUSE + one TRAP_MEPC is not two of the deviation's kind
    v = {"TRAP_MCAUSE:csrrs": 1, "TRAP_MEPC:csrrs": 1}
    assert _ev_sys(v, "TRAP_MCAUSE", "csrrs", detail={"trap": "ecall"}) == ""


def test_uvm_ev_sys_xret_target_not_excluded_stays_empty():
    # XRET_TARGET is deliberately outside _SCAN_EXCLUDED: R1 already sees it
    assert _ev_sys({"XRET_TARGET:mret": 3}, "XRET_TARGET", "mret", detail={}) == ""


def test_r1m_hits_trap_mcause_uvm_vs_mismatch():
    ev = "TRAP_MCAUSE:TRAP_MCAUSE:ecall"
    x = _uvm(1, ev, ev)
    hk = ("MISMATCH", "TRAP_MCAUSE", "ecall")
    cls = frozenset({("TRAP_MCAUSE", "TRAP_MCAUSE:ecall")})
    y = _Atom(hk, [_sig(2, "MISMATCH", hk, sys_viol_class=cls)])
    assert x.ev_sys == frozenset({ev})
    assert _atom_distance(x, y) == D_VIOL
    x0 = _uvm(1, ev, "")
    assert _atom_distance(x0, y) == D_CROSS
