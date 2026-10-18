"""Coverage-driven tests for rfb.cluster: helper edge cases, R7 template
bonus, cluster([]), and _upgma_merge size bookkeeping / rounding tie-break.
"""

import math

import pytest

import rfb.cluster as cluster_mod
from rfb.cluster import (
    ASSERT_BASE,
    ASSERT_SPAN,
    D_UVM_NONE,
    D_UVM_ONESIDED_ASSERT,
    MISMATCH_MCLASS_WEIGHT,
    MISMATCH_VIOL_VEC_WEIGHT,
    TAIL_W,
    TMPL_BONUS,
    _Atom,
    _atom_distance,
    _avg_vector,
    _cos,
    _indicator_diff,
    canon,
    cluster,
)
from rfb.signature import Signature


def _sig(case_id, family, hard_key, **soft_overrides):
    soft = {
        "viol_vec": {},
        "viol_class": None,
        "sys_viol_class": frozenset(),
        "sys_viol_keys": frozenset(),
        "tail_ngrams": {},
        "implicit": 0,
        "assert_set": frozenset(),
        "fatal_tmpl": None,
        "sig_last": None,
    }
    soft.update(soft_overrides)
    return Signature(case_id=case_id, family=family, hard_key=hard_key, soft=soft)


# --- _cos (cluster.py 56-57, 66-67) -------------------------------------------------


def test_cos_empty_vector_is_zero_in_both_orders():
    assert _cos({}, {"a": 1.0}) == 0.0
    assert _cos({"a": 1.0}, {}) == 0.0
    assert _cos({}, {}) == 0.0


def test_cos_all_zero_vector_is_zero_not_division_error():
    assert _cos({"a": 0.0}, {"a": 5.0}) == 0.0
    assert _cos({"a": 5.0}, {"a": 0.0}) == 0.0
    assert _cos({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0}) == 0.0


def test_cos_known_values():
    assert _cos({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 2.0}) == pytest.approx(1.0)
    assert _cos({"a": 1.0}, {"b": 1.0}) == 0.0
    assert _cos({"a": 1.0, "b": 1.0}, {"a": 1.0}) == pytest.approx(1 / math.sqrt(2))


# --- _avg_vector -------------------------------------------------------------


def test_avg_vector_divides_by_member_count_including_members_without_key():
    m1 = _sig(1, "OTHER", ("OTHER", "a"), viol_vec={"x": 1.0, "y": 0.5})
    m2 = _sig(2, "OTHER", ("OTHER", "a"), viol_vec=None)
    m3 = _sig(3, "OTHER", ("OTHER", "a"), viol_vec={"x": 0.5})
    assert _avg_vector([m1, m2, m3], "viol_vec") == pytest.approx(
        {"x": 0.5, "y": 1 / 6}
    )
    assert _avg_vector([m1, m2, m3], "no_such_key") == {}


# --- canon(None) (cluster.py 97-98) ------------------------------------------


def test_canon_none_has_dedicated_marker_distinct_from_string_none():
    # 只釘性質，不釘完整回傳結構
    assert canon(None) != canon("None")
    assert canon(None) != canon("")
    assert canon(None) < canon("")
    assert canon(None) < canon("None")
    assert canon(("FATAL", None)) != canon(("FATAL", "None"))


def test_canon_frozenset_is_distinct_from_tuple_with_same_elements():
    assert canon(frozenset({1})) != canon((1,))
    assert canon(frozenset({1, 2})) != canon((1, 2))
    assert canon(frozenset({0, 8})) == canon(frozenset({8, 0}))


def test_distance_constants_match_architecture_5_4():
    # 期望值寫死，對照 docs/ARCHITECTURE.md 5.4（R1-R7 與常數清單）。
    assert cluster_mod.D_VIOL == 0.10
    assert cluster_mod.D_VIOLVEC == 0.30
    assert cluster_mod.VIOLVEC_COS_MIN == 0.5
    assert cluster_mod.ASSERT_BASE == 0.20
    assert cluster_mod.ASSERT_SPAN == 0.60
    assert cluster_mod.ASSERT_XFAM_MIN_OVL == 0.5
    assert cluster_mod.D_TRAP_LINK == 0.40
    assert cluster_mod.D_UVM_NONE == 0.55
    assert cluster_mod.D_UVM_ONESIDED_ASSERT == 0.70
    assert cluster_mod.TMPL_BONUS == 0.10
    assert cluster_mod.TAIL_W == 0.05
    assert cluster_mod.D_CROSS == 0.90
    # R3：TRAP_LINK 表與 UVM 側 family 集合（ARCHITECTURE R3）。
    assert cluster_mod.TRAP_LINK == {
        "ebreak": ("DEBUG", "EBREAK"),
        "c.ebreak": ("DEBUG", "EBREAK"),
        "dret": ("DRET",),
        "ecall": ("ECALL",),
        "mret": ("MRET",),
    }
    assert cluster_mod._UVM_FAMILIES == frozenset({"FATAL", "ASSERT", "OTHER"})
    # R4：0.5·(1 − cos(vvec)) + 0.5·[mtok 不相交]
    assert cluster_mod.MISMATCH_VIOL_VEC_WEIGHT == 0.5
    assert cluster_mod.MISMATCH_MCLASS_WEIGHT == 0.5


def test_canon_orders_none_template_before_string_template_deterministically():
    keys = [("FATAL", "zzz", frozenset()), ("FATAL", None, frozenset())]
    assert sorted(keys, key=canon) == [
        ("FATAL", None, frozenset()),
        ("FATAL", "zzz", frozenset()),
    ]


# --- cluster([]) (cluster.py 333-334) -----------------------------------------


@pytest.mark.parametrize("k", [0, 1, 5])
def test_cluster_empty_input_gives_empty_mapping(k):
    assert cluster([], k) == {}


# --- R7 template bonus (cluster.py 270-271) -----------------------------------


def _fatal_atom(case_id, tmpl, key, **soft):
    s = _sig(case_id, "FATAL", ("FATAL", tmpl, frozenset({key})), fatal_tmpl=tmpl, **soft)
    return _Atom(s.hard_key, [s])


def test_r7_shared_fatal_template_subtracts_bonus():
    a = _fatal_atom(1, "t", "e1")
    b = _fatal_atom(2, "t", "e2")
    c = _fatal_atom(3, "u", "e3")
    # empty tails -> cosine 0 -> full TAIL_W penalty
    assert _atom_distance(a, b) == pytest.approx(D_UVM_NONE - TMPL_BONUS + TAIL_W)
    assert _atom_distance(a, c) == pytest.approx(D_UVM_NONE + TAIL_W)


def test_r7_identical_tails_remove_tail_penalty():
    tail = {("a", "b", "c"): 1.0}
    a = _fatal_atom(1, "t", "e1", tail_ngrams=tail)
    b = _fatal_atom(2, "t", "e2", tail_ngrams=dict(tail))
    # ARCHITECTURE 5.4 R7：基底 D_UVM_NONE，減 TMPL_BONUS，尾部 cos=1 故 TAIL_W 項為 0
    assert _atom_distance(a, b) == pytest.approx(D_UVM_NONE - TMPL_BONUS)


def test_r7_one_sided_assert_uses_onesided_base_and_still_gets_bonus():
    a = _fatal_atom(1, "t", "e1", assert_set=frozenset({"A1"}))
    b = _fatal_atom(2, "t", "e2")
    c = _fatal_atom(3, "u", "e3")
    assert _atom_distance(a, b) == pytest.approx(
        D_UVM_ONESIDED_ASSERT - TMPL_BONUS + TAIL_W
    )
    assert _atom_distance(a, c) == pytest.approx(D_UVM_ONESIDED_ASSERT + TAIL_W)


def test_r7_template_bonus_decides_pairing_of_three_fatal_atoms():
    # 3 atoms, k=2. Sorted atom order is X(t1) < Y(t2,e1) < Z(t2,e2). Without
    # the bonus all three distances tie at 0.60 and the (i,j) tie-break would
    # merge X+Y; with it Y+Z (0.50) wins.
    sigs = [
        _sig(1, "FATAL", ("FATAL", "t1", frozenset()), fatal_tmpl="t1"),
        _sig(2, "FATAL", ("FATAL", "t2", frozenset({"e1"})), fatal_tmpl="t2"),
        _sig(3, "FATAL", ("FATAL", "t2", frozenset({"e2"})), fatal_tmpl="t2"),
    ]
    assert cluster(sigs, k=2) == {1: "bucket_1", 2: "bucket_2", 3: "bucket_2"}


# --- _indicator_diff (cluster.py 74-76) ---------------------------------------


def test_indicator_diff_values():
    assert _indicator_diff(frozenset(), frozenset()) == 0.0  # no info: same
    assert _indicator_diff(frozenset({"a"}), frozenset({"a", "b"})) == 0.0
    assert _indicator_diff(frozenset({"a"}), frozenset({"b"})) == 1.0
    assert _indicator_diff(frozenset({"a"}), frozenset()) == 1.0
    assert _indicator_diff(frozenset(), frozenset({"a"})) == 1.0


def _mismatch_atom(case_id, key):
    s = _sig(case_id, "MISMATCH", key)
    return _Atom(s.hard_key, [s])


def test_r4_atoms_without_mclass_token_are_treated_as_same_class():
    # 2-tuple hard_keys carry no mclass token (mtok empty on both sides), so
    # only the (zero) violation-vector cosine contributes: 0.5 * (1 - 0).
    a = _mismatch_atom(1, ("MISMATCH", "a"))
    b = _mismatch_atom(2, ("MISMATCH", "b"))
    # ARCHITECTURE 5.4 R4：W_vec*(1-cos) + W_mclass*indicator；cos=0、indicator=0
    assert _atom_distance(a, b) == pytest.approx(MISMATCH_VIOL_VEC_WEIGHT)
    # 3-tuple keys with different class tokens add the full indicator
    c = _mismatch_atom(3, ("MISMATCH", "K", "ADDSUB"))
    d = _mismatch_atom(4, ("MISMATCH", "K", "LOAD"))
    assert _atom_distance(c, d) == pytest.approx(
        MISMATCH_VIOL_VEC_WEIGHT + MISMATCH_MCLASS_WEIGHT)
    e = _mismatch_atom(5, ("MISMATCH", "J", "ADDSUB"))
    assert _atom_distance(c, e) == pytest.approx(MISMATCH_VIOL_VEC_WEIGHT)


# --- R2 template bonus (cluster.py 239-240) -----------------------------------


def _assert_atom(case_id, key, asserts, tmpl=None):
    s = _sig(case_id, "ASSERT", ("ASSERT", frozenset({key})),
             assert_set=frozenset(asserts), fatal_tmpl=tmpl)
    return _Atom(s.hard_key, [s])


def test_r2_shared_template_subtracts_bonus_from_assert_distance():
    a = _assert_atom(1, "k1", {"A"}, "t")
    b = _assert_atom(2, "k2", {"A"}, "t")
    c = _assert_atom(3, "k3", {"A"}, "u")
    # ARCHITECTURE 5.4 R2：ASSERT_BASE + ASSERT_SPAN*(1-ovl) - TMPL_BONUS*[共享] + TAIL_W*(1-cos)
    # full overlap (ovl=1)，空 tail -> cos=0 -> 全額 TAIL_W
    assert _atom_distance(a, b) == pytest.approx(ASSERT_BASE - TMPL_BONUS + TAIL_W)
    assert _atom_distance(a, c) == pytest.approx(ASSERT_BASE + TAIL_W)
    # partial overlap ovl=1/2
    p = _assert_atom(4, "k4", {"A", "B"}, "t")
    q = _assert_atom(5, "k5", {"A", "C"}, "t")
    r = _assert_atom(6, "k6", {"A", "C"}, "u")
    base = ASSERT_BASE + ASSERT_SPAN * (1 - 0.5)
    assert _atom_distance(p, q) == pytest.approx(base - TMPL_BONUS + TAIL_W)
    assert _atom_distance(p, r) == pytest.approx(base + TAIL_W)


def test_r2_template_bonus_decides_pairing_of_three_assert_atoms():
    # 3 atoms, k=2, sorted X < Y < Z. Without the bonus the three distances
    # tie (0.25) and (i,j) order would merge X+Y; with it Y+Z (0.15) wins.
    sigs = [
        _sig(1, "ASSERT", ("ASSERT", frozenset({"a"})), assert_set=frozenset({"A"})),
        _sig(2, "ASSERT", ("ASSERT", frozenset({"b"})), assert_set=frozenset({"A"}),
             fatal_tmpl="t"),
        _sig(3, "ASSERT", ("ASSERT", frozenset({"c"})), assert_set=frozenset({"A"}),
             fatal_tmpl="t"),
    ]
    assert cluster(sigs, k=2) == {1: "bucket_1", 2: "bucket_2", 3: "bucket_2"}


# --- _upgma_merge: size bookkeeping and rounding tie-break --------------------


def _atoms_with_distances(labels, dmat, monkeypatch):
    monkeypatch.setattr(
        cluster_mod,
        "_atom_distance",
        lambda a, b: dmat[frozenset({a.hard_key[1], b.hard_key[1]})],
    )
    sigs = [_sig(i + 1, "OTHER", ("OTHER", lab)) for i, lab in enumerate(labels)]
    atoms = [_Atom(s.hard_key, [s]) for s in sorted(sigs, key=lambda s: canon(s.hard_key))]
    assert [a.hard_key[1] for a in atoms] == list(labels)
    return atoms


def _pairs(labels, values):
    out = {}
    it = iter(values)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            out[frozenset({labels[i], labels[j]})] = next(it)
    return out


def test_upgma_merged_cluster_size_feeds_later_lance_williams_updates(monkeypatch):
    # Five singleton atoms, k=2. Distances chosen (by exhaustive search
    # against a reference implementation) so that the final partition is
    # different for every wrong size bookkeeping after a merge: keeping ni
    # or nj instead of ni+nj, ni+nj+/-1, 1, ni*nj, max(ni, nj).
    labels = ["a0", "a1", "a2", "a3", "a4"]
    # order: 01 02 03 04 12 13 14 23 24 34
    values = [0.36, 0.71, 0.33, 0.48, 0.44, 0.40, 0.78, 0.45, 0.53, 0.38]
    atoms = _atoms_with_distances(labels, _pairs(labels, values), monkeypatch)
    groups = cluster_mod._upgma_merge(atoms, 2)
    assert sorted(sorted(g) for g in groups) == [[0, 1, 3], [2, 4]]


def test_upgma_round_ignores_sub_nanounit_noise_so_index_order_breaks_tie(monkeypatch):
    # d(0,1) exceeds d(0,2) by 3e-10 -> equal after round(., 9) -> the
    # lexicographic (i, j) tie-break picks (0, 1). Any precision > 9 (or no
    # rounding) would see (0, 2) as strictly smaller and merge that instead.
    labels = ["a0", "a1", "a2"]
    dmat = _pairs(labels, [0.5 + 3e-10, 0.5, 0.9])
    atoms = _atoms_with_distances(labels, dmat, monkeypatch)
    groups = cluster_mod._upgma_merge(atoms, 2)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2]]


def test_upgma_round_keeps_differences_above_nanounit(monkeypatch):
    # d(0,1) exceeds d(0,2) by 2e-9 -> still distinct at 9 decimals, so the
    # strictly smaller (0, 2) wins despite the larger index. Any precision
    # < 9 would tie them and pick (0, 1).
    labels = ["a0", "a1", "a2"]
    dmat = _pairs(labels, [0.5 + 2e-9, 0.5, 0.9])
    atoms = _atoms_with_distances(labels, dmat, monkeypatch)
    groups = cluster_mod._upgma_merge(atoms, 2)
    assert sorted(sorted(g) for g in groups) == [[0, 2], [1]]
