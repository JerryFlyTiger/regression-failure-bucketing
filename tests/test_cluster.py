"""Synthetic-Signature tests for rfb.cluster. See docs/ARCHITECTURE.md
sections 5.4 and 5.5.

2026-09-28 批次二修正，依據 Q&A A11: rewritten for the evidence-ladder
distance (R1-R7) + UPGMA merge replacing the M2 per-family rules.
"""

import os
import random
import subprocess
import sys

import pytest

from rfb.cluster import D_VIOLVEC, _Atom, _atom_distance, canon, cluster
from rfb.isa import mclass as isa_mclass
from rfb.record import FailureRecord, RegrInfo, SimInfo, TraceInfo
from rfb.signature import MIN_SYS_VIOL, Signature, signature
from tests.test_public_score import _write_shuffled_input

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET2_INPUT = os.path.join(
    REPO_ROOT, "testcase", "2026_08_20_samples", "problem", "benchmark_set_2", "input.csv"
)


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


def _record(family, violations, deviation=None, raw="x"):
    regr = RegrInfo.empty()
    regr.raw = [raw]
    sim = SimInfo.empty()
    trace = TraceInfo.empty()
    trace.violations = violations
    return FailureRecord(
        case_id=1,
        family=family,
        regr=regr,
        sim=sim,
        trace=trace,
        deviation=deviation,
        parse_errors=[],
    )


def test_merges_same_family_close_viol_vec_and_mclass():
    # Two different hard_keys (different deviation.kind) but identical
    # viol_vec (cos=1) and identical mclass token -> distance 0, should
    # merge into one bucket when k=1.
    a = _sig(
        1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "ADDSUB"),
        viol_vec={"WRONG_RESULT:add": 1.0},
    )
    b = _sig(
        2, "MISMATCH", ("MISMATCH", "BRANCH_NOT_TAKEN", "ADDSUB"),
        viol_vec={"WRONG_RESULT:add": 1.0},
    )
    labels = cluster([a, b], k=1)
    assert labels[1] == labels[2]
    assert set(labels.values()) == {"bucket_1"}


def test_stops_when_group_count_equals_k():
    # Three MISMATCH clusters, same mclass everywhere. A and B share an
    # identical viol_vec (distance 0); C's viol_vec is orthogonal to both
    # (distance 0.5). With k=2 only the closest pair (A, B) should merge.
    a = _sig(
        1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "ADDSUB"),
        viol_vec={"x": 1.0},
    )
    b = _sig(
        2, "MISMATCH", ("MISMATCH", "BRANCH_NOT_TAKEN", "ADDSUB"),
        viol_vec={"x": 1.0},
    )
    c = _sig(
        3, "MISMATCH", ("MISMATCH", "WRONG_ADDR", "ADDSUB"),
        viol_vec={"y": 1.0},
    )
    labels = cluster([a, b, c], k=2)
    assert len(set(labels.values())) == 2
    assert labels[1] == labels[2]
    assert labels[3] != labels[1]


def test_stops_when_min_distance_exceeds_threshold():
    # k is a hard target (Q&A A11): even two cases from unrelated families
    # with no shared evidence must merge into a single bucket when k=1 --
    # there is no longer a distance stop threshold.
    a = _sig(1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "ADDSUB"), viol_vec={"x": 1.0})
    b = _sig(2, "ASSERT", ("ASSERT", frozenset({"AssertA"})), assert_set=frozenset({"AssertA"}))
    labels = cluster([a, b], k=1)
    assert labels[1] == labels[2]
    assert set(labels.values()) == {"bucket_1"}


def test_initial_group_count_below_k_is_not_split():
    a = _sig(1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "ADDSUB"))
    b = _sig(2, "MISMATCH", ("MISMATCH", "WRONG_ADDR", "ADDSUB"))
    labels = cluster([a, b], k=5)
    assert len(set(labels.values())) == 2


def test_cross_family_viol_class_rule_links_mismatch_and_fatal():
    # bug_234-style link: a MISMATCH case and a FATAL case share the same
    # systematic (kind, mclass) violation class -> distance D_VIOL (R1),
    # so they merge even though hard_key/family differ completely.
    a = _sig(
        1, "MISMATCH", ("MISMATCH", "XRET_FALLTHROUGH", "XRET"),
        sys_viol_class=frozenset({("XRET_FALLTHROUGH", "XRET")}),
    )
    b = _sig(
        2, "FATAL", ("FATAL", "No dret detected", frozenset()),
        sys_viol_class=frozenset({("XRET_FALLTHROUGH", "XRET")}),
        fatal_tmpl="No dret detected",
    )
    labels = cluster([a, b], k=1)
    assert labels[1] == labels[2]


def test_deterministic_under_input_shuffle():
    sigs = [
        _sig(1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "ADDSUB"), viol_vec={"x": 1.0}),
        _sig(2, "MISMATCH", ("MISMATCH", "BRANCH_NOT_TAKEN", "ADDSUB"), viol_vec={"x": 1.0}),
        _sig(3, "MISMATCH", ("MISMATCH", "WRONG_ADDR", "SHIFT"), viol_vec={"y": 1.0}),
        _sig(4, "ASSERT", ("ASSERT", frozenset({"AssertA"})), assert_set=frozenset({"AssertA"})),
        _sig(5, "ASSERT", ("ASSERT", frozenset({"AssertA", "AssertB"})), assert_set=frozenset({"AssertA", "AssertB"})),
        _sig(6, "OTHER", ("OTHER", "some raw text")),
        _sig(7, "FATAL", ("FATAL", "tmplX", frozenset()), fatal_tmpl="tmplX"),
    ]
    baseline = cluster(sigs, k=3)

    rng = random.Random(20260928)
    shuffled = list(sigs)
    rng.shuffle(shuffled)
    result = cluster(shuffled, k=3)

    assert result == baseline


def test_label_numbering_follows_min_case_id():
    # bucket_1 must belong to the group containing the smallest case_id.
    a = _sig(10, "OTHER", ("OTHER", "aaa"))
    b = _sig(2, "OTHER", ("OTHER", "bbb"))
    c = _sig(30, "OTHER", ("OTHER", "ccc"))
    labels = cluster([a, b, c], k=3)
    assert labels[2] == "bucket_1"
    assert labels[10] == "bucket_2"
    assert labels[30] == "bucket_3"


def _many_signatures(n):
    sigs = []
    for i in range(n):
        fam = ["MISMATCH", "FATAL", "ASSERT", "OTHER"][i % 4]
        if fam == "MISMATCH":
            hk = ("MISMATCH", f"KIND{i}", f"MCLASS{i % 3}")
        elif fam == "FATAL":
            hk = ("FATAL", f"tmpl{i}", frozenset({f"assert{i}"}))
        elif fam == "ASSERT":
            hk = ("ASSERT", frozenset({f"assert{i}"}))
        else:
            hk = ("OTHER", f"raw{i}")
        sigs.append(_sig(i + 1, fam, hk, viol_vec={f"v{i % 5}": 1.0}))
    return sigs


@pytest.mark.parametrize("k", [1, 10])
def test_k_extremes_do_not_crash(k):
    sigs = _many_signatures(12)
    labels = cluster(sigs, k=k)
    assert set(labels.keys()) == {s.case_id for s in sigs}
    assert len(set(labels.values())) >= 1


# --- rfb.signature: sys_viol_class / sys_viol_keys thresholding -------------


def test_single_violation_does_not_form_sys_viol_class():
    below = _record("OTHER", violations={"TRAP_NOT_TAKEN:ebreak": MIN_SYS_VIOL - 1})
    sig_below = signature(below)
    assert sig_below.soft["sys_viol_class"] == frozenset()
    assert sig_below.soft["sys_viol_keys"] == frozenset()

    at_threshold = _record("OTHER", violations={"TRAP_NOT_TAKEN:ebreak": MIN_SYS_VIOL})
    sig_at = signature(at_threshold)
    assert sig_at.soft["sys_viol_class"] == frozenset(
        {("TRAP_NOT_TAKEN", isa_mclass("ebreak"))}
    )
    assert sig_at.soft["sys_viol_keys"] == frozenset({("TRAP_NOT_TAKEN", "ebreak")})


# --- evidence-ladder ordering (R1-R7) ---------------------------------------


def test_assert_overlap_merges_before_same_template_no_assertion_fatal():
    # A: ASSERT case with {AssertA, AssertB}.
    # B: FATAL case sharing the exact same assertion set (R2, full overlap).
    # C: FATAL case with the same fatal template as B but no assertion
    #    evidence at all (falls through to R7, one-sided-assertion tier).
    a = _sig(1, "ASSERT", ("ASSERT", frozenset({"AssertA", "AssertB"})),
             assert_set=frozenset({"AssertA", "AssertB"}))
    b = _sig(2, "FATAL", ("FATAL", "tmplX", frozenset({"AssertA", "AssertB"})),
             assert_set=frozenset({"AssertA", "AssertB"}), fatal_tmpl="tmplX")
    c = _sig(3, "FATAL", ("FATAL", "tmplX", frozenset()), fatal_tmpl="tmplX")

    labels = cluster([a, b, c], k=2)
    assert labels[1] == labels[2]
    assert labels[3] != labels[1]


def test_ebreak_trap_link_merges_mismatch_and_debug_mode_fatal():
    # MISMATCH case with a systematic TRAP_NOT_TAKEN:ebreak violation
    # (>= MIN_SYS_VIOL occurrences) links, via TRAP_LINK, to a FATAL case
    # whose template mentions IN_DEBUG_MODE -- R3.
    mismatch = _sig(
        1, "MISMATCH", ("MISMATCH", "NONE", "ebreak"),
        sys_viol_keys=frozenset({("TRAP_NOT_TAKEN", "ebreak")}),
    )
    fatal = _sig(
        2, "FATAL", ("FATAL", "Assertion IN_DEBUG_MODE failed", frozenset()),
        fatal_tmpl="Assertion IN_DEBUG_MODE failed",
    )
    labels = cluster([mismatch, fatal], k=1)
    assert labels[1] == labels[2]


def test_distance_order_violation_then_assertion_then_none():
    # One-sided systematic violation (R5, cos below threshold -> D_CROSS)
    # must be a larger distance than one-sided assertion evidence (R7,
    # D_UVM_ONESIDED_ASSERT), which in turn must be larger than neither
    # side having any evidence at all (R7, D_UVM_NONE).
    viol_side = _Atom(
        ("MISMATCH", "K", "M"),
        [_sig(1, "MISMATCH", ("MISMATCH", "K", "M"),
              sys_viol_class=frozenset({("K", "M")}))],
    )
    no_evidence_a = _Atom(
        ("FATAL", "t1", frozenset()),
        [_sig(2, "FATAL", ("FATAL", "t1", frozenset()))],
    )
    d_violation = _atom_distance(viol_side, no_evidence_a)

    assert_side = _Atom(
        ("ASSERT", frozenset({"A1"})),
        [_sig(3, "ASSERT", ("ASSERT", frozenset({"A1"})), assert_set=frozenset({"A1"}))],
    )
    no_evidence_b = _Atom(
        ("FATAL", "t2", frozenset()),
        [_sig(4, "FATAL", ("FATAL", "t2", frozenset()))],
    )
    d_assertion = _atom_distance(assert_side, no_evidence_b)

    none_a = _Atom(
        ("ASSERT", frozenset()),
        [_sig(5, "ASSERT", ("ASSERT", frozenset()))],
    )
    none_b = _Atom(
        ("FATAL", "t3", frozenset()),
        [_sig(6, "FATAL", ("FATAL", "t3", frozenset()))],
    )
    d_none = _atom_distance(none_a, none_b)

    assert d_violation > d_assertion > d_none


# --- UPGMA / Lance-Williams weighting ---------------------------------------


def test_upgma_lance_williams_weighting(monkeypatch):
    import rfb.cluster as cluster_mod

    # Four atoms, A much larger (9 members) than B/C/D (1 member each).
    # Distances are rigged so that a correctly *weighted* Lance-Williams
    # update after merging A+B favors merging {A,B} with C next, while a
    # naive (unweighted) average would instead favor merging C with D.
    # This distinguishes the two update rules.
    dmat = {
        frozenset({"A", "B"}): 0.05,
        frozenset({"A", "C"}): 0.10,
        frozenset({"A", "D"}): 0.80,
        frozenset({"B", "C"}): 0.90,
        frozenset({"B", "D"}): 0.80,
        frozenset({"C", "D"}): 0.30,
    }

    def fake_distance(a, b):
        return dmat[frozenset({a.hard_key[1], b.hard_key[1]})]

    monkeypatch.setattr(cluster_mod, "_atom_distance", fake_distance)

    sizes = {"A": 9, "B": 1, "C": 1, "D": 1}
    sigs = []
    case_id = 1
    for label, n in sizes.items():
        for _ in range(n):
            sigs.append(_sig(case_id, "OTHER", ("OTHER", label)))
            case_id += 1

    labels = cluster_mod.cluster(sigs, k=2)

    a_ids = [s.case_id for s in sigs if s.hard_key[1] == "A"]
    b_ids = [s.case_id for s in sigs if s.hard_key[1] == "B"]
    c_ids = [s.case_id for s in sigs if s.hard_key[1] == "C"]
    d_ids = [s.case_id for s in sigs if s.hard_key[1] == "D"]

    # weighted expectation: {A, B, C} together, {D} alone.
    assert labels[a_ids[0]] == labels[b_ids[0]] == labels[c_ids[0]]
    assert labels[d_ids[0]] != labels[a_ids[0]]


# --- canon() determinism -----------------------------------------------------


def test_canon_stable_across_frozenset_insertion_order():
    a = frozenset(["x", "y", "z"])
    b = frozenset(["z", "y", "x"])
    assert canon(a) == canon(b)

    hk1 = ("FATAL", "tmpl", frozenset(["b", "a", "c"]))
    hk2 = ("FATAL", "tmpl", frozenset(["c", "b", "a"]))
    assert canon(hk1) == canon(hk2)


# --- cross-process determinism (PYTHONHASHSEED, case order) ----------------


def test_cross_process_output_stable_across_hash_seed_and_order(tmp_path):
    # canon()'s frozenset ordering must not depend on PYTHONHASHSEED, and
    # the final labeling must not depend on input case order. Run the real
    # CLI as a subprocess (so PYTHONHASHSEED actually takes effect) against
    # benchmark_set_2, for both orderings and two different hash seeds.
    (tmp_path / "shuffled").mkdir()
    shuffled_input = str(tmp_path / "shuffled" / "input.csv")
    _write_shuffled_input(SET2_INPUT, shuffled_input, seed=20260928)

    program = os.path.join(REPO_ROOT, "bin", "regr_fail_bucketing")
    python = os.path.join(REPO_ROOT, ".venv", "bin", "python")

    outputs = {}
    for order_name, input_csv in (("orig", SET2_INPUT), ("shuffled", shuffled_input)):
        for seed in ("1", "2"):
            output_csv = str(tmp_path / f"output_{order_name}_{seed}.csv")
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [python, program, "--input", input_csv, "--output", output_csv, "--k", "4"],
                cwd=os.path.dirname(input_csv),
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            with open(output_csv, newline="") as f:
                outputs[(order_name, seed)] = f.read()

    baseline = outputs[("orig", "1")]
    for key, content in outputs.items():
        assert content == baseline, f"{key} differed from baseline"


# --- discriminative unit tests (mutation-killing, per reviewer findings) ----


def test_r1_shared_viol_class_changes_merge_pairing():
    # A and B share a systematic violation class (R1, d=0.10) and must merge
    # ahead of B's weaker R2 assertion-overlap link to C (d=0.25) and A-C's
    # R5 fallback (d=0.90, no shared vcls, empty vvecs). Disabling R1 makes
    # A-B fall through to R5/R6 (d=0.90, tied with A-C), which changes the
    # single merge (k=2) to B+C instead of A+B.
    a = _sig(1, "MISMATCH", ("MISMATCH", "K", "M"), sys_viol_class=frozenset({("K", "M")}))
    b = _sig(
        2, "FATAL", ("FATAL", "T1", frozenset({"AssertX"})),
        sys_viol_class=frozenset({("K", "M")}),
        assert_set=frozenset({"AssertX"}), fatal_tmpl="T1",
    )
    c = _sig(
        3, "FATAL", ("FATAL", "T2", frozenset({"AssertX"})),
        assert_set=frozenset({"AssertX"}), fatal_tmpl="T2",
    )
    labels = cluster([a, b, c], k=2)
    assert labels[1] == labels[2]
    assert labels[3] != labels[1]


def test_r3_trap_link_decides_mismatch_fatal_pairing_over_other_candidate():
    # mismatch has a systematic TRAP_NOT_TAKEN:ebreak violation (>=
    # MIN_SYS_VIOL occurrences, represented directly via sys_viol_keys). It
    # links (R3, d=0.40) to fatal_debug's IN_DEBUG_MODE template, which must
    # win over both mismatch's R6 fallback to `other` (d=0.90) and
    # fatal_debug's own R7 link to `other` (d=0.55 + TAIL_W = 0.60). Disabling R3 makes
    # mismatch-fatal_debug fall to R6 (d=0.90), so the single merge (k=2)
    # becomes fatal_debug+other instead of mismatch+fatal_debug.
    mismatch = _sig(
        1, "MISMATCH", ("MISMATCH", "NONE", "ebreak"),
        sys_viol_keys=frozenset({("TRAP_NOT_TAKEN", "ebreak")}),
    )
    fatal_debug = _sig(
        2, "FATAL", ("FATAL", "Assertion IN_DEBUG_MODE failed", frozenset()),
        fatal_tmpl="Assertion IN_DEBUG_MODE failed",
    )
    other = _sig(3, "FATAL", ("FATAL", "some other fatal", frozenset()),
                 fatal_tmpl="some other fatal")
    labels = cluster([mismatch, fatal_debug, other], k=2)
    assert labels[1] == labels[2]
    assert labels[3] != labels[1]


def test_r4_mclass_token_match_beats_equal_viol_vec_cosine():
    # All three atoms share an identical viol_vec (cos == 1 for every pair,
    # so the viol_vec half of R4's distance is 0 everywhere). A and B differ
    # in mclass token (hard_key[2]); C shares A's mclass token. With the
    # viol_vec term neutralized, only the mtok indicator can distinguish the
    # pairs -- A+C (same mtok, d=0.0) must merge before A+B / B+C (different
    # mtok, d=0.5).
    a = _sig(1, "MISMATCH", ("MISMATCH", "K1", "MC1"), viol_vec={"x": 1.0})
    b = _sig(2, "MISMATCH", ("MISMATCH", "K2", "MC2"), viol_vec={"x": 1.0})
    c = _sig(3, "MISMATCH", ("MISMATCH", "K3", "MC1"), viol_vec={"x": 1.0})
    labels = cluster([a, b, c], k=2)
    assert labels[1] == labels[3]
    assert labels[2] != labels[1]


def test_r2_partial_overlap_and_both_uvm_no_vcls_branch():
    # (a) partial overlap (ovl == 0.5, the boundary of 0 < ovl < 1) between a
    # MISMATCH and a FATAL atom -- triggers R2 purely via the overlap
    # condition (both_uvm_no_vcls is False since a is MISMATCH). Hand
    # computed: d = ASSERT_BASE + ASSERT_SPAN*(1-0.5) + TAIL_W*(1-cos(tail));
    # tails are both empty so cos(tail) == 0 (via _cos's "not a or not b"
    # branch) -> d = 0.20 + 0.30 + 0.05 = 0.55.
    a = _Atom(("MISMATCH", "K", "M"), [
        _sig(1, "MISMATCH", ("MISMATCH", "K", "M"), assert_set=frozenset({"a1", "a2"})),
    ])
    b = _Atom(("FATAL", "T", frozenset()), [
        _sig(2, "FATAL", ("FATAL", "T", frozenset()), assert_set=frozenset({"a2", "a3", "a4"})),
    ])
    assert _atom_distance(a, b) == pytest.approx(0.55)

    # (b) both sides UVM (FATAL), no vcls, overlap 1/3 (< ASSERT_XFAM_MIN_OVL
    # = 0.5) -- must still take R2 via the both_uvm_no_vcls branch. Hand
    # computed: d = 0.20 + 0.60*(1 - 1/3) + 0.05 = 0.20 + 0.40 + 0.05 = 0.65.
    c = _Atom(("FATAL", "T1", frozenset()), [
        _sig(3, "FATAL", ("FATAL", "T1", frozenset()), assert_set=frozenset({"c1", "c2", "c3"})),
    ])
    d = _Atom(("FATAL", "T2", frozenset()), [
        _sig(4, "FATAL", ("FATAL", "T2", frozenset()),
             assert_set=frozenset({"c1", "x2", "x3", "x4", "x5"})),
    ])
    assert _atom_distance(c, d) == pytest.approx(0.65)

    # Contrast: same overlap (1/3) but one side is MISMATCH -- not both UVM,
    # so both_uvm_no_vcls is False and R2 does not trigger; falls through to
    # R6 (d = D_CROSS = 0.90), which must differ from the R2 result above.
    e = _Atom(("MISMATCH", "K2", "M2"), [
        _sig(5, "MISMATCH", ("MISMATCH", "K2", "M2"), assert_set=frozenset({"c1", "c2", "c3"})),
    ])
    f = _Atom(("FATAL", "T3", frozenset()), [
        _sig(6, "FATAL", ("FATAL", "T3", frozenset()),
             assert_set=frozenset({"c1", "x2", "x3", "x4", "x5"})),
    ])
    assert _atom_distance(e, f) == pytest.approx(0.90)


def test_r5_violvec_cosine_at_or_above_threshold():
    # a has vcls (no overlap with b's empty vcls, so R1 does not trigger);
    # cos(a.vvec, b.vvec) = 1 / sqrt(2) ~= 0.707 >= VIOLVEC_COS_MIN (0.5) ->
    # R5's D_VIOLVEC branch.
    a = _Atom(("MISMATCH", "K", "M"), [
        _sig(1, "MISMATCH", ("MISMATCH", "K", "M"),
             sys_viol_class=frozenset({("K", "M")}), viol_vec={"a": 1.0, "b": 1.0}),
    ])
    b = _Atom(("FATAL", "T", frozenset()), [
        _sig(2, "FATAL", ("FATAL", "T", frozenset()), viol_vec={"a": 1.0}),
    ])
    from rfb.cluster import D_VIOLVEC
    assert _atom_distance(a, b) == pytest.approx(D_VIOLVEC)


def test_multi_member_atom_average_is_order_independent():
    # >= 10 members sharing one hard_key, each with a fractional viol_vec.
    # The averaged vector (and downstream cluster() output) must not depend
    # on the order the members are handed to _Atom / cluster().
    hk = ("MISMATCH", "K", "M")
    members = [
        _sig(i, "MISMATCH", hk, viol_vec={"a": i / 7.0, "b": (17 - i) / 3.0})
        for i in range(1, 11)
    ]
    other = _sig(100, "OTHER", ("OTHER", "unrelated"))

    atom_fwd = _Atom(hk, list(members))
    atom_rev = _Atom(hk, list(reversed(members)))
    assert atom_fwd.vvec == atom_rev.vvec

    rng = random.Random(9)
    shuffled_members = list(members)
    rng.shuffle(shuffled_members)
    atom_shuf = _Atom(hk, shuffled_members)
    assert atom_fwd.vvec == atom_shuf.vvec

    baseline = cluster(members + [other], k=2)
    shuffled_full = list(members) + [other]
    rng.shuffle(shuffled_full)
    result = cluster(shuffled_full, k=2)
    assert result == baseline


# --- UPGMA vectorization: match the pure-python reference exactly ----------


def _reference_upgma_merge(dist, sizes, k):
    """Pure-python nested-loop UPGMA merge (the pre-vectorization
    implementation), used as an independent oracle for _upgma_merge."""
    n = len(sizes)
    sizes = dict(sizes)
    dist = [row[:] for row in dist]
    members = {i: [i] for i in range(n)}
    alive = list(range(n))
    target = max(k, 1)
    while len(alive) > target:
        best = None
        for x in range(len(alive)):
            for y in range(x + 1, len(alive)):
                i, j = alive[x], alive[y]
                key = (round(dist[i][j], 9), i, j)
                if best is None or key < best:
                    best = key
        _, i, j = best
        ni, nj = sizes[i], sizes[j]
        for m in alive:
            if m in (i, j):
                continue
            dist[i][m] = dist[m][i] = (ni * dist[i][m] + nj * dist[j][m]) / (ni + nj)
        sizes[i] = ni + nj
        members[i] = members[i] + members[j]
        alive.remove(j)
    return [sorted(members[i]) for i in alive]


def test_upgma_merge_matches_pure_python_reference_with_ties(monkeypatch):
    import rfb.cluster as cluster_mod

    n = 60
    rng = random.Random(20260929)
    hk_labels = [f"atom{i}" for i in range(n)]
    dmat: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            # round to 2 decimals to force frequent exact ties -- the
            # scenario where the (round(d, 9), i, j) lexicographic
            # tie-break must match the reference bit-for-bit.
            dmat[frozenset({hk_labels[i], hk_labels[j]})] = round(rng.random(), 2)

    def fake_distance(a, b):
        return dmat[frozenset({a.hard_key[1], b.hard_key[1]})]

    monkeypatch.setattr(cluster_mod, "_atom_distance", fake_distance)

    sigs = [_sig(i + 1, "OTHER", ("OTHER", hk_labels[i])) for i in range(n)]
    groups: dict = {}
    for sig in sigs:
        groups.setdefault(sig.hard_key, []).append(sig)
    hard_keys_sorted = sorted(groups.keys(), key=canon)
    atoms = [_Atom(hk, groups[hk]) for hk in hard_keys_sorted]

    dist_full = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = cluster_mod._atom_distance(atoms[i], atoms[j])
            dist_full[i][j] = dist_full[j][i] = d
    sizes = {i: atoms[i].n for i in range(n)}

    for k in (1, 5, 20, 59):
        expected = _reference_upgma_merge(dist_full, sizes, k)
        actual = cluster_mod._upgma_merge(atoms, k)
        assert sorted(sorted(g) for g in actual) == sorted(sorted(g) for g in expected)


def test_upgma_merged_cluster_keeps_smaller_index_for_tie_break(monkeypatch):
    # After merging atoms 0 and 2 the merged cluster must keep index 0.
    # The next step then has a tie at d=0.5 between (merged, 3) and (1, 3):
    # keeping index 0 makes (0, 3) win the (round(d, 9), i, j) tie-break,
    # whereas keeping index 2 would make (1, 3) win instead.
    import rfb.cluster as cluster_mod

    labels = ["a0", "a1", "a2", "a3"]
    dmat = {
        frozenset({"a0", "a2"}): 0.1,
        frozenset({"a0", "a3"}): 0.5,
        frozenset({"a2", "a3"}): 0.5,
        frozenset({"a1", "a3"}): 0.5,
        frozenset({"a0", "a1"}): 0.9,
        frozenset({"a1", "a2"}): 0.9,
    }
    monkeypatch.setattr(
        cluster_mod,
        "_atom_distance",
        lambda a, b: dmat[frozenset({a.hard_key[1], b.hard_key[1]})],
    )
    sigs = [_sig(i + 1, "OTHER", ("OTHER", labels[i])) for i in range(4)]
    atoms = [_Atom(s.hard_key, [s]) for s in sorted(sigs, key=lambda s: canon(s.hard_key))]
    assert [a.hard_key[1] for a in atoms] == labels

    groups = cluster_mod._upgma_merge(atoms, 2)
    assert sorted(sorted(g) for g in groups) == [[0, 2, 3], [1]]

    dist_full = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            dist_full[i][j] = dist_full[j][i] = dmat[frozenset({labels[i], labels[j]})]
    expected = _reference_upgma_merge(dist_full, {i: 1 for i in range(4)}, 2)
    assert sorted(sorted(g) for g in expected) == [[0, 2, 3], [1]]


# --- canon() cross-process determinism under mutation-sensitive setup ------


_CANON_XPROC_SCRIPT = '''
import itertools
from rfb.signature import Signature
from rfb.cluster import cluster

pool = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
combos = list(itertools.combinations(pool, 5))

sigs = []
for i in range(6):
    hk = ("ASSERT", frozenset(combos[i]))
    soft = dict(
        viol_vec={}, viol_class=None, sys_viol_class=frozenset(),
        sys_viol_keys=frozenset(), tail_ngrams={}, implicit=0,
        assert_set=frozenset(), fatal_tmpl=None, sig_last=None,
    )
    sigs.append(Signature(case_id=i + 1, family="ASSERT", hard_key=hk, soft=soft))

labels = cluster(sigs, k=2)
print(",".join(f"{cid}:{labels[cid]}" for cid in sorted(labels)))
'''


def test_canon_cross_process_stable_across_hash_seed():
    # All 6 atoms are ASSERT family with empty soft evidence -> every
    # pairwise distance is identical (R7, D_UVM_NONE), so which pair merges
    # first (k=2, several merges) is decided purely by initial atom index,
    # i.e. by canon(hard_key) sort order. Each hard_key's frozenset has 5
    # elements, so a hash-seed-dependent sort (e.g. key=repr) would reorder
    # atoms differently per PYTHONHASHSEED and change the output.
    python = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    outputs = {}
    for seed in range(8):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        result = subprocess.run(
            [python, "-c", _CANON_XPROC_SCRIPT],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        outputs[seed] = result.stdout.strip()

    baseline = outputs[0]
    for seed, out in outputs.items():
        assert out == baseline, f"seed {seed} differed from baseline"


# --- 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: R1b -------------


def test_r1b_thin_evidence_sra_and_c_srai_merge_before_fatal_pair():
    # sra and c.srai each have ONLY deviation-level evidence (wr_sys empty:
    # never crossed MIN_SYS_VIOL on their own), so R1b's "both footprints
    # established and disjoint" exception can't fire -- they link at
    # D_VIOLVEC=0.30. Two unrelated FATAL atoms sit at ~0.60 apart (R7, no
    # asserts, no shared template). With k=3 across 4 atoms, exactly one
    # merge happens, and it must be sra+c.srai (0.30 < 0.60 < the 0.90
    # cross-family distances), not either FATAL with a MISMATCH atom.
    sra = _sig(
        1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "sra"),
        sys_viol_class=frozenset({("WRONG_RESULT", "sra")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "sra")}),
    )
    c_srai = _sig(
        2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.srai"),
        sys_viol_class=frozenset({("WRONG_RESULT", "c.srai")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "c.srai")}),
    )
    fatal1 = _sig(3, "FATAL", ("FATAL", "T1", frozenset()), fatal_tmpl="T1")
    fatal2 = _sig(4, "FATAL", ("FATAL", "T2", frozenset()), fatal_tmpl="T2")

    labels = cluster([sra, c_srai, fatal1, fatal2], k=3)
    assert labels[1] == labels[2]
    assert labels[3] != labels[4]
    assert labels[3] != labels[1]
    assert labels[4] != labels[1]


def test_r1b_strong_footprints_keep_c_addi_and_c_addi4spn_separate():
    # c.addi and c.addi4spn both map to alu_op "add", but each has already
    # crossed MIN_SYS_VIOL on its own (wr_sys populated) and the footprints
    # are disjoint (c.addi != c.addi4spn), so R1b's exception fires and
    # they do NOT link at D_VIOLVEC; they fall through to R4 (both
    # MISMATCH, no shared viol_vec/mtok) at d=1.0. The two FATAL atoms
    # (same as above, d~=0.60) merge first instead, leaving c.addi and
    # c.addi4spn each alone.
    c_addi = _sig(
        1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi"),
        sys_viol_class=frozenset({("WRONG_RESULT", "c.addi")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "c.addi")}),
        wr_sys=frozenset({"c.addi"}),
    )
    c_addi4spn = _sig(
        2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
        sys_viol_class=frozenset({("WRONG_RESULT", "c.addi4spn")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "c.addi4spn")}),
        wr_sys=frozenset({"c.addi4spn"}),
    )
    fatal1 = _sig(3, "FATAL", ("FATAL", "T1", frozenset()), fatal_tmpl="T1")
    fatal2 = _sig(4, "FATAL", ("FATAL", "T2", frozenset()), fatal_tmpl="T2")

    labels = cluster([c_addi, c_addi4spn, fatal1, fatal2], k=3)
    assert labels[1] != labels[2]
    assert labels[3] == labels[4]


def test_r1b_one_thin_one_strong_still_links():
    # 2026-10-02 R1b 評估（PLAN 4c「R1b 過度合併（評估）」）: pins the
    # "one side established, the other thin" branch: CUR (and BOTHTHIN)
    # link here, HYBRID/MULTI/OFF don't. c.addi has crossed MIN_SYS_VIOL (wr_sys populated)
    # but c.addi4spn has only deviation-level evidence (wr_sys empty), so
    # R1b's "both established and disjoint" exception can't fire -- they
    # link at D_VIOLVEC=0.30 and merge before the two FATAL atoms (~0.60).
    c_addi = _sig(
        1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi"),
        sys_viol_class=frozenset({("WRONG_RESULT", "c.addi")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "c.addi")}),
        wr_sys=frozenset({"c.addi"}),
    )
    c_addi4spn = _sig(
        2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
        sys_viol_class=frozenset({("WRONG_RESULT", "c.addi4spn")}),
        sys_viol_keys=frozenset({("WRONG_RESULT", "c.addi4spn")}),
    )
    a = _Atom(c_addi.hard_key, [c_addi])
    b = _Atom(c_addi4spn.hard_key, [c_addi4spn])
    assert _atom_distance(a, b) == D_VIOLVEC
    assert _atom_distance(b, a) == D_VIOLVEC

    fatal1 = _sig(3, "FATAL", ("FATAL", "T1", frozenset()), fatal_tmpl="T1")
    fatal2 = _sig(4, "FATAL", ("FATAL", "T2", frozenset()), fatal_tmpl="T2")
    labels = cluster([c_addi, c_addi4spn, fatal1, fatal2], k=3)
    assert labels[1] == labels[2]
    assert labels[3] != labels[4]


def test_r1b_footprint_only_counts_same_op_mnemonics():
    # 2026-10-02 R1b 評估第二輪冷讀 S1: the "both established and
    # disjoint" exception must look only at wr_sys mnemonics of the shared
    # op. a is thin on op "add" (c.addi from the deviation only) but has an
    # established footprint on an unrelated op (mul); b is established on
    # "add" (c.addi4spn). Filtered per op, a's add footprint is empty, so
    # they link. Counting mul would make the footprints look established
    # and disjoint. Both call orders are asserted so that dropping the
    # filter on either side alone is caught.
    a = _Atom(
        ("MISMATCH", "WRONG_RESULT", "c.addi"),
        [_sig(
            1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi"),
            sys_viol_class=frozenset(
                {("WRONG_RESULT", "c.addi"), ("WRONG_RESULT", "mul")}
            ),
            sys_viol_keys=frozenset(
                {("WRONG_RESULT", "c.addi"), ("WRONG_RESULT", "mul")}
            ),
            wr_sys=frozenset({"mul"}),
        )],
    )
    b = _Atom(
        ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
        [_sig(
            2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
            sys_viol_class=frozenset({("WRONG_RESULT", "c.addi4spn")}),
            sys_viol_keys=frozenset({("WRONG_RESULT", "c.addi4spn")}),
            wr_sys=frozenset({"c.addi4spn"}),
        )],
    )
    assert _atom_distance(a, b) == D_VIOLVEC
    assert _atom_distance(b, a) == D_VIOLVEC


def test_r1b_distance_is_d_violvec():
    a = _Atom(
        ("MISMATCH", "WRONG_RESULT", "sra"),
        [_sig(
            1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "sra"),
            sys_viol_class=frozenset({("WRONG_RESULT", "sra")}),
            sys_viol_keys=frozenset({("WRONG_RESULT", "sra")}),
        )],
    )
    b = _Atom(
        ("MISMATCH", "WRONG_RESULT", "c.srai"),
        [_sig(
            2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.srai"),
            sys_viol_class=frozenset({("WRONG_RESULT", "c.srai")}),
            sys_viol_keys=frozenset({("WRONG_RESULT", "c.srai")}),
        )],
    )
    assert _atom_distance(a, b) == D_VIOLVEC


def test_r1b_disjoint_op_does_not_veto_other_shared_op(monkeypatch):
    # 2026-10-04 PLAN 4c「R1b 過度合併（評估）」S6: the "both established
    # and disjoint" exception skips only that op (`continue`); another
    # shared op where one side is thin must still link. Turning the
    # `continue` into `return False` / `break` only shows when the disjoint
    # op is iterated first, which with real mnemonic strings depends on the
    # hash seed. Mapping the ops to small ints pins the iteration order of
    # `ops_a & ops_b` to 0 then 1, so op 0 (disjoint) is always seen first.
    from rfb import isa

    monkeypatch.setattr(
        isa, "ALU_OP", {"sra": 0, "c.srai": 0, "c.addi": 1, "c.addi4spn": 1}
    )

    # op 0: sra vs c.srai, both established, disjoint -> exception fires.
    # op 1: c.addi (established) vs c.addi4spn (thin) -> links.
    a_keys = frozenset({("WRONG_RESULT", "sra"), ("WRONG_RESULT", "c.addi")})
    b_keys = frozenset(
        {("WRONG_RESULT", "c.srai"), ("WRONG_RESULT", "c.addi4spn")}
    )
    a = _Atom(
        ("MISMATCH", "WRONG_RESULT", "sra"),
        [_sig(
            1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "sra"),
            sys_viol_class=a_keys, sys_viol_keys=a_keys,
            wr_sys=frozenset({"sra", "c.addi"}),
        )],
    )
    # b's thin c.addi4spn can only come from the deviation, so it is also
    # b's hard_key anchor (as signature() would build it).
    b = _Atom(
        ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
        [_sig(
            2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "c.addi4spn"),
            sys_viol_class=b_keys, sys_viol_keys=b_keys,
            wr_sys=frozenset({"c.srai"}),
        )],
    )
    # Checked on the real atoms so that the patch silently not taking
    # effect (ops back to hash-seeded strings) fails here, too.
    shared = (
        {isa.alu_op(m) for _, m in a.vkeys} & {isa.alu_op(m) for _, m in b.vkeys}
    )
    assert list(shared) == [0, 1]
    assert _atom_distance(a, b) == D_VIOLVEC
    assert _atom_distance(b, a) == D_VIOLVEC


def test_r1b_ignores_non_alu_wrong_result():
    # 2026-10-04 冷讀 F1: WRONG_RESULT anchored on a non-ALU mnemonic (here
    # LOAD vs CSR, both reachable via pcdelta) has alu_op None. Without the
    # `discard(None)`, None becomes a shared "op" whose wr_sys footprints
    # are both empty, and R1b would link these unrelated atoms.
    a = _Atom(
        ("MISMATCH", "WRONG_RESULT", "LOAD"),
        [_sig(
            1, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "LOAD"),
            sys_viol_class=frozenset({("WRONG_RESULT", "LOAD")}),
            sys_viol_keys=frozenset({("WRONG_RESULT", "lw")}),
        )],
    )
    b = _Atom(
        ("MISMATCH", "WRONG_RESULT", "CSR"),
        [_sig(
            2, "MISMATCH", ("MISMATCH", "WRONG_RESULT", "CSR"),
            sys_viol_class=frozenset({("WRONG_RESULT", "CSR")}),
            sys_viol_keys=frozenset({("WRONG_RESULT", "csrrw")}),
        )],
    )
    assert _atom_distance(a, b) != D_VIOLVEC
    assert _atom_distance(b, a) != D_VIOLVEC


# --- hknegu R1x-: disjoint CSR X-origin signatures (PLAN.md 4c 公開合併 0.96 根因) ---

from rfb.cluster import D_CROSS, D_VIOL, D_TRAP_LINK  # noqa: E402

_XA = frozenset({"mip:.....X.."})
_XB = frozenset({"mip:XXXX...."})


def _xatom(case_id, family, xsig, **soft):
    s = _sig(case_id, family, (family, case_id), xsig=xsig, **soft)
    return _Atom(s.hard_key, [s])


def test_r1x_disjoint_xsig_uvm_pair_is_d_cross():
    a = _xatom(1, "FATAL", _XA)
    b = _xatom(2, "ASSERT", _XB)
    assert _atom_distance(a, b) == D_CROSS
    assert _atom_distance(b, a) == D_CROSS


def test_r1x_disjoint_xsig_beats_shared_asserts_r2():
    asserts = frozenset({"IN_DEBUG_MODE"})
    a = _xatom(1, "FATAL", _XA, assert_set=asserts)
    b = _xatom(2, "FATAL", _XB, assert_set=asserts)
    assert a.asserts and b.asserts
    assert _atom_distance(a, b) == D_CROSS


def test_r1x_shared_xsig_does_not_trigger():
    a = _xatom(1, "FATAL", _XA | _XB)
    b = _xatom(2, "FATAL", _XB)
    assert _atom_distance(a, b) != D_CROSS


def test_r1x_one_sided_xsig_does_not_trigger():
    a = _xatom(1, "FATAL", _XA)
    b = _xatom(2, "FATAL", frozenset())
    assert _atom_distance(a, b) != D_CROSS


def test_r1x_mismatch_vs_uvm_keeps_r3_trap_link():
    mismatch = _xatom(
        1, "MISMATCH", _XA,
        sys_viol_keys=frozenset({("TRAP_NOT_TAKEN", "ebreak")}),
    )
    fatal = _xatom(
        2, "FATAL", _XB,
        fatal_tmpl="Assertion IN_DEBUG_MODE failed",
    )
    assert _atom_distance(mismatch, fatal) == D_TRAP_LINK
    assert _atom_distance(fatal, mismatch) == D_TRAP_LINK


def test_r1_shared_vcls_wins_over_r1x_disjoint_xsig():
    vcls = frozenset({("WRONG_RESULT", "add")})
    a = _xatom(1, "FATAL", _XA, sys_viol_class=vcls)
    b = _xatom(2, "FATAL", _XB, sys_viol_class=vcls)
    assert _atom_distance(a, b) == D_VIOL


# 注意：這條釘的是合成狀態。真實 pipeline 裡 UVM 的 hard_key 已含 xsig，
# 同一個 atom 的成員 xsig 必然相同，不會出現成員 xsig 不同的情況。
# 仍保留，是為了釘住 `_Atom.xsig` 取成員聯集的實作（改成取第一個成員會在這裡失敗）。
def test_atom_xsig_is_union_of_members():
    s1 = _sig(1, "FATAL", ("FATAL", "k"), xsig=_XA)
    s2 = _sig(2, "FATAL", ("FATAL", "k"), xsig=_XB)
    atom = _Atom(s1.hard_key, [s1, s2])
    assert atom.xsig == _XA | _XB
    atom_rev = _Atom(s1.hard_key, [s2, s1])
    assert atom_rev.xsig == _XA | _XB
