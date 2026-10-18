"""Regression test for the beq/bne handling in rfb/isa.py (BR_BEQ / BR_BNE
mclass split plus the is_equal polarity grouping in viol_group), which no
public case exercises.

Benchmarks are synthesised by tools/inject_branch.py from public set-2 cases
(5 cases per bug, BG_CASES = 12 untouched set-2 cases as background), with
the same defaults as the CLI and tools/branch_eq_eval.py.  Five bugs:
  BEQ_ONLY   : beq/c.beqz not taken           (is_equal wrongly false, beq only)
  BNE_ONLY   : bne/c.bnez not taken           (is_equal wrongly true, bne only)
  EQ_BOTH    : shared comparator, is_equal wrongly false: beq NT + bne WT
  EQ_BOTH_FP : shared comparator, is_equal wrongly true : beq WT + bne NT
  BNE_WT_ONLY: bne/c.bnez wrongly taken       (bench_all only; no BA scenario here, see eval scenario G)
Scenarios (the same cut as tools/branch_eq_eval.py; bg = background):
  A : BEQ_ONLY + BNE_ONLY + bg
  B : EQ_BOTH + bg
  C : BEQ_ONLY + BNE_ONLY + EQ_BOTH + bg
  D : thin-evidence EQ_BOTH + bg, sources restricted to set2/13, set1/3,
      set2/12 (a separate benchmark; E2 below is separate too)
  E : EQ_BOTH_FP + bg
  F : BEQ_ONLY + BNE_ONLY + EQ_BOTH + EQ_BOTH_FP + bg
  E2: thin-evidence EQ_BOTH_FP + bg, sources restricted to set2/11 and set1/1
      (a separate benchmark)

Measured 2026-09-29 (polarity = repo / split = no polarity branch / merged =
old BR_EQ):
  A 1.000000 / 1.000000 / 0.905936    B 1.000000 / 1.000000 / 1.000000
  C 0.935737 / 0.935737 / 0.831022    D 1.000000 / 0.883333 / 1.000000
  E 1.000000 / 1.000000 / 1.000000    F 0.905232 / 0.905232 / 0.773389
Scenario G (BEQ_ONLY + BNE_WT_ONLY + bg, the polarity worst case) is only
recorded by tools/branch_eq_eval.py, not asserted: polarity 0.905936 / split
1.000000 / merged 0.905936.
Thresholds are the polarity values; do not lower them to make a change pass.
C and F stay below 1 because the injected bugs have overlapping symptoms at
the anchor: BEQ_ONLY and EQ_BOTH both start with BRANCH_NOT_TAKEN on
beq/c.beqz, and BNE_ONLY and EQ_BOTH_FP both start with BRANCH_NOT_TAKEN on
bne/c.bnez, so they cannot be told apart.  This is an intrinsic limit, not a
defect.
Scenario D separates polarity grouping from plain mclass splitting: with thin
EQ_BOTH evidence (few hits per case) the split configuration falls to 0.8833
while polarity stays at 1.0.  Removing the whole polarity branch of viol_group
therefore fails D, but that only holds for the whole branch: the EQ+ half
(the NOT_TAKEN->BR_BNE and WRONGLY_TAKEN->BR_BEQ cells of _EQ_POLARITY) is
guarded in the regression scenarios only by E2 (thin EQ_BOTH_FP evidence, sources set2/11 and
set1/1, k=5: polarity 1.0, split 0.9038).

Golden labels come from the synthetic benchmark's golden.csv; rfb/ never reads
any golden.csv.
"""

import json
import os

import pytest

from rfb import pipeline
from rfb.cluster import cluster
from rfb.signature import build_signatures
from tools import inject_branch
from tools.score import _read_case_label_csv, balanced_accuracy

# (bugs kept, minimum BA, k); D uses the separate thin-evidence benchmark
SCENARIOS = {
    "A": (["BEQ_ONLY", "BNE_ONLY"], 1.0, 6),
    "B": (["EQ_BOTH"], 1.0, 5),
    "C": (["BEQ_ONLY", "BNE_ONLY", "EQ_BOTH"], 0.9357, 7),  # measured 0.935737
    "D": (["EQ_BOTH"], 1.0, 5),
    "E": (["EQ_BOTH_FP"], 1.0, 5),
    "F": (["BEQ_ONLY", "BNE_ONLY", "EQ_BOTH", "EQ_BOTH_FP"], 0.9052, 8),  # measured 0.905232
    "E2": (["EQ_BOTH_FP"], 1.0, 5),
}
SRC_D = {(2, 13), (1, 3), (2, 12)}
SRC_E2 = {(2, 11), (1, 1)}
# violation kinds each bug may produce (F4)
BUG_KINDS = {
    "BEQ_ONLY": {"BRANCH_NOT_TAKEN"},
    "BNE_ONLY": {"BRANCH_NOT_TAKEN"},
    "EQ_BOTH": {"BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"},
    "EQ_BOTH_FP": {"BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"},
    "BNE_WT_ONLY": {"BRANCH_WRONGLY_TAKEN"},
}


def _load(out):
    with open(os.path.join(out, "meta.json")) as f:
        meta = json.load(f)
    inp = os.path.join(out, "input.csv")
    records = {str(r.case_id): r for r in pipeline.parse_all(inp)}
    gold = _read_case_label_csv(os.path.join(out, "golden.csv"), ("bug",))
    return dict(dir=out, meta=meta, records=records, gold=gold)


@pytest.fixture(scope="module")
def bench_all(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("branch") / "bench")
    inject_branch.generate(out, bg_cases=inject_branch.BG_CASES)
    return _load(out)


@pytest.fixture(scope="module")
def bench_d(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("branch_d") / "bench")
    inject_branch.generate(out, bugs=["EQ_BOTH"], bg_cases=inject_branch.BG_CASES, src_cases=SRC_D)
    return _load(out)


@pytest.fixture(scope="module")
def bench_e2(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("branch_e2") / "bench")
    inject_branch.generate(out, bugs=["EQ_BOTH_FP"], bg_cases=inject_branch.BG_CASES, src_cases=SRC_E2)
    return _load(out)


def _ba(bench, bugs):
    inj = {str(c["case"]): c["bug"] for c in bench["meta"]["cases"]}
    keep = [c for c in bench["gold"] if c not in inj or inj[c] in set(bugs)]
    g = {c: bench["gold"][c] for c in keep}
    k = len(set(g.values()))
    labels = cluster(build_signatures([bench["records"][c] for c in keep]), k)
    bucket = {c: str(labels[bench["records"][c].case_id]) for c in keep}
    return k, g, bucket, balanced_accuracy(g, bucket)[-1]


@pytest.fixture(scope="module")
def scen(bench_all, bench_d, bench_e2):
    res = {}
    separate = {"D": bench_d, "E2": bench_e2}
    for name, (bugs, _, _) in SCENARIOS.items():
        res[name] = _ba(separate.get(name, bench_all), bugs)
    return res


def _viol_kinds(bench, bug):
    """(kind, mnemonic) pairs of the injected hits of one bug."""
    return {(h["kind"], h["mn"]) for c in bench["meta"]["cases"] if c["bug"] == bug for h in c["hits"]}


def test_benchmark_shape(bench_all, bench_d, bench_e2):
    per = {}
    for c in bench_all["meta"]["cases"]:
        per.setdefault(c["bug"], []).append(c)
    assert set(per) == {"BEQ_ONLY", "BNE_ONLY", "EQ_BOTH", "EQ_BOTH_FP", "BNE_WT_ONLY"}
    for bug, cs in per.items():
        assert len(cs) >= 3, bug
    assert {c["bug"] for c in bench_d["meta"]["cases"]} == {"EQ_BOTH"}
    assert {c["src"] for c in bench_d["meta"]["cases"]} <= {"set2/case_13", "set1/case_3", "set2/case_12"}
    # D is only a meaningful separator with exactly 3 thin cases that show both directions
    assert len(bench_d["meta"]["cases"]) == 3
    kinds_d = {h["kind"] for c in bench_d["meta"]["cases"] for h in c["hits"]}
    assert kinds_d == {"BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"}
    assert {c["bug"] for c in bench_e2["meta"]["cases"]} == {"EQ_BOTH_FP"}
    assert {c["src"] for c in bench_e2["meta"]["cases"]} <= {"set2/case_11", "set1/case_1"}
    kinds_e2 = {h["kind"] for c in bench_e2["meta"]["cases"] for h in c["hits"]}
    assert kinds_e2 == {"BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"}
    # BNE_WT_ONLY: only bne/c.bnez wrongly taken
    wt = _viol_kinds(bench_all, "BNE_WT_ONLY")
    assert wt
    assert all(k == "BRANCH_WRONGLY_TAKEN" and m in inject_branch.BNE_SET for k, m in wt), wt
    # EQ_BOTH_FP must carry beq wrongly taken and bne not taken
    fp = _viol_kinds(bench_all, "EQ_BOTH_FP")
    assert any(k == "BRANCH_WRONGLY_TAKEN" and m in inject_branch.BEQ_SET for k, m in fp)
    assert any(k == "BRANCH_NOT_TAKEN" and m in inject_branch.BNE_SET for k, m in fp)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_balanced_accuracy(scen, name):
    _, min_ba, k_want = SCENARIOS[name]
    k, g, _, ba = scen[name]
    assert k == k_want
    # wiring: D / E2 run on their own thin benchmarks, every other scenario on bench_all
    n_want = {"D": 3, "E2": 2}.get(name, 5)
    for b in SCENARIOS[name][0]:
        assert sum(1 for lab in g.values() if lab == b) == n_want, (name, b)
    assert ba >= min_ba, (name, ba)


@pytest.mark.parametrize("which", ["all", "d", "e2"])
def test_detected_deviation_matches_expected(bench_all, bench_d, bench_e2, which):
    b = {"all": bench_all, "d": bench_d, "e2": bench_e2}[which]
    assert b["meta"]["cases"]
    for m in b["meta"]["cases"]:
        d = b["records"][str(m["case"])].deviation
        assert d is not None, m["case"]
        assert f"{d.kind}:{d.mnemonic}" == m["expected"], (m["case"], m["bug"])


@pytest.mark.parametrize("which", ["all", "d", "e2"])
def test_violation_count_and_kinds_match_meta(bench_all, bench_d, bench_e2, which):
    b = {"all": bench_all, "d": bench_d, "e2": bench_e2}[which]
    for m in b["meta"]["cases"]:
        viol = b["records"][str(m["case"])].trace.violations
        assert sum(viol.values()) == len(m["hits"]), (m["case"], m["bug"], viol)
        assert {key.split(":")[0] for key in viol} <= BUG_KINDS[m["bug"]], (m["case"], m["bug"], viol)


def test_beq_only_and_bne_only_split(scen):
    _, g, bucket, _ = scen["A"]
    bk = {b: {bucket[c] for c, lab in g.items() if lab == b} for b in SCENARIOS["A"][0]}
    assert not (bk["BEQ_ONLY"] & bk["BNE_ONLY"])


@pytest.mark.parametrize("per_bug", [1, 2])
@pytest.mark.parametrize("bug", ["EQ_BOTH", "EQ_BOTH_FP"])
def test_eq_both_never_exceeds_per_bug(tmp_path, bug, per_bug):
    out = str(tmp_path / "bench")
    inject_branch.generate(out, bugs=[bug], per_bug=per_bug, bg_cases=inject_branch.BG_CASES)
    with open(os.path.join(out, "meta.json")) as f:
        cases = json.load(f)["cases"]
    assert 1 <= len(cases) <= per_bug
