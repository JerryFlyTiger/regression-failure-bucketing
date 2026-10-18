"""Regression test for READ_MISMATCH (regfile / forwarding bugs), which no
public case triggers.

Benchmarks are synthesised by tools/inject_regfile.py (a port of the architect's
out/sp_shadow_eval/inject_rf.py taint-propagating injector) from public cases:
F (stale forwarding, any register), R2 (regfile write to x2 loses bit 4), Rc
(same on x8..x15) and ALU decoder bugs, plus untouched set-2 cases as
background.  Golden labels: this test reads the golden.csv of the synthetic
benchmark; rfb/ never reads any golden.csv.

Measured 2026-10-08 (after REGREAD_FWD; before it, 2026-10-05, both mixes were 0.949):
  S_mix  (F, R2, Rc, 3 ALU bugs; k = 10): BA = 0.973120
  T_mix  (same, trace cut 20 lines after the anchor): BA = 0.973120
  S_F_a4 (F + inj_addi4spn; k = 6)      : BA = 1.000
Full scale: every source case of both public sets, 5 cases per bug, all of
set 2 as background (the architect's out/sp_shadow_eval runs; reproduced
exactly with the repo rfb).  Each scenario takes ~10 s to generate and score
on an idle machine, so no scaling down was needed.
Thresholds equal the measured values; do not lower them to make a change pass.

Where the defence lives: the three BA thresholds and
`test_regfile_bug_anchors_are_read_mismatch` (the F/R cases must come out of
the pipeline as a READ_MISMATCH deviation, except when the anchor is
c.addi4spn, whose shadow-x2 check fires first).  The generator tests only
guard that the benchmark is reproducible.
"""

import json
import os

import pytest

from rfb import pipeline
from rfb.cluster import cluster
from rfb.signature import build_signatures
from tools import inject_regfile
from tools.score import _read_case_label_csv, balanced_accuracy

# full scale: None = all of set 2 as background, all public cases as sources
BG_CASES = None
PER_BUG = 5
SMALL_SRC = {(2, c) for c in (4, 6, 7, 10, 12, 17, 18, 23, 24)}

BA_S_MIX_MIN = 0.9731  # measured 0.973120
BA_T_MIX_MIN = 0.9731  # measured 0.973120
BA_S_F_A4_MIN = 1.0


def _build(tmp_path_factory, scenario):
    out = tmp_path_factory.mktemp(f"rf_{scenario}")
    bench = str(out / "bench")
    k = inject_regfile.generate(bench, scenario=scenario, per_bug=PER_BUG,
                                bg_cases=BG_CASES)
    # same steps as pipeline.run, split so the deviations can be inspected
    records = pipeline.parse_all(os.path.join(bench, "input.csv"))
    labels = cluster(build_signatures(records), k)
    golden = _read_case_label_csv(os.path.join(bench, "golden.csv"), ("bug",))
    bucket = {str(r.case_id): labels[r.case_id] for r in records}
    with open(os.path.join(bench, "meta.json")) as f:
        meta = json.load(f)
    _, _, _, _, _, _, ba = balanced_accuracy(golden, bucket)
    return dict(k=k, golden=golden, bucket=bucket, meta=meta, ba=ba, dir=bench,
                dev={r.case_id: r.deviation for r in records})


@pytest.fixture(scope="module")
def s_mix(tmp_path_factory):
    return _build(tmp_path_factory, "S_mix")


@pytest.fixture(scope="module")
def t_mix(tmp_path_factory):
    return _build(tmp_path_factory, "T_mix")


@pytest.fixture(scope="module")
def s_f_a4(tmp_path_factory):
    return _build(tmp_path_factory, "S_F_a4")


def test_s_mix_balanced_accuracy(s_mix):
    assert s_mix["ba"] >= BA_S_MIX_MIN, s_mix["ba"]


def test_t_mix_balanced_accuracy(t_mix):
    assert t_mix["ba"] >= BA_T_MIX_MIN, t_mix["ba"]


def test_s_f_a4_balanced_accuracy(s_f_a4):
    assert s_f_a4["ba"] == BA_S_F_A4_MIN, s_f_a4["ba"]


def test_regfile_bug_anchors_are_read_mismatch(s_mix, t_mix):
    seen = 0
    for r in (s_mix, t_mix):
        for c in r["meta"]["cases"]:
            if c["bug"] not in ("F", "R2", "Rc") or c["anchor_mn"] == "c.addi4spn":
                continue
            d = r["dev"][c["case"]]
            assert d.kind == "READ_MISMATCH", (c, d)
            assert d.detail["fwd"] is (c["bug"] == "F"), (c, d)
            seen += 1
    assert seen >= 2  # the benchmark does contain such cases


def test_benchmark_shape(s_mix, s_f_a4):
    bugs = {c["bug"] for c in s_mix["meta"]["cases"]}
    assert bugs == {"F", "R2", "Rc", "inj_addi4spn", "inj_caddi_neg", "inj_sub_borrow"}
    assert s_mix["k"] == len(set(s_mix["golden"].values()))
    assert {c["bug"] for c in s_f_a4["meta"]["cases"]} == {"F", "inj_addi4spn"}
    # every injected case really has a different printed write at its anchor
    assert all(c["events"] >= 1 for c in s_mix["meta"]["cases"])


def test_generator_is_deterministic(tmp_path):
    kw = dict(scenario="S_F_a4", per_bug=1, bg_cases={2}, src_cases=SMALL_SRC)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    inject_regfile.generate(a, **kw)
    inject_regfile.generate(b, **kw)

    def tree(root):
        return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)

    assert tree(a) == tree(b)
    for name in tree(a):
        with open(os.path.join(a, name), "rb") as x, open(os.path.join(b, name), "rb") as y:
            assert x.read() == y.read(), name


def test_out_dir_nonempty_is_refused(tmp_path, capsys):
    out = tmp_path / "o"
    out.mkdir()
    (out / "x").write_text("keep")
    with pytest.raises(SystemExit) as e:
        inject_regfile.main(["--out", str(out), "--scenario", "S_F_a4"])
    assert e.value.code == 1
    assert (out / "x").read_text() == "keep"


def test_out_under_testcase_is_refused(tmp_path, monkeypatch, capsys):
    fake_tc = tmp_path / "fake_tc"
    fake_tc.mkdir()
    monkeypatch.setattr(inject_regfile.base, "TESTCASE", str(fake_tc))
    with pytest.raises(SystemExit) as e:
        inject_regfile.main(["--out", str(fake_tc / "zz_new"), "--scenario", "S_F_a4"])
    assert e.value.code == 1
    # must be the guard, not some earlier InputError (e.g. a renamed scenario)
    assert "testcase guard" in capsys.readouterr().err
    assert os.listdir(fake_tc) == []


def test_scenario_with_thin_is_rejected(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        inject_regfile.main(["--out", str(tmp_path / "x"), "--scenario", "S_F_a4", "--thin"])
    assert e.value.code != 0
    assert not os.path.exists(tmp_path / "x")


def test_unknown_scenario_and_bug_are_rejected(tmp_path, capsys):
    for args in (["--scenario", "nope"], ["--bugs", "bogus"]):
        with pytest.raises(SystemExit) as e:
            inject_regfile.main(["--out", str(tmp_path / "x"), *args])
        assert e.value.code == 1
    assert not os.path.exists(tmp_path / "x") or not os.listdir(tmp_path / "x")
