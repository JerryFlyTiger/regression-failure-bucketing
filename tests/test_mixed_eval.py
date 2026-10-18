"""Wiring and evaluation-logic tests for tools/mixed_eval.py (no BA thresholds; numbers live in PLAN.md 4c)."""

import csv
import json
import os

import pytest

from tools import mixed_eval


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("mixed") / "w")
    d = mixed_eval.build_bench(work, per_bug=1, rf_bugs=["F", "inj_addi4spn"], br_bugs=["EQ_BOTH"], set1=True)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return d, meta


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_case_ids_unique_and_in_ranges(bench):
    d, meta = bench
    ids = [c["case"] for c in meta["cases"]]
    assert len(ids) == len(set(ids))
    rng = dict(set2=(1, 25), set1=(101, 107), rf=(201, 399), br=(401, 599))
    for c in meta["cases"]:
        lo, hi = rng[c["origin"]]
        assert lo <= c["case"] <= hi, c
    assert {c["origin"] for c in meta["cases"]} == {"set1", "set2", "rf", "br"}
    assert {c["family"] for c in meta["cases"]} == {"public", "regfile", "alu", "branch"}


def test_injected_labels_prefixed(bench):
    d, meta = bench
    gold = {int(r["Case"]): r["Bug"] for r in _rows(os.path.join(d, "golden.csv"))}
    for c in meta["cases"]:
        assert gold[c["case"]] == c["label"]
        if c["origin"] == "rf":
            assert c["label"].startswith("rf:")
        elif c["origin"] == "br":
            assert c["label"].startswith("br:")
        else:
            assert not c["label"].startswith(("rf:", "br:"))


def test_k_and_paths(bench):
    d, meta = bench
    gold = _rows(os.path.join(d, "golden.csv"))
    assert meta["k"] == len({r["Bug"] for r in gold})
    rows = _rows(os.path.join(d, "input.csv"))
    assert len(rows) == len(gold) == len(meta["cases"])
    for r in rows:
        for col in ("Regr Log", "Sim Log", "Trace Log"):
            assert os.path.isabs(r[col]) and os.path.exists(r[col]), r


def test_work_nonempty_is_refused(tmp_path, capsys):
    w = tmp_path / "w"
    w.mkdir()
    (w / "x").write_text("keep")
    with pytest.raises(SystemExit) as e:
        mixed_eval.main(["--work", str(w), "--variants", "thick"])
    assert e.value.code == 1
    assert (w / "x").read_text() == "keep"
    assert "not empty" in capsys.readouterr().err


def test_work_under_testcase_is_refused(tmp_path, monkeypatch, capsys):
    fake_tc = tmp_path / "fake_tc"
    fake_tc.mkdir()
    monkeypatch.setattr(mixed_eval.base, "TESTCASE", str(fake_tc))
    with pytest.raises(SystemExit) as e:
        mixed_eval.main(["--work", str(fake_tc / "zz"), "--variants", "thick"])
    assert e.value.code == 1
    assert "testcase guard" in capsys.readouterr().err
    assert os.listdir(fake_tc) == []


# ---- evaluation logic -------------------------------------------------------

@pytest.fixture(scope="module")
def evald(bench):
    return mixed_eval.evaluate(bench[0]), bench[1]


def _bar(res, name):
    return next(b for b in res["bars"] if b["name"] == name)


def test_rf_family_per_label():
    for b in ("F", "F2", "R2", "Rc"):
        assert mixed_eval.rf_family(b) == "regfile", b
    for b in mixed_eval.base.BUGS:
        assert b.startswith("inj_") and mixed_eval.rf_family(b) == "alu", b


def test_diagnose_table():
    g = {"1": "a", "2": "a", "3": "b", "4": "b", "5": "c"}
    bucket = {"1": "x", "2": "y", "3": "y", "4": "y", "5": "z"}
    d = mixed_eval.diagnose(g, bucket)
    assert d["a"] == dict(cases=2, buckets=2, split=True, merged={"b": 2})
    assert d["b"] == dict(cases=2, buckets=1, split=False, merged={"a": 1})
    assert d["c"] == dict(cases=1, buckets=1, split=False, merged={})
    assert set(d) == {"a", "b", "c"}


def test_partition_ignores_bucket_names():
    a = {"1": "x", "2": "x", "3": "y"}
    b = {"1": "q", "2": "q", "3": "p"}
    assert mixed_eval._partition(a) == mixed_eval._partition(b)
    assert mixed_eval._partition(a) != mixed_eval._partition({"1": "x", "2": "y", "3": "y"})


def _case(fam, mn=None, expected=None):
    return dict(family=fam, gen_meta=dict(anchor_mn=mn, expected=expected))


@pytest.mark.parametrize("c,kind,ok", [
    (_case("regfile", "add"), "READ_MISMATCH", True),
    (_case("regfile", "add"), "WRONG_RESULT", False),
    (_case("regfile", "c.addi4spn"), "WRONG_RESULT", True),
    (_case("regfile", "c.addi4spn"), "READ_MISMATCH", False),
    (_case("alu", "mul"), "WRONG_RESULT", True),
    (_case("alu", "mul"), "READ_MISMATCH", False),
    (_case("alu", "c.addi4spn"), "WRONG_RESULT", True),
    (_case("branch", expected="BRANCH_WRONG_TARGET:beq"), "BRANCH_WRONG_TARGET", True),
    (_case("branch", expected="BRANCH_WRONG_TARGET:beq"), "WRONG_RESULT", False),
    (_case("branch", expected="BRANCH_WRONG_TARGET:beq"), "BRANCH", False),  # prefix of expected, not the kind
    (_case("branch", expected="BRANCH_WRONG_TARGET:bne"), "BRANCH_WRONG_TARGET", True),  # only the kind is compared
])
def test_dev_expect_ok_table(c, kind, ok):
    assert mixed_eval._dev_expect_ok(c, kind, "beq") is ok


@pytest.mark.parametrize("c,exp", [
    (_case("regfile", "c.addi4spn"), True),
    (_case("alu", "c.addi4spn"), False),   # same anchor mnemonic, but not a regfile case
    (_case("regfile", "addi"), False),
    (dict(family="regfile", gen_meta=None), False),
])
def test_known_exc_table(c, exp):
    assert mixed_eval._known_exc(c) is exp


def _d(buckets, merged=None):
    return dict(cases=1, buckets=buckets, split=buckets > 1, merged=merged or {})


def test_find_worse_synthetic():
    lf = {"pub_a": "public", "pub_b": "public", "rf:X": "regfile", "br:Y": "branch"}
    diags = {
        "public only": {"pub_a": _d(1), "pub_b": _d(2, {"pub_a": 1})},
        "regfile": {"pub_a": _d(2), "pub_b": _d(2, {"pub_a": 1}), "rf:X": _d(1)},
        "branch": {"pub_a": _d(1), "pub_b": _d(1), "br:Y": _d(1)},
        "all mixed": {"pub_a": _d(2, {"rf:X": 1}), "pub_b": _d(1, {"pub_a": 1}), "rf:X": _d(1, {"br:Y": 2}),
                      "br:Y": _d(1)},
    }
    w = mixed_eval.find_worse(diags, lf)
    got = {(x["label"], x["bench"], x["ref"]): x["reasons"] for x in w}
    assert got == {
        ("pub_a", "regfile", "public only"): ["buckets 1 -> 2"],           # public label is reported (F1)
        ("pub_a", "all mixed", "public only"): ["buckets 1 -> 2", "new MERGED with rf:X(1)"],
        ("rf:X", "all mixed", "regfile"): ["new MERGED with br:Y(2)"],                # vs own family, not public only
    }
    # improvements (pub_b 2 -> 1 buckets) are not reported; equal (not greater) is not worse
    assert not any(x["label"] == "pub_b" for x in w)


def test_find_worse_buckets_strictly_greater_and_direction():
    lf = {"rf:X": "regfile"}
    same = {"all mixed": {"rf:X": _d(2, {"a": 1})}, "regfile": {"rf:X": _d(2, {"a": 1})}}
    assert mixed_eval.find_worse(same, lf) == []
    better = {"all mixed": {"rf:X": _d(1)}, "regfile": {"rf:X": _d(2, {"a": 1})}}
    assert mixed_eval.find_worse(better, lf) == []
    worse = {"all mixed": {"rf:X": _d(3)}, "regfile": {"rf:X": _d(2)}}
    assert mixed_eval.find_worse(worse, lf)[0]["reasons"] == ["buckets 2 -> 3"]


def test_render_marks_already_in_public_only():
    diag_all = {"pub_a": _d(2, {"pub_b": 1, "rf:X": 1}), "rf:X": _d(1, {"pub_a": 1})}
    pub = {"pub_a": _d(2, {"pub_b": 1})}
    res = dict(k=2, n=4, bars=[], worse=[], warnings=[], dev_dist={}, dev_mismatch=[],
               diag={"all mixed": diag_all, "public only": pub})
    text = "\n".join(mixed_eval.render("v", res))
    line = next(x for x in text.splitlines() if x.startswith("- pub_a"))
    assert "SPLIT (already in public only)" in line
    assert "pub_b(1) (already in public only)" in line
    assert "rf:X(1);" in line + ";" and "rf:X(1) (already" not in line
    assert "(already" not in next(x for x in text.splitlines() if x.startswith("- rf:X"))


def test_evaluate_rows_and_diag(evald):
    res, meta = evald
    cases = meta["cases"]
    pub_labels = {c["label"] for c in cases if c["family"] == "public"}
    assert _bar(res, "all mixed")["k"] == meta["k"] == res["k"]
    assert _bar(res, "all mixed")["n"] == len(cases) == res["n"]
    assert _bar(res, "public only")["k"] == len(pub_labels)
    assert _bar(res, "public only")["n"] == sum(1 for c in cases if c["family"] == "public")
    for fam in ("regfile", "alu", "branch"):
        labs = {c["label"] for c in cases if c["family"] == fam}
        assert labs
        b = _bar(res, f"public + {fam}")
        assert b["k"] == len(pub_labels | labs), fam
        assert b["n"] == sum(1 for c in cases if c["family"] in ("public", fam))
        assert pub_labels <= set(res["diag"][fam]) and labs <= set(res["diag"][fam])
    assert res["unstable"] is False
    assert not any("UNSTABLE" in b["name"] for b in res["bars"])
    assert _bar(res, "all mixed (records shuffled)")["k"] == meta["k"]
    assert set(res["diag"]) == {"all mixed", "regfile", "alu", "branch", "public only"}
    assert set(res["diag"]["public only"]) == pub_labels
    assert set(res["diag"]["all mixed"]) == {c["label"] for c in cases}
    assert res["warnings"] == []
    assert not any(i["known_exc"] for i in res["cases"])  # rf:F anchors on addi, rf:inj_addi4spn is family alu


def test_evaluate_unpatched_item_fields_pinned(evald):
    # literal values, not derived from res itself (the other dev_dist tests compare res with res)
    res, meta = evald
    assert res["dev_dist"] == {"rf:F": {"READ_MISMATCH": 1}, "rf:inj_addi4spn": {"WRONG_RESULT": 1},
                               "br:EQ_BOTH": {"BRANCH_WRONGLY_TAKEN": 1}}
    assert res["dev_mismatch"] == []
    by_case = {c["case"]: c for c in meta["cases"]}
    for i in res["cases"]:
        if i["family"] == "branch":
            assert i["expected"] == by_case[i["case"]]["gen_meta"]["expected"] == "BRANCH_WRONGLY_TAKEN:bne"
        else:
            assert "expected" not in i


def test_evaluate_item_bucket_mixed_comes_from_the_all_mixed_run(bench, monkeypatch):
    # the shuffled run (2nd cluster call) is given different bucket names; items must carry the 1st run's
    real = mixed_eval.cluster
    calls = []

    def fake(sigs, k):
        out = real(sigs, k)
        calls.append(dict(out))
        return {cid: f"shuf_{cid}" for cid in out} if len(calls) == 2 else out

    monkeypatch.setattr(mixed_eval, "cluster", fake)
    res = mixed_eval.evaluate(bench[0])
    assert res["cases"]
    for i in res["cases"]:
        assert i["bucket_mixed"] == str(calls[0][i["case"]])
        assert not i["bucket_mixed"].startswith("shuf_")


def test_evaluate_flags_unstable_when_partition_differs(bench, monkeypatch):
    n = iter(range(100))
    monkeypatch.setattr(mixed_eval, "_partition", lambda b: next(n))
    res = mixed_eval.evaluate(bench[0])
    assert res["unstable"] is True
    assert [b["name"] for b in res["bars"]][:2] == ["all mixed", "all mixed (records shuffled) UNSTABLE"]


# ---- guards -----------------------------------------------------------------

def test_id_span_overflow_is_input_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mixed_eval, "ID_SPAN", 0)
    with pytest.raises(mixed_eval.base.InputError, match="at most 0 per family"):
        mixed_eval.build_bench(str(tmp_path / "w"), per_bug=1, rf_bugs=["F"], br_bugs=[], set1=False)


def test_duplicate_ids_is_input_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mixed_eval, "RF_BASE", 0)  # rf ids collide with set 2 ids
    with pytest.raises(mixed_eval.base.InputError, match="duplicate case ids"):
        mixed_eval.build_bench(str(tmp_path / "w"), per_bug=1, rf_bugs=["F"], br_bugs=[], set1=False)


def test_build_bench_own_testcase_guard(tmp_path, monkeypatch):
    fake_tc = tmp_path / "fake_tc"
    fake_tc.mkdir()
    monkeypatch.setattr(mixed_eval.base, "TESTCASE", str(fake_tc))
    with pytest.raises(mixed_eval.base.InputError, match=r"\(testcase guard\)"):
        mixed_eval.build_bench(str(fake_tc / "zz"), rf_bugs=[], br_bugs=[])
    assert os.listdir(fake_tc) == []


def test_bad_bug_names_leave_no_work_dir(tmp_path, capsys):
    w = tmp_path / "w"
    for flag in ("--rf-bugs", "--br-bugs"):
        with pytest.raises(SystemExit) as e:
            mixed_eval.main(["--work", str(w), "--variants", "thick", flag, "NOPE"])
        assert e.value.code == 1
        assert "unknown bug name" in capsys.readouterr().err
        assert not w.exists()


def test_pub_reuses_base_constant():
    assert mixed_eval.PUB == mixed_eval.base.BASE


def test_shortfall_pure():
    w = mixed_eval._shortfall(["a", "b", "c"], 5, {"a": 5, "b": 4})
    assert len(w) == 2
    assert "bug b generated 4" in w[0] and "bug c generated 0" in w[1]
    assert mixed_eval._shortfall(["a"], 5, {"a": 6}) == []


def test_short_generation_warns_in_stdout_and_report(tmp_path, capsys):
    w = tmp_path / "w"
    mixed_eval.main(["--work", str(w), "--variants", "thick", "--rf-bugs", "inj_addi4spn", "--br-bugs", "",
                     "--no-set1", "--per-bug", "5"])
    out = capsys.readouterr().out
    assert "WARNING: bug inj_addi4spn generated" in out and "requested per_bug=5" in out
    report = (w / "report.md").read_text()
    assert "WARNING: bug inj_addi4spn generated" in report


# ---- N4 / N7: render ---------------------------------------------------------

def _res(worse, diag_all, pub):
    return dict(k=2, n=4, bars=[], worse=worse, warnings=[], dev_dist={}, dev_mismatch=[],
                diag={"all mixed": diag_all, "public only": pub})


def test_render_split_not_in_public_only_is_not_marked():
    # SPLIT in all mixed, only a merge (no split) in public only: the SPLIT must not be tagged.
    res = _res([], {"pub_a": _d(2, {"pub_b": 1})}, {"pub_a": _d(1, {"pub_b": 1})})
    line = next(x for x in mixed_eval.render("v", res) if x.startswith("- pub_a"))
    assert "SPLIT" in line and "SPLIT (already" not in line
    assert "pub_b(1) (already in public only)" in line


def test_render_worse_line_without_notes_key_and_full_text():
    w = dict(label="pub_a", family="public", bench="regfile", ref="public only",
             reasons=["buckets 1 -> 2", "new MERGED with rf:X(1)"])
    head = "- pub_a [public] in 'regfile' vs 'public only': buckets 1 -> 2; new MERGED with rf:X(1)"
    out = mixed_eval.render("v", _res([w], {}, {}))  # find_worse output has no "notes" key
    assert head in out
    w2 = dict(w, notes=["rf:X: 1 known-exception case(s)", "br:Y: 2 deviation-mismatch case(s)"])
    out2 = mixed_eval.render("v", _res([w2], {}, {}))
    assert head + " | rf:X: 1 known-exception case(s); br:Y: 2 deviation-mismatch case(s)" in out2
    assert head + " | " not in "\n".join(out)
    out3 = mixed_eval.render("v", _res([dict(w, notes=[])], {}, {}))
    assert head in out3 and head + " |" not in "\n".join(out3)


# ---- N5 / N6: guards ---------------------------------------------------------

def test_id_span_constant_matches_bases():
    assert mixed_eval.ID_SPAN == mixed_eval.BR_BASE - mixed_eval.RF_BASE - 1


def test_id_span_exact_fit_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(mixed_eval, "ID_SPAN", 1)  # exactly one rf case: must be accepted
    d = mixed_eval.build_bench(str(tmp_path / "w"), per_bug=1, rf_bugs=["F"], br_bugs=[], set1=False)
    with open(os.path.join(d, "meta.json")) as f:
        assert sum(1 for c in json.load(f)["cases"] if c["origin"] == "rf") == 1


@pytest.mark.parametrize("kw", [dict(rf_bugs=["NOPE"], br_bugs=[]), dict(rf_bugs=[], br_bugs=["NOPE"])])
def test_build_bench_checks_bug_names_itself(tmp_path, kw):
    w = tmp_path / "w"
    with pytest.raises(mixed_eval.base.InputError, match="unknown bug name"):
        mixed_eval.build_bench(str(w), set1=False, **kw)
    assert not w.exists()


def test_build_bench_missing_path_is_input_error(tmp_path, monkeypatch):
    real = mixed_eval._read_rows

    def broken(path):
        rows = real(path)
        return [(rows[0][0], rows[0][1][:2] + ["no_such_dir/trace_missing.log"])] + rows[1:]

    monkeypatch.setattr(mixed_eval, "_read_rows", broken)
    with pytest.raises(mixed_eval.base.InputError, match="missing path .*trace_missing.log"):
        mixed_eval.build_bench(str(tmp_path / "w"), rf_bugs=[], br_bugs=[], set1=False)


def test_render_deviation_sections_verbatim():
    res = _res([], {}, {})
    res["dev_dist"] = {"rf:F": {"WRONG_RESULT": 2, "READ_MISMATCH": 1}, "br:X": {"BRANCH_WRONGLY_TAKEN": 3}}
    res["dev_mismatch"] = [
        dict(case=401, label="br:X", kind="WRONG_RESULT", mnemonic="bne", expected="BRANCH_WRONGLY_TAKEN:bne"),
        dict(case=201, label="rf:F", kind="WRONG_RESULT", mnemonic="addi"),
    ]
    out = mixed_eval.render("v", res)
    i = out.index("### Deviation kind per injected bug")
    assert out[i:i + 5] == ["### Deviation kind per injected bug", "",
                            "- br:X: BRANCH_WRONGLY_TAKEN=3", "- rf:F: READ_MISMATCH=1, WRONG_RESULT=2", ""]
    j = out.index("### Deviation mismatches (vs. expectation)")
    assert out[j:j + 5] == ["### Deviation mismatches (vs. expectation)", "",
                            "- case 401 br:X: got WRONG_RESULT:bne, expected BRANCH_WRONGLY_TAKEN:bne",
                            "- case 201 rf:F: got WRONG_RESULT:addi", ""]


# ---- N1 / N2: evaluate wiring under controlled diagnostics -------------------

@pytest.fixture(scope="module")
def patched(bench):
    """evaluate() once with: spy on cluster, one regfile case forced known_exc, one branch case forced
    to a deviation mismatch, and diagnose() output doctored so that (a) rf:F gets worse in all mixed,
    (b) a public label gets worse in 'public + regfile' only (all mixed has no injected merge for it)."""
    d, meta = bench
    cs = meta["cases"]
    rf_case = next(c["case"] for c in cs if c["label"] == "rf:F")
    br_case = next(c["case"] for c in cs if c["label"] == "br:EQ_BOTH")
    pub = next(c["label"] for c in cs if c["family"] == "public")
    mp = pytest.MonkeyPatch()
    calls = []
    real_cluster, real_diag = mixed_eval.cluster, mixed_eval.diagnose

    def spy(sigs, k):
        out = real_cluster(sigs, k)
        calls.append((k, dict(out)))
        return out

    def doctored(golden, bucket):
        out = real_diag(golden, bucket)
        labs = set(golden.values())
        if "rf:F" in labs and "br:EQ_BOTH" in labs:                      # all mixed
            out["rf:F"] = dict(out["rf:F"], buckets=99, split=True, merged={"br:EQ_BOTH": 1})
            out[pub] = dict(out[pub], merged={})
        elif "rf:F" in labs and "br:EQ_BOTH" not in labs:               # 'public + regfile' 
            out[pub] = dict(out[pub], buckets=50, split=True, merged={"rf:F": 1})
        return out

    mp.setattr(mixed_eval, "cluster", spy)
    mp.setattr(mixed_eval, "diagnose", doctored)
    mp.setattr(mixed_eval, "_known_exc", lambda c: c["case"] == rf_case)
    mp.setattr(mixed_eval, "_dev_expect_ok", lambda c, kind, mn: c["case"] != br_case)
    try:
        res = mixed_eval.evaluate(d)
    finally:
        mp.undo()
    return res, calls, dict(rf_case=rf_case, br_case=br_case, pub=pub), meta


def test_evaluate_worse_family_and_notes(patched):
    res, _, ids, _ = patched
    got = {(w["label"], w["bench"]): w for w in res["worse"]}
    w = got[("rf:F", "all mixed")]
    assert (w["family"], w["ref"]) == ("regfile", "regfile")  # injected label compared with own family, not 'public only'
    assert w["reasons"] == ["buckets %d -> 99" % _bar_diag(res, "regfile", "rf:F")["buckets"], "new MERGED with br:EQ_BOTH(1)"]
    assert w["notes"] == ["br:EQ_BOTH: 1 deviation-mismatch case(s)",
                          "rf:F: 1 known-exception case(s) (c.addi4spn -> WRONG_RESULT, generator artefact)"]


def _bar_diag(res, bench, lab):
    return res["diag"][bench][lab]


def test_evaluate_notes_use_diag_of_the_worse_bench(patched):
    # public label worse in 'public + regfile': its merged rf:F exists only in that bench's diag (N1)
    res, _, ids, _ = patched
    w = next(x for x in res["worse"] if x["label"] == ids["pub"] and x["bench"] == "regfile")
    assert (w["family"], w["ref"]) == ("public", "public only")
    assert w["notes"] == ["rf:F: 1 known-exception case(s) (c.addi4spn -> WRONG_RESULT, generator artefact)"]
    assert "rf:F" not in res["diag"]["all mixed"][ids["pub"]]["merged"]


def test_evaluate_item_flags_and_dev_mismatch(patched):
    res, _, ids, meta = patched
    by = {i["case"]: i for i in res["cases"]}
    assert {i["known_exc"] for i in res["cases"] if i["case"] != ids["rf_case"]} == {False}
    assert by[ids["rf_case"]]["known_exc"] is True
    assert [i["case"] for i in res["dev_mismatch"]] == [ids["br_case"]]
    assert all(i["ok"] is (i["case"] != ids["br_case"]) for i in res["cases"])
    inj = {c["case"] for c in meta["cases"] if c["origin"] in ("rf", "br")}
    assert set(by) == inj


def test_evaluate_dev_dist_counts(patched):
    res, _, _, meta = patched
    _check_dev_dist(res, meta)


def _check_dev_dist(res, meta):
    from collections import Counter
    exp = {}
    for i in res["cases"]:
        exp.setdefault(i["label"], Counter())[i["kind"]] += 1
    assert {lab: dict(c) for lab, c in exp.items()} == res["dev_dist"]
    for lab, d in res["dev_dist"].items():
        assert sum(d.values()) == sum(1 for c in meta["cases"] if c["label"] == lab)


def test_evaluate_cluster_k_and_ba_per_bar(patched, bench):
    res, calls, _, meta = patched
    from tools.score import _read_case_label_csv, balanced_accuracy
    golden = _read_case_label_csv(os.path.join(bench[0], "golden.csv"), ("bug",))
    assert len(calls) == len(res["bars"])
    for (k, labels), bar in zip(calls, res["bars"]):
        assert k == bar["k"] and len(set(labels.values())) == k, bar["name"]  # k handed to cluster, k buckets back
        g = {str(c): golden[str(c)] for c in labels}
        assert len(set(g.values())) == bar["k"]
        t = balanced_accuracy(g, {str(c): str(v) for c, v in labels.items()})
        assert bar["ba"] == t[-1], bar["name"]
    assert res["bars"][0]["k"] == meta["k"]


def test_evaluate_unpatched_flags_match_generator_meta(evald):
    res, meta = evald
    by_case = {c["case"]: c for c in meta["cases"]}
    for i in res["cases"]:
        assert i["known_exc"] is mixed_eval._known_exc(by_case[i["case"]])
        assert i["ok"] is mixed_eval._dev_expect_ok(by_case[i["case"]], i["kind"], i["mnemonic"])
    assert [i for i in res["cases"] if not i["ok"]] == res["dev_mismatch"]
    _check_dev_dist(res, meta)
