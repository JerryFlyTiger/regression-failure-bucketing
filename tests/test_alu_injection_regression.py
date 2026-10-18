"""Regression test for the ALU-result recompute path (isa.viol_group, shadow x2,
cluster R1b), which no public case triggers.

Benchmarks are synthesised by tools/inject_alu_real.py (a port of the
architect's out/arch_alu experiment) from small public set-2 cases: 10 injected
ALU bugs x 2-4 cases each, plus 12 untouched set-2 cases as background
(k = 14, 46 cases).  Golden labels: this test reads the golden.csv of the
synthetic benchmark; the generator reads the public sets' golden.csv only to
label sources; rfb/ never reads any golden.csv.

Measured 2026-09-29 (scaled-down benchmark, pipeline.run with k=14):
  real    : BA = 1.000000
  trunc20 : BA = 0.968646  (inj_div_unit split in two, inj_addi4spn merged
            with inj_caddi_neg; thin evidence after the anchor)
Full-size benchmark (all 32 sources, 5 cases per bug) measured the same day:
real 1.0, trunc20 0.986591.  Thresholds below equal the measured values; do
not lower them to make a change pass.

Where the defence lives: the two BA thresholds, `test_trunc20_evidence_is_cut`
(the trunc20 traces really are cut) and the generator tests below.  The
structural assertions on `real` and on `trunc20` (which bucket pairs split or
merge) are diagnostic only: they say which relation broke when a BA drops.
They are not an independent line of defence; every perturbation that made one
of them fail also dropped the trunc20 BA below its threshold.

Note: no source case has an srai anchor (srai only occurs as a later hit), so
the sra-family case set has sra and c.srai anchors.
"""

import gzip
import json
import os
import re
import subprocess
import sys

import pytest

from rfb import pipeline
from rfb.parse_trace import _split_trace_line
from tools import inject_alu_real
from tools.score import _read_case_label_csv, balanced_accuracy

BG_CASES = {2, 4, 6, 7, 8, 10, 12, 16, 17, 18, 20, 24}
SRC_CASES = {(2, c) for c in (4, 6, 7, 10, 12, 17, 18, 23, 24)}
PER_BUG = 4

BA_REAL_MIN = 1.0
BA_TRUNC20_MIN = 0.9686  # measured 0.968646


def _build(tmp_path_factory, variant):
    out = tmp_path_factory.mktemp(f"alu_{variant}")
    bench = str(out / "bench")
    k = inject_alu_real.generate(bench, variant, seed=0, per_bug=PER_BUG,
                                 bg_cases=BG_CASES, src_cases=SRC_CASES)
    output_csv = str(out / "output.csv")
    pipeline.run(os.path.join(bench, "input.csv"), output_csv, k)
    golden = _read_case_label_csv(os.path.join(bench, "golden.csv"), ("bug",))
    bucket = _read_case_label_csv(output_csv, ("bucket",))
    with open(os.path.join(bench, "meta.json")) as f:
        meta = json.load(f)
    _, _, _, _, _, _, ba = balanced_accuracy(golden, bucket)
    return dict(k=k, golden=golden, bucket=bucket, meta=meta, ba=ba, dir=bench)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    return _build(tmp_path_factory, "real")


@pytest.fixture(scope="module")
def trunc20(tmp_path_factory):
    return _build(tmp_path_factory, "trunc20")


def _buckets_of(r, label):
    return {r["bucket"][c] for c, g in r["golden"].items() if g == label}


def test_benchmark_shape(real):
    labels = set(real["golden"].values())
    assert real["k"] == 14 and len(labels) == 14
    assert len(real["golden"]) == 46
    per = {}
    for c in real["meta"]["cases"]:
        per.setdefault(c["bug"], []).append(c)
    assert len(per) == 10
    # these bugs need >= 3 cases each (the rules only bite when k < number of atoms)
    for bug in ("inj_mulhsu_neg", "inj_mul_pp", "inj_mulhu_msb", "inj_sra_ge16", "inj_div_unit",
                "inj_caddi_neg"):
        assert len(per[bug]) >= 3, bug
    assert len(per["inj_addi4spn"]) >= 2  # only needs to share a bucket-decision with inj_caddi_neg
    # sra family has both sra and c.srai anchors; div family mixes mnemonics
    assert {c["anchor_mn"] for c in per["inj_sra_ge16"]} >= {"sra", "c.srai"}
    assert len({m for c in per["inj_div_unit"] for m in c["hit_mnemonics"]}) >= 3
    assert sum(1 for g in real["golden"].values() if g == "bug_107") >= 3


def _tree(root):
    out = set()
    for d, _, fs in os.walk(root):
        for f in fs:
            out.add(os.path.relpath(os.path.join(d, f), root))
    return out


def test_generator_is_deterministic(tmp_path):
    kw = dict(per_bug=1, bg_cases={2}, src_cases={(2, 12), (2, 18), (2, 4), (2, 7), (2, 24)})
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    inject_alu_real.generate(a, "real", seed=1, **kw)
    inject_alu_real.generate(b, "real", seed=2, **kw)
    files = _tree(a)
    assert files == _tree(b)
    traces = [p for p in files if p.endswith("trace.log.gz")]
    assert traces
    for p in traces:  # gzip header bytes 4-7 = MTIME
        with open(os.path.join(a, p), "rb") as x, open(os.path.join(b, p), "rb") as y:
            ha, hb = x.read(8), y.read(8)
        assert ha[4:8] == b"\0\0\0\0" and hb[4:8] == b"\0\0\0\0", p
    for name in sorted(files):
        with open(os.path.join(a, name), "rb") as x, open(os.path.join(b, name), "rb") as y:
            da, db = x.read(), y.read()
        if name == "input.csv":  # paths embed the output dir: normalise the prefix
            da, db = da.replace(a.encode(), b"<OUT>"), db.replace(b.encode(), b"<OUT>")
        assert da == db, name


SMALL = dict(per_bug=1, bg_cases={2, 4}, src_cases={(2, 4), (2, 7), (2, 12), (2, 18)})


def _golden_bugs(bench):
    return set(_read_case_label_csv(os.path.join(bench, "golden.csv"), ("bug",)).values())


def test_returned_k_equals_distinct_golden_bugs(tmp_path):
    bench = str(tmp_path / "b")
    k = inject_alu_real.generate(bench, "real", **SMALL)
    with open(os.path.join(bench, "meta.json")) as f:
        meta = json.load(f)
    assert meta["cases"] and k == len(_golden_bugs(bench)) == meta["k"]
    assert k > len({m["bug"] for m in meta["cases"]})  # background bugs are counted too


def _read_input_paths(bench):
    with open(os.path.join(bench, "input.csv")) as f:
        rows = [ln.rstrip("\n").split(",") for ln in f][1:]
    return [p for r in rows for p in r[1:]]


def test_relative_out_gives_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inject_alu_real.generate("rel_out", "real", **SMALL)
    paths = _read_input_paths(str(tmp_path / "rel_out"))
    assert paths
    for p in paths:
        assert os.path.isabs(p) and os.path.exists(p), p


def test_main_smoke_parses_all_options(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inject_alu_real.main(["--out", "cli_out", "--variant", "trunc20", "--seed", "3", "--per-bug", "1",
                          "--bg-cases", "2,4", "--src-cases", "2:4,2:7,2:12,2:18"])
    assert capsys.readouterr().out.startswith("trunc20: k=")
    bench = str(tmp_path / "cli_out")
    with open(os.path.join(bench, "meta.json")) as f:
        meta = json.load(f)
    assert meta["cases"]
    assert {m["src"] for m in meta["cases"]} <= {f"set2/case_{c}" for c in (4, 7, 12, 18)}
    assert all(m["after_anchor"] <= 20 for m in meta["cases"])  # variant honoured
    assert {c for c in _read_case_label_csv(os.path.join(bench, "golden.csv"), ("bug",))
            if int(c) < 26} == {"2", "4"}  # --bg-cases honoured
    for p in _read_input_paths(bench):
        assert os.path.isabs(p) and os.path.exists(p), p
    per = {}
    for m in meta["cases"]:
        per[m["bug"]] = per.get(m["bug"], 0) + 1
    assert max(per.values()) == 1  # --per-bug honoured


@pytest.mark.parametrize("args", [
    ["--src-cases", "2"], ["--src-cases", "2:x"], ["--src-cases", "2:3:4"], ["--src-cases", "a:3"],
    ["--src-cases", "2:4,,2:x"], ["--per-bug", "0"], ["--per-bug", "-2"], ["--per-bug", "x"],
])
def test_cli_rejects_bad_arguments(tmp_path, args):
    out = tmp_path / "never"
    with pytest.raises(SystemExit) as e:
        inject_alu_real.main(["--out", str(out)] + args)
    assert e.value.code == 2
    assert not out.exists()


def _snapshot(root):
    return {os.path.relpath(os.path.join(d, n), root): (open(os.path.join(d, n), "rb").read() if n in fs else None)
            for d, ds, fs in os.walk(root) for n in ds + fs}


def test_out_dir_nonempty_is_refused_and_untouched(tmp_path, capsys):
    d = tmp_path / "precious"
    d.mkdir()
    (d / "keep.txt").write_text("keep")
    (d / "case_1").mkdir()
    (d / "case_1" / "regr.log").write_text("user data")
    (d / "input.csv").write_text("user csv")
    before = _snapshot(str(d))
    with pytest.raises(inject_alu_real.InputError, match="not empty"):
        inject_alu_real.generate(str(d), "real", **SMALL)
    assert _main_exit(["--out", str(d)] + ARGS_OK) == 1
    assert "not empty" in capsys.readouterr().err
    assert _snapshot(str(d)) == before


def test_out_dir_empty_and_missing_are_usable(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    inject_alu_real.generate(str(empty), "real", **SMALL)
    assert (empty / "input.csv").exists() and (empty / "case_26").is_dir()
    inject_alu_real.generate(str(tmp_path / "new" / "deep"), "real", **SMALL)
    assert (tmp_path / "new" / "deep" / "input.csv").exists()
    inject_alu_real.generate(str(tmp_path / "slash") + "/", "real", **SMALL)  # trailing slash
    assert (tmp_path / "slash" / "input.csv").exists()


def test_regenerating_into_a_generated_dir_is_refused(tmp_path, capsys):
    d = tmp_path / "gen"
    inject_alu_real.generate(str(d), "real", **SMALL)
    before = _snapshot(str(d))
    with pytest.raises(inject_alu_real.InputError, match="not empty"):
        inject_alu_real.generate(str(d), "real", **SMALL)
    assert _main_exit(["--out", str(d)] + ARGS_OK) == 1
    assert _snapshot(str(d)) == before


DET_KW = ["--per-bug", "2", "--bg-cases", "2,4", "--src-cases", "2:4,2:6,2:7,2:10,2:12,2:17,2:18,2:23,2:24"]


def _cli_run(out, hashseed, variant="real"):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    r = subprocess.run([sys.executable, inject_alu_real.__file__, "--out", str(out), "--variant", variant] + DET_KW,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]


def test_generator_is_deterministic_across_hash_seeds(tmp_path):
    outs = []
    for seed in (0, 1, 2):
        d = tmp_path / f"s{seed}"
        _cli_run(d, seed)
        outs.append(str(d))
    base = outs[0]
    files = _tree(base)
    assert any(p.endswith("trace.log.gz") for p in files) and "meta.json" in files
    for other in outs[1:]:
        assert _tree(other) == files
        for name in sorted(files):
            with open(os.path.join(base, name), "rb") as x, open(os.path.join(other, name), "rb") as y:
                da, db = x.read(), y.read()
            if name == "input.csv":
                da, db = da.replace(base.encode(), b"<OUT>"), db.replace(other.encode(), b"<OUT>")
            elif name.endswith(".gz") and "trace" in name:
                assert da[4:8] == db[4:8] == b"\0\0\0\0", name
                da, db = gzip.decompress(da), gzip.decompress(db)
            assert da == db, (name, other)


def test_real_balanced_accuracy(real):
    assert real["ba"] >= BA_REAL_MIN


def test_real_same_mclass_bugs_split(real):
    mulhsu = _buckets_of(real, "inj_mulhsu_neg")
    mul = _buckets_of(real, "inj_mul_pp")
    mulhu = _buckets_of(real, "inj_mulhu_msb")
    assert not (mulhsu & mul) and not (mulhsu & mulhu) and not (mul & mulhu)


def test_real_sra_family_merges(real):
    assert len(_buckets_of(real, "inj_sra_ge16")) == 1  # sra / c.srai anchors


def test_real_div_family_merges(real):
    assert len(_buckets_of(real, "inj_div_unit")) == 1  # div/divu/rem/remu


def test_real_different_decoders_split(real):
    assert not (_buckets_of(real, "inj_caddi_neg") & _buckets_of(real, "inj_addi4spn"))


def test_real_sra_and_sll_split(real):
    assert not (_buckets_of(real, "inj_sra_ge16") & _buckets_of(real, "inj_sll_ge16"))


def test_real_injected_not_merged_with_background(real):
    inj = {real["bucket"][c] for c, g in real["golden"].items() if g.startswith("inj_")}
    bg = {real["bucket"][c] for c, g in real["golden"].items() if g == "bug_107"}
    assert not (inj & bg)


def test_trunc20_balanced_accuracy(trunc20):
    assert trunc20["ba"] >= BA_TRUNC20_MIN


def _trace_lines(bench, cid):
    with gzip.open(os.path.join(bench, f"case_{cid}", "trace.log.gz"), "rt") as f:
        lines = f.read().splitlines()
    return lines[0], [ln for ln in lines[1:] if ln]


def _anchor_pos(bench, cid, body):
    """Index of the anchor in the actual trace, found from regr.log (ibex pc and
    wrong value), not from meta."""
    with open(os.path.join(bench, f"case_{cid}", "regr.log")) as f:
        m = re.search(r"ibex\[\d+\] : pc\[([0-9a-f]+)\].*: \w+:([0-9a-f]{8})", f.read())
    pc, val = int(m.group(1), 16), m.group(2)
    for i, ln in enumerate(body):
        fields = _split_trace_line(ln.split("\t"))
        if fields and int(fields[0], 16) == pc and f"=0x{val}" in fields[4]:
            return i
    raise AssertionError(f"anchor not found in trace of case {cid}")


# Diagnostic structural assertions on trunc20 follow (see the module docstring).
def test_trunc20_evidence_is_cut(real, trunc20):
    cut = 0
    for c in trunc20["meta"]["cases"]:
        cid = c["case"]
        _, body = _trace_lines(trunc20["dir"], cid)
        _, full = _trace_lines(real["dir"], cid)  # same case id = same source, uncut
        assert c["trace_len"] == len(body), c
        pos = _anchor_pos(trunc20["dir"], cid, body)
        assert pos == _anchor_pos(real["dir"], cid, full)
        after = len(body) - 1 - pos
        assert 1 <= after <= 20, c
        assert after == min(len(full) - 1 - pos, 20), c  # cut at exactly 20, or the clean prefix end
        assert body == full[:len(body)], c
        cut += len(full) - len(body) > 0
    assert cut  # the benchmark does exercise the cut


def test_trunc20_same_mclass_bugs_split(trunc20):
    mulhsu = _buckets_of(trunc20, "inj_mulhsu_neg")
    mul = _buckets_of(trunc20, "inj_mul_pp")
    mulhu = _buckets_of(trunc20, "inj_mulhu_msb")
    assert mulhsu and mul and mulhu
    assert not (mulhsu & mul) and not (mulhsu & mulhu) and not (mul & mulhu)


def test_trunc20_sra_family_merges(trunc20):
    assert len(_buckets_of(trunc20, "inj_sra_ge16")) == 1


def test_trunc20_sra_and_sll_split(trunc20):
    assert not (_buckets_of(trunc20, "inj_sra_ge16") & _buckets_of(trunc20, "inj_sll_ge16"))


def test_trunc20_injected_not_merged_with_background(trunc20):
    inj = {trunc20["bucket"][c] for c, g in trunc20["golden"].items() if g.startswith("inj_")}
    bg = {trunc20["bucket"][c] for c, g in trunc20["golden"].items() if g == "bug_107"}
    assert bg and not (inj & bg)


def _main_exit(argv):
    with pytest.raises(SystemExit) as e:
        inject_alu_real.main(argv)
    return e.value.code


ARGS_OK = ["--per-bug", "1", "--bg-cases", "2", "--src-cases", "2:4"]


def test_cli_rejects_empty_src_and_bad_bg_cases_with_exit_2(tmp_path):
    out = str(tmp_path / "never")
    for extra in (["--src-cases", ""], ["--src-cases", ","], ["--bg-cases", "2,x"], ["--bg-cases", "a"]):
        assert _main_exit(["--out", out] + extra) == 2, extra
    assert not os.path.exists(out)


def test_nonexistent_cases_raise_value_error_and_cli_exit_1(tmp_path, capsys):
    out = str(tmp_path / "never")
    with pytest.raises(ValueError, match="2:999"):
        inject_alu_real.generate(out, "real", per_bug=1, bg_cases={2}, src_cases={(2, 4), (2, 999)})
    with pytest.raises(ValueError, match="77"):
        inject_alu_real.generate(out, "real", per_bug=1, bg_cases={2, 77}, src_cases={(2, 4)})
    with pytest.raises(ValueError):
        inject_alu_real.generate(out, "real", per_bug=1, src_cases=set())
    assert not os.path.exists(out)
    assert _main_exit(["--out", out, "--per-bug", "1", "--src-cases", "2:4,2:999"]) == 1
    assert "2:999" in capsys.readouterr().err
    assert _main_exit(["--out", out, "--per-bug", "1", "--bg-cases", "2,77", "--src-cases", "2:4"]) == 1
    assert "77" in capsys.readouterr().err
    assert not os.path.exists(out)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, "2", None])
def test_generate_rejects_bad_per_bug(tmp_path, bad):
    out = str(tmp_path / "never")
    with pytest.raises(inject_alu_real.InputError, match="per_bug"):
        inject_alu_real.generate(out, "real", per_bug=bad, bg_cases={2}, src_cases={(2, 4)})
    assert not os.path.exists(out)


@pytest.mark.parametrize("bad", ["", "Real", "trunc", None, 1])
def test_generate_rejects_bad_variant(tmp_path, bad):
    out = str(tmp_path / "never")
    with pytest.raises(inject_alu_real.InputError, match="variant"):
        inject_alu_real.generate(out, bad, per_bug=1, bg_cases={2}, src_cases={(2, 4)})
    assert not os.path.exists(out)


def test_empty_bg_cases_means_no_background(tmp_path):
    api, cli = tmp_path / "api", tmp_path / "cli"
    ka = inject_alu_real.generate(str(api), "real", per_bug=1, bg_cases=set(), src_cases=SMALL["src_cases"])
    assert inject_alu_real.main(["--out", str(cli), "--per-bug", "1", "--bg-cases", "",
                                 "--src-cases", "2:4,2:7,2:12,2:18"]) is None
    for d, k in ((api, ka), (cli, None)):
        labels = _read_case_label_csv(str(d / "golden.csv"), ("bug",))
        assert labels and all(int(c) >= 26 for c in labels), d  # no set-2 background case
        assert all(g.startswith("inj_") for g in labels.values())
        if k is not None:
            assert k == len(set(labels.values()))


def test_out_is_a_file(tmp_path, capsys):
    f = tmp_path / "afile"
    f.write_text("data")
    with pytest.raises(ValueError, match="not a directory"):
        inject_alu_real.generate(str(f), "real", **SMALL)
    assert _main_exit(["--out", str(f)] + ARGS_OK) == 1
    assert "not a directory" in capsys.readouterr().err
    assert f.read_text() == "data"


def test_out_parent_is_a_file(tmp_path, capsys):
    f = tmp_path / "afile"
    f.write_text("data")
    with pytest.raises(inject_alu_real.InputError, match="cannot create"):
        inject_alu_real.generate(str(f / "sub" / "out"), "real", **SMALL)
    assert _main_exit(["--out", str(f / "out")] + ARGS_OK) == 1
    assert "cannot create" in capsys.readouterr().err
    assert f.read_text() == "data"


def test_out_symlinks_are_refused_and_never_followed(tmp_path, capsys):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    empty_target = tmp_path / "empty_target"
    empty_target.mkdir()
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "nowhere")
    links = {"dangling": dangling}
    for name, dest in (("l_full", target), ("l_empty", empty_target)):
        links[name] = tmp_path / name
        links[name].symlink_to(dest)
    for name, link in links.items():
        for out in (str(link), str(link) + "/"):  # trailing slash must not hide the link
            if name == "dangling" and out.endswith("/"):
                continue  # "dangling/" cannot even be resolved; the plain form is the case
            with pytest.raises(inject_alu_real.InputError, match="symlink"):
                inject_alu_real.generate(out, "real", **SMALL)
            assert _main_exit(["--out", out] + ARGS_OK) == 1, name
            assert "symlink" in capsys.readouterr().err
        assert link.is_symlink()
    assert (target / "keep.txt").read_text() == "keep" and len(list(target.iterdir())) == 1
    assert list(empty_target.iterdir()) == []
    assert not os.path.exists(tmp_path / "nowhere")


def test_testcase_constant_is_the_repo_testcase_dir():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.realpath(inject_alu_real.TESTCASE) == os.path.realpath(os.path.join(repo, "testcase"))


GUARD = "testcase guard"


@pytest.fixture
def fake_tc(tmp_path, monkeypatch):
    """A stand-in testcase/ under tmp_path; the real testcase/ is never touched."""
    tc = tmp_path / "fake_tc"
    (tc / "sub").mkdir(parents=True)
    monkeypatch.setattr(inject_alu_real, "TESTCASE", str(tc))
    return tc


def test_out_equal_to_testcase_root_is_refused(fake_tc, capsys):
    before = _snapshot(str(fake_tc))
    for out in (str(fake_tc), str(fake_tc) + "/"):  # an empty root would otherwise be usable
        empty = fake_tc / "sub"
        assert not os.listdir(empty)
        with pytest.raises(inject_alu_real.InputError, match=GUARD):
            inject_alu_real.generate(out, "real", **SMALL)
        assert _main_exit(["--out", out] + ARGS_OK) == 1
        assert GUARD in capsys.readouterr().err
    assert _snapshot(str(fake_tc)) == before


def test_out_under_testcase_is_refused(fake_tc, capsys):
    before = _snapshot(str(fake_tc))
    for out in (fake_tc / "zz_new", fake_tc / "sub", fake_tc / "a" / "b" / "c",
                fake_tc / "sub" / ".." / "zz_dotdot"):
        with pytest.raises(inject_alu_real.InputError, match=GUARD):
            inject_alu_real.generate(str(out), "real", **SMALL)
        assert _main_exit(["--out", str(out)] + ARGS_OK) == 1
        assert GUARD in capsys.readouterr().err
    assert _snapshot(str(fake_tc)) == before


def test_out_via_symlinked_parent_into_testcase_is_refused(fake_tc, tmp_path):
    before = _snapshot(str(fake_tc))
    link = tmp_path / "via_link"
    link.symlink_to(fake_tc)
    for out in (link / "zz_new", link / "sub"):
        with pytest.raises(inject_alu_real.InputError, match=GUARD):
            inject_alu_real.generate(str(out), "real", **SMALL)
    assert _snapshot(str(fake_tc)) == before


def test_sibling_of_testcase_is_allowed(fake_tc, tmp_path):
    out = tmp_path / "fake_tc2" / "x"  # shares the string prefix but is not inside
    inject_alu_real.generate(str(out), "real", **SMALL)
    assert (out / "input.csv").exists()


def test_main_only_catches_input_error(monkeypatch, tmp_path):
    for exc in (RuntimeError("boom"), ValueError("plain value error")):
        def boom(*a, **k):
            raise exc
        monkeypatch.setattr(inject_alu_real, "generate", boom)
        with pytest.raises(type(exc)):
            inject_alu_real.main(["--out", str(tmp_path / "o")])
