#!/usr/bin/env python3
"""Evaluate BR_EQ grouping variants for the Ibex shared is_equal comparator.

Three configurations are run by subprocess (PYTHONPATH decides which rfb/ is
imported):
  polarity : the repo's rfb/ (BR_BEQ / BR_BNE mclass; viol_group groups
             NOT_TAKEN/WRONGLY_TAKEN on beq/bne by is_equal error polarity)
  split    : copy of rfb/ at out/branch_eq/rfb_split/rfb/ with the polarity
             branch of viol_group removed (pure mclass grouping)
  merged   : split copy plus the reverse string-replaced edits that merge
             BR_BEQ/BR_BNE back into BR_EQ (registration, _CONTROL_FLOW_CLASSES,
             the two `cls in (...)` tuples in check())
Every replacement is asserted to hit exactly once.
Scenarios (bench_all = inject_branch.py output with all five bugs, background
= BG_CASES; bench_d = EQ_BOTH from sources set2/13, set1/3, set2/12 only):
  A : BEQ_ONLY + BNE_ONLY + bg
  B : EQ_BOTH + bg
  C : BEQ_ONLY + BNE_ONLY + EQ_BOTH + bg
  D : thin-evidence EQ_BOTH (bench_d) + bg
  E : EQ_BOTH_FP + bg
  F : BEQ_ONLY + BNE_ONLY + EQ_BOTH + EQ_BOTH_FP + bg
  E2: thin-evidence EQ_BOTH_FP (bench_e2, sources set2/11, set1/1) + bg
  G : BEQ_ONLY + BNE_WT_ONLY + bg (polarity worst case; numbers only, no threshold)
  P1: public set 1 (k=2), P2: public set 2 (k=4)   (no regression)
Writes out/branch_eq/report.md.  Never touches rfb/ or testcase/.

Usage: branch_eq_eval.py [--work out/branch_eq] [--skip-inject]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
PUB = os.path.join(REPO, "testcase/2026_08_20_samples/problem")


def worker(inp, out, k, dev_json):
    """Runs inside the subprocess: uses whatever `rfb` PYTHONPATH selects."""
    import rfb
    from rfb import isa, pipeline
    from rfb.cluster import cluster
    from rfb.signature import build_signatures
    records = pipeline.parse_all(inp)
    labels = cluster(build_signatures(records), int(k))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for r in sorted(records, key=lambda r: r.case_id):
            w.writerow([r.case_id, labels[r.case_id]])
    dev = {}
    for r in records:
        d = r.deviation
        dev[str(r.case_id)] = None if d is None else dict(kind=d.kind, mn=d.mnemonic, mclass=d.mclass, src=d.source)
    with open(dev_json, "w") as f:
        json.dump(dict(rfb=os.path.dirname(rfb.__file__), beq=isa.mclass("beq"), bne=isa.mclass("c.bnez"),
                       vg=[isa.viol_group(k, m) for k, m in (("BRANCH_NOT_TAKEN", "beq"), ("BRANCH_WRONGLY_TAKEN", "beq"),
                                                          ("BRANCH_NOT_TAKEN", "c.bnez"), ("BRANCH_WRONGLY_TAKEN", "bne"))],
                       dev=dev), f)


def _sub(text, old, new, path):
    assert text.count(old) == 1, f"{path}: expected exactly one match of {old!r}, got {text.count(old)}"
    return text.replace(old, new)


MARKER = ".branch_eq_eval_copy"


def _copy_rfb(src_rfb, dst_rfb):
    """Fresh copy of rfb/.  An existing dst is rebuilt only when it carries this
    tool's marker (i.e. a previous copy); anything else is an error."""
    if os.path.lexists(dst_rfb):
        if os.path.islink(dst_rfb) or not os.path.isfile(os.path.join(dst_rfb, MARKER)):
            sys.exit(f"{dst_rfb} exists and was not created by this tool (no {MARKER}); refusing to reuse or delete it")
        shutil.rmtree(dst_rfb)
    shutil.copytree(src_rfb, dst_rfb, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(dst_rfb, MARKER), "w") as f:
        f.write("created by tools/branch_eq_eval.py\n")


def make_split(work):
    dst = os.path.join(work, "rfb_split", "rfb")
    _copy_rfb(os.path.join(REPO, "rfb"), dst)
    p = os.path.join(dst, "isa.py")
    t = open(p).read()
    t = _sub(t, "    if mclass(mn) in (\"BR_BEQ\", \"BR_BNE\") and kind in _EQ_POLARITY:\n"
                "        return _EQ_POLARITY[kind][mclass(mn)]\n", "", p)
    with open(p, "w") as f:
        f.write(t)
    return os.path.dirname(dst)


def make_merged(work):
    dst = os.path.join(work, "rfb_merged", "rfb")
    _copy_rfb(os.path.join(make_split(work), "rfb"), dst)
    p = os.path.join(dst, "isa.py")
    t = open(p).read()
    t = _sub(t, '_register("BR_BEQ", "beq c.beqz")\n_register("BR_BNE", "bne c.bnez")',
             '_register("BR_EQ", "beq bne c.beqz c.bnez")', p)
    t = _sub(t, '_CONTROL_FLOW_CLASSES = {"BR_BEQ", "BR_BNE", "BR_LT",', '_CONTROL_FLOW_CLASSES = {"BR_EQ", "BR_LT",', p)
    t = _sub(t, 'if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE", "TRAP_INSTR", "XRET") and next_pc',
             'if cls in ("BR_EQ", "BR_LT", "BR_GE", "TRAP_INSTR", "XRET") and next_pc', p)
    t = _sub(t, 'if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE"):', 'if cls in ("BR_EQ", "BR_LT", "BR_GE"):', p)
    with open(p, "w") as f:
        f.write(t)
    return os.path.dirname(dst)


def run_cfg(cfg, pypath, inp, k, outdir):
    os.makedirs(outdir, exist_ok=True)
    out, dev = os.path.join(outdir, "output.csv"), os.path.join(outdir, "dev.json")
    env = dict(os.environ, PYTHONPATH=pypath, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([os.path.join(REPO, ".venv/bin/python"), "-B", os.path.abspath(__file__), "--worker",
                        inp, out, str(k), dev], env=env, cwd=outdir, capture_output=True, text=True, timeout=900)
    with open(os.path.join(outdir, "run.log"), "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{cfg} run failed ({outdir}/run.log): {r.stderr.strip().splitlines()[-1:]}")
    d = json.load(open(dev))
    want = os.path.realpath(os.path.join(pypath, "rfb"))
    assert os.path.realpath(d["rfb"]) == want, f"imported {d['rfb']}, wanted {want}"
    assert (d["beq"], d["bne"]) == (("BR_EQ", "BR_EQ") if cfg == "merged" else ("BR_BEQ", "BR_BNE")), d
    want_vg = {"polarity": ["BR_EQ-", "BR_EQ+", "BR_EQ+", "BR_EQ-"], "split": ["BR_BEQ", "BR_BEQ", "BR_BNE", "BR_BNE"],
               "merged": ["BR_EQ"] * 4}[cfg]
    assert d["vg"] == want_vg, d["vg"]
    return out, d["dev"]


def read_csv(path, col):
    with open(path, newline="") as f:
        return {str(r["Case"]).strip(): str(r[col]).strip() for r in csv.DictReader(f)}


def make_scenario(bench, meta, name, bugs, work):
    d = os.path.join(work, "scen", name)
    os.makedirs(d, exist_ok=True)
    gold = read_csv(os.path.join(bench, "golden.csv"), "Bug")
    inj = {str(c["case"]): c["bug"] for c in meta["cases"]}
    keep = [c for c in gold if c not in inj or inj[c] in bugs]
    with open(os.path.join(bench, "input.csv"), newline="") as f:
        rows = {r["Case"]: r for r in csv.DictReader(f)}
    with open(os.path.join(d, "input.csv"), "w") as f:
        f.write("Case,Regr Log,Sim Log,Trace Log\n")
        for c in keep:
            r = rows[c]
            f.write(f"{c},{os.path.join(bench, r['Regr Log'])},{os.path.join(bench, r['Sim Log'])},"
                    f"{os.path.join(bench, r['Trace Log'])}\n")
    with open(os.path.join(d, "golden.csv"), "w") as f:
        f.write("Case,Bug\n")
        for c in keep:
            f.write(f"{c},{gold[c]}\n")
    return os.path.join(d, "input.csv"), os.path.join(d, "golden.csv"), len(set(gold[c] for c in keep))


def analyse(golden_csv, output_csv, dev, meta, bugs):
    from score import balanced_accuracy
    g, b = read_csv(golden_csv, "Bug"), read_csv(output_csv, "bucket")
    ba = balanced_accuracy(g, b)[-1]
    inj = {str(c["case"]): c for c in meta["cases"] if str(c["case"]) in g}
    lines = []
    for bug in bugs:
        cs = sorted((c for c in inj if inj[c]["bug"] == bug), key=int)
        bks = sorted({b[c] for c in cs})
        mates = sorted({g[c2] for c2, bk in b.items() if bk in bks and g[c2] != bug and c2 not in inj}
                       | {inj[c2]["bug"] for c2, bk in b.items() if bk in bks and c2 in inj and inj[c2]["bug"] != bug})
        lines.append(f"- {bug}: {len(cs)} cases in {len(bks)} bucket(s); other labels in those buckets: "
                     f"{', '.join(mates) if mates else '(none)'}")
    det = []
    ok = 0
    for c in sorted(inj, key=int):
        m, d = inj[c], dev.get(c)
        got = "none" if d is None else f"{d['kind']}:{d['mn']}"
        mark = "ok" if got == m["expected"] else ("~" if got in m["hit_kinds"] else "MISS")
        ok += mark != "MISS"
        det.append(f"| {c} | {m['bug']} | {m['src']} | {m['expected']} | {got} | {d['mclass'] if d else '-'} "
                   f"| {d['src'] if d else '-'} | {mark} | {b[c]} |")
    return ba, lines, det, ok, len(inj)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--work", default=os.path.join(REPO, "out/branch_eq"))
    ap.add_argument("--skip-inject", action="store_true", help="reuse an existing <work>/bench")
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    tc = os.path.normcase(os.path.realpath(os.path.join(REPO, "testcase")))
    rw = os.path.normcase(os.path.realpath(work))
    if rw == tc or rw.startswith(tc + os.sep):
        sys.exit(f"--work {work} resolves into the repo testcase/ directory (testcase guard); refusing")
    bench = os.path.join(work, "bench_all")
    bench_d = os.path.join(work, "bench_d")
    bench_e2 = os.path.join(work, "bench_e2")
    py = os.path.join(REPO, ".venv/bin/python")
    if not a.skip_inject:
        for b in (bench, bench_d, bench_e2):
            if os.path.exists(b) and os.listdir(b):
                sys.exit(f"{b} exists and is not empty; use --skip-inject to reuse it (nothing is deleted)")
        subprocess.run([py, os.path.join(TOOLS, "inject_branch.py"), "--out", bench], check=True, timeout=900)
        subprocess.run([py, os.path.join(TOOLS, "inject_branch.py"), "--out", bench_d, "--bugs", "EQ_BOTH",
                        "--src-cases", "2:13,1:3,2:12"], check=True, timeout=900)
        subprocess.run([py, os.path.join(TOOLS, "inject_branch.py"), "--out", bench_e2, "--bugs", "EQ_BOTH_FP",
                        "--src-cases", "2:11,1:1"], check=True, timeout=900)
    metas = {bench: json.load(open(os.path.join(bench, "meta.json"))),
             bench_d: json.load(open(os.path.join(bench_d, "meta.json"))),
             bench_e2: json.load(open(os.path.join(bench_e2, "meta.json")))}
    cfgs = {"polarity": REPO, "split": make_split(work), "merged": make_merged(work)}
    sys.path.insert(0, TOOLS)

    md = ["# BR_EQ grouping evaluation", "",
          "polarity = repo rfb/ (viol_group groups beq/bne NT/WT by is_equal error polarity); "
          "split = copy without the polarity branch (pure mclass BR_BEQ/BR_BNE); merged = split copy with the old BR_EQ", ""]
    table = [("scenario",) + tuple(cfgs)]
    scen = [("A", bench, ["BEQ_ONLY", "BNE_ONLY"]), ("B", bench, ["EQ_BOTH"]),
            ("C", bench, ["BEQ_ONLY", "BNE_ONLY", "EQ_BOTH"]), ("D", bench_d, ["EQ_BOTH"]),
            ("E", bench, ["EQ_BOTH_FP"]), ("F", bench, ["BEQ_ONLY", "BNE_ONLY", "EQ_BOTH", "EQ_BOTH_FP"]),
            ("E2", bench_e2, ["EQ_BOTH_FP"]), ("G", bench, ["BEQ_ONLY", "BNE_WT_ONLY"])]
    for name, bdir, bugs in scen:
        meta = metas[bdir]
        inp, gold, k = make_scenario(bdir, meta, name, bugs, work)
        md += [f"## Scenario {name}: {' + '.join(bugs)} + background (k={k})", ""]
        bas = []
        for cfg, pp in cfgs.items():
            out, dev = run_cfg(cfg, pp, inp, k, os.path.join(work, "run", f"{name}_{cfg}"))
            ba, lines, det, ok, n = analyse(gold, out, dev, meta, bugs)
            bas.append(f"{ba:.6f}")
            md += [f"### {cfg}: BA={ba:.6f}, detection kind:mnemonic ok/~ = {ok}/{n}", ""] + lines + [""]
            md += ["| case | bug | source | expected | detected | mclass | dev.source | match | bucket |",
                   "|---|---|---|---|---|---|---|---|---|"] + det + [""]
        table.append((name, *bas))
    for name, s, k in (("P1", 1, 2), ("P2", 2, 4)):
        inp = os.path.join(PUB, f"benchmark_set_{s}", "input.csv")
        gold = os.path.join(PUB, f"benchmark_set_{s}", "golden.csv")
        bas = []
        for cfg, pp in cfgs.items():
            out, _ = run_cfg(cfg, pp, inp, k, os.path.join(work, "run", f"{name}_{cfg}"))
            from score import balanced_accuracy
            bas.append(f"{balanced_accuracy(read_csv(gold, 'Bug'), read_csv(out, 'bucket'))[-1]:.6f}")
        table.append((f"{name} (set {s}, k={k})", *bas))
    md = md[:4] + ["## Summary", "", "| " + " | ".join(table[0]) + " |", "|---|---|---|---|"] \
        + ["| " + " | ".join(r) + " |" for r in table[1:]] + [""] + md[4:]
    with open(os.path.join(work, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for r in table:
        print("\t".join(r))
    print(f"report: {os.path.join(work, 'report.md')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:6])
    else:
        main()
