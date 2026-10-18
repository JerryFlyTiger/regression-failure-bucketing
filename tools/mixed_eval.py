#!/usr/bin/env python3
"""Mixed-bench evaluation: public cases + regfile/ALU + branch injections in ONE bench.

Merges the outputs of tools/inject_regfile.py and tools/inject_branch.py (each
generated into its own directory, then renumbered) with the public cases, runs
the pipeline once and measures balanced accuracy, to see whether bug families
interfere with each other when mixed.

Case ids in the merged bench: set 2 = 1..25 (original), set 1 = 101.. (unless
--no-set1), regfile/ALU injections = 201.., branch injections = 401..
Labels of injected cases are prefixed "rf:" / "br:".  Paths in the merged
input.csv are absolute (public cases point into testcase/, injected cases into
gen_rf / gen_br); nothing is copied.

Per variant (thick / thin regfile generator; branch is always thick):
all-mixed BA, shuffled-record BA (UNSTABLE if it differs), per-family
(public background + one injected family) BA, public-only BA, per-bug
SPLIT/MERGED diagnostics, "worse when mixed" labels (injected: vs. its own family
bench; public: vs. public only; SPLIT/MERGED lines mark what public only already has),
deviation kind per injected case vs. expectation, and a warning when a requested bug
generated fewer than per_bug cases.  The shuffled row only checks that clustering is
insensitive to record order (parsing happens once); it is not cross-run stability.

Usage: .venv/bin/python tools/mixed_eval.py --work DIR [--per-bug 5]
       [--rf-bugs ...] [--br-bugs ...] [--no-set1] [--variants thick,thin]
Reads golden.csv (evaluation use only; rfb/ never does).  Deletes nothing.
"""
import argparse
import csv
import json
import os
import random
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
for p in (REPO, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

import inject_alu_real as base  # noqa: E402
import inject_branch  # noqa: E402
import inject_regfile  # noqa: E402
from rfb import pipeline  # noqa: E402
from rfb.cluster import cluster  # noqa: E402
from rfb.signature import build_signatures  # noqa: E402
from tools.score import _read_case_label_csv, balanced_accuracy  # noqa: E402

PUB = base.BASE
SET2_BASE, SET1_BASE, RF_BASE, BR_BASE = 0, 100, 200, 400
ID_SPAN = 199  # injected ids of one family must stay in BASE+1..BASE+199
KNOWN_EXC_MN = "c.addi4spn"  # regfile case whose shadow x2 triggers WRONG_RESULT first
DEFAULT_RF_BUGS = ["F", "R2", "Rc"] + list(base.BUGS)
DEFAULT_BR_BUGS = ["EQ_BOTH", "EQ_BOTH_FP"]
FAMILIES = ("regfile", "alu", "branch")
SHUFFLE_SEED = 20261006
FIELDS = ["Case", "Regr Log", "Sim Log", "Trace Log"]


def rf_family(bug):
    return "alu" if bug.startswith("inj_") else "regfile"


def _read_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hm = {h.strip().lower(): h for h in r.fieldnames}
        return [(int(str(row[hm["case"]]).strip()), [str(row[hm[n]]).strip() for n in ("regr log", "sim log", "trace log")])
                for row in r]


def _gen_meta(d):
    with open(os.path.join(d, "meta.json")) as f:
        m = json.load(f)
    return {int(c["case"]): c for c in m["cases"]}


def _check_bugs(rf_bugs, br_bugs):
    bad_rf = [b for b in rf_bugs if b not in inject_regfile.BUG_NAMES]
    bad_br = [b for b in br_bugs if b not in inject_branch.BUG_NAMES]
    if bad_rf or bad_br:
        raise base.InputError(f"unknown bug name(s): rf {bad_rf} (choose from {list(inject_regfile.BUG_NAMES)}), "
                              f"br {bad_br} (choose from {list(inject_branch.BUG_NAMES)})")


def _shortfall(bugs, per_bug, counts):
    """Warnings for requested bugs whose generated case count is below per_bug (0 included)."""
    return [f"WARNING: bug {b} generated {counts.get(b, 0)} case(s), requested per_bug={per_bug}"
            for b in bugs if counts.get(b, 0) < per_bug]


def build_bench(work, per_bug=5, rf_bugs=None, br_bugs=None, set1=True, thin=False, br_dir=None):
    """Generate the injected benches under `work` and merge them into
    `work`/bench.  `work` must be missing or empty (testcase guard).  If br_dir
    is given, that already generated branch bench is reused instead of
    generating one.  Returns the bench dir."""
    rf_bugs = list(DEFAULT_RF_BUGS if rf_bugs is None else rf_bugs)
    br_bugs = list(DEFAULT_BR_BUGS if br_bugs is None else br_bugs)
    _check_bugs(rf_bugs, br_bugs)
    base.prepare_out_dir(work)
    work = os.path.abspath(work)
    gen_rf = os.path.join(work, "gen_rf")
    if rf_bugs:
        inject_regfile.generate(gen_rf, bugs=rf_bugs, thin=thin, per_bug=per_bug, bg_cases=set())
    if br_bugs and br_dir is None:
        br_dir = os.path.join(work, "gen_br")
        inject_branch.generate(br_dir, bugs=br_bugs, per_bug=per_bug, bg_cases=set())

    entries = []  # (case, origin, label, family, paths, gen_case, gen_meta)

    def public(s, id_base, origin):
        d = os.path.join(PUB, f"benchmark_set_{s}")
        gold = _read_case_label_csv(os.path.join(d, "golden.csv"), ("bug",))
        for c, rel in _read_rows(os.path.join(d, "input.csv")):
            entries.append((id_base + c if id_base else c, origin, gold[str(c)], "public",
                            [os.path.join(d, p) for p in rel], c, None))

    public(2, SET2_BASE, "set2")
    if set1:
        public(1, SET1_BASE, "set1")

    def injected(gen, id_base, origin, prefix, fam_of):
        meta = _gen_meta(gen)
        gold = _read_case_label_csv(os.path.join(gen, "golden.csv"), ("bug",))
        rows = _read_rows(os.path.join(gen, "input.csv"))
        if len(rows) > ID_SPAN:
            raise base.InputError(f"{origin} injection produced {len(rows)} cases; at most {ID_SPAN} per family "
                                  f"(ids {id_base + 1}..{id_base + ID_SPAN}); lower --per-bug or the bug list")
        n = 0
        for c, rel in rows:
            m = meta[c]
            n += 1
            entries.append((id_base + n, origin, prefix + gold[str(c)], fam_of(m["bug"]),
                            [os.path.join(gen, p) for p in rel], c, m))

    if rf_bugs:
        injected(gen_rf, RF_BASE, "rf", "rf:", rf_family)
    if br_bugs:
        injected(br_dir, BR_BASE, "br", "br:", lambda b: "branch")

    ids = [e[0] for e in entries]
    if len(ids) != len(set(ids)):
        raise base.InputError("duplicate case ids in merged bench")
    for e in entries:
        for p in e[4]:
            if not os.path.exists(p):
                raise base.InputError(f"missing path {p}")
    counts = {}
    for e in entries:
        if e[6] is not None:
            counts[e[6]["bug"]] = counts.get(e[6]["bug"], 0) + 1
    warnings = _shortfall(rf_bugs + br_bugs, per_bug, counts)

    bench = os.path.join(work, "bench")
    os.makedirs(bench)
    with open(os.path.join(bench, "input.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for e in entries:
            w.writerow([e[0]] + e[4])
    with open(os.path.join(bench, "golden.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Bug"])
        for e in entries:
            w.writerow([e[0], e[2]])
    cases = [dict(case=e[0], origin=e[1], label=e[2], family=e[3], gen_case=e[5], gen_meta=e[6]) for e in entries]
    with open(os.path.join(bench, "meta.json"), "w") as f:
        json.dump({"k": len({e[2] for e in entries}), "cases": cases, "warnings": warnings}, f, indent=1)
    return bench


def _partition(bucket):
    d = {}
    for c, b in bucket.items():
        d.setdefault(b, set()).add(c)
    return {frozenset(v) for v in d.values()}


def _run(records, golden):
    """records subset -> (k, bucket, ba); k = distinct golden labels in subset."""
    g = {str(r.case_id): golden[str(r.case_id)] for r in records}
    k = len(set(g.values()))
    labels = cluster(build_signatures(records), k)
    bucket = {str(r.case_id): str(labels[r.case_id]) for r in records}
    return k, g, bucket, balanced_accuracy(g, bucket)[-1]


def diagnose(golden, bucket):
    """label -> dict(buckets=n, split=bool, merged={foreign label: cases shared})."""
    bks = {}
    for c, b in bucket.items():
        bks.setdefault(b, []).append(c)
    out = {}
    for lab in sorted(set(golden.values())):
        cs = [c for c in golden if golden[c] == lab]
        mine = {bucket[c] for c in cs}
        merged = {}
        for b in mine:
            for c2 in bks[b]:
                if golden[c2] != lab:
                    merged[golden[c2]] = merged.get(golden[c2], 0) + 1
        out[lab] = dict(cases=len(cs), buckets=len(mine), split=len(mine) > 1, merged=merged)
    return out


def _known_exc(c):
    """Regfile case anchored on c.addi4spn: the shadow x2 triggers WRONG_RESULT first (known exception)."""
    return c["family"] == "regfile" and (c["gen_meta"] or {}).get("anchor_mn") == KNOWN_EXC_MN


def _dev_expect_ok(c, kind, mn):
    """Does the observed deviation kind match the case's expectation?  Compares the kind
    only, it does not verify the position/mnemonic (mn is unused)."""
    fam, m = c["family"], c["gen_meta"]
    if fam == "regfile":
        return kind == ("WRONG_RESULT" if _known_exc(c) else "READ_MISMATCH")
    if fam == "alu":
        return kind == "WRONG_RESULT"
    return m["expected"].split(":")[0] == kind


def _cmp(dm, ds):
    reasons = []
    if dm["buckets"] > ds["buckets"]:
        reasons.append(f"buckets {ds['buckets']} -> {dm['buckets']}")
    new = sorted(set(dm["merged"]) - set(ds["merged"]))
    if new:
        reasons.append("new MERGED with " + ", ".join(f"{x}({dm['merged'][x]})" for x in new))
    return reasons


def find_worse(diags, label_family):
    """Labels that got worse.  Public labels: every other bench (all mixed, each family
    subset) vs. 'public only'.  Injected labels: all mixed vs. their own family subset.
    Only degradations are reported (more buckets, new foreign merges)."""
    out = []
    for lab in diags["all mixed"]:
        fam = label_family[lab]
        if fam == "public":
            pairs = [(b, "public only") for b in diags if b != "public only"]
        else:
            pairs = [("all mixed", fam)]
        for bench, ref in pairs:
            if lab not in diags[bench] or lab not in diags[ref]:
                continue
            reasons = _cmp(diags[bench][lab], diags[ref][lab])
            if reasons:
                out.append(dict(label=lab, family=fam, bench=bench, ref=ref, reasons=reasons))
    return out


def _notes(w, items, diag):
    """Which injected labels are involved in this worse entry and are they known exceptions / mismatches."""
    involved = {w["label"]} | set(diag[w["label"]]["merged"])
    notes = []
    for lab in sorted(involved):
        exc = sum(1 for i in items if i["label"] == lab and i["known_exc"])
        mis = sum(1 for i in items if i["label"] == lab and not i["ok"])
        if exc:
            notes.append(f"{lab}: {exc} known-exception case(s) (c.addi4spn -> WRONG_RESULT, generator artefact)")
        if mis:
            notes.append(f"{lab}: {mis} deviation-mismatch case(s)")
    return notes


def evaluate(bench_dir):
    """Run the pipeline on a merged bench and return a result dict."""
    with open(os.path.join(bench_dir, "meta.json")) as f:
        meta = json.load(f)
    cases = {str(c["case"]): c for c in meta["cases"]}
    golden = _read_case_label_csv(os.path.join(bench_dir, "golden.csv"), ("bug",))
    records = pipeline.parse_all(os.path.join(bench_dir, "input.csv"))
    res = dict(bars=[], diag={}, worse=[], dev_mismatch=[], dev_dist={}, cases=[])

    def row(name, recs):
        k, g, bucket, ba = _run(recs, golden)
        res["bars"].append(dict(name=name, k=k, n=len(recs), ba=ba))
        return g, bucket, ba

    g_all, b_all, ba_all = row("all mixed", records)
    shuffled = list(records)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    _, b_sh, ba_sh = row("all mixed (records shuffled)", shuffled)  # parse happens once; order-insensitivity only
    unstable = ba_sh != ba_all or _partition(b_sh) != _partition(b_all)
    res["unstable"] = unstable
    if unstable:
        res["bars"][-1]["name"] += " UNSTABLE"
    res["diag"]["all mixed"] = diagnose(g_all, b_all)

    def subset(fam):
        return [r for r in records if cases[str(r.case_id)]["family"] in ("public", fam)] if fam else \
               [r for r in records if cases[str(r.case_id)]["family"] == "public"]

    for fam in FAMILIES:
        if not any(c["family"] == fam for c in cases.values()):
            continue
        g, b, ba = row(f"public + {fam}", subset(fam))
        res["diag"][fam] = diagnose(g, b)
    g, b, _ = row("public only", subset(None))
    res["diag"]["public only"] = diagnose(g, b)

    for r in records:
        c = cases[str(r.case_id)]
        if c["origin"] not in ("rf", "br"):
            continue
        d = r.deviation
        kind, mn = d.kind, d.mnemonic
        ok = _dev_expect_ok(c, kind, mn)
        lab = c["label"]
        res["dev_dist"].setdefault(lab, {})
        res["dev_dist"][lab][kind] = res["dev_dist"][lab].get(kind, 0) + 1
        item = dict(case=c["case"], label=lab, family=c["family"], kind=kind, mnemonic=mn,
                    mclass=d.mclass, bucket_mixed=b_all[str(c["case"])], ok=ok, known_exc=_known_exc(c))
        if c["family"] == "branch":
            item["expected"] = c["gen_meta"]["expected"]
        res["cases"].append(item)
        if not ok:
            res["dev_mismatch"].append(item)
    label_family = {c["label"]: c["family"] for c in cases.values()}
    res["worse"] = find_worse(res["diag"], label_family)
    for w in res["worse"]:
        w["notes"] = _notes(w, res["cases"], res["diag"][w["bench"]])
    res["warnings"] = list(meta.get("warnings", []))
    res["k"] = len(set(golden.values()))
    res["n"] = len(records)
    return res


def _table(rows, head):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def render(variant, res):
    L = [f"## Variant: {variant}", "", f"k = {res['k']}, cases = {res['n']}", ""]
    L += [f"**{w}**" for w in res["warnings"]] + [""] * bool(res["warnings"])
    L += _table([(b["name"], b["k"], b["n"], f"{b['ba']:.4f}") for b in res["bars"]], ["bench", "k", "cases", "BA"])
    L += ["", "### Worse when mixed (injected: vs. own family bench; public: vs. public only)", ""]
    L += [f"- {w['label']} [{w['family']}] in '{w['bench']}' vs '{w['ref']}': " + "; ".join(w["reasons"])
          + (" | " + "; ".join(w["notes"]) if w.get("notes") else "") for w in res["worse"]] or ["- (none)"]
    L += ["", "### SPLIT / MERGED (all mixed)", ""]
    n0 = len(L)
    pub = res["diag"]["public only"]
    for lab, d in res["diag"]["all mixed"].items():
        if d["split"] or d["merged"]:
            p = pub.get(lab)  # None for injected labels: everything is new
            tag = lambda already: " (already in public only)" if p is not None and already else ""  # noqa: E731
            L.append(f"- {lab} ({d['cases']} cases): {d['buckets']} bucket(s)"
                     + (" SPLIT" + tag(p["split"] if p else False) if d["split"] else "")
                     + ("; MERGED with " + ", ".join(f"{k}({v}){tag(k in p['merged'] if p else False)}"
                                                      for k, v in sorted(d["merged"].items())) if d["merged"] else ""))
    if len(L) == n0:
        L.append("- (none)")
    L += ["", "### Deviation kind per injected bug", ""]
    L += [f"- {lab}: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())) for lab, d in sorted(res["dev_dist"].items())]
    L += ["", "### Deviation mismatches (vs. expectation)", ""]
    L += [f"- case {m['case']} {m['label']}: got {m['kind']}:{m['mnemonic']}"
          + (f", expected {m['expected']}" if "expected" in m else "") for m in res["dev_mismatch"]] or ["- (none)"]
    return L + [""]


def run_all(work, per_bug=5, rf_bugs=None, br_bugs=None, set1=True, variants=("thick", "thin")):
    _check_bugs(list(DEFAULT_RF_BUGS if rf_bugs is None else rf_bugs),
                list(DEFAULT_BR_BUGS if br_bugs is None else br_bugs))
    base.prepare_out_dir(work)
    work = os.path.abspath(work)
    md = ["# Mixed bench evaluation", "",
          f"per_bug={per_bug}, set1={'yes' if set1 else 'no'}, shuffle seed={SHUFFLE_SEED}", ""]
    summary, br_dir = [], None
    for v in variants:
        vdir = os.path.join(work, v)
        bench = build_bench(vdir, per_bug, rf_bugs, br_bugs, set1, thin=(v == "thin"), br_dir=br_dir)
        if br_dir is None and os.path.isdir(os.path.join(vdir, "gen_br")):
            br_dir = os.path.join(vdir, "gen_br")  # branch is always thick: reuse
        res = evaluate(bench)
        with open(os.path.join(vdir, "cases.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        md += render(v, res)
        summary.append(f"[{v}] k={res['k']} n={res['n']} " + "; ".join(f"{b['name']}={b['ba']:.4f}" for b in res["bars"])
                       + f"; worse={len(res['worse'])}; dev_mismatch={len(res['dev_mismatch'])}")
        summary += [f"[{v}] {w}" for w in res["warnings"]]
    with open(os.path.join(work, "report.md"), "w") as f:
        f.write("\n".join(md))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-bug", type=int, default=5)
    ap.add_argument("--rf-bugs", default=",".join(DEFAULT_RF_BUGS))
    ap.add_argument("--br-bugs", default=",".join(DEFAULT_BR_BUGS))
    ap.add_argument("--no-set1", action="store_true")
    ap.add_argument("--variants", default="thick,thin")
    a = ap.parse_args(argv)
    variants = [v for v in a.variants.split(",") if v]
    if not variants or set(variants) - {"thick", "thin"}:
        sys.exit("--variants must be a subset of thick,thin")
    split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    try:
        for line in run_all(a.work, a.per_bug, split(a.rf_bugs), split(a.br_bugs), not a.no_set1, variants):
            print(line)
    except base.InputError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
