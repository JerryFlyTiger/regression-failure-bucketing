#!/usr/bin/env python3
"""Synthetic branch-comparator bug injection (BR_EQ split evaluation).

Keeps the source trace's real control flow and rewrites only the *reads* of
beq/bne/c.beqz/c.bnez lines so that the checker sees "condition contradicts
the observed next PC".  Bugs:

  BEQ_ONLY : beq/c.beqz that really did not jump (rs1 != rs2): reads made
             equal            -> BRANCH_NOT_TAKEN:beq / c.beqz
  BNE_ONLY : bne/c.bnez that really did not jump (rs1 == rs2): reads made
             unequal          -> BRANCH_NOT_TAKEN:bne / c.bnez
  EQ_BOTH  : shared is_equal comparator (ibex_alu.sv EQ/NE share is_equal):
             the BEQ_ONLY rewrite on beq/c.beqz, plus bne/c.bnez that really
             jumped (rs1 != rs2) made equal -> BRANCH_WRONGLY_TAKEN:bne
  EQ_BOTH_FP: is_equal wrongly true (opposite polarity): beq/c.beqz that
             really jumped (rs1 == rs2) made unequal -> BRANCH_WRONGLY_TAKEN:
             beq / c.beqz, plus the BNE_ONLY rewrite on bne/c.bnez
  BNE_WT_ONLY: bne/c.bnez that really jumped (rs1 != rs2) made equal ->
             BRANCH_WRONGLY_TAKEN:bne / c.bnez (single-instruction bug; with
             BEQ_ONLY it is the polarity-grouping worst case, scenario G)

Every hit line in the clean prefix (< E) of a source case is rewritten (a
systematic bug).  regr.log gets a Mismatch[1] block anchored at the first
instruction with a register write after the first hit: ibex side = that trace
line, spike side pc = the correct path (target if the branch should have
jumped, fallthrough otherwise), spike text = a trace line with the same PC if
one exists, else "nop" (recorded in meta.json as spike_text = "nop").  The
trace is the source's clean prefix (like inject_alu_real's "real" variant).

Deterministic (no randomness).  Reads golden.csv only to label sources.
Reuses inject_alu_real (load_sources, prepare_out_dir, ...).

CLI: inject_branch.py --out DIR [--bugs BEQ_ONLY,BNE_ONLY,EQ_BOTH,EQ_BOTH_FP,BNE_WT_ONLY]
     [--per-bug N] [--bg-cases 2,4,..] [--src-cases 2:4,..]
--out must not exist or be an empty directory; nothing is ever deleted; a
path at or under testcase/ is refused (exit 1).
"""
import argparse
import gzip
import json
import os
import shutil
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import inject_alu_real as base  # noqa: E402
from inject_alu_real import (  # noqa: E402
    InputError, PASSED_SIM, SET, abi, golden, isa, load_sources, prepare_out_dir, rows,
)
from rfb.parse_trace import _pass1_trap_entries  # noqa: E402

M32 = 0xFFFFFFFF
BEQ_SET = {"beq", "c.beqz"}
BNE_SET = {"bne", "c.bnez"}
# same background as tests/test_alu_injection_regression.py
BG_CASES = {2, 4, 6, 7, 8, 10, 12, 16, 17, 18, 20, 24}
BUG_NAMES = ("BEQ_ONLY", "BNE_ONLY", "EQ_BOTH", "EQ_BOTH_FP", "BNE_WT_ONLY")


def rewrite_read(line, reg, new):
    """Replace the `reg:0x..` read token of a trace line (write tokens
    `reg=0x..`, including a trailing `x0=0x00000000`, are untouched)."""
    parts = line.split("\t")
    k = 6 if len(parts) >= 7 else 5
    toks = parts[k].split(" ")
    hit = [i for i, t in enumerate(toks) if t.startswith(f"{reg}:0x")]
    assert len(hit) == 1, line
    toks[hit[0]] = f"{reg}:0x{new & M32:08x}"
    parts[k] = " ".join(toks)
    return "\t".join(parts)


def branch_hit(src, traps, i, cls, want):
    """cls 'beq'|'bne'; want 'nt' (really not taken -> NOT_TAKEN) or 'wt'
    (really taken and rewritten -> WRONGLY_TAKEN: bne made equal, beq made
    unequal).  Returns
    (idx, reg, newval, kind, mnemonic, correct_pc) or None."""
    instrs = src["instrs"]
    ins = instrs[i]
    if i + 1 >= src["E"]:
        return None
    ops = isa._split_operands(ins.operands)
    parsed = isa._branch_operands(ins.mnemonic, ops)
    if parsed is None:
        return None
    rs1, rs2, target = parsed
    if rs1 == rs2 or target is None:
        return None
    a, b = isa._regval(ins.reads, rs1), isa._regval(ins.reads, rs2)
    fall = (ins.pc + ins.size) & M32
    nxt = instrs[i + 1].pc
    if a is None or b is None or target == fall or nxt in traps:
        return None
    taken = nxt == target
    if taken != isa._branch_taken(ins.mnemonic, a, b):  # source line must be consistent
        return None
    if cls == "beq" and want == "nt":
        if taken:
            return None
        mk_eq, kind, correct = True, "BRANCH_NOT_TAKEN", target
    elif cls == "bne" and want == "nt":
        if taken:
            return None
        mk_eq, kind, correct = False, "BRANCH_NOT_TAKEN", target
    elif cls == "bne":  # wt
        if not taken:
            return None
        mk_eq, kind, correct = True, "BRANCH_WRONGLY_TAKEN", fall
    else:  # beq, wt
        if not taken:
            return None
        mk_eq, kind, correct = False, "BRANCH_WRONGLY_TAKEN", fall
    # modify one operand's read: prefer rs2, never x0 / x2 (x2 would desync the sp shadow)
    for reg, other in ((rs2, a), (rs1, b)):
        if reg not in ("x0", "x2"):
            return (i, reg, other if mk_eq else (other + 1) & M32, kind, ins.mnemonic, correct)
    return None


def compute_hits(src, traps):
    """Per source: dict of hit lists beq_nt / bne_nt / bne_wt / beq_wt (index order)."""
    out = {"beq_nt": [], "bne_nt": [], "bne_wt": [], "beq_wt": []}
    for i, ins in enumerate(src["instrs"][: src["E"]]):
        if ins.mnemonic in BEQ_SET:
            h = branch_hit(src, traps, i, "beq", "nt")
            if h:
                out["beq_nt"].append(h)
            h = branch_hit(src, traps, i, "beq", "wt")
            if h:
                out["beq_wt"].append(h)
        elif ins.mnemonic in BNE_SET:
            h = branch_hit(src, traps, i, "bne", "nt")
            if h:
                out["bne_nt"].append(h)
            h = branch_hit(src, traps, i, "bne", "wt")
            if h:
                out["bne_wt"].append(h)
    return out


def _has_write(ins):
    return any(k != "x0" for k in ins.writes)


def _candidate(src, hits):
    """(hits sorted, anchor index) or None when no usable anchor exists."""
    if not hits:
        return None
    hits = sorted(hits)
    a = hits[0][0]
    anchor = next((j for j in range(a + 1, src["E"]) if _has_write(src["instrs"][j])), None)
    if anchor is None:
        return None
    return hits, anchor


def choose(bug, srcs, per_bug, used):
    """Deterministic source selection; returns [(src, hits, anchor)]."""
    cands = []
    for src in srcs:
        h = src["_hits"]
        if bug == "BEQ_ONLY":
            hits, cat = h["beq_nt"], "x"
        elif bug == "BNE_ONLY":
            hits, cat = h["bne_nt"], "x"
        elif bug == "BNE_WT_ONLY":
            hits, cat = h["bne_wt"], "x"
        elif bug == "EQ_BOTH":
            hits = h["beq_nt"] + h["bne_wt"]
            nb, nw = len(h["beq_nt"]), len(h["bne_wt"])
            cat = "both" if nb and nw else ("beq" if nb else "bne")
        else:  # EQ_BOTH_FP
            hits = h["beq_wt"] + h["bne_nt"]
            nb, nw = len(h["beq_wt"]), len(h["bne_nt"])
            cat = "both" if nb and nw else ("beq" if nb else "bne")
        c = _candidate(src, hits)
        if c:
            cands.append((cat, src, c[0], c[1]))
    def key_default(t):
        return ((t[1]["set"], t[1]["case"]) in used, -len(t[2]), t[1]["set"], int(t[1]["case"]))
    key = key_default
    if bug in ("EQ_BOTH", "EQ_BOTH_FP"):
        k1, k2 = ("beq_nt", "bne_wt") if bug == "EQ_BOTH" else ("beq_wt", "bne_nt")
        # both-kind sources rank by the weaker side so both classes are well represented
        def key_eq_both(t):
            h = t[1]["_hits"]
            return ((t[1]["set"], t[1]["case"]) in used, -min(len(h[k1]), len(h[k2])),
                    -len(t[2]), t[1]["set"], int(t[1]["case"]))
        key = key_eq_both
        # single-class first: the distinct-test rule would otherwise starve them
        quota = [("bne", 1), ("beq", 1), ("both", per_bug - 2)]
    else:
        quota = [("x", per_bug)]
    chosen, tests = [], set()
    for cat, n in quota:
        got = 0
        n = min(n, per_bug - len(chosen))  # never more than per_bug cases (EQ_BOTH*, small per_bug)
        if n <= 0:
            continue
        for t in sorted((c for c in cands if c[0] == cat), key=key):
            if t[1]["test"] in tests:
                continue
            chosen.append(t[1:]); tests.add(t[1]["test"]); got += 1
            if got == n:
                break
    for src, _, _ in chosen:
        used.add((src["set"], src["case"]))
    return chosen


def _write_rd(ins):
    rd = next(k for k in ins.writes if k != "x0")
    return rd, ins.writes[rd]


def _build(bugs, srcs, out_dir, per_bug, bg_cases):
    table = []
    g2 = golden(2)
    for r in rows(2):
        if bg_cases is not None and int(r["Case"]) not in bg_cases:
            continue
        table.append((int(r["Case"]), os.path.join(SET[2], r["Regr Log"]), os.path.join(SET[2], r["Sim Log"]),
                      os.path.join(SET[2], r["Trace Log"]), g2[r["Case"]]))
    for src in srcs:
        _, traps, _ = _pass1_trap_entries(iter([src["header"]] + src["body"]))
        src["_hits"] = compute_hits(src, set(traps))
    cid, meta, used = 26, [], set()
    for bug in bugs:
        for src, hits, anchor in choose(bug, srcs, per_bug, used):
            hmap = {h[0]: h for h in hits}
            end = src["E"]
            body = []
            for i in range(end):
                ln = src["body"][i]
                if i in hmap:
                    ln = rewrite_read(ln, hmap[i][1], hmap[i][2])
                body.append(ln)
            first = hits[0]
            anc = src["instrs"][anchor]
            rd, val = _write_rd(anc)
            idx = sum(1 for ins in src["instrs"][:anchor] if _has_write(ins))
            after_w = sum(1 for ins in src["instrs"][anchor + 1:end] if _has_write(ins))
            spike_pc = first[5]
            same = next((ins for ins in src["instrs"] if ins.pc == spike_pc), None)
            if same is not None:
                sops = ", ".join(abi(o) if o.startswith("x") and o[1:].isdigit() else o
                                 for o in isa._split_operands(same.operands))
                spike_txt = f"{same.mnemonic:<7} {sops}"
                if _has_write(same):
                    srd, sval = _write_rd(same)
                    spike_txt += f": {abi(srd)}:{sval:08x}"
                spike_src = "trace"
            else:
                spike_txt, spike_src = "nop", "nop"
            regr = (f"Mismatch[1]:\n"
                    f"ibex[{idx}] : pc[{anc.pc:08x}] {anc.mnemonic}\t{anc.operands}: {abi(rd)}:{val:08x}\n"
                    f"spike[{idx}] : pc[ffffffff{spike_pc:08x}] {spike_txt}\n"
                    f"[FAILED]: {max(idx - 6, 0)} matched, {after_w + 1} mismatch")
            cdir = os.path.join(out_dir, f"case_{cid}")
            os.makedirs(cdir)
            with open(os.path.join(cdir, "regr.log"), "w") as f:
                f.write(regr)
            shutil.copyfile(src["sim"] if src["family"] == "MISMATCH" else PASSED_SIM,
                            os.path.join(cdir, "sim.log.gz"))
            with open(os.path.join(cdir, "trace.log.gz"), "wb") as raw, \
                    gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write((src["header"] + "\n" + "\n".join(body) + "\n").encode())
            meta.append(dict(
                case=cid, bug=bug, src=f"set{src['set']}/case_{src['case']}", src_bug=src["bug"],
                src_family=src["family"], trace_len=end, anchor_line=anchor, anchor_mn=anc.mnemonic,
                first_hit_line=first[0], expected=f"{first[3]}:{first[4]}", spike_text=spike_src,
                hits=[dict(line=h[0], mn=h[4], kind=h[3], reg=h[1]) for h in hits],
                hit_kinds=sorted({f"{h[3]}:{h[4]}" for h in hits}),
                n_beq_nt=sum(1 for h in hits if h[3] == "BRANCH_NOT_TAKEN" and h[4] in BEQ_SET),
                n_bne_nt=sum(1 for h in hits if h[3] == "BRANCH_NOT_TAKEN" and h[4] in BNE_SET),
                n_bne_wt=sum(1 for h in hits if h[3] == "BRANCH_WRONGLY_TAKEN" and h[4] in BNE_SET),
                n_beq_wt=sum(1 for h in hits if h[3] == "BRANCH_WRONGLY_TAKEN" and h[4] in BEQ_SET)))
            table.append((cid, os.path.join(cdir, "regr.log"), os.path.join(cdir, "sim.log.gz"),
                          os.path.join(cdir, "trace.log.gz"), bug))
            cid += 1
    with open(os.path.join(out_dir, "input.csv"), "w") as f:
        f.write("Case,Regr Log,Sim Log,Trace Log\n")
        for t in table:
            f.write(f"{t[0]},{os.path.relpath(t[1], out_dir)},{os.path.relpath(t[2], out_dir)},"
                    f"{os.path.relpath(t[3], out_dir)}\n")
    with open(os.path.join(out_dir, "golden.csv"), "w") as f:
        f.write("Case,Bug\n")
        for t in table:
            f.write(f"{t[0]},{t[4]}\n")
    k = len({t[4] for t in table})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(k=k, cases=meta), f, indent=1)
    return k


def generate(out_dir, bugs=BUG_NAMES, per_bug=5, bg_cases=BG_CASES, src_cases=None):
    bugs = list(bugs)
    bad = [b for b in bugs if b not in BUG_NAMES]
    if bad or not bugs:
        raise InputError(f"--bugs must be a non-empty subset of {BUG_NAMES}, got {bugs}")
    if isinstance(per_bug, bool) or not isinstance(per_bug, int) or per_bug < 1:
        raise InputError(f"per_bug must be an integer >= 1, got {per_bug!r}")
    if bg_cases:
        have = {int(r["Case"]) for r in rows(2)}
        missing = sorted(set(bg_cases) - have)
        if missing:
            raise InputError("background case(s) do not exist in set 2: " + ", ".join(map(str, missing)))
    prepare_out_dir(out_dir)
    srcs = load_sources(src_cases)
    return _build(bugs, srcs, os.path.abspath(out_dir), per_bug, bg_cases)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bugs", default=",".join(BUG_NAMES), help="comma list of BEQ_ONLY,BNE_ONLY,EQ_BOTH,EQ_BOTH_FP,BNE_WT_ONLY")
    ap.add_argument("--per-bug", type=base._per_bug, default=5)
    ap.add_argument("--bg-cases", type=base._bg_cases, default=set(BG_CASES),
                    help="set 2 cases kept as background (default: the ALU regression test's BG_CASES)")
    ap.add_argument("--src-cases", type=base._src_cases, default=None, help="restrict sources, e.g. 2:4,2:7")
    a = ap.parse_args(argv)
    try:
        k = generate(a.out, [b for b in a.bugs.split(",") if b], a.per_bug, a.bg_cases, a.src_cases)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"k={k}, wrote {a.out}")


if __name__ == "__main__":
    main()
