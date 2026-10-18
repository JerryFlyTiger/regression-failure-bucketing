#!/usr/bin/env python3
"""Synthetic register-file / forwarding bug injection (READ_MISMATCH evaluation).

A taint-propagating simulation on the clean prefix (< E) of public cases keeps
the buggy world self-consistent: printed reads carry the buggy value, ALU
consumers are recomputed with it, and the trace is cut where behaviour can no
longer be synthesised.  Bugs (registers are x-names, "armed" = from line
arm_from on):

  F  : stale forwarding on any register.  When an instruction reads a register
       that the previous instruction wrote with a different value than the
       one printed, the read token is replaced by the value the register had
       *before* that previous write (the forwarding path missed the new value).
  F2 : same as F, restricted to x2.
  R2 : regfile write corruption on x2: the value stored has bit 4 cleared
       (only when bit 4 was set).  The printed write value stays the correct
       one; later reads of x2 print the corrupted value.
  Rc : same as R2 on x8..x15 (the compressed-register window).
  inj_*: any ALU bug of tools/inject_alu_real.BUGS, run through the same
       propagation (result recomputed from tainted reads, then bug function).

Per line: read tokens `xN:0x..` get the tainted/stale value; ALU-class lines
(ADDSUB LOGIC SHIFT SLT MUL DIV, except auipc) are recomputed from the new
reads and the printed write tokens `xN=0x..` follow; lines that are not
recomputable are only allowed when no read changed.  A taint map records the
registers whose stored value differs from the original trace.

Truncation: the simulation stops at the first line whose behaviour cannot be
synthesised: a branch whose outcome would change (or cannot be decided), a
load/store/jump/CSR (anything non-ALU) with a tainted read, or an ALU line that
cannot be recomputed.  If nothing visible happened before the cut, arming is
moved past the cut and the simulation restarts (up to 40 times).  With
thin=True the trace is also cut 20 lines after the anchor (thin evidence).

Anchor = first line whose printed write differs from the original trace.
regr.log gets a Mismatch[1] block anchored there: ibex side = the buggy
(printed) write, spike side = the original write; idx counts earlier lines with
a non-x0 write; "[FAILED]: max(idx-6,0) matched, <later writes + 1> mismatch".

Limitations: because of the cut rule the evidence after the anchor is thinner
than in a real regression, and a bug that never produces a visible different
write in any clean prefix yields no case.  Sources are chosen per bug by most
injected events (then set, case number), one per distinct sim test name.

Deterministic (no randomness).  Reads golden.csv only to label sources and
background.  Reuses inject_alu_real (load_sources, prepare_out_dir, ...).

CLI: inject_regfile.py --out DIR (--scenario S_mix | --bugs F,R2,..) [--thin]
     [--per-bug N] [--bg-cases 2,4,..] [--src-cases 2:4,..]
Scenarios: S_F, S_F2, S_R2, S_Rc, S_F_a4, S_R2_a4, S_mix, S_alu, T_F_a4, T_mix
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

M32 = 0xFFFFFFFF
ALU = ("ADDSUB", "LOGIC", "SHIFT", "SLT", "MUL", "DIV")
_MIX = ["F", "R2", "Rc", "inj_addi4spn", "inj_caddi_neg", "inj_sub_borrow"]
# scenario name -> (bugs, thin)
SCENARIOS = {
    "S_F": (["F"], False), "S_F2": (["F2"], False), "S_R2": (["R2"], False), "S_Rc": (["Rc"], False),
    "S_F_a4": (["F", "inj_addi4spn"], False),
    "S_R2_a4": (["R2", "inj_addi4spn"], False),
    "S_mix": (_MIX, False),
    "S_alu": (list(base.BUGS), False),
    "T_F_a4": (["F", "inj_addi4spn"], True),
    "T_mix": (_MIX, True),
}
BUG_NAMES = ("F", "F2", "R2", "Rc") + tuple(base.BUGS)


def bit4_clear(v):
    return v & ~0x10 if v & 0x10 else None

# bug spec: dict(kind=R|F|A, regs=set or None, corrupt=fn, mset, fn)
BUGSPEC = {
    "R2": dict(kind="R", regs={"x2"}, corrupt=bit4_clear),
    "Rc": dict(kind="R", regs={f"x{i}" for i in range(8, 16)}, corrupt=bit4_clear),
    "F":  dict(kind="F", regs=None),
    "F2": dict(kind="F", regs={"x2"}),
}
for tag, (mset, fn, _c, _d) in base.BUGS.items():
    BUGSPEC[tag] = dict(kind="A", mset=mset, fn=fn)

def retok(line, reads_new, writes_new):
    parts = line.split("\t"); k = 6 if len(parts) >= 7 else 5
    toks = parts[k].split(" ")
    for i, t in enumerate(toks):
        if ":0x" in t and t.startswith("x"):
            r = t.split(":")[0]
            if r in reads_new: toks[i] = f"{r}:0x{reads_new[r]:08x}"
        elif "=0x" in t and t.startswith("x"):
            r = t.split("=")[0]
            if r in writes_new: toks[i] = f"{r}=0x{writes_new[r]:08x}"
    parts[k] = " ".join(toks)
    return "\t".join(parts)

def simulate(src, spec, arm_from, thin):
    instrs = src["instrs"][: src["E"]]
    known, taint = {}, {}
    prev = None  # dict r -> buggy value before prev instr's write
    out, anchor, events, cut = [], None, 0, None
    for j, ins in enumerate(instrs):
        armed = j >= arm_from
        mn, ops = ins.mnemonic, isa._split_operands(ins.operands)
        origr = dict(ins.reads)
        if mn == "c.addi4spn" and "x2" not in ins.reads and "x2" in known:
            origr["x2"] = known["x2"]
        rb = {r: taint.get(r, v) for r, v in origr.items()}
        if spec["kind"] == "F" and armed and prev is not None:
            for r in rb:
                if r in prev and (spec["regs"] is None or r in spec["regs"]) and prev[r] is not None and prev[r] != rb[r]:
                    rb[r] = prev[r]; events += 1
        changed = any(rb[r] != origr[r] for r in rb)
        cls = isa.mclass(mn)
        wn = dict(ins.writes)
        ok = True
        if cls in ALU and mn != "auipc":
            res = isa._arith_operands(mn, ops, rb)
            if res is None:
                ok = not changed
            else:
                rd, val = res; val &= M32
                if spec["kind"] == "A" and armed and mn in spec["mset"] and rd not in ("x0",):
                    nv = spec["fn"](mn, ops, rb, val)
                    if nv is not None and (nv & M32) != val:
                        val = nv & M32; events += 1
                if rd != "x0" and rd in wn:
                    wn[rd] = val
        elif changed:
            if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE"):
                p = isa._branch_operands(mn, ops)
                if p is None: ok = False
                else:
                    r1, r2, _t = p
                    def tk(rr):
                        a = isa._regval(rr, r1); b = isa._regval(rr, r2) if r2 is not None else 0
                        return None if a is None or b is None else isa._branch_taken(mn, a, b)
                    ok = tk(origr) is not None and tk(origr) == tk(rb)
            else:
                ok = False
        if not ok:
            cut = j; break
        # printed write differs from original -> mismatch point
        if anchor is None and any(r != "x0" and wn[r] != ins.writes[r] for r in wn):
            anchor = j
        # regfile store (R corruption) and taint update
        before = {}
        for r in wn:
            if r == "x0": continue
            before[r] = taint.get(r, known.get(r))
            stored = wn[r]
            if spec["kind"] == "R" and armed and r in spec["regs"]:
                c = spec["corrupt"](stored)
                if c is not None: stored = c; events += 1
            if stored != ins.writes[r]: taint[r] = stored
            else: taint.pop(r, None)
        prev = before if spec["kind"] == "F" else None
        for r, v in ins.reads.items(): known[r] = v
        for r, v in ins.writes.items(): known[r] = v
        out.append(retok(src["body"][j], {r: rb[r] for r in ins.reads}, wn))
        if thin and anchor is not None and j >= anchor + 20:
            break
    return out, anchor, events, cut

def build_case(src, spec, thin):
    arm = 0
    for _ in range(40):
        out, anchor, events, cut = simulate(src, spec, arm, thin)
        if anchor is not None:
            return out, anchor, events
        if cut is None:
            return None
        arm = cut + 1
    return None

def write_case(cdir, src, out, a):
    ins = src["instrs"][a]
    orig = ins.writes
    new = isa.parse_regs(out[a].split("\t")[6 if len(out[a].split("\t")) >= 7 else 5])[1]
    rd = next(r for r in new if r != "x0" and new[r] != orig[r])
    idx = sum(1 for x in src["instrs"][:a] if any(k != "x0" for k in x.writes))
    after_w = sum(1 for x in src["instrs"][a + 1:len(out)] if any(k != "x0" for k in x.writes))
    ops = isa._split_operands(ins.operands)
    spike_ops = ", ".join(abi(o) if o.startswith("x") and o[1:].isdigit() else o for o in ops)
    regr = (f"Mismatch[1]:\n"
            f"ibex[{idx}] : pc[{ins.pc:08x}] {ins.mnemonic}\t{ins.operands}: {abi(rd)}:{new[rd]:08x}\n"
            f"spike[{idx}] : pc[ffffffff{ins.pc:08x}] {ins.mnemonic:<7} {spike_ops}: {abi(rd)}:{orig[rd]:08x}\n"
            f"[FAILED]: {max(idx - 6, 0)} matched, {after_w + 1} mismatch")
    os.makedirs(cdir)
    with open(os.path.join(cdir, "regr.log"), "w") as f:
        f.write(regr)
    shutil.copyfile(src["sim"] if src["family"] == "MISMATCH" else PASSED_SIM, os.path.join(cdir, "sim.log.gz"))
    with open(os.path.join(cdir, "trace.log.gz"), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write((src["header"] + "\n" + "\n".join(out) + "\n").encode())
    return ins.mnemonic



def _build(bugs, srcs, out_dir, thin, per_bug, bg_cases):
    table, meta = [], []
    g2 = golden(2)
    for r in rows(2):
        if bg_cases is not None and int(r["Case"]) not in bg_cases:
            continue
        table.append((int(r["Case"]), os.path.join(SET[2], r["Regr Log"]), os.path.join(SET[2], r["Sim Log"]),
                      os.path.join(SET[2], r["Trace Log"]), g2[r["Case"]]))
    cid = 26
    for tag in bugs:
        spec = BUGSPEC[tag]
        cands = []
        for src in srcs:
            res = build_case(src, spec, thin)
            if res:
                cands.append((res[2], src, res))
        cands.sort(key=lambda t: (-t[0], t[1]["set"], int(t[1]["case"])))
        tests, n = set(), 0
        for ev, src, (out, a, _e) in cands:
            if src["test"] in tests:
                continue
            tests.add(src["test"])
            cdir = os.path.join(out_dir, f"case_{cid}")
            amn = write_case(cdir, src, out, a)
            meta.append(dict(case=cid, bug=tag, src=f"set{src['set']}/case_{src['case']}", anchor_mn=amn,
                             events=ev, anchor=a, length=len(out)))
            table.append((cid, os.path.join(cdir, "regr.log"), os.path.join(cdir, "sim.log.gz"),
                          os.path.join(cdir, "trace.log.gz"), tag))
            cid += 1
            n += 1
            if n == per_bug:
                break
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


def generate(out_dir, bugs=None, thin=False, scenario=None, per_bug=5, bg_cases=None, src_cases=None):
    """bg_cases None = every set-2 case is background (the architect's runs);
    an empty set = no background.  scenario (a SCENARIOS key) overrides bugs
    and thin."""
    if scenario is not None:
        if scenario not in SCENARIOS:
            raise InputError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        bugs, thin = SCENARIOS[scenario]
    bugs = list(bugs or [])
    bad = [b for b in bugs if b not in BUG_NAMES]
    if bad or not bugs:
        raise InputError(f"bugs must be a non-empty subset of {BUG_NAMES}, got {bugs}")
    if isinstance(per_bug, bool) or not isinstance(per_bug, int) or per_bug < 1:
        raise InputError(f"per_bug must be an integer >= 1, got {per_bug!r}")
    if bg_cases:
        have = {int(r["Case"]) for r in rows(2)}
        missing = sorted(set(bg_cases) - have)
        if missing:
            raise InputError("background case(s) do not exist in set 2: " + ", ".join(map(str, missing)))
    prepare_out_dir(out_dir)
    srcs = load_sources(src_cases)
    return _build(bugs, srcs, os.path.abspath(out_dir), bool(thin), per_bug, bg_cases)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", help="one of " + ",".join(SCENARIOS))
    g.add_argument("--bugs", help="comma list from " + ",".join(BUG_NAMES))
    ap.add_argument("--thin", action="store_true", help="with --bugs: cut 20 lines after the anchor")
    ap.add_argument("--per-bug", type=base._per_bug, default=5)
    ap.add_argument("--bg-cases", type=base._bg_cases, default=None,
                    help="set 2 cases kept as background (default: all of set 2)")
    ap.add_argument("--src-cases", type=base._src_cases, default=None, help="restrict sources, e.g. 2:4,2:7")
    a = ap.parse_args(argv)
    if a.scenario and a.thin:
        ap.error("--thin only applies with --bugs (a scenario fixes its own thin setting)")
    try:
        k = generate(a.out, [b for b in (a.bugs or "").split(",") if b], a.thin, a.scenario, a.per_bug,
                     a.bg_cases, a.src_cases)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"k={k}, wrote {a.out}")


if __name__ == "__main__":
    main()
