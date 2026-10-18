#!/usr/bin/env python3
"""Quantify the cost of "post-divergence noise" violations on systematic-violation
grouping (sys_viol_class / vcls: R1 merge, R2 both_uvm_no_vcls, R3 _link, R5).

Noise injection: in a copy of a public case's trace.log(.gz), pick c control-flow
instructions with a chosen mnemonic (default jal) that currently have no violation
(next_pc is the target or the fallthrough, not a trap entry) and insert, right after
each, a fake non-control-flow line `addi x0,x0,0` (format copied from the trace) whose
PC is a unique address H absent from the original trace.  The control-flow
instruction's next_pc then becomes H (not target, not fallthrough, not a trap entry)
-> one BRANCH_WRONG_TARGET:<mnemonic> per site.  The original next line follows the
fake line unchanged.  For MISMATCH cases sites are chosen after the regr anchor
(clear of the anchor's 4-instruction after-window); only if there are too few are
sites before the anchor used (clear of the 16-instruction before-window).  Sites are
never in the first 10 instructions (head) and are placed outside the last 64 (tail)
whenever possible.

Scenarios (set 1: k=2, set 2: k=4; score = pairwise balanced accuracy of tools/score.py):
  N0 original public set (must be 1.0, else abort)
  N1 every case c=1             N2 every case c=2
  N3 only FATAL/ASSERT c=2      N4 one case per golden bug (smallest id with
                                 enough sites) c=2
  N5 every case c=2, one distinct mnemonic group per golden bug (no R1 collision)
  N6 every case c=5
  N7 every case c=5 "returning" noise (fake line at H, `mret` at H+4, then the original
     next line = an interrupt entered and returned from; trap_entries unchanged)
  N8 returning c=5, then one non-returning site (c=1) on the result
  N9  every case c=2 "vector" noise (like N2, but the fake line sits in an interrupt vector slot
     mtvec_base + 4k, k in {3, 7, 11, 16..31} (the slots Ibex interrupts use), instead of at
     the far-away H_BASE)
  N10 every case c=5 "vector" noise (like N6)
N2-N6 model the residual risk after rfb ignores WRONG_TARGET that is followed by an xret
return; N7/N8 model what that fix removes.  Verification of returning sites adapts to
whether rfb.isa.ScanContext has the `xret_returns` field.
Every injected case is verified through rfb's own parser/signature (violation count
+c, sys_viol_class membership, deviation/hard_key of MISMATCH cases unchanged); a
failed check aborts the run.  Reads golden.csv (experiment tool; rfb/ never does).
Writes <work>/report.md.  Never touches rfb/ or testcase/.

Usage: noise_eval.py [--work out/noise_eval]
"""
import argparse
import csv
import dataclasses
import gzip
import itertools
import bisect
import os
import re
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rfb import isa, pipeline  # noqa: E402
from rfb.io_utils import open_text  # noqa: E402
from rfb.parse_regr import parse_regr  # noqa: E402
from rfb.parse_trace import _pass1_trap_entries, _split_trace_line  # noqa: E402
from rfb.record import Instr  # noqa: E402
from rfb.signature import signature  # noqa: E402

PUB = os.path.join(REPO, "testcase/2026_08_20_samples/problem")
SET_K = {1: 2, 2: 4}
H_BASE = 0xA0000000
HEAD_SKIP = 10       # sites at instruction index < 10 are never used (head = first 8)
TAIL_N = 64          # prefer sites outside the last 64 lines (trace tail features)
AFTER_WIN = 4        # regr anchor after/before windows (parse_trace._ANCHOR_AFTER/_BEFORE)
BEFORE_WIN = 16
# mnemonic groups for N5 (viol_group of BRANCH_WRONG_TARGET == mclass); preference order
GROUPS = [("JUMP", ["jal", "c.j", "c.jal"]), ("BR_GE", ["bge", "bgeu"]), ("BR_LT", ["blt", "bltu"]),
          ("BR_BEQ", ["beq", "c.beqz"]), ("BR_BNE", ["bne", "c.bnez"])]
NOISE_KIND = "BRANCH_WRONG_TARGET"
MTVEC_NAMES = ("mtvec", "0x305")
# interrupt vector slots base + 4k Ibex really uses (ibex_pkg.sv:321-328; ibex_controller.sv:734-758;
# ibex_if_stage.sv:214-218); k = 0 is the synchronous-exception entry.  Same set as rfb.isa._IRQ_KS.
VEC_KS = (3, 7, 11, *range(16, 32))


class InjectionError(Exception):
    pass


# ---------------------------------------------------------------- trace handling
def read_trace(path):
    """Trace as a list of lines (no "\\n"); a trailing newline gives a final ''."""
    with open_text(path) as f:
        return f.read().split("\n")


def write_trace(path, lines):
    data = "\n".join(lines)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            f.write(data)
    else:
        with open(path, "wt", encoding="utf-8", newline="") as f:
            f.write(data)


def _to_instr(idx, line):
    f = _split_trace_line(line.rstrip("\r").split("\t"))
    if f is None:
        return None
    pc_hex, insn_hex, mn, ops, regs = f
    try:
        pc = int(pc_hex, 16)
    except ValueError:
        return None
    reads, writes, pa, store, load = isa.parse_regs(regs)
    return Instr(index=idx, pc=pc, size=2 if len(insn_hex) <= 4 else 4, mnemonic=mn, operands=ops,
                 reads=reads, writes=writes, pa=pa, store=store, load=load)


class Scan:
    """Parsed view of one trace: entries[i] = (line_no, Instr) for each instruction line."""

    def __init__(self, lines, regr_path):
        self.lines = lines
        self.entries = []
        for ln, line in enumerate(lines):
            if ln == 0 or not line.strip():
                continue
            ins = _to_instr(len(self.entries), line)
            if ins is not None:
                self.entries.append((ln, ins))
        self.traps = set(_pass1_trap_entries(iter(lines))[1])
        self.pcs = {ins.pc for _, ins in self.entries}
        # PC -> latest index of the XRET entry whose next line has that PC.  Same coordinate as
        # rfb.parse_trace._pass1_trap_entries' xret_returns (the xret's own index, not the return line's).
        self.xret_ret = {}
        for i in range(1, len(self.entries)):
            if isa.mclass(self.entries[i - 1][1].mnemonic) == "XRET":
                self.xret_ret[self.entries[i][1].pc] = i - 1
        # (entry index, mtvec base) of every csrrw/csrrwi mtvec, in trace order.  Independent of
        # rfb.isa.observe: base = written value & ~0xFF (csrrw: rs1's read value, csrrwi: decimal uimm).
        # Matches rfb only within csrrw/csrrwi; csrrs/csrrc are not tracked here, so between a
        # CSR set/clear and the next write the two bases may differ.  verify_vector's violation-count
        # assertion guards that case.
        self.mtvec_writes = []
        for i, (_, ins) in enumerate(self.entries):
            if ins.mnemonic not in ("csrrw", "csrrwi"):
                continue
            ops = ins.operands.split(",")
            if len(ops) != 3 or ops[1].strip().lower() not in MTVEC_NAMES:
                continue
            if ins.mnemonic == "csrrwi":
                val = int(ops[2].strip(), 10) if re.fullmatch(r"\s*\d+\s*", ops[2]) else None
            else:
                val = ins.reads.get(ops[2].strip())
            # an unreadable value or a base of 0 means "unknown" (None), as in rfb.isa for csrrw/csrrwi
            base = None if val is None else val & ~0xFF & 0xFFFFFFFF
            self.mtvec_writes.append((i, base or None))
        self._mtvec_idx = [i for i, _ in self.mtvec_writes]
        self.hits = []
        ri = parse_regr(regr_path) if regr_path else None
        if ri is not None and getattr(ri, "mismatches", None):
            m = ri.mismatches[0]
            mn0 = m.ibex_text.split()[0] if m.ibex_text.split() else ""
            for i, (_, ins) in enumerate(self.entries):
                if ins.pc == m.ibex_pc:
                    if m.ibex_val is not None:
                        if m.ibex_val in ins.writes.values():
                            self.hits.append(i)
                    elif ins.mnemonic == mn0:
                        self.hits.append(i)

    def mtvec_before(self, i):
        """mtvec base of the last csrrw/csrrwi mtvec before entry index i (None if never written,
        or if that write left the base unknown/0)."""
        j = bisect.bisect_left(self._mtvec_idx, i)
        return self.mtvec_writes[j - 1][1] if j else None

    def candidates(self, mn, mode="nonreturn"):
        """Entry indices of clean `mn` instructions usable as injection sites, in
        placement-priority order (after the regr anchor first, tail last).  With an rfb
        that has xret_returns, non-returning sites whose target/fallthrough is returned
        to by a *later* xret in the trace are dropped: rfb would suppress their
        WRONG_TARGET, and the noise would then no longer be the unreturned residual."""
        drop_returned = mode in ("nonreturn", "vector") and rfb_has_xret_returns()
        ctx = isa.ScanContext(trap_entries=set(self.traps))
        n = len(self.entries)
        out = []
        for i, (_, ins) in enumerate(self.entries):
            if ins.mnemonic != mn or i + 1 >= n or i < HEAD_SKIP:
                continue
            if mode == "vector" and self.mtvec_before(i) is None:
                continue
            nxt = self.entries[i + 1][1].pc
            if nxt in self.traps:
                continue
            v, err = isa.check_safe(None, ins, nxt, ctx)
            if v is not None or err is not None:
                continue
            target, fall = _target_fall(mn, ins)
            if target is None or nxt not in (target, fall):
                continue
            if drop_returned and max(self.xret_ret.get(target, -1), self.xret_ret.get(fall, -1)) > i:
                continue
            out.append(i)
        after_from = (max(self.hits) + AFTER_WIN + 1) if self.hits else 0
        before_to = (min(self.hits) - BEFORE_WIN - 2) if self.hits else -1
        tail_from = n - TAIL_N - 1

        def rank(i):
            if i >= after_from:
                grp = 0
            elif i <= before_to:
                grp = 1
            else:
                return None
            return (grp, i > tail_from, i if grp == 0 else -i)
        ranked = [(rank(i), i) for i in out]
        return [i for r, i in sorted(x for x in ranked if x[0] is not None)]


def _target_fall(mn, ins):
    """(target, fallthrough) of control-flow instruction `ins` (target None when unparsable)."""
    ops = isa._split_operands(ins.operands)
    if mn in ("jal", "c.j", "c.jal", "jalr", "c.jr", "c.jalr"):
        target = isa._jump_target(mn, ops, ins.reads)
    else:
        parsed = isa._branch_operands(mn, ops)
        target = None if parsed is None else parsed[2]
    return target, (ins.pc + ins.size) & 0xFFFFFFFF


def _template(scan):
    """(line, synthesized?) of a `addi x0,x0,0` line in the trace's own format."""
    for ln, ins in scan.entries:
        if ins.mnemonic == "addi" and ins.operands == "x0,x0,0":
            return scan.lines[ln], False
    first = scan.lines[scan.entries[0][0]].split("\t")
    first[3], first[4], first[5], first[6] = "00000013", "addi", "x0,x0,0", "  x0:0x00000000  x0=0x00000000"
    return "\t".join(first), True


def _mret_template(scan):
    """(line, synthesized?) of an `mret` line in the trace's own format (a real one
    when the trace has any, else built from the first line's columns)."""
    for ln, ins in scan.entries:
        if ins.mnemonic == "mret":
            return scan.lines[ln], False
    first = scan.lines[scan.entries[0][0]].split("\t")
    return "\t".join(first[:3] + ["30200073", "mret", ""]), True


def _fake_line(tparts, jparts, pc, cr):
    """Copy of template columns with the time/cycle columns of the site line and PC `pc`."""
    p = list(tparts)
    p[0], p[1] = jparts[0], jparts[1]
    p[2] = f"{pc:0{len(tparts[2])}x}"
    if cr and not p[-1].endswith("\r"):
        p[-1] += cr
    return "\t".join(p)


def inject_trace(scan, mn, c, mode="nonreturn"):
    """Return (new_lines, info).  info: n_req, n_inj, h (list of H), placed
    (list of (entry_idx, 'after'|'before', in_tail)), synth_template, synth_mret.

    mode "nonreturn": after the site insert one fake non-control-flow line at PC H
    (the interrupt is never returned from).  mode "returning": insert the fake line at
    H and then an `mret` line at PC H+4, followed by the original next line (the
    interrupt is entered and returned from to the site's correct next PC); H is chosen
    so that the original next PC is not H+8 (the mret's own fallthrough).
    mode "vector": like "nonreturn", but H = (mtvec base before the site) + 4k, k in VEC_KS, a
    PC absent from the trace; sites need a csrrw/csrrwi mtvec before them."""
    cand = scan.candidates(mn, mode)[:c]
    vec_next = 0
    tmpl, synth = _template(scan)
    tparts = tmpl.split("\t")
    cr = "\r" if tmpl.endswith("\r") else ""
    mparts, msynth = None, False
    if mode == "returning":
        mtmpl, msynth = _mret_template(scan)
        mparts = mtmpl.split("\t")
        if cr and not mparts[-1].endswith("\r"):
            mparts[-1] += cr
    n = len(scan.entries)
    after_from = (max(scan.hits) + AFTER_WIN + 1) if scan.hits else 0
    ins_after = {}
    hs = []
    placed = []
    used = set(scan.pcs)
    h = H_BASE
    for i in sorted(cand):
        nxt = scan.entries[i + 1][1].pc
        if mode == "vector":
            # slot base + 4k, k in VEC_KS, not an existing PC; rotate through the usable k's
            base = scan.mtvec_before(i)
            # H must also differ from the site's own target/fallthrough, else no violation at all
            skip = set(_target_fall(mn, scan.entries[i][1]))
            ks = [k for k in VEC_KS if base + 4 * k not in scan.pcs and base + 4 * k not in skip]
            if not ks:
                raise InjectionError(f"no free interrupt vector slot at mtvec base {base:#x}")
            hv = base + 4 * ks[vec_next % len(ks)]
            vec_next += 1
        else:
            while h in used or h + 4 in used or h + 8 == nxt:
                h += 0x40
            used.add(h)
            used.add(h + 4)
        ln = scan.entries[i][0]
        jparts = scan.lines[ln].split("\t")
        hh = hv if mode == "vector" else h
        add = [_fake_line(tparts, jparts, hh, cr)]
        if mode == "returning":
            add.append(_fake_line(mparts, jparts, h + 4, ""))
        ins_after[ln] = add
        hs.append(hh)
        placed.append((i, "after" if i >= after_from else "before", i > n - TAIL_N - 1))
    new = []
    for ln, line in enumerate(scan.lines):
        new.append(line)
        if ln in ins_after:
            new.extend(ins_after[ln])
    return new, dict(n_req=c, n_inj=len(cand), h=hs, placed=placed, synth_template=synth, synth_mret=msynth)


# ---------------------------------------------------------------- verification
def parse_case(case_id, case_dir, trace_path):
    return pipeline._parse_one_case(case_id, os.path.join(case_dir, "regr.log"),
                                    os.path.join(case_dir, "sim.log.gz"), trace_path)


def rfb_has_xret_returns():
    """True when rfb's ScanContext already has the xret_returns field (the fix that
    stops a WRONG_TARGET followed by an xret return from counting as a violation)."""
    return "xret_returns" in {f.name for f in dataclasses.fields(isa.ScanContext)}


def rfb_has_mtvec_base():
    """True when rfb's ScanContext tracks mtvec_base (the fix that stops a jump into an interrupt
    vector slot from counting as BRANCH_WRONG_TARGET)."""
    return "mtvec_base" in {f.name for f in dataclasses.fields(isa.ScanContext)}


def verify_vector(orig_scan, new_scan, orig, new, mn, info):
    """Fix-independent wiring assertions for "vector" sites.  Every site, checked with a
    ScanContext that has only trap_entries/xret_returns (mtvec_base None, i.e. what an rfb without
    the vector fix sees), must give BRANCH_WRONG_TARGET; each fake PC must be a legal slot of the
    mtvec base in force before the site (base + 4k, k in VEC_KS) and absent
    from the original trace.  When rfb tracks mtvec_base, its violation count for the key must stay
    unchanged (Scan.mtvec_writes ignores csrrs/csrrc, rfb does not, so the two bases can differ in a
    CSR set/clear span; such a site would still be reported by rfb and skew the report).  A drop is
    refused too: it would mean the fake line changed something besides the noise (e.g. its
    successor became a trap entry and swallowed an original violation).
    Returns notes; raises InjectionError."""
    cid = orig.case_id
    placed = sorted(i for i, _, _ in info["placed"])
    if len(placed) != len(info["h"]):
        raise InjectionError(f"case {cid}: vector: {len(placed)} sites but {len(info['h'])} H")
    if new.trace.n_instr != orig.trace.n_instr + len(placed):
        raise InjectionError(f"case {cid}: vector: n_instr {orig.trace.n_instr} -> {new.trace.n_instr}, "
                             f"expected +{len(placed)}")
    ctx = isa.ScanContext(trap_entries=set(new_scan.traps), xret_returns=dict(new_scan.xret_ret))
    for r, (i, h) in enumerate(zip(placed, info["h"])):
        base = orig_scan.mtvec_before(i)
        d = None if base is None else h - base
        if d is None or d % 4 != 0 or (d >> 2) not in VEC_KS or h in orig_scan.pcs:
            raise InjectionError(f"case {cid}: vector: site {i} H={h:#x} is not a free slot of base "
                                 f"{base if base is None else hex(base)}")
        j = i + r   # index in the new trace (r earlier fake lines)
        site, nxt = new_scan.entries[j][1], new_scan.entries[j + 1][1]
        if site.mnemonic != mn or nxt.pc != h:
            raise InjectionError(f"case {cid}: vector: site {i} wiring (mnemonic {site.mnemonic}, next pc {nxt.pc:#x})")
        v, err = isa.check_safe(None, site, nxt.pc, ctx)
        if err is not None or v is None or v.kind != NOISE_KIND:
            raise InjectionError(f"case {cid}: vector: site {i} not a {NOISE_KIND} without the vector fix: {v} {err}")
    if orig.family == "MISMATCH":
        d0, d1 = orig.deviation, new.deviation
        if (d0.kind, d0.mnemonic) != (d1.kind, d1.mnemonic) or signature(orig).hard_key != signature(new).hard_key:
            raise InjectionError(f"case {cid}: vector: MISMATCH deviation/hard_key changed")
    key = f"{NOISE_KIND}:{mn}"
    if rfb_has_mtvec_base():
        n0, n1 = orig.trace.violations.get(key, 0), new.trace.violations.get(key, 0)
        if n1 != n0:
            raise InjectionError(f"case {cid}: vector: rfb still reports {key} {n0}->{n1} with mtvec_base "
                                 f"(must stay unchanged; a rise means noise_eval and rfb disagree on the base)")
    return [f"vector: {key} {orig.trace.violations.get(key, 0)}->{new.trace.violations.get(key, 0)} "
            f"(rfb mtvec_base={rfb_has_mtvec_base()})"]


def verify_injection(orig, new, mn, c_eff, mode="nonreturn", traps=None):
    """Wiring assertions through rfb's own parse/signature.  `traps` = (trap_entries
    before, after) for mode "returning".  Returns notes (list of str); raises
    InjectionError on a failed assertion."""
    key = f"{NOISE_KIND}:{mn}"
    cid = orig.case_id
    n0, n1 = orig.trace.violations.get(key, 0), new.trace.violations.get(key, 0)
    if mode == "returning":
        want_n = c_eff if not rfb_has_xret_returns() else 0
        if n1 != n0 + want_n:
            raise InjectionError(f"case {cid}: returning: {key} count {n0} -> {n1}, expected +{want_n} "
                                 f"(rfb xret_returns={rfb_has_xret_returns()})")
        if traps[0] != traps[1]:
            raise InjectionError(f"case {cid}: returning: trap_entries changed "
                                 f"(+{sorted(traps[1] - traps[0])} -{sorted(traps[0] - traps[1])})")
        if new.trace.n_instr != orig.trace.n_instr + 2 * c_eff:
            raise InjectionError(f"case {cid}: returning: n_instr {orig.trace.n_instr} -> {new.trace.n_instr}, "
                                 f"expected +{2 * c_eff}")
        o0 = {k: v for k, v in orig.trace.violations.items() if k != key}
        o1 = {k: v for k, v in new.trace.violations.items() if k != key}
        if o0 != o1:
            raise InjectionError(f"case {cid}: returning: other violation counts changed {o0} -> {o1}")
        if orig.family == "MISMATCH":
            d0, d1 = orig.deviation, new.deviation
            s0, s1 = signature(orig), signature(new)
            if (d0.kind, d0.mnemonic) != (d1.kind, d1.mnemonic) or s0.hard_key != s1.hard_key:
                raise InjectionError(f"case {cid}: MISMATCH deviation/hard_key changed")
        return [f"returning: {key} {n0}->{n1}"]
    if n1 != n0 + c_eff:
        raise InjectionError(f"case {cid}: {key} count {n0} -> {n1}, expected +{c_eff}")
    group = isa.viol_group(NOISE_KIND, mn)
    s0, s1 = signature(orig), signature(new)
    c0, c1 = s0.soft["sys_viol_class"], s1.soft["sys_viol_class"]
    want = (NOISE_KIND, group) in c0 or n0 + c_eff >= 2
    if ((NOISE_KIND, group) in c1) != want:
        raise InjectionError(f"case {cid}: ({NOISE_KIND},{group}) in sys_viol_class = "
                             f"{(NOISE_KIND, group) in c1}, expected {want} (orig count {n0}, +{c_eff})")
    notes = []
    if n0:
        notes.append(f"orig already had {n0}x {key}")
    if orig.family == "MISMATCH":
        d0, d1 = orig.deviation, new.deviation
        if (d0.kind, d0.mnemonic) != (d1.kind, d1.mnemonic) or s0.hard_key != s1.hard_key:
            raise InjectionError(f"case {cid}: MISMATCH deviation/hard_key changed: "
                                 f"{(d0.kind, d0.mnemonic)} {s0.hard_key} -> {(d1.kind, d1.mnemonic)} {s1.hard_key}")
    elif s0.hard_key != s1.hard_key:
        notes.append(f"hard_key changed {s0.hard_key} -> {s1.hard_key}")
    others0 = {k: v for k, v in orig.trace.violations.items() if k != key}
    others1 = {k: v for k, v in new.trace.violations.items() if k != key}
    if others0 != others1:
        notes.append(f"other violation counts changed {others0} -> {others1}")
    return notes


def inject_case(set_no, case_id, mn, c, out_case_dir, trace_name, orig=None, mode="nonreturn", steps=None):
    """Inject c noise sites of `mn` into set `set_no` case `case_id`; writes the
    modified trace as <out_case_dir>/<trace_name>.  Returns (info, notes).  `orig` is
    the original FailureRecord (parsed here when None).  `steps` = list of
    (mode, mn, c) applied in order (each on the previous result and verified against
    the previous step); overrides mode/mn/c."""
    src = os.path.join(PUB, f"benchmark_set_{set_no}", f"case_{case_id}")
    src_trace = os.path.join(src, trace_name)
    regr = os.path.join(src, "regr.log")
    steps = steps or [(mode, mn, c)]
    lines = read_trace(src_trace)
    os.makedirs(out_case_dir, exist_ok=True)
    if orig is None:
        orig = parse_case(case_id, src, src_trace)
    prev, notes = orig, []
    agg = dict(n_req=0, n_inj=0, step_inj=[], h=[], placed=[], synth_template=False, synth_mret=False)
    for si, (smode, smn, sc) in enumerate(steps):
        scan = Scan(lines, regr)
        lines, info = inject_trace(scan, smn, sc, smode)
        for k in ("n_req", "n_inj"):
            agg[k] += info[k]
        agg["step_inj"].append(info["n_inj"])
        agg["h"] += info["h"]
        agg["placed"] += info["placed"]
        agg["synth_template"] |= info["synth_template"]
        agg["synth_mret"] |= info["synth_mret"]
        if not info["n_inj"]:
            continue
        last = si == len(steps) - 1
        dst = os.path.join(out_case_dir, trace_name if last else f"_step{si}_{trace_name}")
        write_trace(dst, lines)
        new = parse_case(case_id, src, dst)
        if smode == "vector":
            notes += verify_vector(scan, Scan(lines, regr), prev, new, smn, info)
        else:
            traps = (scan.traps, Scan(lines, regr).traps) if smode == "returning" else None
            notes += verify_injection(prev, new, smn, info["n_inj"], smode, traps)
        prev = new
    if not os.path.exists(os.path.join(out_case_dir, trace_name)):
        write_trace(os.path.join(out_case_dir, trace_name), lines)
    agg["mn"] = steps[0][1] + ("" if all(st[0] == "nonreturn" for st in steps) else
                               " (" + ", ".join(f"{st[0]} {st[2]}" for st in steps) + ")")
    return agg, notes


# ---------------------------------------------------------------- scenarios
def read_csv_map(path, col):
    with open(path, newline="") as f:
        return {str(r["Case"]).strip(): str(r[col]).strip() for r in csv.DictReader(f)}


def site_counts(set_no, cases, mns):
    """{(case, mn): number of usable sites} (no injection performed)."""
    out = {}
    for c in cases:
        d = os.path.join(PUB, f"benchmark_set_{set_no}", f"case_{c}")
        scan = Scan(read_trace(os.path.join(d, "trace.log.gz")), os.path.join(d, "regr.log"))
        for mn in mns:
            out[(c, mn)] = len(scan.candidates(mn))
    return out


def assign_n5(bugs, counts, c):
    """bugs: {bug: [cases]}.  Pick one distinct mnemonic group per bug maximising the
    total injected sites (sum of min(c, available)), then fewest short cases, then
    preference order.  Returns {bug: mnemonic}."""
    opts = {}
    for bug, cs in bugs.items():
        o = []
        for gi, (g, mns) in enumerate(GROUPS):
            best = max(mns, key=lambda m: (sum(min(c, counts[(x, m)]) for x in cs), -mns.index(m)))
            tot = sum(min(c, counts[(x, best)]) for x in cs)
            o.append((gi, g, best, tot, sum(counts[(x, best)] < c for x in cs)))
        opts[bug] = o
    names = sorted(bugs)
    best_assign, best_key = None, None
    for combo in itertools.product(*(opts[b] for b in names)):
        if len({x[1] for x in combo}) < len(combo):
            continue
        key = (-sum(x[3] for x in combo), sum(x[4] for x in combo), tuple(x[0] for x in combo))
        if best_key is None or key < best_key:
            best_key, best_assign = key, combo
    if best_assign is None:
        raise InjectionError("no distinct mnemonic-group assignment for N5")
    return {b: x[2] for b, x in zip(names, best_assign)}


def plan_scenario(name, set_no, golden, families, counts):
    """{case: (mnemonic, c)} for the cases that get noise."""
    cases = sorted(golden, key=int)
    if name == "N1":
        return {c: ("jal", 1) for c in cases}
    if name == "N2":
        return {c: ("jal", 2) for c in cases}
    if name == "N6":
        return {c: ("jal", 5) for c in cases}
    if name == "N7":
        return {c: [("returning", "jal", 5)] for c in cases}
    if name == "N8":
        # returning first, then one unreturned site on the result (its candidates exclude any site
        # a later xret, including the injected mrets, returns to, so the fix cannot suppress it)
        return {c: [("returning", "jal", 5), ("nonreturn", "jal", 1)] for c in cases}
    if name == "N9":
        return {c: [("vector", "jal", 2)] for c in cases}
    if name == "N10":
        return {c: [("vector", "jal", 5)] for c in cases}
    if name == "N3":
        return {c: ("jal", 2) for c in cases if families[c] in ("FATAL", "ASSERT")}
    if name == "N4":
        plan = {}
        for bug in sorted(set(golden.values())):
            cs = [c for c in cases if golden[c] == bug]
            ok = [c for c in cs if counts[(c, "jal")] >= 2]
            plan[(ok or cs)[0]] = ("jal", 2)
        return plan
    if name == "N5":
        bugs = {b: [c for c in cases if golden[c] == b] for b in sorted(set(golden.values()))}
        asg = assign_n5(bugs, counts, 2)
        return {c: (asg[golden[c]], 2) for c in cases}
    raise ValueError(name)


# ---------------------------------------------------------------- running
def check_injected_coverage(name, set_no, plan, results, golden):
    """Every step of a scenario must really inject into cases of at least two different golden
    bugs: an R1 collision (or, for N5/N7/N8, its absence) needs noise on two bugs to be
    observable at all.  A case counts for a step when it got min(c, 2) sites: with c >= 2 a
    single site never reaches MIN_SYS_VIOL, so it cannot put the noise into sys_viol_class.
    inject_case skips verification for a step with 0 sites, so without this a scenario could
    silently degrade to N0.  Returns the per-step "injected/total" strings (cases with >= 1
    site, for the report)."""
    total = len(plan)
    nsteps = max(len(v) if isinstance(v, list) else 1 for v in plan.values())
    out = []
    for si in range(nsteps):
        def got(c):
            return results[c][0]["step_inj"][si]  # no guard: a misaligned step_inj must fail loudly

        def req(c):
            return plan[c][si][2] if isinstance(plan[c], list) else plan[c][1]

        hit = [c for c in plan if got(c) > 0]
        full = [c for c in plan if got(c) >= min(req(c), 2)]
        bugs = {golden[c] for c in full}
        if len(bugs) < 2:
            raise InjectionError(f"{name} set {set_no} step {si}: cases with min(c, 2) sites "
                                 f"{sorted(full, key=int)} span {len(bugs)} golden bug(s), need >= 2")
        out.append(f"{len(hit)}/{total}")
    return out


def make_bench(bdir, set_no, plan, golden):
    """Copy of the public set: injected cases get a rewritten trace, the rest are
    symlinks to the originals.  Returns {case: (info, notes)}."""
    src_set = os.path.join(PUB, f"benchmark_set_{set_no}")
    os.makedirs(bdir)
    results = {}
    for c in sorted(golden, key=int):
        cd = os.path.join(bdir, f"case_{c}")
        os.makedirs(cd)
        for fn in ("regr.log", "sim.log.gz"):
            os.symlink(os.path.join(src_set, f"case_{c}", fn), os.path.join(cd, fn))
        if c in plan:
            if isinstance(plan[c], list):
                results[c] = inject_case(set_no, c, None, None, cd, "trace.log.gz", orig=ORIG[(set_no, c)],
                                         steps=plan[c])
            else:
                mn, cnt = plan[c]
                results[c] = inject_case(set_no, c, mn, cnt, cd, "trace.log.gz", orig=ORIG[(set_no, c)])
        else:
            os.symlink(os.path.join(src_set, f"case_{c}", "trace.log.gz"), os.path.join(cd, "trace.log.gz"))
    with open(os.path.join(bdir, "input.csv"), "w") as f:
        f.write("Case,Regr Log,Sim Log,Trace Log\n")
        for c in sorted(golden, key=int):
            f.write(f"{c},case_{c}/regr.log,case_{c}/sim.log.gz,case_{c}/trace.log.gz\n")
    with open(os.path.join(bdir, "golden.csv"), "w") as f:
        f.write("Case,Bug\n")
        for c in sorted(golden, key=int):
            f.write(f"{c},{golden[c]}\n")
    return results


def worker(inp, out, k):
    """Runs in a subprocess: parse + signature + cluster, write Case,bucket."""
    from rfb.cluster import cluster
    from rfb.signature import build_signatures
    records = pipeline.parse_all(inp)
    labels = cluster(build_signatures(records), int(k))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for r in sorted(records, key=lambda r: r.case_id):
            w.writerow([r.case_id, labels[r.case_id]])


def run_cluster(inp, k, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "output.csv")
    env = dict(os.environ, PYTHONPATH=REPO, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([os.path.join(REPO, ".venv/bin/python"), "-B", os.path.abspath(__file__), "--worker",
                        inp, out, str(k)], env=env, cwd=outdir, capture_output=True, text=True, timeout=900)
    with open(os.path.join(outdir, "run.log"), "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"cluster run failed ({outdir}/run.log): {r.stderr.strip().splitlines()[-1:]}")
    return out


def analyse(golden, bucket):
    from tools.score import balanced_accuracy
    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, bucket)
    lines = []
    bks = {}
    for c in sorted(bucket, key=int):
        bks.setdefault(bucket[c], []).append(c)
    for bug in sorted(set(golden.values())):
        cs = [c for c in sorted(golden, key=int) if golden[c] == bug]
        mine = sorted({bucket[c] for c in cs}, key=lambda x: (len(x), x))
        foreign = sorted({golden[c2] for b in mine for c2 in bks[b] if golden[c2] != bug})
        lines.append(f"- {bug} (cases {','.join(cs)}): {len(mine)} bucket(s)"
                     + (" SPLIT" if len(mine) > 1 else "")
                     + f"; foreign bugs sharing its buckets: {', '.join(foreign) if foreign else '(none)'}"
                     + (" MERGED" if foreign else ""))
    detail = []
    for b in sorted(bks, key=lambda x: (len(x), x)):
        detail.append(f"  - bucket {b}: " + " ".join(f"{c}({golden[c]})" for c in bks[b]))
    return ba, (tp, fp, tn, fn), lines, detail


ORIG = {}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--work", default=os.path.join(REPO, "out/noise_eval"))
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    rw = os.path.normcase(os.path.realpath(work))
    tc = os.path.normcase(os.path.realpath(os.path.join(REPO, "testcase")))
    rf = os.path.normcase(os.path.realpath(os.path.join(REPO, "rfb")))
    for g in (tc, rf):
        if rw == g or rw.startswith(g + os.sep):
            sys.exit(f"--work {work} resolves into {g}; refusing")
    if os.path.lexists(work) and (not os.path.isdir(work) or os.listdir(work)):
        sys.exit(f"{work} exists and is not an empty directory; refusing to reuse it (nothing is deleted)")
    os.makedirs(work, exist_ok=True)

    golden, families, counts = {}, {}, {}
    mns = [m for _, ms in GROUPS for m in ms]
    for s in (1, 2):
        golden[s] = read_csv_map(os.path.join(PUB, f"benchmark_set_{s}", "golden.csv"), "Bug")
        for c in golden[s]:
            d = os.path.join(PUB, f"benchmark_set_{s}", f"case_{c}")
            ORIG[(s, c)] = parse_case(int(c), d, os.path.join(d, "trace.log.gz"))
        families[s] = {c: ORIG[(s, c)].family for c in golden[s]}
        counts[s] = site_counts(s, sorted(golden[s], key=int), mns)

    xr = rfb_has_xret_returns()
    ver = ("rfb WITH ScanContext.xret_returns (WRONG_TARGET followed by an xret return is not a violation)"
           if xr else "rfb WITHOUT ScanContext.xret_returns (every WRONG_TARGET counts)")
    print(f"detected: {ver}")
    md = ["# Noise injection evaluation (post-divergence BRANCH_WRONG_TARGET noise)", "",
          f"Detected: {ver}.", "",
          "- N2-N6 (non-returning noise: interrupt entered, never returned from) are the residual risk after the "
          "xret_returns fix.",
          "- N7, N8 (returning noise: fake line at H, `mret` at H+4, then the original next line) are the case the "
          "fix removes.",
          "- N9, N10 (vector noise: non-returning, but the fake line sits in an interrupt vector slot "
          "mtvec_base + 4k, k in {3, 7, 11, 16..31}) model what the mtvec-vector fix removes; they are verified without the fix.",
          "- N1 is the single non-returning site; N8 = returning c=5, then one non-returning site (c=1).  With xret_returns, non-returning sites whose target/fallthrough a later xret returns to are not used.",
          "- Wiring assertion for returning sites: with xret_returns the BRANCH_WRONG_TARGET:jal count must not "
          "increase from them; without it it increases by c.  trap_entries and all other violation keys must "
          "be unchanged.", ""]
    table = [("scenario", "set 1 (k=2)", "set 2 (k=4)", "set 1 injected cases", "set 2 injected cases")]
    sections = []
    # N0
    row = ["N0 original"]
    cov = []
    for s in (1, 2):
        inp = os.path.join(PUB, f"benchmark_set_{s}", "input.csv")
        out = run_cluster(inp, SET_K[s], os.path.join(work, "run", f"N0_set{s}"))
        ba, cnt, lines, detail = analyse(golden[s], read_csv_map(out, "bucket"))
        row.append(f"{ba:.6f}")
        cov.append("-")
        if ba != 1.0:
            sys.exit(f"N0 set {s}: BA={ba:.6f}, expected 1.0; aborting")
    table.append(tuple(row + cov))
    for name in ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"):
        row = [name]
        cov = []
        sec = [f"## {name}", ""]
        for s in (1, 2):
            gs = golden[s]
            counts_s = counts[s]
            plan = plan_scenario(name, s, gs, families[s], counts_s)
            bdir = os.path.join(work, "bench", name, f"set{s}")
            res = make_bench(bdir, s, plan, gs)
            cov.append(", ".join(check_injected_coverage(name, s, plan, res, gs)))
            out = run_cluster(os.path.join(bdir, "input.csv"), SET_K[s], os.path.join(work, "run", f"{name}_set{s}"))
            ba, (tp, fp, tn, fn), lines, detail = analyse(gs, read_csv_map(out, "bucket"))
            row.append(f"{ba:.6f}")
            sec += [f"### {name} set {s} (k={SET_K[s]}): BA={ba:.6f}  TP={tp} FP={fp} TN={tn} FN={fn}", ""]
            if name == "N5":
                sec += ["mnemonic per bug: " + ", ".join(sorted({f"{gs[c]}={plan[c][0]}" for c in plan})), ""]
            sec += ["| case | family | bug | mnemonic | c req | c injected | placed (idx:where) | notes |",
                    "|---|---|---|---|---|---|---|---|"]
            for c in sorted(gs, key=int):
                if c in res:
                    info, notes = res[c]
                    pl = " ".join(f"{i}:{w}{'/tail' if t else ''}" for i, w, t in info["placed"]) or "-"
                    nt = list(notes)
                    if info["n_inj"] < info["n_req"]:
                        nt.append(f"SHORT: only {info['n_inj']} usable site(s)")
                    if info["synth_template"]:
                        nt.append("synthesized nop line")
                    if info["synth_mret"]:
                        nt.append("synthesized mret line")
                    sec.append(f"| {c} | {families[s][c]} | {gs[c]} | {info['mn']} | {info['n_req']} | {info['n_inj']} "
                               f"| {pl} | {'; '.join(nt)} |")
                else:
                    sec.append(f"| {c} | {families[s][c]} | {gs[c]} | - | 0 | 0 | not injected | |")
            sec += ["", "Per bug:"] + lines + ["", "Buckets:"] + detail + [""]
        table.append(tuple(row + cov))
        sections.append("\n".join(sec))
    md += ["## Summary (balanced accuracy)", "", "| " + " | ".join(table[0]) + " |", "|---|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in table[1:]] + [""] + sections
    with open(os.path.join(work, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for r in table:
        print("\t".join(r))
    print(f"report: {os.path.join(work, 'report.md')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        main()
