#!/usr/bin/env python3
"""Realistic ALU-result bug injection (MISMATCH family) for regression tests.

Synthesises "ALU result wrong" cases from the clean prefix of the public
cases: the first instruction hitting a bug's condition is the anchor, its
(and every later hit's) written value is rewritten, regr.log becomes a
Mismatch[1] block (ibex wrong value / spike correct value / same PC) plus a
[FAILED] line, the trace is cut at the clean prefix end, sim.log is the
PASSED log of the source case.  Ported verbatim from the architect's
one-off experiment (out/arch_alu/{common,bugs,gen_real}.py).

Variants:
  real    : trace = source clean prefix with all condition hits mutated.
  trunc20 : same, cut 20 instructions after the anchor (thin evidence).

Output is fully deterministic (no randomness is involved; --seed is accepted
for interface stability and has no effect).  Reads golden.csv of the public
sets only to label sources (tools may do this; rfb/ never does).

CLI: inject_alu_real.py --out DIR --variant real|trunc20 [--seed S]
     [--per-bug N (>=1)] [--bg-cases 1,2,..] [--src-cases 2:4,2:7,..]
     (bg = set 2 cases kept as background, "" = no background; src = source
     cases to inject from)
--out must not exist or must be an empty directory.  The generator never
deletes any file; a non-empty --out, a file or a symlink is refused (exit 1),
as is anything at or under the repo's testcase/ directory.
"""
import argparse
import csv
import gzip
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rfb import isa  # noqa: E402
from rfb.io_utils import open_text  # noqa: E402
from rfb.record import Instr  # noqa: E402
from rfb.parse_trace import _pass1_trap_entries, _split_trace_line  # noqa: E402
from rfb.parse_regr import parse_regr  # noqa: E402
from rfb.parse_sim import parse_sim  # noqa: E402

BASE = os.path.join(REPO, "testcase/2026_08_20_samples/problem")
SET = {1: os.path.join(BASE, "benchmark_set_1"), 2: os.path.join(BASE, "benchmark_set_2")}
M32 = 0xFFFFFFFF
ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"] + [f"a{i}" for i in range(8)] \
    + [f"s{i}" for i in range(2, 12)] + ["t3", "t4", "t5", "t6"]
PASSED_SIM = os.path.join(SET[2], "case_3/sim.log.gz")


def read_lines(path):
    with open_text(path) as f:
        return f.read().splitlines()


def abi(reg):
    return ABI[int(reg[1:])]


def s32(v):
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def to_instr(i, line):
    parts = line.split("\t")
    f = _split_trace_line(parts)
    if f is None:
        return None
    pc_hex, insn_hex, mn, ops, regs = f
    reads, writes, pa, store, load = isa.parse_regs(regs)
    return Instr(index=i, pc=int(pc_hex, 16), size=2 if len(insn_hex) <= 4 else 4, mnemonic=mn,
                 operands=ops, reads=reads, writes=writes, pa=pa, store=store, load=load)


def scan(lines_with_header):
    """Return (instrs, violations list aligned with instrs) using rfb's own
    trap-entry pass + isa.check_safe.

    Not identical to parse_trace: there is no isa.observe call, so no shadow x2
    is maintained.  This does not change the violation position on the clean
    prefix of the source cases (measured 2026-09-29 on 32 sources), which is
    all load_sources() uses it for.

    Known exception: because observe() is not called, ctx.mtvec_base stays
    None, so the interrupt-vector-slot exclusion (isa._irq_vector_entry) never
    applies here.  The BRANCH_WRONG_TARGET:c.beqz of set 2 case 14 (which
    parse_trace no longer reports) is therefore still a violation in this
    scan, so "clean prefix position unchanged" is outdated for c14.  Left as
    is on purpose (changing it would change the generated benchmarks)."""
    _, targets, _ = _pass1_trap_entries(iter(lines_with_header))
    # xret_returns deliberately not passed: keeps source-selection (and thus
    # the generated benchmarks) identical to before the async-entry rule.
    # If xret_returns is ever passed, Instr.index below (enumerate line number,
    # blank lines counted) must first become rfb's valid-line count, or the
    # coordinates are misaligned.
    ctx = isa.ScanContext(trap_entries=set(targets))
    instrs = [to_instr(i, ln) for i, ln in enumerate(lines_with_header[1:]) if ln]
    viol = []
    for j, ins in enumerate(instrs):
        nxt = instrs[j + 1].pc if j + 1 < len(instrs) else None
        v, _ = isa.check_safe(None, ins, nxt, ctx)
        viol.append(v)
    return instrs, viol


def rows(s):
    with open(os.path.join(SET[s], "input.csv"), newline="") as f:
        return list(csv.DictReader(f))


def golden(s):
    with open(os.path.join(SET[s], "golden.csv"), newline="") as f:
        return {r["Case"]: r["Bug"] for r in csv.DictReader(f)}


def _rv(reads, r):
    return isa._regval(reads, r)


def _imm(tok):
    try:
        return isa._parse_imm(tok)
    except ValueError:
        return None


def _shift_src(mn, ops, reads):
    """(a, shamt) for sll/srl/sra-type incl. imm and compressed forms."""
    if mn in ("sll", "srl", "sra"):
        a, b = _rv(reads, ops[1]), _rv(reads, ops[2])
        return (None, None) if a is None or b is None else (a, b & 31)
    if mn in ("slli", "srli", "srai"):
        a, i = _rv(reads, ops[1]), _imm(ops[2])
        return (None, None) if a is None or i is None else (a, i & 31)
    if mn in ("c.slli", "c.srli", "c.srai"):
        a, i = _rv(reads, ops[0]), _imm(ops[1])
        return (None, None) if a is None or i is None else (a, i & 31)
    return None, None


def b_mulhsu_neg(mn, ops, reads, orig):
    # MULH path with signed_mode=01 (mulhsu only): rs1 negative -> result - 1
    a = _rv(reads, ops[1])
    if a is None or s32(a) >= 0:
        return None
    return (orig - 1) & M32


def b_mul_pp(mn, ops, reads, orig):
    # MULL (mul only): the a_lo*b_hi partial product is dropped
    a, b = _rv(reads, ops[1]), _rv(reads, ops[2])
    if a is None or b is None:
        return None
    return (orig - (((a & 0xFFFF) * (b >> 16)) << 16)) & M32


def b_mulhu_msb(mn, ops, reads, orig):
    # MULH path with signed_mode=00 (mulhu only): rs2 msb set -> result lsb flipped
    b = _rv(reads, ops[2])
    if b is None or not (b >> 31):
        return None
    return orig ^ 1


def b_sra_ge16(mn, ops, reads, orig):
    # ALU_SRA (sra/srai/c.srai share the operator): shamt >= 16 -> zero fill
    a, sh = _shift_src(mn, ops, reads)
    if a is None or sh < 16:
        return None
    return (a & M32) >> sh


def b_sll_ge16(mn, ops, reads, orig):
    # ALU_SLL (sll/slli/c.slli): shamt[4] ignored in the left-shift path
    a, sh = _shift_src(mn, ops, reads)
    if a is None or sh < 16:
        return None
    return (a << (sh & 15)) & M32


def b_sltu_msb(mn, ops, reads, orig):
    # ALU_SLTU (sltu/sltiu): operands' msb differ -> result inverted
    a = _rv(reads, ops[1])
    b = _rv(reads, ops[2]) if mn == "sltu" else _imm(ops[2])
    if a is None or b is None:
        return None
    b &= M32
    if (a >> 31) == (b >> 31):
        return None
    return orig ^ 1


def b_div_unit(mn, ops, reads, orig):
    # shared divider FSM (div/divu/rem/remu): dividend bit31 set and
    # divisor != 0 -> result lsb flipped
    a, b = _rv(reads, ops[1]), _rv(reads, ops[2])
    if a is None or b is None or not (a >> 31) or b == 0:
        return None
    return orig ^ 1


def b_caddi_neg(mn, ops, reads, orig):
    # compressed decoder, c.addi only: negative imm -> result + 1
    i = _imm(ops[1])
    if i is None or i >= 0:
        return None
    return (orig + 1) & M32


def b_sub_borrow(mn, ops, reads, orig):
    # ALU_SUB (sub/c.sub): unsigned borrow (rs1 < rs2) -> result - 1
    if mn == "sub":
        a, b = _rv(reads, ops[1]), _rv(reads, ops[2])
    else:
        a, b = _rv(reads, ops[0]), _rv(reads, ops[1])
    if a is None or b is None or (a & M32) >= (b & M32):
        return None
    return (orig - 1) & M32


def b_addi4spn(mn, ops, reads, orig):
    # compressed decoder, c.addi4spn only: nzuimm[4] dropped
    i = _imm(ops[2])
    if i is None or not (i & 16):
        return None
    return (orig - 16) & M32


BUGS = {
    "inj_mulhsu_neg": ({"mulhsu"}, b_mulhsu_neg, "MUL", "mulhsu rs1<0 結果 -1"),
    "inj_mul_pp":     ({"mul"}, b_mul_pp, "MUL", "mul 漏掉 a_lo*b_hi 部分積"),
    "inj_mulhu_msb":  ({"mulhu"}, b_mulhu_msb, "MUL", "mulhu rs2 msb=1 結果 lsb 反相"),
    "inj_sra_ge16":   ({"sra", "srai", "c.srai"}, b_sra_ge16, "SHIFT", "sra/srai/c.srai shamt>=16 做邏輯右移"),
    "inj_sll_ge16":   ({"sll", "slli", "c.slli"}, b_sll_ge16, "SHIFT", "sll/slli/c.slli shamt[4] 被忽略"),
    "inj_sltu_msb":   ({"sltu", "sltiu"}, b_sltu_msb, "SLT", "sltu/sltiu msb 不同時反相"),
    "inj_div_unit":   ({"div", "divu", "rem", "remu"}, b_div_unit, "DIV", "除法器（四個 mnemonic）被除數 bit31=1 時 lsb 反相"),
    "inj_caddi_neg":  ({"c.addi"}, b_caddi_neg, "ADDSUB", "c.addi 負立即數 結果 +1（壓縮解碼器）"),
    "inj_sub_borrow": ({"sub", "c.sub"}, b_sub_borrow, "ADDSUB", "sub/c.sub 無號借位時 結果 -1"),
    "inj_addi4spn":   ({"c.addi4spn"}, b_addi4spn, "ADDSUB", "c.addi4spn nzuimm[4] 遺失（壓縮解碼器）"),
}


def load_sources(only=None):
    """only: optional set of (set, case) pairs restricting the source cases."""
    srcs = []
    for s in (1, 2):
        g = golden(s)
        for r in rows(s):
            if only is not None and (s, int(r["Case"])) not in only:
                continue
            lines = read_lines(os.path.join(SET[s], r["Trace Log"]))
            body = [ln for ln in lines[1:] if ln]
            instrs, viol = scan([lines[0]] + body)
            regr = parse_regr(os.path.join(SET[s], r["Regr Log"]))
            sim = parse_sim(os.path.join(SET[s], r["Sim Log"]))
            E = next((i for i, v in enumerate(viol) if v is not None), len(instrs))
            if regr.kind == "MISMATCH":
                m = regr.mismatches[0]
                for ins in instrs:
                    if ins.pc == m.ibex_pc and (m.ibex_val is None or m.ibex_val in ins.writes.values()):
                        E = min(E, ins.index); break
            srcs.append(dict(set=s, case=r["Case"], bug=g[r["Case"]], header=lines[0], body=body,
                             instrs=instrs, E=E, family=regr.kind, test=sim.test_name,
                             sim=os.path.join(SET[s], r["Sim Log"])))
    return srcs


def hits(src, fn, mset):
    out = []
    for ins in src["instrs"][: src["E"]]:
        if ins.mnemonic not in mset:
            continue
        ops = isa._split_operands(ins.operands)
        if not ops or ops[0] in ("x0", "x2"):  # x2: non-propagated sp write would fake shadow inconsistency
            continue
        rd = ops[0]
        orig = ins.writes.get(rd)
        if orig is None:
            continue
        new = fn(ins.mnemonic, ops, ins.reads, orig)
        if new is None or (new & M32) == orig:
            continue
        out.append((ins.index, rd, orig, new & M32))
    return out


def rewrite(line, rd, orig, new):
    parts = line.split("\t")
    k = 6 if len(parts) >= 7 else 5
    toks = parts[k].split(" ")
    done = False
    for i, t in enumerate(toks):
        if t.startswith(f"{rd}=0x") and not done:
            toks[i] = f"{rd}=0x{new:08x}"; done = True
    assert done, line
    parts[k] = " ".join(toks)
    return "\t".join(parts)


TESTCASE = os.path.join(REPO, "testcase")


class InputError(ValueError):
    """Bad user input (arguments / output location); main() reports it as exit 1."""


def prepare_out_dir(out_dir):
    """Prepare out_dir for a fresh build; this generator never deletes anything.
    Missing -> create (with parents); existing empty directory -> use; anything
    else (non-empty dir, file, symlink even if dangling, uncreatable path) ->
    InputError.  Locations at or under the repo's testcase/ (after resolving
    symlinks) are refused first."""
    out_dir = os.path.abspath(out_dir)  # a trailing slash would hide a symlink from islink
    real = os.path.normcase(os.path.realpath(out_dir))
    tc = os.path.normcase(os.path.realpath(TESTCASE))
    if real == tc or real.startswith(tc + os.sep):
        raise InputError(f"--out {out_dir} resolves into the repo testcase/ directory (testcase guard); refusing")
    if os.path.islink(out_dir):
        raise InputError(f"--out {out_dir} is a symlink; refusing to use it")
    if os.path.lexists(out_dir):
        if not os.path.isdir(out_dir):
            raise InputError(f"--out {out_dir} exists and is not a directory")
        if os.listdir(out_dir):
            raise InputError(f"--out {out_dir} is not empty; it must be missing or empty (nothing is ever deleted)")
        return
    try:
        os.makedirs(out_dir)
    except OSError as e:
        raise InputError(f"cannot create --out {out_dir}: {e}")


def _build(variant, srcs, out_root, per_bug=5, bg_cases=None):
    """Write the benchmark.  Private: the caller must have run _validate and
    prepare_out_dir first (generate does)."""
    out_dir = os.path.abspath(out_root)
    table = []  # (cid, regr, sim, trace, label)
    g2 = golden(2)
    for r in rows(2):
        if bg_cases is not None and int(r["Case"]) not in bg_cases:
            continue
        table.append((int(r["Case"]), os.path.join(SET[2], r["Regr Log"]), os.path.join(SET[2], r["Sim Log"]),
                      os.path.join(SET[2], r["Trace Log"]), g2[r["Case"]]))
    cid = 26
    meta = []
    for tag, (mset, fn, mcls, desc) in BUGS.items():
        cands = []
        for src in srcs:
            h = hits(src, fn, mset)
            if not h:
                continue
            first_idx, rd, orig, new = h[0]
            anc = src["instrs"][first_idx]
            # anchor must be unique: no earlier same-PC instr writing the wrong value
            if any(i.pc == anc.pc and new in i.writes.values() for i in src["instrs"][:first_idx]):
                continue
            cands.append((len(h), src, h))
        cands.sort(key=lambda t: (-t[0], t[1]["set"], int(t[1]["case"])))
        chosen, tests = [], set()
        for c in cands:
            if c[1]["test"] in tests:
                continue
            chosen.append(c); tests.add(c[1]["test"])
            if len(chosen) == per_bug:
                break
        for nh, src, h in chosen:
            hmap = {i: (rd, o, n) for i, rd, o, n in h}
            a_idx = h[0][0]
            end = src["E"] if variant == "real" else min(src["E"], a_idx + 21)
            body = []
            for i in range(end):
                ln = src["body"][i]
                if i in hmap:
                    body.append(rewrite(ln, *hmap[i]))
                else:
                    body.append(ln)
            anc = src["instrs"][a_idx]
            rd, orig, new = hmap[a_idx]
            idx = sum(1 for ins in src["instrs"][:a_idx] if any(k != "x0" for k in ins.writes))
            after_w = sum(1 for ins in src["instrs"][a_idx + 1:end] if any(k != "x0" for k in ins.writes))
            ops = isa._split_operands(anc.operands)
            spike_ops = ", ".join(abi(o) if o.startswith("x") and o[1:].isdigit() else o for o in ops)
            regr = (f"Mismatch[1]:\n"
                    f"ibex[{idx}] : pc[{anc.pc:08x}] {anc.mnemonic}\t{anc.operands}: {abi(rd)}:{new:08x}\n"
                    f"spike[{idx}] : pc[ffffffff{anc.pc:08x}] {anc.mnemonic:<7} {spike_ops}: {abi(rd)}:{orig:08x}\n"
                    f"[FAILED]: {max(idx - 6, 0)} matched, {after_w + 1} mismatch")
            cdir = os.path.join(out_dir, f"case_{cid}")
            os.makedirs(cdir)
            with open(os.path.join(cdir, "regr.log"), "w") as f:
                f.write(regr)
            shutil.copyfile(src["sim"] if src["family"] == "MISMATCH" else PASSED_SIM, os.path.join(cdir, "sim.log.gz"))
            with open(os.path.join(cdir, "trace.log.gz"), "wb") as raw, \
                    gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:  # mtime=0: byte-deterministic
                gz.write((src["header"] + "\n" + "\n".join(body) + "\n").encode())
            n_after = sum(1 for i in hmap if a_idx < i < end)
            meta.append(dict(case=cid, bug=tag, src=f"set{src['set']}/case_{src['case']}", src_bug=src["bug"],
                             src_family=src["family"], anchor_mn=anc.mnemonic, anchor_line=a_idx,
                             trace_len=end, after_anchor=end - 1 - a_idx, hits_total=1 + n_after,
                             hit_mnemonics=sorted({src["instrs"][i].mnemonic for i in hmap if i < end})))
            table.append((cid, os.path.join(cdir, "regr.log"), os.path.join(cdir, "sim.log.gz"),
                          os.path.join(cdir, "trace.log.gz"), tag))
            cid += 1
    with open(os.path.join(out_dir, "input.csv"), "w") as f:
        f.write("Case,Regr Log,Sim Log,Trace Log\n")
        for t in table:
            f.write(f"{t[0]},{t[1]},{t[2]},{t[3]}\n")
    with open(os.path.join(out_dir, "golden.csv"), "w") as f:
        f.write("Case,Bug\n")
        for t in table:
            f.write(f"{t[0]},{t[4]}\n")
    k = len({t[4] for t in table})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(k=k, cases=meta), f, indent=1)
    return k


def _validate(variant, per_bug, bg_cases, src_cases):
    if variant not in ("real", "trunc20"):
        raise InputError(f"variant must be 'real' or 'trunc20', got {variant!r}")
    if isinstance(per_bug, bool) or not isinstance(per_bug, int) or per_bug < 1:
        raise InputError(f"per_bug must be an integer >= 1, got {per_bug!r}")
    if src_cases is not None:
        if not src_cases:
            raise InputError("src_cases is empty")
        have = {(s, int(r["Case"])) for s in (1, 2) for r in rows(s)}
        missing = sorted(set(src_cases) - have)
        if missing:
            raise InputError("source case(s) do not exist: " + ", ".join(f"{s}:{c}" for s, c in missing))
    if bg_cases:
        have = {int(r["Case"]) for r in rows(2)}
        missing = sorted(set(bg_cases) - have)
        if missing:
            raise InputError("background case(s) do not exist in set 2: " + ", ".join(map(str, missing)))


def generate(out_dir, variant="real", seed=0, per_bug=5, bg_cases=None, src_cases=None):
    """Build one benchmark under out_dir; returns k (number of distinct buckets).
    out_dir must not exist or be an empty directory (nothing is ever deleted).
    bg_cases: None = all set-2 cases as background, empty set = no background.
    src_cases: None = all sources; an empty set is an error.
    Raises InputError (a ValueError) on bad arguments or an unusable out_dir."""
    del seed  # deterministic: nothing random
    _validate(variant, per_bug, bg_cases, src_cases)
    prepare_out_dir(out_dir)
    srcs = load_sources(src_cases)
    return _build(variant, srcs, out_dir, per_bug=per_bug, bg_cases=bg_cases)


def _per_bug(text):
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--per-bug must be an integer, got {text!r}")
    if n < 1:
        raise argparse.ArgumentTypeError("--per-bug must be >= 1")
    return n


def _src_cases(text):
    out = set()
    for x in text.split(","):
        if not x:
            continue
        p = x.split(":")
        try:
            if len(p) != 2:
                raise ValueError
            out.add((int(p[0]), int(p[1])))
        except ValueError:
            raise argparse.ArgumentTypeError(f"--src-cases item {x!r} must be set:case (integers)")
    if not out:
        raise argparse.ArgumentTypeError("--src-cases is empty")
    return out


def _bg_cases(text):
    out = set()
    for x in text.split(","):
        if not x:
            continue
        try:
            out.add(int(x))
        except ValueError:
            raise argparse.ArgumentTypeError(f"--bg-cases item {x!r} must be an integer")
    return out  # empty ("" or ",") = no background, same as generate(bg_cases=set())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=("real", "trunc20"), default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-bug", type=_per_bug, default=5)
    ap.add_argument("--bg-cases", type=_bg_cases, default=None, help="set 2 cases kept as background, e.g. 2,4")
    ap.add_argument("--src-cases", type=_src_cases, default=None, help="restrict sources, e.g. 2:4,2:7 (set:case)")
    a = ap.parse_args(argv)
    try:
        k = generate(a.out, a.variant, a.seed, a.per_bug, a.bg_cases, a.src_cases)
    except InputError as e:  # only bad input; other errors keep their traceback
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{a.variant}: k={k}, wrote {a.out}")


if __name__ == "__main__":
    main()
