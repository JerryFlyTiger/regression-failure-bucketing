#!/usr/bin/env python3
"""Synthetic ALU-result-bug injection experiment (task B, see PLAN.md).

Generates two synthetic benchmarks under out/alu_inject/<tag>/ from
public benchmark_set_2 (25 cases, 4 bugs):

- out/alu_inject/mix/  ("mixed"): the 25 original set-2 cases unmodified
  (golden = their real bug), plus 1 injected case per (bug, source case)
  pair for 6 injected ALU bugs, sources drawn from anywhere in set 2.
  k = 4 + (number of injected bug tags actually used).
- out/alu_inject/pure/ ("pure"): same 25-case background, but injected
  cases are only built from *scan-clean* source cases (trace.violations
  is completely empty -- no pre-existing bge/dret/ebreak/... symptom to
  get entangled with the injected one). See PLAN.md task B for why: set 2
  source cases already carry their own bug, so an injected case is really
  "two bugs at once" unless the source itself has zero violations.

Each injected bug is a (mnemonic-class, data condition) pair chosen to
resemble a plausible RTL corner-case bug (see BUGS below). A case is only
used as an injection source for a given bug if that bug's condition is
met at least twice in its trace (a single hit would never reach
MIN_SYS_VIOL=2 in rfb/signature.py and could never form a systematic
violation class regardless of how correct the semantic checker is).

Only regr.log's mtime and the trace.log.gz payload are touched: for a
matched instruction, only its own write value (Regs column, the
"xN=0x...." token for the injected bug's rd) is overwritten with the
buggy recomputed value; regr.log and sim.log(.gz) are copied byte-for-byte
(the injected bug is invisible to the comparator log on purpose -- this
experiment measures whether rfb's trace-level WRONG_RESULT scan alone can
recover it, matching the M2 "second stage" gap noted in PLAN.md section 6).

Does not read golden.csv for anything other than building this script's
*own* synthetic golden.csv (which is the injection experiment's answer
key, not the real one) -- rfb/ itself is never invoked from this file.
"""

from __future__ import annotations

import csv
import gzip
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfb import isa
from rfb.io_utils import open_text

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET2_DIR = os.path.join(REPO_ROOT, "testcase/2026_08_20_samples/problem/benchmark_set_2")
SET1_DIR = os.path.join(REPO_ROOT, "testcase/2026_08_20_samples/problem/benchmark_set_1")
OUT_ROOT = os.path.join(REPO_ROOT, "out/alu_inject")

_MASK32 = 0xFFFFFFFF


def _signed32(v: int) -> int:
    v &= _MASK32
    return v - 0x100000000 if v & 0x80000000 else v


# --- bug definitions --------------------------------------------------------
# Each entry: mnemonics (set), condition(mnemonic, ops, reads) -> bool,
# mutate(mnemonic, ops, reads, orig_write) -> new_write (masked by caller),
# description (for the report).


def _cond_mulhsu_neg(mnemonic, ops, reads):
    if mnemonic != "mulhsu" or len(ops) < 3:
        return False
    a = isa._regval(reads, ops[1])
    return a is not None and _signed32(a) < 0


def _mut_mulhsu_neg(mnemonic, ops, reads, orig):
    return (orig - 1) & _MASK32


def _cond_sra_ge16(mnemonic, ops, reads):
    if mnemonic == "sra" and len(ops) >= 3:
        b = isa._regval(reads, ops[2])
        return b is not None and (b & 0x1F) >= 16
    if mnemonic == "srai" and len(ops) >= 3:
        try:
            imm = isa._parse_imm(ops[2])
        except ValueError:
            return False
        return (imm & 0x1F) >= 16
    if mnemonic == "c.srai" and len(ops) >= 2:
        try:
            imm = isa._parse_imm(ops[1])
        except ValueError:
            return False
        return (imm & 0x1F) >= 16
    return False


def _mut_sra_ge16(mnemonic, ops, reads, orig):
    # Bug: for shamt >= 16, the shifter's sign-extend mux picks the logical
    # (zero-fill) path instead of arithmetic (sign-fill). Recompute the
    # *source* value the same way isa.py would and redo the shift as srl.
    if mnemonic == "sra":
        a = isa._regval(reads, ops[1])
        shamt = isa._regval(reads, ops[2]) & 0x1F
    elif mnemonic == "srai":
        a = isa._regval(reads, ops[1])
        shamt = isa._parse_imm(ops[2]) & 0x1F
    else:  # c.srai: rd doubles as rs1
        a = isa._regval(reads, ops[0])
        shamt = isa._parse_imm(ops[1]) & 0x1F
    if a is None:
        return orig
    return (isa._to_unsigned32(a) >> shamt) & _MASK32


def _cond_sltu_msbdiff(mnemonic, ops, reads):
    if mnemonic != "sltu" or len(ops) < 3:
        return False
    a = isa._regval(reads, ops[1])
    b = isa._regval(reads, ops[2])
    if a is None or b is None:
        return False
    return ((a >> 31) & 1) != ((b >> 31) & 1)


def _mut_sltu_msbdiff(mnemonic, ops, reads, orig):
    return 1 - orig  # comparator bit inverted when the sign bits disagree


def _cond_divu_by1(mnemonic, ops, reads):
    if mnemonic != "divu" or len(ops) < 3:
        return False
    b = isa._regval(reads, ops[2])
    return b is not None and (b & _MASK32) == 1

def _mut_divu_by1(mnemonic, ops, reads, orig):
    return (orig - 1) & _MASK32  # divide-by-one fast path off by one


def _cond_sub_ovf(mnemonic, ops, reads):
    if mnemonic != "sub" or len(ops) < 3:
        return False
    a = isa._regval(reads, ops[1])
    b = isa._regval(reads, ops[2])
    if a is None or b is None:
        return False
    return ((a - b) & _MASK32) == 0x80000000


def _mut_sub_ovf(mnemonic, ops, reads, orig):
    return 0x7FFFFFFF  # signed-overflow corner case saturates instead of wrapping


def _cond_caddi_neg(mnemonic, ops, reads):
    if mnemonic != "c.addi" or len(ops) < 2:
        return False
    try:
        imm = isa._parse_imm(ops[1])
    except ValueError:
        return False
    return imm < 0


def _mut_caddi_neg(mnemonic, ops, reads, orig):
    return (orig + 1) & _MASK32  # compressed-immediate sign extension off by one


BUGS = {
    "inj_mulhsu_neg": (_cond_mulhsu_neg, _mut_mulhsu_neg,
                        "mulhsu 當 rs1（有號）為負時，結果少 1"),
    "inj_sra_shamt_ge16": (_cond_sra_ge16, _mut_sra_ge16,
                            "sra/srai/c.srai 當 shamt >= 16 時做成邏輯右移"),
    "inj_sltu_msbdiff": (_cond_sltu_msbdiff, _mut_sltu_msbdiff,
                          "sltu 當兩運算元最高位不同時，結果反相"),
    "inj_divu_by1": (_cond_divu_by1, _mut_divu_by1,
                      "divu 除數為 1 時，結果少 1"),
    "inj_sub_overflow": (_cond_sub_ovf, _mut_sub_ovf,
                          "sub 當結果為 0x80000000 時，結果變成 0x7fffffff"),
    "inj_c_addi_neg": (_cond_caddi_neg, _mut_caddi_neg,
                        "c.addi 立即數為負時，結果多 1"),
}

RD_INDEX = {
    "mulhsu": 0, "sra": 0, "srai": 0, "c.srai": 0,
    "sltu": 0, "divu": 0, "sub": 0, "c.addi": 0,
}


# --- trace line I/O ----------------------------------------------------------

def _read_trace_lines(path: str) -> list[str]:
    with open_text(path) as f:
        return f.read().splitlines()


def _write_trace_gz(path: str, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    with gzip.open(path, "wt") as f:
        f.write(text)


def _mutate_trace(src_trace_path: str, tag: str) -> tuple[list[str], int]:
    """Return (new_lines, n_mutated)."""
    cond, mut, _ = BUGS[tag]
    lines = _read_trace_lines(src_trace_path)
    out_lines = []
    n_mutated = 0
    header = True
    for line in lines:
        if header:
            out_lines.append(line)
            header = False
            continue
        if not line:
            out_lines.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            out_lines.append(line)
            continue
        mnemonic = parts[4]
        operands = parts[5] if len(parts) >= 6 else ""
        regs_field = parts[6] if len(parts) >= 7 else (parts[5] if len(parts) == 6 else "")
        ops = isa._split_operands(operands)
        reads, writes, pa, store, load = isa.parse_regs(regs_field)
        if not cond(mnemonic, ops, reads):
            out_lines.append(line)
            continue
        rd_idx = RD_INDEX.get(mnemonic)
        if rd_idx is None or rd_idx >= len(ops):
            out_lines.append(line)
            continue
        rd = ops[rd_idx]
        orig = isa._writeval(writes, rd)
        if orig is None:
            out_lines.append(line)
            continue
        new_val = mut(mnemonic, ops, reads, orig) & _MASK32
        token_old = f"{rd}=0x{orig:08x}"
        token_new = f"{rd}=0x{new_val:08x}"
        if token_old not in regs_field:
            # tolerate a different hex-digit width than 8 by falling back to
            # a token-level rewrite instead of skipping the instruction.
            new_regs_tokens = []
            replaced = False
            for tok in regs_field.split():
                if tok.startswith(f"{rd}=0x") and not replaced:
                    new_regs_tokens.append(f"{rd}=0x{new_val:08x}")
                    replaced = True
                else:
                    new_regs_tokens.append(tok)
            if not replaced:
                out_lines.append(line)
                continue
            new_regs_field = " ".join(new_regs_tokens)
        else:
            new_regs_field = regs_field.replace(token_old, token_new, 1)
        if len(parts) >= 7:
            parts[6] = new_regs_field
        elif len(parts) == 6:
            parts[5] = new_regs_field
        out_lines.append("\t".join(parts))
        n_mutated += 1
    return out_lines, n_mutated


# --- source-case condition counting -----------------------------------------

def _count_condition(trace_path: str, tag: str) -> int:
    cond, _, _ = BUGS[tag]
    n = 0
    for line in _read_trace_lines(trace_path)[1:]:
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        mnemonic = parts[4]
        operands = parts[5] if len(parts) >= 6 else ""
        regs_field = parts[6] if len(parts) >= 7 else (parts[5] if len(parts) == 6 else "")
        ops = isa._split_operands(operands)
        reads, writes, pa, store, load = isa.parse_regs(regs_field)
        if cond(mnemonic, ops, reads):
            n += 1
    return n


def _read_set2_rows():
    with open(os.path.join(SET2_DIR, "input.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _read_golden(set_dir):
    with open(os.path.join(set_dir, "golden.csv"), newline="") as f:
        return {row["Case"]: row["Bug"] for row in csv.DictReader(f)}


def _scan_violations_empty(trace_path: str) -> bool:
    """True iff isa.check flags zero violations anywhere in the trace
    (reuses isa.check_safe / ScanContext, same as rfb.parse_trace's own
    full-trace scan, not a re-derivation of the semantics)."""
    from rfb.parse_trace import parse_trace

    info = parse_trace(trace_path, regr_info=None)
    return not info.violations


def _clean_pool():
    """Cases (set_dir, case_row) whose trace.violations is empty, from both
    public sets (PLAN.md task B explicitly allows set 1 for the pure pool)."""
    pool = []
    for set_dir in (SET2_DIR, SET1_DIR):
        with open(os.path.join(set_dir, "input.csv"), newline="") as f:
            for row in csv.DictReader(f):
                trace_path = os.path.join(set_dir, row["Trace Log"])
                if _scan_violations_empty(trace_path):
                    pool.append((set_dir, row))
    return pool


def _select_sources(candidates: list[tuple], golden: dict, set_dir_of, tag: str,
                     target: int = 5) -> list[tuple]:
    """candidates: list of (set_dir, row, count). Greedily diversify by
    original golden-bug label, falling back to highest count."""
    by_label: dict[str, list[tuple]] = {}
    for set_dir, row, n in candidates:
        label = golden.get(row["Case"], "?")
        by_label.setdefault(label, []).append((set_dir, row, n))
    for label in by_label:
        by_label[label].sort(key=lambda t: -t[2])
    chosen = []
    labels_cycle = sorted(by_label)
    i = 0
    while len(chosen) < target and any(by_label[l] for l in labels_cycle):
        label = labels_cycle[i % len(labels_cycle)]
        if by_label[label]:
            chosen.append(by_label[label].pop(0))
        i += 1
    return chosen


def _build_variant(tag_name: str, use_clean_pool: bool, report: list[str]) -> dict:
    """Build one injected benchmark directory. Returns
    {tag: [(set_dir, case_row_used_as_source, out_case_id, n_mutated), ...]}."""
    out_dir = os.path.join(OUT_ROOT, tag_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    set2_rows = _read_set2_rows()
    golden_set2 = _read_golden(SET2_DIR)
    golden_set1 = _read_golden(SET1_DIR)

    def golden_for(set_dir, case):
        return golden_set2[case] if set_dir == SET2_DIR else golden_set1[case]

    # background: the 25 original set-2 cases, unmodified, case ids kept as-is
    rows = []  # (case_id_out, regr, sim, trace, bug_label)
    for row in set2_rows:
        cid = int(row["Case"])
        rows.append((cid, os.path.join(SET2_DIR, row["Regr Log"]),
                     os.path.join(SET2_DIR, row["Sim Log"]),
                     os.path.join(SET2_DIR, row["Trace Log"]),
                     golden_set2[row["Case"]]))

    next_id = max(r[0] for r in rows) + 1

    if use_clean_pool:
        pool = _clean_pool()  # [(set_dir, row)]
    else:
        pool = [(SET2_DIR, row) for row in set2_rows]

    bug_summary = {}
    used_tags = []
    for tag in BUGS:
        candidates = []
        for set_dir, row in pool:
            trace_path = os.path.join(set_dir, row["Trace Log"])
            n = _count_condition(trace_path, tag)
            if n >= 2:
                candidates.append((set_dir, row, n))
        if not candidates:
            chosen = []
        elif not use_clean_pool:
            chosen = _select_sources(candidates, golden_set2, SET2_DIR, tag, target=5)
        else:
            # for the clean pool, diversify by (set label) since golden bug
            # differs across set1/set2 numbering; use case-id string only.
            chosen = sorted(candidates, key=lambda t: -t[2])[:5]

        if not chosen:
            report.append(f"  [{tag_name}] {tag}: 無可用來源 case (需要條件 >= 2 次), 跳過")
            continue
        used_tags.append(tag)
        entries = []
        for set_dir, row, n in chosen:
            src_trace = os.path.join(set_dir, row["Trace Log"])
            src_regr = os.path.join(set_dir, row["Regr Log"])
            src_sim = os.path.join(set_dir, row["Sim Log"])
            new_lines, n_mut = _mutate_trace(src_trace, tag)
            cid = next_id
            next_id += 1
            case_dir = os.path.join(out_dir, f"case_{cid}")
            os.makedirs(case_dir, exist_ok=True)
            shutil.copyfile(src_regr, os.path.join(case_dir, "regr.log"))
            shutil.copyfile(src_sim, os.path.join(case_dir, "sim.log.gz"))
            _write_trace_gz(os.path.join(case_dir, "trace.log.gz"), new_lines)
            rows.append((cid, os.path.join(case_dir, "regr.log"),
                         os.path.join(case_dir, "sim.log.gz"),
                         os.path.join(case_dir, "trace.log.gz"), tag))
            src_label = os.path.basename(set_dir) + "/case_" + row["Case"]
            entries.append((src_label, cid, n, n_mut))
        bug_summary[tag] = entries
        report.append(f"  [{tag_name}] {tag}: {len(entries)} 個來源 case")
        for src_label, cid, n_cond, n_mut in entries:
            report.append(f"      source={src_label} -> case_{cid}  condition_hits={n_cond} mutated_lines={n_mut}")

    # write input.csv / golden.csv
    with open(os.path.join(out_dir, "input.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        for cid, regr, sim, trace, _label in sorted(rows, key=lambda r: r[0]):
            w.writerow([cid, os.path.relpath(regr, out_dir), os.path.relpath(sim, out_dir),
                        os.path.relpath(trace, out_dir)])
    with open(os.path.join(out_dir, "golden.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Bug"])
        for cid, _regr, _sim, _trace, label in sorted(rows, key=lambda r: r[0]):
            w.writerow([cid, label])

    k = 4 + len(used_tags)
    report.append(f"  [{tag_name}] 總 case 數={len(rows)}  k={k} (4 + {len(used_tags)} 個注入 bug)")
    return {"k": k, "n_cases": len(rows), "bug_summary": bug_summary, "used_tags": used_tags}


def main() -> int:
    report: list[str] = []
    report.append("ALU 注入實驗來源選擇報告")
    report.append("")
    for tag_name, desc, use_clean in (
        ("mix", "混合版（來源可為 set2 任一 case）", False),
        ("pure", "純淨版（來源限掃描零違規 case）", True),
    ):
        report.append(f"== {tag_name}: {desc} ==")
        info = _build_variant(tag_name, use_clean, report)
        report.append("")

    report.append("bug 定義:")
    for tag, (_c, _m, desc) in BUGS.items():
        report.append(f"  {tag}: {desc}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, "inject_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
