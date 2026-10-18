#!/usr/bin/env python3
"""ALU semantic-check coverage report over the public set (task A, see
PLAN.md "ALU 覆蓋量測").

For each ADDSUB/LOGIC/SHIFT/SLT/MUL/DIV mnemonic that appears in the 32
public-set traces, counts:

- appearances: how many trace lines have that mnemonic.
- recomputed: how many of those were actually recomputed and compared
  against the traced write value (i.e. ``rfb.isa.check`` did not bail out
  early for missing data).
- skipped, broken down by reason: missing_reads (a required source
  register's traced value wasn't printed), operand_unparseable (too few
  operands / bad immediate token), rd_x0 (destination is x0, always 0,
  ``isa.check`` deliberately never flags it), missing_write (rd not among
  the traced writes), other (an exception from parsing).
- wrong_result: how many of the recomputed instances rfb.isa.check actually
  flagged as WRONG_RESULT (should be 0 on the public set -- no public bug
  is an ALU result bug).

This reuses rfb.isa's own operand-resolution helpers and compute functions
(``_arith_operands``, ``_compute_reg_reg``, ``_compute_reg_imm``,
``_regval``, the ``_REG_REG``/``_REG_IMM``/... mnemonic-class tables) rather
than reimplementing ISA semantics: the only new code here is the
book-keeping needed to tell "skipped" apart from "recomputed and matched"
-- ``isa.check`` returns ``None`` for both, by design, since only the
caller (this diagnostic tool) needs the distinction.

Does NOT read golden.csv (coverage is a property of the trace + isa.py
only, no answer key involved).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfb import isa
from rfb.io_utils import open_text
from rfb.parse_trace import _split_trace_line  # file-format parsing, not semantics
from rfb.record import Instr

ARITH_CLASSES = {"ADDSUB", "LOGIC", "SHIFT", "SLT", "MUL", "DIV"}

_REASON_LABELS = {
    "missing_reads": "缺 reads",
    "operand_unparseable": "operand 無法解析",
    "rd_x0": "rd 為 x0",
    "missing_write": "writes 缺 rd",
    "other": "其他",
}


def _has_reg(reg: str, reads: dict) -> bool:
    """Mirrors isa._regval's "is this operand resolvable" check without
    duplicating the arithmetic itself."""
    return reg == "x0" or reg in reads


def _looks_missing_reads(mnemonic: str, ops: list, reads: dict) -> bool:
    """True iff the operand *shape* is fine (isa._arith_operands would have
    proceeded to compute) but a required source register's value wasn't in
    ``reads`` -- i.e. the reason isa._arith_operands returned None is a
    missing read, not a malformed operand list. Uses isa's own mnemonic-set
    tables (_REG_REG, _REG_IMM, ...) so the operand shape per mnemonic is
    defined in exactly one place (isa.py)."""
    if mnemonic in isa._REG_REG:
        return len(ops) >= 3 and not (_has_reg(ops[1], reads) and _has_reg(ops[2], reads))
    if mnemonic in isa._REG_IMM:
        return len(ops) >= 3 and not _has_reg(ops[1], reads)
    if mnemonic in isa._COMPR_RD_RS2:
        if len(ops) < 2:
            return False
        if mnemonic == "c.mv":
            return not _has_reg(ops[1], reads)
        return not (_has_reg(ops[1], reads) and _has_reg(ops[0], reads))
    if mnemonic in isa._COMPR_RD_IMM:
        if len(ops) < 2:
            return False
        return not _has_reg(ops[0], reads)
    if mnemonic == "c.li":
        return False
    if mnemonic == "c.addi4spn":
        return len(ops) >= 3 and not _has_reg(ops[1], reads)
    if mnemonic == "c.addi16sp":
        return len(ops) >= 2 and not _has_reg(ops[0], reads)
    if mnemonic in ("lui", "c.lui"):
        return False
    return False


def _diagnose(mnemonic: str, instr: Instr, sp_shadow: int | None = None) -> tuple[str, str | None]:
    """Return (status, reason). status is "ok" (recomputed & compared,
    caller should also run isa.check_safe to learn match/mismatch) or
    "skip" (reason is one of _REASON_LABELS' keys).

    2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: ``sp_shadow``
    mirrors the ``ScanContext.sp_shadow`` isa.check() now consults for
    c.addi4spn (see docs/ARCHITECTURE.md 4.3), so this diagnostic reflects
    the same effective reads isa.check() would use.
    """
    ops = isa._split_operands(instr.operands)
    reads = instr.reads
    if (
        mnemonic == "c.addi4spn"
        and "x2" not in reads
        and sp_shadow is not None
    ):
        reads = dict(reads)
        reads["x2"] = sp_shadow
    try:
        if mnemonic == "auipc":
            if len(ops) < 2:
                return "skip", "operand_unparseable"
            rd = ops[0]
            try:
                isa._parse_imm(ops[1])
            except ValueError:
                return "skip", "operand_unparseable"
        else:
            res = isa._arith_operands(mnemonic, ops, reads)
            if res is None:
                if _looks_missing_reads(mnemonic, ops, reads):
                    return "skip", "missing_reads"
                return "skip", "operand_unparseable"
            rd, _ = res
    except (ValueError, IndexError):
        return "skip", "other"

    if rd == "x0":
        return "skip", "rd_x0"
    if rd not in instr.writes:
        return "skip", "missing_write"
    return "ok", None


def _iter_instrs(trace_path: str):
    with open_text(trace_path) as f:
        header_skipped = False
        index = 0
        for line in f:
            line = line.rstrip("\r\n")
            if not header_skipped:
                header_skipped = True
                continue
            if not line:
                continue
            parts = line.split("\t")
            fields_ = _split_trace_line(parts)
            if fields_ is None:
                continue
            pc_hex, insn_hex, mnemonic, operands, regs_field = fields_
            try:
                pc = int(pc_hex, 16)
            except ValueError:
                continue
            size = 2 if len(insn_hex) <= 4 else 4
            reads, writes, pa, store, load = isa.parse_regs(regs_field)
            yield Instr(
                index=index, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
                reads=reads, writes=writes, pa=pa, store=store, load=load,
            )
            index += 1


def scan_case(case_id, trace_path: str, stats: dict, wrong_result_instances: list) -> None:
    ctx = isa.ScanContext(trap_entries=set())
    for instr in _iter_instrs(trace_path):
        mnemonic = instr.mnemonic
        if isa.mclass(mnemonic) not in ARITH_CLASSES:
            # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗: observe() runs for
            # every instruction (not just ARITH ones), mirroring
            # parse_trace.py -- an sp-updating instruction outside the
            # ARITH classes (e.g. a STORE using x2 as base) must still keep
            # the shadow in sync.
            isa.observe(ctx, instr)
            continue
        row = stats.setdefault(mnemonic, {
            "appearances": 0, "recomputed": 0, "wrong_result": 0,
            **{f"skip_{r}": 0 for r in _REASON_LABELS},
        })
        row["appearances"] += 1
        status, reason = _diagnose(mnemonic, instr, ctx.sp_shadow)
        if status == "skip":
            row[f"skip_{reason}"] += 1
            isa.observe(ctx, instr)
            continue
        row["recomputed"] += 1
        v, err = isa.check_safe(None, instr, None, ctx)
        if err is not None:
            # _diagnose said "ok" but check() itself raised -- extremely
            # unlikely (same helpers), but keep the counts consistent.
            row["recomputed"] -= 1
            row["skip_other"] += 1
            isa.observe(ctx, instr)
            continue
        if v is not None and v.kind == "WRONG_RESULT":
            row["wrong_result"] += 1
            wrong_result_instances.append((case_id, trace_path, instr.index, instr.pc, mnemonic))
        isa.observe(ctx, instr)


def _read_input_csv(input_csv: str):
    base_dir = os.path.dirname(os.path.abspath(input_csv))
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        header_map = {h.strip().lower(): h for h in reader.fieldnames}
        case_key = header_map["case"]
        trace_key = header_map.get("trace log")
        for row in reader:
            case_id = row[case_key].strip()
            trace_path = os.path.join(base_dir, row[trace_key].strip())
            yield case_id, trace_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alu_coverage")
    parser.add_argument("--out", default="out/alu_coverage.txt")
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csvs = [
        os.path.join(repo_root, "testcase/2026_08_20_samples/problem/benchmark_set_1/input.csv"),
        os.path.join(repo_root, "testcase/2026_08_20_samples/problem/benchmark_set_2/input.csv"),
    ]

    stats: dict[str, dict] = {}
    wrong_result_instances: list = []
    n_cases = 0
    for input_csv in input_csvs:
        set_name = os.path.basename(os.path.dirname(input_csv))
        for case_id, trace_path in _read_input_csv(input_csv):
            n_cases += 1
            scan_case(f"{set_name}/case_{case_id}", trace_path, stats, wrong_result_instances)

    lines = []
    lines.append(f"ALU coverage over {n_cases} public-set cases (set 1 + set 2)")
    lines.append("")
    header = (
        f"{'mnemonic':<10} {'mclass':<8} {'appear':>7} {'recomp':>7} {'recomp%':>8} "
        f"{'wrong':>6} {'miss_rd':>8} {'bad_op':>7} {'rd_x0':>6} {'miss_wr':>8} {'other':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    low_coverage = []
    for mnemonic in sorted(stats):
        row = stats[mnemonic]
        appear = row["appearances"]
        recomp = row["recomputed"]
        pct = (100.0 * recomp / appear) if appear else 0.0
        lines.append(
            f"{mnemonic:<10} {isa.mclass(mnemonic):<8} {appear:>7} {recomp:>7} {pct:>7.1f}% "
            f"{row['wrong_result']:>6} {row['skip_missing_reads']:>8} "
            f"{row['skip_operand_unparseable']:>7} {row['skip_rd_x0']:>6} "
            f"{row['skip_missing_write']:>8} {row['skip_other']:>6}"
        )
        if appear > 0 and pct < 90.0:
            low_coverage.append((mnemonic, appear, recomp, pct, row))

    lines.append("")
    lines.append("mnemonics with recompute ratio < 90%:")
    if not low_coverage:
        lines.append("  (none)")
    else:
        for mnemonic, appear, recomp, pct, row in sorted(low_coverage, key=lambda t: t[3]):
            reasons = sorted(
                ((r, row[f"skip_{r}"]) for r in _REASON_LABELS if row[f"skip_{r}"] > 0),
                key=lambda t: -t[1],
            )
            reason_str = ", ".join(f"{_REASON_LABELS[r]}={n}" for r, n in reasons)
            lines.append(
                f"  {mnemonic}: {recomp}/{appear} = {pct:.1f}% -- 主要跳過原因: {reason_str}"
            )

    lines.append("")
    lines.append(f"WRONG_RESULT instances found on public set: {len(wrong_result_instances)}")
    for case_id, trace_path, index, pc, mnemonic in wrong_result_instances:
        lines.append(f"  {case_id} instr#{index} pc=0x{pc:x} {mnemonic} ({trace_path})")

    out_text = "\n".join(lines) + "\n"
    out_path = os.path.join(repo_root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out_text)
    print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
