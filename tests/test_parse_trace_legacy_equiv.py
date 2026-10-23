"""Regression oracle: rfb.parse_trace.parse_trace (M4 two-pass streaming)
must produce byte-identical TraceInfo to tests/_legacy_parse_trace.py (the
frozen pre-M4 single-pass, full-in-memory implementation) for the same
inputs, including edge cases around truncation (_MAX_INSTR) and the
anchor-window cap (_MAX_ANCHOR_WINDOWS).

All synthetic traces here are deliberately semantically-meaningless
(arbitrary pcs/mnemonics/operands): the point is not RISC-V correctness
but that both implementations walk the same field values through
isa.parse_regs/isa.check/isa.is_control_flow identically.
"""

from __future__ import annotations

import gzip
import random

import pytest

from rfb import isa as rfb_isa
from rfb import parse_trace as pt
from rfb.record import Mismatch, RegrInfo
from tests import _legacy_parse_trace as legacy

@pytest.fixture(autouse=True)
def _no_reg_addr(monkeypatch):
    """REG_ADDR (isa._reg_addr) compares the printed registers with the Insn
    bits.  The frozen legacy oracle builds Instr without the Insn column, so it
    can never report REG_ADDR.  Unlike READ_MISMATCH it can be first_violation,
    so stripping it from the result is not enough; turn it off on both sides.
    This is more than a defensive opt-out (2026-10-18 R3-3 correction): the
    fuzz has random next_pc, which looks like a trap after every instruction,
    so REG_ADDR skips those, but the last line of a trace has next_pc None and
    is still checked.  Of 3000 fuzz traces 14 (about 0.47%) fire on that last
    line.  The 0.47% is per fuzz trace, not per seed: test_fuzz_random_traces
    is one seed with 200 traces, so with the opt-out removed about two thirds
    of seeds fail (27 of 40 measured); the fixed seed happens not to.
    REG_ADDR through the same pipeline (in-memory vs streaming) is checked by
    tests/test_reg_addr.py."""
    monkeypatch.setattr(rfb_isa, "_RA_ON", False)


def test_reg_addr_opt_out_is_active():
    # The opt-out guards against failures that only appear with another seed;
    # with the current seed removing the fixture turns no test red, so pin it.
    assert rfb_isa._RA_ON is False


_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents"


def mk_line(pc: int, insn_hex: str, mnemonic: str, operands: str = "", regs: str = "",
            drop_operands_field: bool = False) -> str:
    """One trace.log data line. ``drop_operands_field`` produces a 6-column
    line (operands folded away, regs takes column 5) instead of 7."""
    if drop_operands_field:
        return f"0\t0\t{pc:x}\t{insn_hex}\t{mnemonic}\t{regs}"
    return f"0\t0\t{pc:x}\t{insn_hex}\t{mnemonic}\t{operands}\t{regs}"


def write_trace(tmp_path, name: str, lines: list[str], gz: bool) -> str:
    text = "\n".join([_HEADER] + lines) + "\n"
    if gz:
        path = tmp_path / f"{name}.log.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path = tmp_path / f"{name}.log"
        path.write_text(text)
    return str(path)


def _make_regr(ibex_pc: int, ibex_text: str, ibex_val: int | None) -> RegrInfo:
    m = Mismatch(
        n=1, ibex_idx=0, ibex_pc=ibex_pc, ibex_text=ibex_text, ibex_rd=None,
        ibex_val=ibex_val, spike_idx=0, spike_pc=0, spike_text="", spike_rd=None,
        spike_val=None,
    )
    return RegrInfo(kind="FAILED", test_name=None, mismatches=[m], matched=0,
                     mismatch_count=1, raw=[])


def _run(mod, path, regr_info):
    """Run ``mod.parse_trace``, returning ("ok", dict) or ("err", (type,
    message)). isa.py's operand parsing has pre-existing gaps that can
    raise on sufficiently adversarial operand text (out of scope for this
    equivalence check -- both implementations call the exact same
    isa.check/isa.parse_regs with the same field values, so a crash there
    is not a legacy/new divergence and comparing crash-for-crash is the
    correct equivalence check)."""
    try:
        return "ok", mod.parse_trace(path, regr_info).to_dict()
    except Exception as e:  # noqa: BLE001
        return "err", (type(e).__name__, str(e))


def _strip_read_mismatch(result):
    """READ_MISMATCH (isa.check_safe's register read-after-write check) does
    not exist in the frozen legacy oracle, and the fuzz traces are random
    (reads and writes of the same register never agree), so the new result
    is full of it. Filter it out before comparing: from ``violations``,
    ``violation_samples`` and ``anchor_window_violations`` (where it is the
    only violation at that instruction, legacy has None). first_violation
    is unaffected: parse_trace never records READ_MISMATCH there.

    LOAD_EXT / MEM_MISMATCH / CSR_MISMATCH (2026-10-09) are not stripped: the
    fuzz cannot reach them.
    LOAD_EXT: ``check()`` runs ``_mem_operands`` first, and the fuzz operands
    are not in ``imm(rs1)`` form, so it returns before LOAD_EXT (a next_pc
    condition would not help: the last instruction of every fuzz trace has
    next_pc None).  If a future fuzz change makes the operands ``imm(rs1)``,
    do NOT strip LOAD_EXT: it comes from the ``check()`` that both sides
    share, so both sides report it identically and stripping would create a
    false divergence.
    MEM_MISMATCH / CSR_MISMATCH: the legacy oracle never calls ``observe``, so
    its shadows stay empty and it can never report them; the new side only
    could with a same-PA access pair or a CSR mnemonic, which the fuzz does
    not produce.  If those become reachable, strip them here, and handle
    ``first_violation`` as well as ``violations`` (parse_trace can record
    them there)."""
    if result[0] != "ok":
        return result
    d = dict(result[1])
    d["violations"] = {
        k: n for k, n in d["violations"].items() if not k.startswith("READ_MISMATCH:")
    }
    d["violation_samples"] = [
        v for v in d["violation_samples"] if v["kind"] != "READ_MISMATCH"
    ]
    d["anchor_window_violations"] = [
        [None if (v is not None and v["kind"] == "READ_MISMATCH") else v for v in w]
        for w in d["anchor_window_violations"]
    ]
    return "ok", d


def _assert_equiv(monkeypatch, path, regr_info=None, force_stream=False):
    if force_stream:
        monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", 0)
    new_result = _strip_read_mismatch(_run(pt, path, regr_info))
    legacy_result = _run(legacy, path, regr_info)
    assert new_result == legacy_result


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("force_stream", [False, True])
def test_truncation_with_trailing_bad_lines(tmp_path, monkeypatch, gz, force_stream):
    monkeypatch.setattr(pt, "_MAX_INSTR", 10)
    monkeypatch.setattr(legacy, "_MAX_INSTR", 10)
    lines = []
    for i in range(15):
        lines.append(mk_line(0x1000 + i * 4, "00000013", "addi", "x1,x0,1"))
    # bad lines interleaved, including some after the truncation cutoff
    lines.insert(3, "too\tfew\tcols")
    lines.append("bad\tpc\tZZZZ\t13\taddi\tx1,x0,1\t")
    lines.append(mk_line(0x2000, "13", "addi", "x1,x0,1"))
    path = write_trace(tmp_path, "trunc", lines, gz)
    _assert_equiv(monkeypatch, path, force_stream=force_stream)


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("force_stream", [False, True])
def test_anchor_windows_overlap_and_cap(tmp_path, monkeypatch, gz, force_stream):
    monkeypatch.setattr(pt, "_MAX_ANCHOR_WINDOWS", 8)
    monkeypatch.setattr(legacy, "_MAX_ANCHOR_WINDOWS", 8)
    anchor_pc = 0x1000
    anchor_mnemonic = "myspecial"
    lines = []
    gap = 3  # < _ANCHOR_AFTER (4): consecutive anchor windows overlap
    n_total = 30
    for i in range(n_total):
        if i % gap == 0:
            lines.append(mk_line(anchor_pc, "00000013", anchor_mnemonic, "x1,x0,1"))
        else:
            lines.append(mk_line(0x2000 + i * 4, "00000013", "addi", "x1,x0,1"))
    n_hits = sum(1 for i in range(n_total) if i % gap == 0)
    assert n_hits >= 9
    path = write_trace(tmp_path, "anchor", lines, gz)
    regr_info = _make_regr(anchor_pc, anchor_mnemonic, ibex_val=None)
    _assert_equiv(monkeypatch, path, regr_info=regr_info, force_stream=force_stream)


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("force_stream", [False, True])
def test_anchor_at_head_and_tail(tmp_path, monkeypatch, gz, force_stream):
    anchor_pc = 0x3000
    anchor_mnemonic = "headtail"
    lines = []
    # anchor as the very first instruction: lookback (< 16) is empty
    lines.append(mk_line(anchor_pc, "00000013", anchor_mnemonic, "x1,x0,1"))
    for i in range(10):
        lines.append(mk_line(0x4000 + i * 4, "00000013", "addi", "x1,x0,1"))
    # anchor two instructions before EOF: "after" window (< 4) is truncated
    lines.append(mk_line(anchor_pc, "00000013", anchor_mnemonic, "x1,x0,1"))
    lines.append(mk_line(0x5000, "00000013", "addi", "x1,x0,1"))
    path = write_trace(tmp_path, "headtail", lines, gz)
    regr_info = _make_regr(anchor_pc, anchor_mnemonic, ibex_val=None)
    _assert_equiv(monkeypatch, path, regr_info=regr_info, force_stream=force_stream)


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("force_stream", [False, True])
def test_malformed_lines(tmp_path, monkeypatch, gz, force_stream):
    lines = [
        "not\tenough\tcols\there",              # < 5 columns
        "0\t0\tZZZZZZZZ\t13\taddi\tx1,x0,1\t",  # bad pc hex
        "0\t0\t1000\tGGGG\taddi\tx1,x0,1\t",     # bad insn hex
        mk_line(0x1004, "13", "addi", "x1,x0,1"),   # compressed (size 2)
        mk_line(0x1006, "00000013", "addi", "x1,x0,1"),  # 32-bit (size 4)
        mk_line(0x100A, "0013", "addi", "", "", drop_operands_field=True),  # 6 cols
    ]
    path = write_trace(tmp_path, "malformed", lines, gz)
    _assert_equiv(monkeypatch, path, force_stream=force_stream)


@pytest.mark.parametrize("gz", [False, True])
def test_n_instr_below_tail_deque_size(tmp_path, monkeypatch, gz):
    lines = [mk_line(0x1000 + i * 4, "00000013", "addi", "x1,x0,1") for i in range(5)]
    assert len(lines) < 64  # < _TAIL_N
    path = write_trace(tmp_path, "short", lines, gz)
    _assert_equiv(monkeypatch, path)


def test_c_addi4spn_shadow_diverges_from_legacy(tmp_path):
    # instr0 writes x2 = 0x100 (an ordinary addi, no c.addi4spn involved
    # yet). instr1 is a c.addi4spn whose Regs field carries no x2 read (as
    # real tracer output never does) and whose actual x8 write (0x999)
    # does not equal x2 + imm (0x100 + 16 = 0x110). rfb.parse_trace tracks
    # x2 via isa.observe()/ctx.sp_shadow and so recomputes the expected
    # value and flags WRONG_RESULT; the frozen legacy oracle never calls
    # isa.observe() at all, so ctx.sp_shadow stays None and legacy's check
    # for this instruction is skipped entirely (see the comment on
    # _MNEMONICS above). This is the intended, deliberate divergence.
    lines = [
        mk_line(0x1000, "00000013", "addi", "x2,x0,0x100", "x2=0x00000100"),
        mk_line(0x1004, "0000e013", "c.addi4spn", "x8,x2,16", "x8=0x00000999"),
    ]
    path = write_trace(tmp_path, "addi4spn_diverge", lines, gz=False)

    new_info = pt.parse_trace(path)
    legacy_info = legacy.parse_trace(path)

    assert new_info.violations.get("WRONG_RESULT:c.addi4spn") == 1
    assert "WRONG_RESULT:c.addi4spn" not in legacy_info.violations


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("force_stream", [False, True])
def test_branch_jump_trap_violations(tmp_path, monkeypatch, gz, force_stream):
    lines = [
        # beq taken (rs1==rs2) but next line falls through -> BRANCH_NOT_TAKEN
        mk_line(0x1000, "00000013", "beq", "x1,x1,1008", "x1:0x00000005"),
        mk_line(0x1004, "00000013", "addi", "x1,x0,1"),
        # bne not-taken condition wrongly jumps to target -> BRANCH_WRONGLY_TAKEN
        mk_line(0x1008, "00000013", "bne", "x2,x2,1020", "x2:0x00000009"),
        mk_line(0x1020, "00000013", "addi", "x1,x0,1"),
        # jal with a target that doesn't match next_pc or fallthrough -> JUMP violation
        mk_line(0x1024, "00000013", "jal", "x1,2000"),
        mk_line(0x1028, "00000013", "addi", "x1,x0,1"),  # wrong target on purpose
        # ecall (trap instr): next_pc unrelated to fallthrough
        mk_line(0x102C, "00000013", "ecall", ""),
        mk_line(0x9000, "00000013", "addi", "x1,x0,1"),
        # mret (xret): fallthrough mismatch
        mk_line(0x9004, "00000013", "mret", ""),
        mk_line(0xA000, "00000013", "addi", "x1,x0,1"),
    ]
    path = write_trace(tmp_path, "violations", lines, gz)
    _assert_equiv(monkeypatch, path, force_stream=force_stream)


# --------------------------------------------------------------------------
# Fuzz: many small random traces, compared against the legacy oracle, both
# in-memory and forced-streaming.
# --------------------------------------------------------------------------

# c.addi4spn is deliberately excluded from this fuzz list. Its rs1 (x2) is
# never printed as a read by the real tracer, so rfb.parse_trace relies on
# isa.observe()/ctx.sp_shadow (see rfb/isa.py, "2026-09-28 ALU 覆蓋") to
# recompute it -- but this frozen legacy oracle never calls isa.observe()
# at all, so ctx.sp_shadow stays None forever and legacy's c.addi4spn check
# is always skipped (isa.check's "x2" not in reads and ctx.sp_shadow is
# not None guard never enters the shadow path). Including c.addi4spn here
# would make new-vs-legacy fuzz comparisons diverge on every trial that
# happens to draw it with a mismatching x8 write, which is an intended
# behavior change (new code correctly flags it), not a bug. That intended
# divergence is pinned down explicitly by
# test_c_addi4spn_shadow_diverges_from_legacy below instead.
_MNEMONICS = [
    "addi", "add", "sub", "and", "or", "xor", "sll", "srl", "sra",
    "beq", "bne", "blt", "bltu", "bge", "bgeu",
    "jal", "jalr", "c.j", "c.jr", "c.jalr",
    "ecall", "ebreak", "mret", "dret",
    "lw", "sw", "lb", "sb",
    "c.li", "c.mv", "c.addi",
    "unknownop", "c.nop",
]


def _fuzz_operands(rng: random.Random, mnemonic: str) -> str:
    # Only digit-only numeric tokens (no letters, no parens): safe as both
    # a decimal/hex immediate (_parse_imm) and a bare-hex branch/jump
    # target (_parse_target) regardless of which operand position a given
    # mnemonic's isa.py handler happens to read it from -- unlike e.g.
    # "16(x5)", a memory-operand-shaped token, which crashes _parse_imm if
    # some *other* mnemonic (whose handler expects a plain immediate,
    # e.g. c.addi) is paired with it by chance. Exercising the mem-operand
    # form specifically is out of scope here: equivalence, not isa.py
    # coverage.
    # isa.py's jal/c.j/c.jal handler now guards against an empty operand
    # list (returns None instead of indexing ops[-1]), so an empty string
    # is safe to include here too.
    choices = [
        "x1,x2,3", "x1,x2,1008", f"{rng.randrange(0, 0x10000)}",
        "x0,x0,0", "x1,x2", "x1,4", "",
    ]
    return rng.choice(choices)


def _fuzz_regs(rng: random.Random) -> str:
    toks = []
    for _ in range(rng.randrange(0, 4)):
        kind = rng.choice(["read", "write", "pa", "store", "load", "junk"])
        reg = f"x{rng.randrange(0, 32)}"
        val = rng.randrange(0, 0xFFFFFFFF)
        if kind == "read":
            toks.append(f"{reg}:0x{val:x}")
        elif kind == "write":
            toks.append(f"{reg}=0x{val:x}")
        elif kind == "pa":
            toks.append(f"PA:0x{val:x}")
        elif kind == "store":
            toks.append(f"store:0x{val:x}")
        elif kind == "load":
            toks.append(f"load:0x{val:x}")
        else:
            toks.append("???notatoken")
    return " ".join(toks)


def _gen_random_trace(rng: random.Random, n_lines: int) -> tuple[list[str], int | None, str | None]:
    lines = []
    pcs_seen: list[int] = []
    mnems_seen: list[str] = []
    for _ in range(n_lines):
        bad_kind = rng.random() < 0.08
        if bad_kind:
            variant = rng.choice(["short", "badpc", "badinsn"])
            if variant == "short":
                lines.append("x\ty")
                continue
            if variant == "badpc":
                lines.append(mk_line(0, "00000013", "addi", drop_operands_field=False)
                              .replace("\t0\t", "\tZZZZZZZG\t", 1))
                continue
            pc = rng.randrange(0, 0xFFFFFFFF)
            mnemonic = rng.choice(_MNEMONICS)
            lines.append(mk_line(pc, "GGGG", mnemonic, _fuzz_operands(rng, mnemonic),
                                  _fuzz_regs(rng)))
            continue

        # occasionally repeat a previously used pc, to exercise implicit
        # jumps / anchor pc matching in fuzz too
        if pcs_seen and rng.random() < 0.3:
            pc = rng.choice(pcs_seen)
        else:
            pc = rng.randrange(0, 0xFFFFFFFF)
        insn_hex = rng.choice(["13", "0013", "00000013", "deadbeef", "1"])
        mnemonic = rng.choice(_MNEMONICS)
        operands = _fuzz_operands(rng, mnemonic)
        regs = _fuzz_regs(rng)
        drop = rng.random() < 0.1
        lines.append(mk_line(pc, insn_hex, mnemonic, operands, regs, drop_operands_field=drop))
        pcs_seen.append(pc)
        mnems_seen.append(mnemonic)

    anchor_pc = None
    anchor_text = None
    if pcs_seen and rng.random() < 0.6:
        idx = rng.randrange(0, len(pcs_seen))
        anchor_pc = pcs_seen[idx]
        anchor_text = mnems_seen[idx]
    return lines, anchor_pc, anchor_text


class _SafeCheckIsaProxy:
    """Stand-in for the ``isa`` name inside tests/_legacy_parse_trace.py.

    Delegates every attribute to the real ``rfb.isa`` module except
    ``check()``, which goes through ``isa.check_safe`` instead (discarding
    the error string, keeping only the violation-or-None). This gives the
    frozen legacy implementation the exact same safety net
    ``rfb.parse_trace`` gives the new module, so a trial that would
    otherwise crash legacy can still be compared byte-for-byte.
    """

    def __getattr__(self, name):
        return getattr(rfb_isa, name)

    def check(self, prev, cur, next_pc, ctx):
        v, _err = rfb_isa.check_safe(prev, cur, next_pc, ctx)
        return v


@pytest.mark.parametrize("force_stream", [False, True])
def test_fuzz_random_traces(tmp_path, monkeypatch, force_stream):
    rng = random.Random(20260928)
    monkeypatch.setattr(pt, "_MAX_INSTR", 500)
    monkeypatch.setattr(legacy, "_MAX_INSTR", 500)
    if force_stream:
        monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", 0)
    n_proxy_trials = 0
    for trial in range(200):
        n_lines = rng.randrange(0, 40)
        lines, anchor_pc, anchor_text = _gen_random_trace(rng, n_lines)
        path = write_trace(tmp_path, f"fuzz_{trial}", lines, gz=(trial % 2 == 0))
        regr_info = None
        if anchor_pc is not None:
            regr_info = _make_regr(anchor_pc, anchor_text, ibex_val=None)
        new_result = _strip_read_mismatch(_run(pt, path, regr_info))
        legacy_result = _run(legacy, path, regr_info)
        if (
            legacy_result[0] == "err"
            and legacy_result[1][0] in ("ValueError", "IndexError")
        ):
            # The frozen legacy oracle calls isa.check() directly and
            # crashes on this trial's malformed operand token; the new
            # parse_trace isolates the same isa.check() failure per
            # instruction via isa.check_safe instead of letting it abort
            # the whole trace (the intended behavior change). To still get
            # a byte-identical comparison here, rerun legacy with its
            # ``isa`` name swapped for _SafeCheckIsaProxy -- i.e. "legacy
            # with the same safety net new code has" -- which must now
            # succeed. Its to_dict() must equal the new module's to_dict()
            # once the "check failed: ..." parse_errors entries (which only
            # the new module records) are stripped back out.
            n_proxy_trials += 1
            with monkeypatch.context() as m:
                m.setattr(legacy, "isa", _SafeCheckIsaProxy())
                proxy_legacy_result = _run(legacy, path, regr_info)
            assert proxy_legacy_result[0] == "ok", f"trial={trial} n_lines={n_lines}"
            assert new_result[0] == "ok", f"trial={trial} n_lines={n_lines}"
            new_dict = new_result[1]
            filtered_errors = [
                e for e in new_dict["parse_errors"]
                if not (e.startswith("instr ") and "check failed:" in e)
                and "instructions total" not in e
            ]
            assert len(filtered_errors) < len(new_dict["parse_errors"]), (
                f"trial={trial} n_lines={n_lines}"
            )
            new_dict_filtered = dict(new_dict, parse_errors=filtered_errors)
            assert new_dict_filtered == proxy_legacy_result[1], (
                f"trial={trial} n_lines={n_lines}"
            )
            continue
        assert new_result == legacy_result, f"trial={trial} n_lines={n_lines}"
    print(f"\n[fuzz] proxy-comparison trials: {n_proxy_trials}/200")
