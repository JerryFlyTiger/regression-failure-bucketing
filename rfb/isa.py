"""RV32IMC + Zicsr semantic checker.

``MCLASS`` table, ``mclass()``, ``is_control_flow()``, ``ScanContext`` and
the ``Regs`` field parsing helper ``parse_regs()`` are M1 scope.
``check()`` implements docs/ARCHITECTURE.md section 4.3 (M2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rfb.record import Violation

# ARCHITECTURE.md section 4.2
MCLASS: dict[str, str] = {}


def _register(cls: str, mnemonics: str) -> None:
    for m in mnemonics.split():
        MCLASS[m] = cls


# beq/bne are split: Ibex ibex_alu.sv:126,155-156 shares is_equal between EQ and NE,
# but ibex_decoder.sv:934-935 decodes them separately by funct3, so a bug can hit
# only one of them.  A shared-comparator bug still lands in one bucket (see PLAN.md section 6).
_register("BR_BEQ", "beq c.beqz")
_register("BR_BNE", "bne c.bnez")
_register("BR_LT", "blt bltu")
_register("BR_GE", "bge bgeu")
_register("JUMP", "jal jalr c.j c.jal c.jr c.jalr")
_register("TRAP_INSTR", "ecall ebreak c.ebreak")
_register("XRET", "mret dret")
_register(
    "ADDSUB",
    "add addi sub c.add c.addi c.sub c.mv c.li c.addi4spn c.addi16sp lui auipc c.lui",
)
_register("LOGIC", "and or xor andi ori xori c.and c.or c.xor c.andi")
_register("SHIFT", "sll srl sra slli srli srai c.slli c.srli c.srai")
_register("SLT", "slt sltu slti sltiu")
_register("MUL", "mul mulh mulhsu mulhu")
_register("DIV", "div divu rem remu")
_register("LOAD", "lb lh lw lbu lhu c.lw c.lwsp")
_register("STORE", "sb sh sw c.sw c.swsp")
_register("CSR", "csrrw csrrs csrrc csrrwi csrrsi csrrci")
_register("FENCE", "fence fence.i")


def mclass(mnemonic: str) -> str:
    """Return the instruction class for a mnemonic. Never raises."""
    return MCLASS.get(mnemonic, "OTHER")


_CONTROL_FLOW_CLASSES = {"BR_BEQ", "BR_BNE", "BR_LT", "BR_GE", "JUMP", "TRAP_INSTR", "XRET"}


def is_control_flow(mnemonic: str) -> bool:
    return mclass(mnemonic) in _CONTROL_FLOW_CLASSES


# 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: per-mnemonic base ALU
# operation, one level finer than ``mclass`` for the M/DIV/ADDSUB/LOGIC/
# SHIFT/SLT groups -- e.g. "sra" and "c.srai" share ALU_OP but differ in
# mclass only if their forms diverge (they don't; the point of this table
# is to let ``cluster._alu_op_link`` treat *different encodings of the same
# base ALU op* as linked evidence). See docs/ARCHITECTURE.md 4.2.
ALU_OP: dict[str, str] = {}


def _register_op(op: str, mnemonics: str) -> None:
    for m in mnemonics.split():
        ALU_OP[m] = op


_register_op("add", "add addi c.add c.addi c.mv c.li c.addi4spn c.addi16sp")
_register_op("sub", "sub c.sub")
_register_op("and", "and andi c.and c.andi")
_register_op("or", "or ori c.or")
_register_op("xor", "xor xori c.xor")
_register_op("sll", "sll slli c.slli")
_register_op("srl", "srl srli c.srli")
_register_op("sra", "sra srai c.srai")
_register_op("slt", "slt slti")
_register_op("sltu", "sltu sltiu")
_register_op("lui", "lui c.lui")
_register_op("auipc", "auipc")
_register_op("mul", "mul")
_register_op("mulh", "mulh")
_register_op("mulhsu", "mulhsu")
_register_op("mulhu", "mulhu")
_register_op("div", "div")
_register_op("divu", "divu")
_register_op("rem", "rem")
_register_op("remu", "remu")


def alu_op(mn: str) -> str | None:
    """Base ALU operation for ``mn``, or None if ``mn`` isn't an ALU
    mnemonic (control flow, LOAD/STORE, CSR, ...). Never raises."""
    return ALU_OP.get(mn)


# kind -> mclass -> polarity token of the is_equal error ("EQ-": is_equal
# wrongly false, "EQ+": wrongly true). Tokens can't collide with mclass names.
_EQ_POLARITY = {
    "BRANCH_NOT_TAKEN": {"BR_BEQ": "BR_EQ-", "BR_BNE": "BR_EQ+"},
    "BRANCH_WRONGLY_TAKEN": {"BR_BEQ": "BR_EQ+", "BR_BNE": "BR_EQ-"},
}


def viol_group(kind: str, mn: str, detail: dict | None = None) -> str:
    """2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: grouping key for
    a violation ``(kind, mn)`` used by hard_key/sys_viol_class (see
    docs/ARCHITECTURE.md 5.3). WRONG_RESULT on an ALU mnemonic groups at
    mnemonic granularity (finer than mclass, so e.g. sra vs mulhsu never
    share a hard_key); BRANCH_NOT_TAKEN / BRANCH_WRONGLY_TAKEN on beq/bne
    (BR_BEQ / BR_BNE) group by is_equal error polarity (``_EQ_POLARITY``:
    BR_EQ- / BR_EQ+, see docs/ARCHITECTURE.md 4.2); everything else
    (including WRONG_RESULT on LOAD/CSR via pcdelta, and every other
    control-flow kind, e.g. TRAP_NOT_TAKEN, BRANCH_WRONG_TARGET) keeps the coarser
    mclass grouping. READ_MISMATCH (a register read differing from the last
    value written to it) groups as "REGREAD" whatever the mnemonic, except
    that ``detail["fwd"]`` true gives "REGREAD_FWD" (2026-10-08 REGREAD 細分,
    PLAN.md 4c): Ibex has a single WB->ID forwarding path, so a missed
    forward reads the value from before the write; hence only a read of the
    stale value right after the write (the immediately preceding retired
    instruction) counts.  The hardware rationale is from the architect's
    recollection of the Ibex source; no RTL is available locally, so it has
    not been verified line by line.

    2026-10-09: LOAD_EXT -> "LD_SEXT" (lb/lh) or "LD_ZEXT" (lbu/lhu);
    MEM_MISMATCH -> "MEMREAD"; CSR_MISMATCH -> "CSR" + detail["pol"]
    ("CSR+", "CSR-", "CSR~"; "CSR~" when detail or pol is missing).
    2026-10-14 CSR／MEMREAD 細分: MEM_MISMATCH with detail["st"] (every
    MEM_MISMATCH of the trace is next to a later sub-word store, set by
    parse_trace) -> "MEMREAD_ST"; CSR_MISMATCH with detail["op"] (the read
    is explained by one write op behaving as another, ``_csr_src_op``) ->
    "CSR@" + op, before the polarity.

    REG_ADDR_RD / REG_ADDR_RS: compressed mnemonics group per mnemonic
    ("RA_RD:<mn>" / "RA_RS:<mn>"); 32-bit ones group by detail["field"]
    ("RA_RS1:I", "RA_RS2:I", "RA_RS:I", "RA_RD:I").  A missing field
    defaults by kind (RD -> "RA_RD:I", RS -> "RA_RS:I"); in the real pipeline
    ``_reg_addr`` always supplies the field, so only unit tests see it."""
    if kind == "LOAD_EXT":
        return "LD_SEXT" if mn in ("lb", "lh") else "LD_ZEXT"
    if kind == "MEM_MISMATCH":
        return "MEMREAD_ST" if (detail or {}).get("st") else "MEMREAD"
    if kind == "CSR_MISMATCH":
        d = detail or {}
        if d.get("op"):
            return "CSR@" + d["op"]
        return "CSR" + (d.get("pol") or "~")
    if kind == "MSTATUS_MISMATCH":
        return "MST:" + ((detail or {}).get("field") or "?")
    if kind in _TRAP_KINDS_SET:
        t = (detail or {}).get("trap")
        if t and kind == "TRAP_MEPC":
            t = "irq" if t == "irq" else "sync"
        return f"{kind}:{t}" if t else kind
    if kind in ("REG_ADDR_RD", "REG_ADDR_RS"):
        # Compressed expansions are one ibex_compressed_decoder line per
        # mnemonic; the 32-bit decoder shares its rd/rs1/rs2 fields.
        if mn.startswith("c."):
            return ("RA_RD:" if kind == "REG_ADDR_RD" else "RA_RS:") + mn
        return "RA_" + ((detail or {}).get("field") or ("RD" if kind == "REG_ADDR_RD" else "RS")) + ":I"
    if kind == "READ_MISMATCH":
        if detail and detail.get("fwd"):
            return "REGREAD_FWD"
        return "REGREAD"
    if kind == "WRONG_RESULT" and alu_op(mn) is not None:
        return mn
    # Ibex EQ/NE share is_equal (ibex_alu.sv:155-156: ALU_EQ = is_equal,
    # ALU_NE = ~is_equal), so a broken comparator makes beq and bne fail in
    # opposite directions. Group by polarity of the is_equal error instead
    # of by mnemonic (see PLAN.md section 6).
    if mclass(mn) in ("BR_BEQ", "BR_BNE") and kind in _EQ_POLARITY:
        return _EQ_POLARITY[kind][mclass(mn)]
    return mclass(mn)


@dataclass(slots=True)
class ScanContext:
    trap_entries: set[int] = field(default_factory=set)
    # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗: shadow x2 (sp) value, kept in
    # sync by ``observe()`` so c.addi4spn (whose rs1=x2 is never printed in
    # ``reads`` by the tracer) can still be recomputed. See
    # docs/ARCHITECTURE.md 4.3.
    sp_shadow: int | None = None
    # PC R -> latest Instr.index of an XRET whose next line has pc R
    # (filled by parse_trace pass 1). See ``_async_entry_after_cf``.
    xret_returns: dict[int, int] = field(default_factory=dict)
    # 2026-10-04 mtvec base (value & ~0xFF) as last written by a csrrw/
    # csrrwi (or a csrrs/csrrc on a known value), tracked by ``observe()``.
    # Ibex mtvec is always vectored with only [31:8] kept (ibex_cs_registers.sv
    # :733-740). None = unknown: before the trace's first write, after a
    # write whose value the trace does not show, or when the base is 0
    # (see ``_observe_mtvec``). See ``_irq_vector_entry``.
    mtvec_base: int | None = None
    # 2026-10-05 影子 x2 評估（方案 c1g）: last known value of each register
    # (x0 excluded), used by ``_read_mismatch`` to flag a read whose value
    # differs from the last write. Updated only by ``observe()``: a write
    # always sets it; a read only seeds a register that has no entry yet
    # (never overwrites, so a corrupted read value keeps being reported
    # instead of re-syncing). parse_trace clears it after a skipped
    # malformed line (that line may have written registers).
    regs: dict[str, int] = field(default_factory=dict)
    # 2026-10-08 REGREAD_FWD: value each register held before its last write
    # (None = unknown) and the Instr.index of that write; used by
    # ``_read_mismatch`` to decide ``detail["fwd"]``. Cleared with ``regs``.
    regs_prev: dict[str, int | None] = field(default_factory=dict)
    regs_wr_at: dict[str, int] = field(default_factory=dict)
    # 2026-10-09 LSU/CSR shadows: byte address -> last known byte; CSR name ->
    # last known read-back value. Updated by ``observe()``; see
    # ``_lsu_csr_mismatch``. parse_trace clears them with ``regs``.
    mem: dict[int, int] = field(default_factory=dict)
    csr: dict[str, int] = field(default_factory=dict)
    # 2026-10-14 CSR／MEMREAD 細分 (PLAN.md 4c): per tracked CSR, the value
    # at the last synced read and the writes since then, [(op, operand)]
    # (``_csr_src_op``); per shadow byte, the Instr.index of the store or
    # load that set it; per aligned word, (index, lo, hi) of its last
    # sub-word store (``_lsu_csr_mismatch`` detail "nb").  Cleared with
    # ``mem``/``csr``.
    csr_chain: dict[str, tuple[int | None, list[tuple[str, int]]]] = field(default_factory=dict)
    mem_at: dict[int, int] = field(default_factory=dict)
    word_sub: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Hardware-written trap CSR predictions (csr -> (trap, values)) and the
    # mepc an mret must return to (None = unknown).
    trap_pred: dict[str, tuple[str, frozenset[int]]] = field(default_factory=dict)
    xmepc: int | None = None
    priv: str | None = "M"
    mpp: int | None = None
    # 2026-10-20 mstatus model (``_observe_mstatus``): value, known-bit mask,
    # the event that last set each field ("csr" / "trap" / "mret"), whether an
    # NMI handler is active (its mret restores MPIE/MPP from mstack), and
    # whether the last control-flow break landed somewhere unrecognised.
    mst: int = 0
    mst_known: int = 0
    mst_src: dict[str, str] = field(default_factory=dict)
    mst_nmi: bool = False
    mst_unk: bool = False
    # ``ctx.priv`` may be stale (an unrecognised landing / unknown mtvec, e.g.
    # a reset out of U mode): a trap entry then treats MPP as unknown, until an
    # mret whose MPP the model knew. An ecall's mcause prediction also reads it (R2-6).
    mst_priv_unk: bool = False


def observe(ctx: ScanContext, instr, next_pc: int | None = None) -> None:
    """2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗: update ``ctx.sp_shadow``
    from x2 reads/writes on ``instr``. Reads are applied first (any
    instruction that reads the current x2 value re-syncs the shadow to it),
    then writes override (an instruction that writes x2 updates the shadow
    to the new value). A CSR instruction targeting mtvec also updates
    ``ctx.mtvec_base`` (``_observe_mtvec``). Callers MUST call this *after* ``check``/
    ``check_safe`` has evaluated ``instr`` (see docs/ARCHITECTURE.md 4.3):
    an instruction's own register writes must never influence its own
    semantic check. Also maintains ``ctx.regs`` (see ``ScanContext.regs``):
    read values seed unknown registers, then writes override.

    2026-10-09: ``next_pc`` (the PC of the following instruction, or None if
    unknown) lets ``_observe_lsu_csr`` and ``_observe_trap`` (trap CSR prediction,
    ``ctx.trap_pred`` / ``ctx.xmepc`` / ``ctx.priv`` / ``ctx.mpp``) detect a control-flow break (trap, xret,
    interrupt); it also updates the memory shadow ``ctx.mem`` (stores write,
    loads seed unknown bytes) and the CSR shadow ``ctx.csr``.  A store with
    unknown PA clears ``ctx.mem``."""
    regs = ctx.regs
    reads = instr.reads
    writes = instr.writes
    for reg, val in reads.items():
        if reg != "x0" and reg not in regs:
            regs[reg] = val
    if writes:
        regs_prev = ctx.regs_prev
        regs_wr_at = ctx.regs_wr_at
        index = instr.index
        for reg, val in writes.items():
            if reg != "x0":
                regs_prev[reg] = regs.get(reg)
                regs_wr_at[reg] = index
                regs[reg] = val
    if "x2" in reads:
        ctx.sp_shadow = reads["x2"]
    if "x2" in writes:
        ctx.sp_shadow = writes["x2"]
    mn = instr.mnemonic
    brk = _flow_break(instr, next_pc)  # computed once, shared by both observers
    if MCLASS.get(mn) == "CSR":
        _observe_mtvec(ctx, instr)
    # _observe_lsu_csr is a no-op for a non-store/load/CSR mnemonic when
    # next_pc is the fallthrough (or unknown); skip the call in that case.
    if (mn in _ST_W or mn in _LD_W or mn in _CSR_OP
            or (next_pc is not None and next_pc != (instr.pc + instr.size) & _MASK32)):
        _observe_lsu_csr(ctx, instr, next_pc, brk)
    if mn in _CSR_OP or mn in ("mret", "dret") or (next_pc is not None and next_pc != (instr.pc + instr.size) & _MASK32):
        # an NMI's mret restores MPP from mstack, not 0; read the flag before
        # _observe_mstatus clears it
        nmi_ret = mn == "mret" and ctx.mst_nmi
        _observe_mstatus(ctx, instr, next_pc, brk)  # before _observe_trap (needs old priv/mpp/xmepc)
        _observe_trap(ctx, instr, next_pc, brk, nmi_ret=nmi_ret)


_MTVEC_NAMES = ("mtvec", "0x305")


def _observe_mtvec(ctx: ScanContext, instr) -> None:
    """Track ``ctx.mtvec_base`` from a CSR instruction targeting mtvec.
    Operands look like ``rd,mtvec,rs1`` / ``rd,mtvec,uimm``. csrrw/csrrwi
    replace the value; for every csrr* form an unreadable or unparseable value means the
    old base is stale -> base unknown (None). csrrs/csrrc (and i forms) with rs1=x0 or value 0
    are pure reads and omitted;
    otherwise set/clear bits only if the base is already known. csrrsi/
    csrrci take uimm <= 31, which cannot touch bits [31:8], so they are
    equivalent to a no-op on the base; the generic path handles them anyway.
    A computed base of 0 is stored as None (unknown): a base of 0 would turn
    the exclusion window into low addresses, exactly where a wrong jalr
    target (rs1 read as 0) lands. Never raises."""
    ops = _split_operands(instr.operands)
    if len(ops) != 3 or ops[1].lower() not in _MTVEC_NAMES:
        return
    mn = instr.mnemonic
    src = ops[2]
    try:
        if mn.endswith("i"):
            val = _parse_imm(src)
        elif src == "x0":
            val = 0
        else:
            val = instr.reads.get(src)
    except ValueError:
        val = None
    if val is None:
        ctx.mtvec_base = None
        return
    val &= _MASK32
    if mn in ("csrrw", "csrrwi"):
        new = val
    elif mn in ("csrrs", "csrrsi", "csrrc", "csrrci"):
        if val == 0 or ctx.mtvec_base is None:
            return
        if mn in ("csrrs", "csrrsi"):
            new = ctx.mtvec_base | val
        else:
            new = ctx.mtvec_base & ~val
    else:
        return
    base = new & ~0xFF & _MASK32
    ctx.mtvec_base = base if base != 0 else None


# Regs field tokens: "xN:0x..." (read), "xN=0x..." (write), "PA:0x...",
# "store:0x...", "load:0x...". Confirmed against public trace samples
# (benchmark_set_1/case_2, set_2/case_1 etc): all hex values carry a "0x"
# prefix. Unknown tokens are ignored.
_PA_RE = re.compile(r"^PA:0x([0-9a-fA-F]+)$")
_STORE_RE = re.compile(r"^store:0x([0-9a-fA-F]+)$")
_LOAD_RE = re.compile(r"^load:0x([0-9a-fA-F]+)$")
# parse_regs fast path: reads and writes in one pattern, bound match methods.
_XRW_RE = re.compile(r"^x(\d+)([:=])0x([0-9a-fA-F]+)$").match
_PA_RE_M = _PA_RE.match
_STORE_RE_M = _STORE_RE.match
_LOAD_RE_M = _LOAD_RE.match


def parse_regs(regs_field: str):
    """Parse the trailing "Regs" column of a trace.log line.

    Returns (reads, writes, pa, store, load) where reads/writes map
    register name ("xN") to int value, and pa/store/load are int or None.
    """
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    pa: int | None = None
    store: int | None = None
    load: int | None = None

    # Dispatch on the first character (every token kind starts with a distinct
    # one) so each token costs one regex match instead of up to five.
    for tok in regs_field.split():
        c = tok[0]
        if c == "x":
            m = _XRW_RE(tok)
            if m:
                n, sep, h = m.groups()
                if sep == ":":
                    reads["x" + n] = int(h, 16)
                else:
                    writes["x" + n] = int(h, 16)
        elif c == "P":
            m = _PA_RE_M(tok)
            if m:
                pa = int(m.group(1), 16)
        elif c == "s":
            m = _STORE_RE_M(tok)
            if m:
                store = int(m.group(1), 16)
        elif c == "l":
            m = _LOAD_RE_M(tok)
            if m:
                load = int(m.group(1), 16)
        # unknown token, ignore

    return reads, writes, pa, store, load


# ---------------------------------------------------------------------------
# section 4.3: semantic checker
# ---------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF


def _to_signed32(v: int) -> int:
    v &= _MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def _to_unsigned32(v: int) -> int:
    return v & _MASK32


def _parse_imm(tok: str) -> int:
    """Parse a decimal (optionally signed) or 0x-hex immediate token."""
    tok = tok.strip()
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:]
    if tok.lower().startswith("0x"):
        val = int(tok, 16)
    else:
        val = int(tok, 10)
    return -val if neg else val


def _parse_target(tok: str) -> int:
    """Bare hex PC target (no 0x prefix)."""
    return int(tok, 16) & _MASK32


_MEM_OPERAND_RE = re.compile(r"^(-?(?:0x[0-9a-fA-F]+|\d+))\((x\d+)\)$")


def _split_operands(operands: str) -> list[str]:
    return [p.strip() for p in operands.split(",")] if operands else []


def _regval(reads: dict[str, int], reg: str) -> int | None:
    if reg == "x0":
        return 0
    return reads.get(reg)


def _writeval(writes: dict[str, int], reg: str) -> int | None:
    if reg == "x0":
        return 0
    return writes.get(reg)


# --- branch / jump target and condition helpers ----------------------------


def _branch_operands(mnemonic: str, ops: list[str]):
    """Return (rs1, rs2_or_None, target) for a branch mnemonic."""
    if mnemonic in ("c.beqz", "c.bnez"):
        if len(ops) < 2:
            return None
        return ops[0], "x0", _parse_target(ops[1])
    if len(ops) < 3:
        return None
    return ops[0], ops[1], _parse_target(ops[2])


def _branch_taken(mnemonic: str, rs1_val: int, rs2_val: int) -> bool:
    if mnemonic in ("beq", "c.beqz"):
        return rs1_val == rs2_val
    if mnemonic in ("bne", "c.bnez"):
        return rs1_val != rs2_val
    if mnemonic == "blt":
        return _to_signed32(rs1_val) < _to_signed32(rs2_val)
    if mnemonic == "bltu":
        return _to_unsigned32(rs1_val) < _to_unsigned32(rs2_val)
    if mnemonic == "bge":
        return _to_signed32(rs1_val) >= _to_signed32(rs2_val)
    if mnemonic == "bgeu":
        return _to_unsigned32(rs1_val) >= _to_unsigned32(rs2_val)
    raise AssertionError(mnemonic)


def _jump_target(mnemonic: str, ops: list[str], reads: dict[str, int]):
    """Return computed target PC, or None if not computable (missing reads)."""
    if mnemonic in ("jal", "c.j", "c.jal"):
        if not ops:
            return None
        tok = ops[-1]
        return _parse_target(tok)
    if mnemonic == "jalr":
        if len(ops) < 2:
            return None
        m = _MEM_OPERAND_RE.match(ops[1])
        if not m:
            return None
        imm = _parse_imm(m.group(1))
        rs1 = m.group(2)
        rs1_val = _regval(reads, rs1)
        if rs1_val is None:
            return None
        return (rs1_val + imm) & ~1 & _MASK32
    if mnemonic in ("c.jr", "c.jalr"):
        if not ops:
            return None
        rs1_val = _regval(reads, ops[0])
        if rs1_val is None:
            return None
        return rs1_val & ~1 & _MASK32
    return None


def _link_rd(mnemonic: str, ops: list[str]) -> str | None:
    if mnemonic == "jal":
        return ops[0] if ops else None
    if mnemonic == "jalr":
        return ops[0] if ops else None
    if mnemonic == "c.jal":
        return "x1"
    if mnemonic == "c.jalr":
        return "x1"
    return None


# --- arithmetic recompute ----------------------------------------------------

_REG_REG = {"add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu",
            "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"}
_REG_IMM = {"addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai"}
_COMPR_RD_RS2 = {"c.add", "c.sub", "c.and", "c.or", "c.xor", "c.mv"}
_COMPR_RD_IMM = {"c.addi", "c.andi", "c.slli", "c.srli", "c.srai"}


def _arith_operands(mnemonic: str, ops: list[str], reads: dict[str, int]):
    """Return (rd, computed_value) or None if not computable.

    ``rd`` is the destination register name; computed_value is the full
    (possibly out-of-range, caller masks) recomputed result.
    """
    if mnemonic in _REG_REG:
        if len(ops) < 3:
            return None
        rd, rs1, rs2 = ops[0], ops[1], ops[2]
        a = 0 if rs1 == "x0" else reads.get(rs1)
        b = 0 if rs2 == "x0" else reads.get(rs2)
        if a is None or b is None:
            return None
        return rd, _compute_reg_reg(mnemonic, a, b)

    if mnemonic in _REG_IMM:
        if len(ops) < 3:
            return None
        rd, rs1, imm_tok = ops[0], ops[1], ops[2]
        a = 0 if rs1 == "x0" else reads.get(rs1)
        if a is None:
            return None
        imm = _parse_imm(imm_tok)
        return rd, _compute_reg_imm(mnemonic, a, imm)

    if mnemonic in _COMPR_RD_RS2:
        # rd,rs2 ; rd doubles as rs1 (except c.mv which discards rs1)
        if len(ops) < 2:
            return None
        rd, rs2 = ops[0], ops[1]
        b = _regval(reads, rs2)
        if b is None:
            return None
        if mnemonic == "c.mv":
            return rd, b
        a = _regval(reads, rd)
        if a is None:
            return None
        if mnemonic == "c.add":
            return rd, (a + b) & _MASK32
        if mnemonic == "c.sub":
            return rd, (a - b) & _MASK32
        if mnemonic == "c.and":
            return rd, a & b
        if mnemonic == "c.or":
            return rd, a | b
        if mnemonic == "c.xor":
            return rd, a ^ b

    if mnemonic in _COMPR_RD_IMM:
        if len(ops) < 2:
            return None
        rd, imm_tok = ops[0], ops[1]
        imm = _parse_imm(imm_tok)
        if mnemonic == "c.addi":
            a = _regval(reads, rd)
            if a is None:
                return None
            return rd, (a + imm) & _MASK32
        if mnemonic == "c.andi":
            a = _regval(reads, rd)
            if a is None:
                return None
            return rd, a & imm & _MASK32
        if mnemonic in ("c.slli", "c.srli", "c.srai"):
            a = _regval(reads, rd)
            if a is None:
                return None
            shamt = imm & 0x1F
            if mnemonic == "c.slli":
                return rd, (a << shamt) & _MASK32
            if mnemonic == "c.srli":
                return rd, (_to_unsigned32(a) >> shamt) & _MASK32
            return rd, (_to_signed32(a) >> shamt) & _MASK32

    if mnemonic == "c.li":
        if len(ops) < 2:
            return None
        rd, imm_tok = ops[0], ops[1]
        return rd, _parse_imm(imm_tok) & _MASK32

    if mnemonic == "c.addi4spn":
        if len(ops) < 3:
            return None
        rd, rs1, imm_tok = ops[0], ops[1], ops[2]
        a = _regval(reads, rs1)
        if a is None:
            return None
        return rd, (a + _parse_imm(imm_tok)) & _MASK32

    if mnemonic == "c.addi16sp":
        if len(ops) < 2:
            return None
        rd, imm_tok = ops[0], ops[1]
        a = _regval(reads, rd)
        if a is None:
            return None
        return rd, (a + _parse_imm(imm_tok)) & _MASK32

    if mnemonic == "lui":
        if len(ops) < 2:
            return None
        rd, imm_tok = ops[0], ops[1]
        imm = _parse_imm(imm_tok)
        return rd, (imm << 12) & _MASK32

    if mnemonic == "c.lui":
        if len(ops) < 2:
            return None
        rd, imm_tok = ops[0], ops[1]
        imm = _parse_imm(imm_tok)
        return rd, (imm << 12) & _MASK32

    if mnemonic == "auipc":
        return None  # handled by caller (needs pc)

    return None


def _compute_reg_reg(mnemonic: str, a: int, b: int) -> int:
    if mnemonic == "add":
        return (a + b) & _MASK32
    if mnemonic == "sub":
        return (a - b) & _MASK32
    if mnemonic == "and":
        return a & b & _MASK32
    if mnemonic == "or":
        return (a | b) & _MASK32
    if mnemonic == "xor":
        return (a ^ b) & _MASK32
    if mnemonic == "sll":
        return (a << (b & 0x1F)) & _MASK32
    if mnemonic == "srl":
        return (_to_unsigned32(a) >> (b & 0x1F)) & _MASK32
    if mnemonic == "sra":
        return (_to_signed32(a) >> (b & 0x1F)) & _MASK32
    if mnemonic == "slt":
        return 1 if _to_signed32(a) < _to_signed32(b) else 0
    if mnemonic == "sltu":
        return 1 if _to_unsigned32(a) < _to_unsigned32(b) else 0
    if mnemonic == "mul":
        return (a * b) & _MASK32
    if mnemonic == "mulh":
        return ((_to_signed32(a) * _to_signed32(b)) >> 32) & _MASK32
    if mnemonic == "mulhsu":
        return ((_to_signed32(a) * _to_unsigned32(b)) >> 32) & _MASK32
    if mnemonic == "mulhu":
        return ((_to_unsigned32(a) * _to_unsigned32(b)) >> 32) & _MASK32
    if mnemonic in ("div", "divu", "rem", "remu"):
        return _compute_divrem(mnemonic, a, b)
    raise AssertionError(mnemonic)


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _compute_divrem(mnemonic: str, a: int, b: int) -> int:
    sa = _to_signed32(a)
    sb = _to_signed32(b)
    ua = _to_unsigned32(a)
    ub = _to_unsigned32(b)
    if mnemonic == "div":
        if sb == 0:
            return _MASK32  # -1
        if sa == -0x80000000 and sb == -1:
            return 0x80000000
        return _trunc_div(sa, sb) & _MASK32
    if mnemonic == "divu":
        if ub == 0:
            return _MASK32
        return (ua // ub) & _MASK32
    if mnemonic == "rem":
        if sb == 0:
            return ua  # dividend, bit pattern unchanged
        if sa == -0x80000000 and sb == -1:
            return 0
        q = _trunc_div(sa, sb)
        return (sa - sb * q) & _MASK32
    if mnemonic == "remu":
        if ub == 0:
            return ua
        return (ua % ub) & _MASK32
    raise AssertionError(mnemonic)


def _compute_reg_imm(mnemonic: str, a: int, imm: int) -> int:
    if mnemonic == "addi":
        return (a + imm) & _MASK32
    if mnemonic == "andi":
        return a & imm & _MASK32
    if mnemonic == "ori":
        return (a | (imm & _MASK32)) & _MASK32
    if mnemonic == "xori":
        return (a ^ (imm & _MASK32)) & _MASK32
    if mnemonic == "slti":
        return 1 if _to_signed32(a) < imm else 0
    if mnemonic == "sltiu":
        return 1 if _to_unsigned32(a) < _to_unsigned32(imm) else 0
    if mnemonic == "slli":
        return (a << (imm & 0x1F)) & _MASK32
    if mnemonic == "srli":
        return (_to_unsigned32(a) >> (imm & 0x1F)) & _MASK32
    if mnemonic == "srai":
        return (_to_signed32(a) >> (imm & 0x1F)) & _MASK32
    raise AssertionError(mnemonic)


# 2026-10-09 LSU / CSR shadows (PLAN.md 4c "LOAD／CSR 訊號缺口").
_EXT = {"lb": (8, True), "lh": (16, True), "lbu": (8, False), "lhu": (16, False)}
_LD_W = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4, "c.lw": 4, "c.lwsp": 4}
_ST_W = {"sb": 1, "sh": 2, "sw": 4, "c.sw": 4, "c.swsp": 4}
_CSR_OP = {"csrrw": "W", "csrrwi": "W", "csrrs": "S", "csrrsi": "S", "csrrc": "C", "csrrci": "C"}
# CSR read-back legalisation of the Ibex configuration used by the DV.  Observed on
# the public traces (riscv_csr_test write/read pairs); not verified against RTL.
# 依公開 trace 觀察，未查證 Ibex RTL.
# None = read-only (writes ignored).
_CSR_LEGAL = {
    "mscratch": lambda v: v,
    "mtvec": lambda v: (v & 0xFFFFFF00) | 0x1,
    "mie": lambda v: v & 0x7FFF0888,
    "mepc": lambda v: v & 0xFFFFFFFE,
    "mcause": lambda v: v & 0x8000001F,
    "mtval": lambda v: v,
    "misa": None,
}
# Written by hardware on trap entry: forgotten after any control-flow break.
_CSR_TRAP = ("mepc", "mcause", "mtval")


def _flow_break(instr, next_pc) -> bool:
    """next_pc is neither the fallthrough nor (branch/jump) the target: a trap,
    interrupt, debug entry or xret happened after ``instr``."""
    if next_pc is None:
        return False
    if next_pc == (instr.pc + instr.size) & _MASK32:
        return False
    cls = mclass(instr.mnemonic)
    try:
        ops = _split_operands(instr.operands)
        if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE"):
            p = _branch_operands(instr.mnemonic, ops)
            return p is None or next_pc != p[2]
        if cls == "JUMP":
            t = _jump_target(instr.mnemonic, ops, instr.reads)
            return t is None or next_pc != t
    except (ValueError, IndexError):
        return True
    return True


def _csr_operand(instr, src):
    if instr.mnemonic.endswith("i"):
        try:
            return _parse_imm(src) & _MASK32
        except ValueError:
            return None
    if src == "x0":
        return 0
    return instr.reads.get(src)


def _csr_pol(expected: int, actual: int) -> str:
    extra = actual & ~expected & _MASK32
    miss = expected & ~actual & _MASK32
    if extra and not miss:
        return "+"
    if miss and not extra:
        return "-"
    return "~"


def _lsu_csr_mismatch(cur, next_pc, ctx) -> Violation | None:
    """MEM_MISMATCH: a load's rd bytes differ from the memory shadow.
    CSR_MISMATCH: a CSR read differs from the CSR shadow.  Skipped when the
    instruction is followed by a control-flow break (it may have trapped)."""
    mn = cur.mnemonic
    if mn not in _LD_W and mn not in _CSR_OP:
        return None
    if next_pc is not None and next_pc != (cur.pc + cur.size) & _MASK32:
        return None
    if mn in _LD_W:
        if cur.pa is None:
            return None
        ops = _split_operands(cur.operands)
        if not ops or ops[0] == "x0":
            return None
        v = cur.writes.get(ops[0])
        if v is None:
            return None
        for k in range(_LD_W[mn]):
            b = ctx.mem.get((cur.pa + k) & _MASK32)
            got = (v >> (8 * k)) & 0xFF
            if b is not None and b != got:
                a = (cur.pa + k) & _MASK32
                ws = ctx.word_sub.get(a & ~3)
                # nb: the byte is a neighbour lane of a sub-word store made
                # after the byte's shadow value was set (store-side spill).
                nb = ws is not None and not ws[1] <= a < ws[2] and ws[0] > ctx.mem_at.get(a, -1)
                return Violation(index=cur.index, kind="MEM_MISMATCH", mnemonic=mn,
                                 detail={"pa": cur.pa, "byte": k, "expected": b, "actual": got,
                                         "nb": nb},
                                 pc=cur.pc, next_pc=next_pc)
        return None
    if mn in _CSR_OP:
        ops = _split_operands(cur.operands)
        if len(ops) != 3 or ops[1] not in _CSR_LEGAL or ops[0] == "x0":
            return None
        rv = cur.writes.get(ops[0])
        known = ctx.csr.get(ops[1])
        if rv is None or known is None or rv == known:
            return None
        return Violation(index=cur.index, kind="CSR_MISMATCH", mnemonic=mn,
                         detail={"csr": ops[1], "expected": known, "actual": rv,
                                 "pol": _csr_pol(known, rv),
                                 "op": _csr_src_op(ctx, ops[1], rv)},
                         pc=cur.pc, next_pc=next_pc)
    return None


_CSR_CHAIN_MAX = 8


def _csr_fold(base, chain, legal, swap):
    """Replay ``chain`` from ``base`` with op ``swap[0]`` executed as
    ``swap[1]`` ("R" = the write is dropped); None if a value is unknown."""
    v = base
    for op, sv in chain:
        if op == swap[0]:
            op = swap[1]
        if op == "R":
            continue
        if op == "W":
            v = sv
        elif v is None:
            return None
        elif op == "S":
            v = v | sv
        else:
            v = v & ~sv
        v = legal(v & _MASK32)
    return v


def _csr_src_op(ctx, cn, actual) -> str | None:
    """2026-10-14 CSR／MEMREAD 細分: the write op ("W"/"S"/"C") whose
    replacement by another op (W/S/C, or the write dropped) reproduces
    ``actual`` from the writes since the last synced read of ``cn``.  When
    hypotheses with different source ops fit, the most recent write's op
    wins; None if no hypothesis (or no unique source op) fits.  Models the
    csr_op decode / csr_wdata_int mux bug class (architect's recollection
    of ibex_cs_registers.sv, not verified against RTL)."""
    ent = ctx.csr_chain.get(cn)
    legal = _CSR_LEGAL.get(cn)
    if ent is None or legal is None or not ent[1]:
        return None
    base, chain = ent
    xs = set()
    for x in sorted({op for op, _ in chain}):
        for y in ("W", "S", "C", "R"):
            if y != x and _csr_fold(base, chain, legal, (x, y)) == actual:
                xs.add(x)
    if len(xs) > 1:
        xs &= {chain[-1][0]}
    return xs.pop() if len(xs) == 1 else None


def _observe_lsu_csr(ctx, instr, next_pc, brk) -> None:
    mn = instr.mnemonic
    if mn in _ST_W and instr.pa is not None:
        ops = _split_operands(instr.operands)
        v = None
        if ops:
            v = 0 if ops[0] == "x0" else instr.reads.get(ops[0])
        for k in range(_ST_W[mn]):
            a = (instr.pa + k) & _MASK32
            if v is None or brk:
                ctx.mem.pop(a, None)
            else:
                ctx.mem[a] = (v >> (8 * k)) & 0xFF
                ctx.mem_at[a] = instr.index
        if _ST_W[mn] < 4:
            ctx.word_sub[instr.pa & ~3] = (instr.index, instr.pa, instr.pa + _ST_W[mn])
    elif mn in _ST_W:
        # Store whose address is unknown: any byte may have changed.
        ctx.mem.clear()
        ctx.mem_at.clear()
        ctx.word_sub.clear()
    elif mn in _LD_W and instr.pa is not None and not brk:
        ops = _split_operands(instr.operands)
        v = instr.writes.get(ops[0]) if ops and ops[0] != "x0" else None
        if v is not None:
            for k in range(_LD_W[mn]):
                a = (instr.pa + k) & _MASK32
                if a not in ctx.mem:
                    ctx.mem[a] = (v >> (8 * k)) & 0xFF
                    ctx.mem_at[a] = instr.index
    elif mn in _CSR_OP:
        ops = _split_operands(instr.operands)
        if len(ops) == 3 and ops[1] in _CSR_LEGAL:
            rd, cn, src = ops
            legal = _CSR_LEGAL[cn]
            rv = instr.writes.get(rd) if rd != "x0" else None
            old = rv if rv is not None else ctx.csr.get(cn)
            sv = _csr_operand(instr, src)
            op = _CSR_OP[mn]
            pure = op != "W" and sv == 0
            if brk or (sv is None and not pure):
                ctx.csr.pop(cn, None)
                ctx.csr_chain.pop(cn, None)
            elif pure or legal is None:
                if old is not None and cn not in ctx.csr:
                    ctx.csr[cn] = old
                    ctx.csr_chain[cn] = (old, [])
            else:
                if op == "W":
                    new = sv
                elif old is None:
                    new = None
                elif op == "S":
                    new = old | sv
                else:
                    new = old & ~sv
                if new is None:
                    ctx.csr.pop(cn, None)
                    ctx.csr_chain.pop(cn, None)
                else:
                    ctx.csr[cn] = legal(new & _MASK32)
                    ent = ctx.csr_chain.get(cn)
                    if rv is not None or ent is None:
                        ctx.csr_chain[cn] = (old, [(op, sv)])
                    elif len(ent[1]) < _CSR_CHAIN_MAX:
                        ctx.csr_chain[cn] = (ent[0], ent[1] + [(op, sv)])
                    else:
                        ctx.csr_chain.pop(cn, None)
    if brk:
        for c in _CSR_TRAP:
            ctx.csr.pop(c, None)
            ctx.csr_chain.pop(c, None)


def _width_mask(mnemonic: str) -> int:
    if mnemonic in ("sb",):
        return 0xFF
    if mnemonic in ("sh",):
        return 0xFFFF
    return _MASK32  # sw, c.sw, c.swsp


def _mem_operands(mnemonic: str, ops: list[str]):
    """Return (reg, imm, rs1) for a load/store operand list "reg,offset(rs1)"."""
    if len(ops) < 2:
        return None
    o1 = ops[1]
    m = _MEM_OPERAND_RE.match(o1)
    if not m:
        return None
    return ops[0], _parse_imm(m.group(1)), m.group(2)


def _async_entry_after_cf(cur, target: int, fallthrough: int, ctx: ScanContext) -> bool:
    """True when some *later* xret returns to this control-flow instruction's
    correct target or fallthrough: an asynchronous interrupt took the
    handler entry as next_pc and mret resumed at mepc (docs/ARCHITECTURE.md
    4.3). Only used to suppress BRANCH_WRONG_TARGET."""
    xr = ctx.xret_returns
    return max(xr.get(target, -1), xr.get(fallthrough, -1)) > cur.index


_IRQ_KS = frozenset({3, 7, 11, *range(16, 32)})


def _irq_vector_entry(next_pc: int, ctx: ScanContext) -> bool:
    """True when next_pc is an Ibex interrupt vector slot ``base + 4k``,
    k in {3, 7, 11, 16..31} (docs/ARCHITECTURE.md 4.3; ibex_if_stage.sv:
    213-231, ibex_cs_registers.sv:733-740). Ibex only raises these 19
    interrupt ids: software 3, timer 7, external 11, fast 16..30, NMI 31
    (ibex_pkg.sv:321-328; ibex_controller.sv:734-758, fast is
    ``{1'b1, mfip_id}``; ibex_if_stage.sv:214-218, internal NMI also 31).
    The other 12 slots (1, 2, 4..6, 8..10, 12..15) are never entered by an
    interrupt, so excluding them would only hide real bugs.
    k=0 (``base`` itself) is deliberately NOT excluded: a synchronous
    exception enters there, so a jal that jumped to an illegal address and
    then faulted lands on base, which is a real bug signal. Conversely a
    correct core also lands on base when a test deliberately jumps to a
    faulting address; the public set has no control-flow -> base case at
    all, so staying conservative is reasonable.
    Needs ``ctx.mtvec_base`` known. Before the first mtvec write it is
    unknown on purpose: Ibex PC_BOOT is ``{boot_addr[31:8], 8'h80}`` so the
    reset base is derivable, but interrupts are not yet enabled in the
    first ~55 lines, so not excluding costs nothing. Only used to suppress
    BRANCH_WRONG_TARGET."""
    base = ctx.mtvec_base
    if base is None or next_pc is None:
        return False
    d = next_pc - base
    return d % 4 == 0 and (d >> 2) in _IRQ_KS


_UNCHECKED_CLASSES = frozenset({"CSR", "FENCE", "OTHER"})


def check(prev, cur, next_pc, ctx: ScanContext) -> Violation | None:
    """Semantic violation check. See docs/ARCHITECTURE.md section 4.3."""
    mnemonic = cur.mnemonic
    cls = MCLASS.get(mnemonic, "OTHER")
    if cls in _UNCHECKED_CLASSES:
        return None  # CSR, FENCE, OTHER: not checked.
    ops = _split_operands(cur.operands)
    fallthrough = (cur.pc + cur.size) & _MASK32

    # --- control flow -------------------------------------------------
    # Exclusion rule (section 4.3): an interrupt can land anywhere, so any
    # next_pc that matches a known trap-entry PC is never a violation for
    # the control-flow kinds (BRANCH_*, TRAP_NOT_TAKEN, XRET_FALLTHROUGH).
    if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE", "TRAP_INSTR", "XRET") and next_pc is not None \
            and next_pc in ctx.trap_entries:
        return None

    if cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE"):
        if next_pc is not None:
            parsed = _branch_operands(mnemonic, ops)
            if parsed is not None:
                rs1, rs2, target = parsed
                rs1_val = _regval(cur.reads, rs1)
                rs2_val = _regval(cur.reads, rs2) if rs2 is not None else None
                if rs1_val is not None and rs2_val is not None and target != fallthrough:
                    # When target == fallthrough (degenerate branch encoding,
                    # e.g. a branch whose offset is exactly its own size),
                    # next_pc can never distinguish taken from not-taken, so
                    # the NOT_TAKEN/WRONGLY_TAKEN checks are undecidable.
                    taken = _branch_taken(mnemonic, rs1_val, rs2_val)
                    if taken and next_pc == fallthrough:
                        return Violation(
                            index=cur.index, kind="BRANCH_NOT_TAKEN",
                            mnemonic=mnemonic, detail={"target": target},
                            pc=cur.pc, next_pc=next_pc,
                        )
                    if not taken and next_pc == target:
                        return Violation(
                            index=cur.index, kind="BRANCH_WRONGLY_TAKEN",
                            mnemonic=mnemonic, detail={"target": target},
                            pc=cur.pc, next_pc=next_pc,
                        )
                if (
                    next_pc != target
                    and next_pc != fallthrough
                    and next_pc not in ctx.trap_entries
                    and not _async_entry_after_cf(cur, target, fallthrough, ctx)
                    and not _irq_vector_entry(next_pc, ctx)
                ):
                    return Violation(
                        index=cur.index, kind="BRANCH_WRONG_TARGET",
                        mnemonic=mnemonic, detail={"target": target},
                        pc=cur.pc, next_pc=next_pc,
                    )
        return None

    if cls == "JUMP":
        if next_pc is not None:
            target = _jump_target(mnemonic, ops, cur.reads)
            if (
                target is not None
                and next_pc != target
                and next_pc != fallthrough
                and next_pc not in ctx.trap_entries
                and not _async_entry_after_cf(cur, target, fallthrough, ctx)
                and not _irq_vector_entry(next_pc, ctx)
            ):
                return Violation(
                    index=cur.index, kind="BRANCH_WRONG_TARGET",
                    mnemonic=mnemonic, detail={"target": target},
                    pc=cur.pc, next_pc=next_pc,
                )
        rd = _link_rd(mnemonic, ops)
        if rd is not None and rd != "x0":
            wv = _writeval(cur.writes, rd)
            if wv is not None and wv != fallthrough:
                return Violation(
                    index=cur.index, kind="WRONG_LINK",
                    mnemonic=mnemonic, detail={"expected": fallthrough, "actual": wv},
                    pc=cur.pc, next_pc=next_pc,
                )
        return None

    if cls == "TRAP_INSTR":
        if next_pc is not None and next_pc == fallthrough:
            return Violation(
                index=cur.index, kind="TRAP_NOT_TAKEN", mnemonic=mnemonic,
                detail={}, pc=cur.pc, next_pc=next_pc,
            )
        return None

    if cls == "XRET":
        if next_pc is not None and next_pc == fallthrough:
            return Violation(
                index=cur.index, kind="XRET_FALLTHROUGH", mnemonic=mnemonic,
                detail={}, pc=cur.pc, next_pc=next_pc,
            )
        return None

    # --- arithmetic -----------------------------------------------------
    if cls in ("ADDSUB", "LOGIC", "SHIFT", "SLT", "MUL", "DIV"):
        if mnemonic == "auipc":
            if len(ops) < 2:
                return None
            rd, imm_tok = ops[0], ops[1]
            expected = (cur.pc + (_parse_imm(imm_tok) << 12)) & _MASK32
        else:
            # 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗: c.addi4spn's rs1 is
            # implicit x2, which the tracer never prints in ``reads``. Fall
            # back to the shadow value only when reads doesn't already have
            # it (reads always takes priority per docs/ARCHITECTURE.md 4.3).
            if mnemonic == "c.addi4spn" and "x2" not in cur.reads and ctx.sp_shadow is not None:
                reads_eff = dict(cur.reads)
                reads_eff["x2"] = ctx.sp_shadow
                res = _arith_operands(mnemonic, ops, reads_eff)
            else:
                res = _arith_operands(mnemonic, ops, cur.reads)
            if res is None:
                return None
            rd, expected = res
            expected &= _MASK32
        if rd == "x0":
            return None  # writes to x0 are always 0, never a violation
        wv = cur.writes.get(rd)
        if wv is None:
            return None
        if wv != expected:
            return Violation(
                index=cur.index, kind="WRONG_RESULT", mnemonic=mnemonic,
                detail={"expected": expected, "actual": wv},
                pc=cur.pc, next_pc=next_pc,
            )
        return None

    if cls in ("LOAD", "STORE"):
        mem = _mem_operands(mnemonic, ops)
        if mem is None:
            return None
        reg, imm, rs1 = mem
        rs1_val = _regval(cur.reads, rs1)
        if rs1_val is not None and cur.pa is not None:
            expected_addr = (rs1_val + imm) & _MASK32
            if cur.pa != expected_addr:
                return Violation(
                    index=cur.index, kind="WRONG_ADDR", mnemonic=mnemonic,
                    detail={"expected": expected_addr, "actual": cur.pa},
                    pc=cur.pc, next_pc=next_pc,
                )
        if cls == "LOAD" and mnemonic in _EXT and reg != "x0" \
                and (next_pc is None or next_pc == fallthrough):
            wv = cur.writes.get(reg)
            if wv is not None:
                bits, signed = _EXT[mnemonic]
                low = wv & ((1 << bits) - 1)
                exp = (low | (_MASK32 ^ ((1 << bits) - 1))) if (signed and low >> (bits - 1)) else low
                if wv != exp:
                    return Violation(
                        index=cur.index, kind="LOAD_EXT", mnemonic=mnemonic,
                        detail={"expected": exp, "actual": wv},
                        pc=cur.pc, next_pc=next_pc,
                    )
        if cls == "STORE":
            rs2_val = _regval(cur.reads, reg)
            if rs2_val is not None and cur.store is not None:
                mask = _width_mask(mnemonic)
                expected_data = rs2_val & mask
                if (cur.store & mask) != expected_data:
                    return Violation(
                        index=cur.index, kind="WRONG_STORE_DATA", mnemonic=mnemonic,
                        detail={"expected": expected_data, "actual": cur.store & mask},
                        pc=cur.pc, next_pc=next_pc,
                    )
        return None

    # CSR, FENCE, OTHER: not checked.
    return None


def _read_mismatch(cur, next_pc, ctx: ScanContext) -> Violation | None:
    """READ_MISMATCH: the first register read (in ``cur.reads`` order, x0
    excluded) whose value differs from the last known value in ``ctx.regs``.
    Catches regfile corruption / forwarding bugs no other check sees.
    ``detail["fwd"]``: the read value equals the register's value from before
    its last write (known, not None) and that write was the immediately
    preceding instruction (``cur.index - regs_wr_at == 1``): the signature of
    a missed forward (see ``viol_group``)."""
    for reg, val in cur.reads.items():
        if reg == "x0":
            continue
        known = ctx.regs.get(reg)
        if known is not None and known != val:
            prev = ctx.regs_prev.get(reg)
            return Violation(
                index=cur.index,
                kind="READ_MISMATCH",
                mnemonic=cur.mnemonic,
                detail={
                    "reg": reg, "expected": known, "actual": val,
                    "fwd": (prev is not None and val == prev
                            and reg in ctx.regs_wr_at
                            and cur.index - ctx.regs_wr_at[reg] == 1),
                },
                pc=cur.pc,
                next_pc=next_pc,
            )
    return None


_TRAP_KINDS = {"mcause": "TRAP_MCAUSE", "mepc": "TRAP_MEPC", "mtval": "TRAP_MTVAL"}
_TRAP_KINDS_SET = frozenset(_TRAP_KINDS.values())
_TRAP_CSR_NAMES = {"mcause": "mcause", "0x342": "mcause", "mepc": "mepc", "0x341": "mepc",
                   "mtval": "mtval", "0x343": "mtval"}

def _observe_trap(ctx: ScanContext, instr, next_pc, brk, *, nmi_ret: bool = False) -> None:
    """Predict mcause/mepc/mtval at a trap entry (Ibex mtvec is vectored: sync
    exceptions enter base, interrupt k enters base + 4k) and track the mepc an
    mret must return to."""
    mn = instr.mnemonic
    xmepc_before = ctx.xmepc  # an mret's expected return address, before this instruction
    if mn == "mret":
        ctx.priv = {0: "U", 3: "M"}.get(ctx.mpp) if ctx.mpp is not None else None
        ctx.mpp = None if nmi_ret else 0
    if mn in _CSR_OP:
        ops = _split_operands(instr.operands)
        if len(ops) == 3 and ops[1] in ("mstatus", "0x300"):
            rd, cn, src = ops
            val = _csr_operand(instr, src)
            op = _CSR_OP[mn]
            if op == "W":
                m = None if val is None else (val >> 11) & 3
                ctx.mpp = m if m in (0, 3) else None
            elif val is None or val & 0x1800:
                ctx.mpp = None
            elif rd != "x0" and rd in instr.writes:
                m = (instr.writes[rd] >> 11) & 3
                ctx.mpp = m if m in (0, 3) else None
        if len(ops) == 3:
            rd, cn, src = ops
            name = _TRAP_CSR_NAMES.get(cn)
            if name is not None:
                ctx.trap_pred.pop(name, None)
                if name == "mepc":
                    val = _csr_operand(instr, src)
                    pure = _CSR_OP[mn] != "W" and val == 0
                    if pure:
                        if rd != "x0" and rd in instr.writes:
                            ctx.xmepc = instr.writes[rd]
                    elif _CSR_OP[mn] == "W" and val is not None:
                        ctx.xmepc = val & ~1 & _MASK32
                    else:
                        ctx.xmepc = None
    if next_pc is None or not brk:
        return
    ctx.trap_pred = {}
    prev_priv = ctx.priv
    cls = MCLASS.get(mn, "OTHER")
    base = ctx.mtvec_base
    if cls == "XRET":
        _observe_xret_entry(ctx, next_pc, base, xmepc_before, mn == "mret")
        return
    if base is None:
        # The entry point cannot be recognised: privilege and the mret target
        # are no longer known.
        ctx.xmepc = None
        ctx.priv = None
        ctx.mpp = None
        return
    fallthrough = (instr.pc + instr.size) & _MASK32
    off = (next_pc - base) & _MASK32
    if next_pc == base:
        epcs, tval = (instr.pc,), None
        if mn == "ecall":
            # a stale priv (mst_priv_unk) cannot pick the ecall cause
            trap, cause, tval = ("ecall", (8, 11) if ctx.mst_priv_unk
                                 else {"U": (8,), "M": (11,)}.get(ctx.priv, (8, 11)), 0)
        elif mn in ("ebreak", "c.ebreak"):
            trap, cause = "ebreak", (3,)
        elif cls == "LOAD":
            ops = _split_operands(instr.operands)
            # c.lw (rd' is only x8-x15) and c.lwsp (rd=x0 is a reserved
            # encoding) can never print rd x0 when they retire, and there is
            # no sample of how the tracer prints a faulting one: always take
            # the fault set, regardless of the printed rd.
            if mn in ("c.lw", "c.lwsp") or (ops and ops[0] == "x0"):
                # Printed rd x0: the load itself faulted (the tracer prints a
                # faulting load with rd x0). mepc is this pc, or the next one
                # when the fault belongs to a following, unprinted instruction.
                trap, cause = "ldfault", (1, 2, 4, 5)
                epcs = (instr.pc, fallthrough)
            else:
                # The load retired; the trap came from the next instruction.
                trap, cause, epcs = "other", (1, 2), ()
        elif cls == "STORE":
            trap, cause, epcs = "stfault", (1, 2, 6, 7), ()
        else:
            # The trapping instruction itself is not in the trace.
            trap, cause, epcs = "other", (1, 2), ()
        pred = {"mcause": (trap, frozenset(cause))}
        if epcs:
            pred["mepc"] = (trap, frozenset(epcs))
        if tval is not None:
            pred["mtval"] = (trap, frozenset((tval,)))
        ctx.trap_pred = pred
        ctx.xmepc = epcs[0] if len(epcs) == 1 else None
        ctx.mpp = {"M": 3, "U": 0}.get(prev_priv)
        ctx.priv = "M"
        return
    if off % 4 == 0 and off // 4 in _IRQ_KS:
        pred = {"mcause": ("irq", frozenset((0x80000000 | (off // 4),)))}
        if cls not in _CONTROL_FLOW_CLASSES:
            pred["mepc"] = ("irq", frozenset((fallthrough,)))
            ctx.xmepc = fallthrough
        else:
            ctx.xmepc = None
        ctx.trap_pred = pred
        ctx.mpp = {"M": 3, "U": 0}.get(prev_priv)
        ctx.priv = "M"
        return
    # Unrecognised landing (debug entry, or an entry this model does not
    # know): the mret target is no longer known.
    ctx.xmepc = None


def _observe_xret_entry(ctx: ScanContext, next_pc, base, xmepc_before, is_mret: bool) -> None:
    """mret/dret whose next pc is not its fallthrough. A normal return to the
    expected mepc changes nothing. Otherwise, if it lands on a vector entry, a
    trap (interrupt slot) or an ambiguous entry (base) follows immediately.
    A dret landing in an interrupt slot predicts the irq mcause and sets
    priv=M, mpp and xmepc unknown. Either real situation (an irq masked during
    debug is taken right after dret; or the irq was already taken and a debug
    halt sits at the vector entry, which the model did not see) leaves the
    hart in M. MPP and mepc differ between the two (pre-debug privilege and
    dpc, or pre-irq privilege and the interrupted pc), so both are left
    unknown: that only skips checks and cannot raise a false report.
    mret in a slot: mpp is the returned-to privilege, priv=M, xmepc unknown."""
    if base is None or next_pc == xmepc_before:
        return
    off = (next_pc - base) & _MASK32
    if next_pc == base:
        ctx.xmepc = None
        ctx.mpp = None
        ctx.priv = "M"
    elif off % 4 == 0 and off // 4 in _IRQ_KS:
        ctx.trap_pred = {"mcause": ("irq", frozenset((0x80000000 | (off // 4),)))}
        ctx.xmepc = None
        ctx.mpp = {"M": 3, "U": 0}.get(ctx.priv) if is_mret else None
        ctx.priv = "M"


def _trap_csr(cur, next_pc, ctx: ScanContext) -> Violation | None:
    if not ctx.trap_pred:
        return None
    if next_pc is not None and next_pc != (cur.pc + cur.size) & _MASK32:
        return None
    ops = _split_operands(cur.operands)
    if len(ops) != 3 or ops[0] == "x0":
        return None
    name = _TRAP_CSR_NAMES.get(ops[1])
    pred = ctx.trap_pred.get(name) if name else None
    if pred is None:
        return None
    rv = cur.writes.get(ops[0])
    if rv is None or rv in pred[1]:
        return None
    return Violation(index=cur.index, kind=_TRAP_KINDS[name], mnemonic=cur.mnemonic,
                     detail={"csr": name, "trap": pred[0], "expected": sorted(pred[1]), "actual": rv},
                     pc=cur.pc, next_pc=next_pc)


def _xret_target(cur, next_pc, ctx: ScanContext) -> Violation | None:
    if cur.mnemonic != "mret" or next_pc is None or ctx.xmepc is None:
        return None
    if next_pc == ctx.xmepc or next_pc == (cur.pc + cur.size) & _MASK32:
        return None
    if next_pc in ctx.trap_entries or _irq_vector_entry(next_pc, ctx) or next_pc == ctx.mtvec_base:
        return None
    return Violation(index=cur.index, kind="XRET_TARGET", mnemonic=cur.mnemonic,
                     detail={"expected": ctx.xmepc, "actual": next_pc}, pc=cur.pc, next_pc=next_pc)


_XN = tuple("x%d" % i for i in range(32))
_XN_SET = tuple(frozenset((n,)) for n in _XN)
# Compressed: mnemonic -> (quadrant, funct3, src fields, dst field).  Field
# codes: "a" = [11:7], "b" = [6:2], "p" = rs1' [9:7], "q" = rs2' [4:2],
# or a fixed register number (x2 = sp, x1 = ra, x0 for c.mv's expansion).
_RA_C = {
    "c.addi4spn": (0, 0, (2,), "q"), "c.lw": (0, 2, ("p",), "q"), "c.sw": (0, 6, ("p", "q"), None),
    "c.addi": (1, 0, ("a",), "a"), "c.jal": (1, 1, (), 1), "c.li": (1, 2, (), "a"),
    "c.lui": (1, 3, (), "a"), "c.addi16sp": (1, 3, (2,), 2),
    "c.srli": (1, 4, ("p",), "p"), "c.srai": (1, 4, ("p",), "p"), "c.andi": (1, 4, ("p",), "p"),
    "c.sub": (1, 4, ("p", "q"), "p"), "c.xor": (1, 4, ("p", "q"), "p"),
    "c.or": (1, 4, ("p", "q"), "p"), "c.and": (1, 4, ("p", "q"), "p"),
    "c.j": (1, 5, (), None), "c.beqz": (1, 6, ("p",), None), "c.bnez": (1, 7, ("p",), None),
    "c.slli": (2, 0, ("a",), "a"), "c.lwsp": (2, 2, (2,), "a"), "c.jr": (2, 4, ("a",), None),
    "c.mv": (2, 4, (0, "b"), "a"), "c.jalr": (2, 4, ("a",), 1), "c.add": (2, 4, ("a", "b"), "a"),
    "c.swsp": (2, 6, (2, "b"), None),
}
# 32-bit: mnemonic -> (opcode, rs1 used, rs2 used, rd used).  The fields are
# shared by every opcode (ibex_decoder instr_rs1/rs2/rd), so the group names
# the field, not the mnemonic.
_RA_32: dict[str, tuple[int, bool, bool, bool]] = {}
for _mn, _cls in MCLASS.items():
    if _mn.startswith("c."):
        continue
    if _cls in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE"):
        _RA_32[_mn] = (0x63, True, True, False)
    elif _cls == "LOAD":
        _RA_32[_mn] = (0x03, True, False, True)
    elif _cls == "STORE":
        _RA_32[_mn] = (0x23, True, True, False)
    elif _cls == "CSR":
        _RA_32[_mn] = (0x73, not _mn.endswith("i"), False, True)
    elif _cls in ("ADDSUB", "LOGIC", "SHIFT", "SLT", "MUL", "DIV"):
        if _mn in ("lui", "auipc"):
            _RA_32[_mn] = (0x37 if _mn == "lui" else 0x17, False, False, True)
        elif _mn in ("addi", "andi", "ori", "xori", "slli", "srli", "srai", "slti", "sltiu"):
            _RA_32[_mn] = (0x13, True, False, True)
        else:
            _RA_32[_mn] = (0x33, True, True, True)
_RA_32["jal"] = (0x6F, False, False, True)
_RA_32["jalr"] = (0x67, True, False, True)
_RA_NONE = (None, None, None, None, ())
_ra_cache: dict = {}
_RA_CAP = 20000  # entries per cache dict
# Entries with the same content share one tuple (decoded entries repeat a lot
# across Insn strings; without this the two caches doubled the peak RSS).
_ra_pool: dict = {}
# REG_ADDR on/off: tests/test_parse_trace_legacy_equiv.py turns it off (the
# frozen legacy oracle has no Insn column, see that file).
_RA_ON = True


def _ra_decode(mn: str, v: int | None, compressed: bool):
    ent = None
    if v is not None and compressed:
        c = _RA_C.get(mn)
        if c is not None and (v & 3) == c[0] and ((v >> 13) & 7) == c[1]:
            f = {"a": (v >> 7) & 31, "b": (v >> 2) & 31, "p": 8 + ((v >> 7) & 7), "q": 8 + ((v >> 2) & 7)}
            srcs = frozenset(_XN[f.get(x, x)] for x in c[2])
            d = c[3]
            dst = None if d is None else _XN[f.get(d, d)]
            ent = (mn, srcs, frozenset(("x0", dst)) if dst else _XN_SET[0], dst, ())
    elif v is not None:
        e = _RA_32.get(mn)
        if e is not None and (v & 0x7F) == e[0] and (e[0] != 0x73 or (v >> 12) & 3):
            r1 = _XN[(v >> 15) & 31] if e[1] else None
            r2 = _XN[(v >> 20) & 31] if e[2] else None
            dst = _XN[(v >> 7) & 31] if e[3] else None
            fields = tuple(fr for fr in (("RS1", r1), ("RS2", r2)) if fr[1] is not None)
            ent = (mn, frozenset(fr[1] for fr in fields), frozenset(("x0", dst)) if dst else _XN_SET[0],
                   dst, fields)
    if ent is None:
        ent = (mn,) + _RA_NONE[1:]
    pooled = _ra_pool.get(ent)
    if pooled is None:
        if len(_ra_pool) >= _RA_CAP:
            _ra_pool.clear()
        _ra_pool[ent] = pooled = ent
    return pooled


_ra_fields: dict = {}


def _ra_entry(mn: str, insn: str):
    """Cache entry for one Insn string: (mnemonic, allowed reads, allowed
    writes, dst, src fields).  The sets are None when the bits do not decode
    to the printed mnemonic (then nothing is checked).  Random tests rarely
    repeat an Insn, so a miss first looks up the register/opcode bits only
    (immediates masked off) before decoding."""
    try:
        v = int(insn, 16) if insn else None
    except ValueError:
        v = None
    compressed = len(insn) <= 4
    key = (mn, compressed, None if v is None else (v if compressed else v & 0x01FFFFFF))
    ent = _ra_fields.get(key)
    if ent is None:
        ent = _ra_decode(mn, v, compressed)
        if len(_ra_fields) >= _RA_CAP:
            _ra_fields.clear()
        _ra_fields[key] = ent
    if len(_ra_cache) >= _RA_CAP:
        _ra_cache.clear()
    _ra_cache[insn] = ent
    return ent


def _reg_addr(cur, next_pc, ent) -> Violation | None:
    """REG_ADDR (docs/ARCHITECTURE.md 4.3), called only after the inline
    screen in ``check_safe`` found a printed register outside ``ent``.
    ibex_tracer prints the register numbers (operand text and the xN:/xN=
    tokens) from rvfi_rs1/rs2/rd_addr, i.e. the registers the core used,
    but the Insn column is rvfi_insn.  A printed register that is not a
    field encoded in the Insn bits means the decoder picked a different
    register.  A printed x0 write is never checked (branches print x0=, a
    faulting load prints rd x0); a printed x0 read is (only c.mv encodes one)."""
    _mn, srcs, _aw, dst, fields = ent
    for r in cur.writes:
        if r != "x0" and r != dst:
            return Violation(index=cur.index, kind="REG_ADDR_RD", mnemonic=cur.mnemonic,
                             detail={"reg": r, "expected": dst, "field": "RD"},
                             pc=cur.pc, next_pc=next_pc)
    reads = cur.reads
    r = next(r for r in reads if r not in srcs)
    miss = [f for f, e in fields if e not in reads]
    if fields and not miss:
        # 32-bit: rs1 and rs2 encode the same register, so the faulty field
        # cannot be named; stay silent (another instruction of the same bug
        # names it) rather than emit a field-less RS group.
        return None
    return Violation(index=cur.index, kind="REG_ADDR_RS", mnemonic=cur.mnemonic,
                     detail={"reg": r, "field": miss[0] if len(miss) == 1 else "RS"},
                     pc=cur.pc, next_pc=next_pc)


def check_safe(prev, cur, next_pc, ctx: ScanContext) -> tuple[Violation | None, str | None]:
    """Like ``check``, but never raises on malformed operand tokens.

    Catches ``ValueError``/``IndexError`` from immediate/target parsing and
    returns ``(None, "<exception class>: <message>")`` instead. On success
    returns ``(violation_or_none, None)``. When ``check()`` finds no violation,
    REG_ADDR runs first (``_reg_addr``; skipped when ``_flow_break`` says a
    trap or xret followed the instruction); a REG_ADDR hit stops the later
    checks. Otherwise ``_trap_csr`` (CSR ops) and ``_xret_target`` (mret) run,
    then ``_read_mismatch`` (``check()`` itself never reports READ_MISMATCH),
    then ``_lsu_csr_mismatch`` (MEM_MISMATCH / CSR_MISMATCH against ``ctx.mem`` /
    ``ctx.csr``; LOAD_EXT comes from ``check()`` itself). A CSR read also runs
    ``_trap_csr`` (TRAP_* against ``ctx.trap_pred`` set by ``_observe_trap``) and
    an mret runs ``_xret_target`` (XRET_TARGET against ``ctx.xmepc``).
    """
    try:
        v = check(prev, cur, next_pc, ctx)
        if v is None and _RA_ON:
            ent = _ra_cache.get(cur.insn)
            if ent is None or ent[0] != cur.mnemonic:
                ent = _ra_entry(cur.mnemonic, cur.insn)
            if ent[1] is not None and not (cur.reads.keys() <= ent[1] and cur.writes.keys() <= ent[2]):
                # An instruction followed by a trap entry may have been
                # illegal; Ibex can then zero rvfi_rs*_addr and the tracer
                # prints x0 reads.  Do not judge it (flow break = next_pc is
                # not the fallthrough / branch or jump target).
                if not _flow_break(cur, next_pc):
                    v = _reg_addr(cur, next_pc, ent)
        if v is None and cur.mnemonic in _CSR_OP:
            v = _trap_csr(cur, next_pc, ctx)
            if v is None:
                v = _mstatus_check(cur, next_pc, ctx)
        if v is None and cur.mnemonic == "mret":
            v = _xret_target(cur, next_pc, ctx)
        if v is None:
            # Inline no-mismatch screen; the full helper runs only on a hit.
            regs_get = ctx.regs.get
            for reg, val in cur.reads.items():
                if reg != "x0":
                    known = regs_get(reg)
                    if known is not None and known != val:
                        v = _read_mismatch(cur, next_pc, ctx)
                        break
        if v is None:
            mn = cur.mnemonic
            if mn in _LD_W or mn in _CSR_OP:
                v = _lsu_csr_mismatch(cur, next_pc, ctx)
        return v, None
    except (ValueError, IndexError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


# 2026-10-20 mstatus 追蹤 (PLAN.md 4c): model of the readable Ibex mstatus
# fields, checked against every mstatus read.  Field semantics are the
# privileged spec plus the architect's recollection of ibex_cs_registers.sv
# (not verified against RTL); every transition used here was confirmed on the
# public traces (1,244 reads, all equal), except MPP legalisation (MPP becomes
# unknown) and the NMI mstack restore (MPIE/MPP become unknown).
_MST_NAMES = ("mstatus", "0x300")
# Readable bits: MIE 3, MPIE 7, MPP 12:11, MPRV 17, TW 21; all others read 0.
_MST_R = 0x00221888
_MST_MIE, _MST_MPIE, _MST_MPP, _MST_MPRV = 0x8, 0x80, 0x1800, 0x20000
_MST_FIELDS = (("MIE", _MST_MIE), ("MPIE", _MST_MPIE), ("MPP", _MST_MPP),
               ("MPRV", _MST_MPRV), ("TW", 0x200000))
_MST_NMI_K = 31


def _mst_legal(v: int, k: int) -> tuple[int, int]:
    """Legalise a written value ``v`` with known mask ``k``.  A written MPP of
    01/10 is converted to a supported mode; which one is not verified, so MPP
    becomes unknown.  Bits outside ``_MST_R`` read 0 and are known."""
    v &= _MST_R
    if k & _MST_MPP != _MST_MPP or (v >> 11) & 3 in (1, 2):
        k &= ~_MST_MPP
    return v, (k | ~_MST_R) & _MASK32


def _mst_set(ctx: ScanContext, bits: int, val: int, src: str) -> None:
    ctx.mst = (ctx.mst & ~bits) | (val & bits)
    ctx.mst_known |= bits
    for name, m in _MST_FIELDS:
        if bits & m:
            ctx.mst_src[name] = src


def _mst_trap(ctx: ScanContext, priv: str | None) -> None:
    """Trap entry: MPIE <- MIE, MIE <- 0, MPP <- the privilege trapped from."""
    if ctx.mst_priv_unk:
        priv = None
    if ctx.mst_known & _MST_MIE:
        _mst_set(ctx, _MST_MPIE, _MST_MPIE if ctx.mst & _MST_MIE else 0, "trap")
    else:
        ctx.mst_known &= ~_MST_MPIE
    _mst_set(ctx, _MST_MIE, 0, "trap")
    m = {"M": _MST_MPP, "U": 0}.get(priv)
    if m is None:
        ctx.mst_known &= ~_MST_MPP
    else:
        _mst_set(ctx, _MST_MPP, m, "trap")


def _mst_note_nmi(ctx: ScanContext, slot: bool, off: int) -> None:
    """Landing in the NMI slot (31): the handler's mret restores from mstack."""
    if slot and off // 4 == _MST_NMI_K:
        ctx.mst_nmi = True


def _mst_after_csr(instr, ctx: ScanContext) -> tuple[int, int]:
    """(value, known mask) of mstatus after a retired CSR op on mstatus: the
    printed read value (if any) re-syncs every bit, then the write applies."""
    v, k = ctx.mst, ctx.mst_known
    ops = _split_operands(instr.operands)
    rd, src = ops[0], ops[2]
    rv = instr.writes.get(rd) if rd != "x0" else None
    if rv is not None:
        v, k = rv, _MASK32
    sv = _csr_operand(instr, src)
    op = _CSR_OP[instr.mnemonic]
    if op == "W":
        return (v, 0) if sv is None else _mst_legal(sv, _MASK32)
    if sv is None:
        return v, 0
    if sv == 0:
        return v, k
    if op == "S":
        return _mst_legal((v | sv) & _MASK32, k | sv)
    return _mst_legal(v & ~sv & _MASK32, k | sv)


def _mstatus_check(cur, next_pc, ctx: ScanContext) -> Violation | None:
    """MSTATUS_MISMATCH: an mstatus read differs from the model on a known
    bit.  Skipped when the read is followed by a control-flow break (it may
    have trapped).  detail["field"] names the differing field(s)."""
    if not ctx.mst_known:
        return None
    if next_pc is not None and next_pc != (cur.pc + cur.size) & _MASK32:
        return None
    ops = _split_operands(cur.operands)
    if len(ops) != 3 or ops[1] not in _MST_NAMES or ops[0] == "x0":
        return None
    rv = cur.writes.get(ops[0])
    if rv is None:
        return None
    k, e = ctx.mst_known, ctx.mst
    diff = (e ^ rv) & k
    if not diff:
        return None
    fields = [n for n, m in _MST_FIELDS if diff & m] or ["OTHER"]
    return Violation(index=cur.index, kind="MSTATUS_MISMATCH", mnemonic=cur.mnemonic,
                     detail={"field": "+".join(fields), "expected": e & k, "actual": rv,
                             "src": ctx.mst_src.get(fields[0]) if len(fields) == 1 else None},
                     pc=cur.pc, next_pc=next_pc)


def _observe_mstatus(ctx: ScanContext, instr, next_pc, brk: bool) -> None:
    """Update the mstatus model (``ctx.mst`` / ``ctx.mst_known``).  Must run
    before ``_observe_trap``: it needs the pre-instruction ``priv`` / ``mpp``
    / ``xmepc``.  Landings follow ``_observe_trap``: base = sync exception,
    base + 4k (k in ``_IRQ_KS``) = interrupt, anything else = debug entry or
    an unprinted instruction, which forgets the whole model."""
    mn = instr.mnemonic
    is_csr = False
    if mn in _CSR_OP:
        ops = _split_operands(instr.operands)
        is_csr = len(ops) == 3 and ops[1] in _MST_NAMES
    is_mret = mn == "mret"
    if not brk and not is_mret:
        if mn == "dret":
            ctx.mst_known &= ~_MST_MPRV  # dret that falls through: only MPRV may change
        elif is_csr:
            v, k = _mst_after_csr(instr, ctx)
            changed = ((v ^ ctx.mst) | (k ^ ctx.mst_known)) & k
            ctx.mst, ctx.mst_known = v, k
            for name, m in _MST_FIELDS:
                if changed & m:
                    ctx.mst_src[name] = "csr"
        return
    base = ctx.mtvec_base
    if brk and base is None:
        ctx.mst_known = 0
        ctx.mst_unk = True
        ctx.mst_priv_unk = True
        return
    if brk:
        off = (next_pc - base) & _MASK32
        at_base = next_pc == base
        slot = not at_base and off % 4 == 0 and off // 4 in _IRQ_KS
    else:
        # mret falling through / with an unknown next pc: an ordinary mret.
        off, at_base, slot = 0, False, False
    if is_mret:
        ret_to_mepc = brk and next_pc == ctx.xmepc
        if ctx.mst_unk or (at_base and not ret_to_mepc):
            # mret reached through an unrecognised landing (not a modelled
            # handler), or an mret that may itself have been illegal: only a
            # following trap entry's MIE=0 is certain.
            ctx.mst_unk = False
            ctx.mst_known = 0
            if (at_base or slot) and not ret_to_mepc:
                _mst_set(ctx, _MST_MIE, 0, "trap")
                _mst_note_nmi(ctx, slot, off)
            return
        # MPP known to the model (and not an NMI restore): the privilege
        # after this mret is known, so ``priv`` is trustworthy again.
        if ctx.mst_known & _MST_MPP and not ctx.mst_nmi and ctx.mpp is not None:
            ctx.mst_priv_unk = False
        # MIE <- MPIE, MPIE <- 1, MPP <- U, MPRV <- 0 unless MPP was M.
        if ctx.mst_known & _MST_MPIE:
            _mst_set(ctx, _MST_MIE, _MST_MIE if ctx.mst & _MST_MPIE else 0, "mret")
        else:
            ctx.mst_known &= ~_MST_MIE
        if ctx.mst_known & _MST_MPP:
            if ctx.mst & _MST_MPP != _MST_MPP:
                _mst_set(ctx, _MST_MPRV, 0, "mret")
        else:
            ctx.mst_known &= ~_MST_MPRV
        new_priv = {0: "U", 3: "M"}.get(ctx.mpp) if ctx.mpp is not None else None
        if ctx.mst_nmi:
            # Return from an NMI handler: Ibex restores MPIE/MPP from mstack.
            ctx.mst_known &= ~(_MST_MPIE | _MST_MPP)
            ctx.mst_nmi = False
        else:
            _mst_set(ctx, _MST_MPIE, _MST_MPIE, "mret")
            _mst_set(ctx, _MST_MPP, 0, "mret")
        if slot and not ret_to_mepc:
            _mst_trap(ctx, new_priv)
            _mst_note_nmi(ctx, slot, off)
        return
    if mn == "dret":
        # Debug may have changed the privilege (MPRV is cleared when
        # dcsr.prv != M).  A landing on base / a slot (an irq right after
        # dret, or a debug halt at a vector entry already taken) leaves MIE 0.
        ctx.mst_known &= ~_MST_MPRV
        if (at_base or slot) and next_pc != ctx.xmepc:
            if at_base:
                ctx.mst_known = 0
            else:
                ctx.mst_known &= ~(_MST_MPIE | _MST_MPP)
            _mst_set(ctx, _MST_MIE, 0, "trap")
            ctx.mst_unk = False
            _mst_note_nmi(ctx, slot, off)
        return
    if not (at_base or slot):
        ctx.mst_known = 0
        ctx.mst_unk = True
        ctx.mst_priv_unk = True
        return
    ctx.mst_unk = False
    if is_csr:
        if at_base:
            # The CSR op may itself have trapped (illegal): no write.
            ctx.mst_known = 0
        else:
            # An interrupt is taken after the CSR op retired.
            ctx.mst, ctx.mst_known = _mst_after_csr(instr, ctx)
    _mst_trap(ctx, ctx.priv)
    _mst_note_nmi(ctx, slot, off)
