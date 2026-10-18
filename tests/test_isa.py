"""Unit tests for rfb.isa.check() (ARCHITECTURE.md section 4.3)."""

from rfb import isa
from rfb.isa import ScanContext
from rfb.record import Instr


def _instr(index, pc, mnemonic, operands, reads=None, writes=None, pa=None,
           store=None, load=None, size=4):
    return Instr(
        index=index, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
        reads=reads or {}, writes=writes or {}, pa=pa, store=store, load=load,
    )


def _ctx(trap_entries=()):
    return ScanContext(trap_entries=set(trap_entries))


# --- BRANCH_NOT_TAKEN / BRANCH_WRONGLY_TAKEN -------------------------------


def test_branch_not_taken_positive():
    # bge x1,x2,target ; 5 >= 3 -> should be taken, but next_pc is fallthrough
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"
    assert v.mnemonic == "bge"


def test_branch_not_taken_negative_when_taken():
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is None


def test_branch_wrongly_taken_positive():
    # bge x1,x2,target ; 1 >= 5 -> false, but next_pc is target
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_WRONGLY_TAKEN"


def test_branch_wrongly_taken_negative_when_not_taken():
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_bltu_unsigned_comparison():
    # bltu: 0xffffffff (unsigned huge) < 1 is False -> not taken is correct
    i = _instr(0, 0x2000, "bltu", "x1,x2,2008", reads={"x1": 0xFFFFFFFF, "x2": 1})
    v = isa.check(None, i, 0x2004, _ctx())
    assert v is None


def test_c_beqz_compares_to_zero():
    i = _instr(0, 0x2000, "c.beqz", "x1,2008", reads={"x1": 0}, size=2)
    v = isa.check(None, i, 0x2002, _ctx())  # equal to zero -> should take
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"


def test_missing_reads_skips_condition_check():
    i = _instr(0, 0x1000, "bge", "x1,x2,1008")  # no reads recorded
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


# --- BRANCH_WRONG_TARGET ----------------------------------------------------


def test_branch_wrong_target_positive():
    i = _instr(0, 0x1000, "beq", "x1,x2,1008", reads={"x1": 1, "x2": 1})
    v = isa.check(None, i, 0x9000, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_WRONG_TARGET"


def test_branch_wrong_target_excluded_by_trap_entry():
    i = _instr(0, 0x1000, "beq", "x1,x2,1008", reads={"x1": 1, "x2": 1})
    v = isa.check(None, i, 0x9000, _ctx(trap_entries={0x9000}))
    assert v is None


def test_jump_wrong_target_positive():
    i = _instr(0, 0x1000, "jal", "x1,1008")
    v = isa.check(None, i, 0x9000, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_WRONG_TARGET"


def test_jump_wrong_target_negative():
    i = _instr(0, 0x1000, "jal", "x1,1008", writes={"x1": 0x1004})
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is None


# --- TRAP_NOT_TAKEN ----------------------------------------------------------


def test_trap_not_taken_positive():
    i = _instr(0, 0x1000, "ecall", "")
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "TRAP_NOT_TAKEN"
    assert v.mnemonic == "ecall"


def test_trap_not_taken_negative():
    i = _instr(0, 0x1000, "ecall", "")
    v = isa.check(None, i, 0x8000_0000, _ctx())
    assert v is None


def test_c_ebreak_not_taken_positive():
    i = _instr(0, 0x1000, "c.ebreak", "", size=2)
    v = isa.check(None, i, 0x1002, _ctx())
    assert v is not None
    assert v.kind == "TRAP_NOT_TAKEN"
    assert v.mnemonic == "c.ebreak"


# --- XRET_FALLTHROUGH ---------------------------------------------------


def test_xret_fallthrough_positive():
    i = _instr(0, 0x1000, "dret", "")
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "XRET_FALLTHROUGH"
    assert v.mnemonic == "dret"


def test_xret_fallthrough_negative():
    i = _instr(0, 0x1000, "mret", "")
    v = isa.check(None, i, 0x8000_f000, _ctx())
    assert v is None


# --- exclusion rule: next_pc in trap_entries never a violation -------------


def test_trap_entry_excludes_xret_and_trap():
    i = _instr(0, 0x1000, "dret", "")
    v = isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004}))
    assert v is None
    i2 = _instr(1, 0x2000, "ecall", "")
    v2 = isa.check(None, i2, 0x2004, _ctx(trap_entries={0x2004}))
    assert v2 is None


# --- exclusion rule coverage for BR_LT / BR_GE ------------------------------


def test_blt_zero_disp_trap_entry_excluded():
    # blt taken, but next_pc == fallthrough (not target) and is a known
    # trap entry -> excluded, not a violation.
    i = _instr(0, 0x1000, "blt", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004}))
    assert v is None


def test_blt_not_taken_when_next_pc_not_trap_entry():
    i = _instr(0, 0x1000, "blt", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"
    assert v.mnemonic == "blt"


def test_bltu_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x1000, "bltu", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004}))
    assert v is None


def test_bltu_not_taken_when_next_pc_not_trap_entry():
    i = _instr(0, 0x1000, "bltu", "x1,x2,1008", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"
    assert v.mnemonic == "bltu"


def test_bge_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004}))
    assert v is None


def test_bge_not_taken_when_next_pc_not_trap_entry():
    i = _instr(0, 0x1000, "bge", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"
    assert v.mnemonic == "bge"


def test_bgeu_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x1000, "bgeu", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004}))
    assert v is None


# BR_BEQ / BR_BNE share the same exclusion tuple in check(); each case below is
# taken with next_pc == fallthrough (target != fallthrough), so without the
# exclusion it would be BRANCH_NOT_TAKEN (not stopped by the WRONG_TARGET guard).


def test_beq_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x1000, "beq", "x1,x2,1008", reads={"x1": 4, "x2": 4})
    assert isa.check(None, i, 0x1004, _ctx()).kind == "BRANCH_NOT_TAKEN"
    assert isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004})) is None


def test_bne_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x1000, "bne", "x1,x2,1008", reads={"x1": 4, "x2": 5})
    assert isa.check(None, i, 0x1004, _ctx()).kind == "BRANCH_NOT_TAKEN"
    assert isa.check(None, i, 0x1004, _ctx(trap_entries={0x1004})) is None


def test_c_beqz_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x2000, "c.beqz", "x1,2008", reads={"x1": 0}, size=2)
    assert isa.check(None, i, 0x2002, _ctx()).kind == "BRANCH_NOT_TAKEN"
    assert isa.check(None, i, 0x2002, _ctx(trap_entries={0x2002})) is None


def test_c_bnez_zero_disp_trap_entry_excluded():
    i = _instr(0, 0x2000, "c.bnez", "x1,2008", reads={"x1": 7}, size=2)
    assert isa.check(None, i, 0x2002, _ctx()).kind == "BRANCH_NOT_TAKEN"
    assert isa.check(None, i, 0x2002, _ctx(trap_entries={0x2002})) is None


def test_bgeu_not_taken_when_next_pc_not_trap_entry():
    i = _instr(0, 0x1000, "bgeu", "x1,x2,1008", reads={"x1": 5, "x2": 3})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"
    assert v.mnemonic == "bgeu"


# --- zero-displacement branches (target == pc + size) -----------------------


def test_zero_disp_branch_taken_skipped():
    # blt (4-byte): target == pc+size (zero displacement). Condition is
    # true, next_pc == target == fallthrough -> undecidable, must be None.
    i = _instr(0, 0x1000, "blt", "x1,x2,1004", reads={"x1": 1, "x2": 5})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_zero_disp_branch_not_taken_skipped():
    # blt (4-byte): target == pc+size, condition false, next_pc still ==
    # fallthrough == target -> undecidable, must be None.
    i = _instr(0, 0x1000, "blt", "x1,x2,1004", reads={"x1": 5, "x2": 1})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_zero_disp_c_beqz_taken_skipped():
    # c.beqz (2-byte): target == pc+size (zero displacement).
    i = _instr(0, 0x2000, "c.beqz", "x1,2002", reads={"x1": 0}, size=2)
    v = isa.check(None, i, 0x2002, _ctx())
    assert v is None


def test_zero_disp_c_bnez_not_taken_skipped():
    # c.bnez (2-byte): target == pc+size, condition false (x1 == 0).
    i = _instr(0, 0x2000, "c.bnez", "x1,2002", reads={"x1": 0}, size=2)
    v = isa.check(None, i, 0x2002, _ctx())
    assert v is None


# --- WRONG_RESULT ------------------------------------------------------------


def test_wrong_result_add_positive():
    i = _instr(0, 0x1000, "add", "x3,x1,x2", reads={"x1": 2, "x2": 3}, writes={"x3": 999})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "WRONG_RESULT"
    assert v.mnemonic == "add"


def test_wrong_result_add_negative():
    i = _instr(0, 0x1000, "add", "x3,x1,x2", reads={"x1": 2, "x2": 3}, writes={"x3": 5})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_wrong_result_x0_dest_never_violates():
    i = _instr(0, 0x1000, "addi", "x0,x0,5", reads={"x0": 0}, writes={"x0": 0})
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_wrong_result_sub_and_shift():
    i = _instr(0, 0x1000, "sub", "x3,x1,x2", reads={"x1": 10, "x2": 3}, writes={"x3": 7})
    assert isa.check(None, i, 0x1004, _ctx()) is None
    i2 = _instr(0, 0x1000, "slli", "x3,x1,0x2", reads={"x1": 1}, writes={"x3": 4})
    assert isa.check(None, i2, 0x1004, _ctx()) is None
    i3 = _instr(0, 0x1000, "slli", "x3,x1,0x2", reads={"x1": 1}, writes={"x3": 99})
    v3 = isa.check(None, i3, 0x1004, _ctx())
    assert v3 is not None and v3.kind == "WRONG_RESULT"


def test_wrong_result_lui_and_auipc():
    i = _instr(0, 0x80000128, "auipc", "x19,0x27", writes={"x19": 0x80027128})
    assert isa.check(None, i, 0x8000012C, _ctx()) is None
    i2 = _instr(0, 0x1000, "lui", "x2,0x5296a", writes={"x2": 0x5296A000})
    assert isa.check(None, i2, 0x1004, _ctx()) is None
    i3 = _instr(0, 0x1000, "lui", "x2,0x5296a", writes={"x2": 0x12345678})
    v3 = isa.check(None, i3, 0x1004, _ctx())
    assert v3 is not None and v3.kind == "WRONG_RESULT"


def test_wrong_result_compressed_forms():
    # c.add: rd = rd + rs2
    i = _instr(0, 0x1000, "c.add", "x17,x1", reads={"x17": 1, "x1": 2}, writes={"x17": 3}, size=2)
    assert isa.check(None, i, 0x1002, _ctx()) is None
    # c.mv: rd = rs2
    i2 = _instr(0, 0x1000, "c.mv", "x22,x4", reads={"x4": 7}, writes={"x22": 7}, size=2)
    assert isa.check(None, i2, 0x1002, _ctx()) is None
    # c.li: rd = imm
    i3 = _instr(0, 0x1000, "c.li", "x1,0", writes={"x1": 0}, size=2)
    assert isa.check(None, i3, 0x1002, _ctx()) is None


# --- WRONG_ADDR --------------------------------------------------------------


def test_wrong_addr_load_positive():
    i = _instr(0, 0x1000, "lw", "x5,4(x6)", reads={"x6": 0x1000}, pa=0x2000)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "WRONG_ADDR"


def test_wrong_addr_load_negative():
    i = _instr(0, 0x1000, "lw", "x5,4(x6)", reads={"x6": 0x1000}, pa=0x1004)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_wrong_addr_store_negative_offset():
    i = _instr(0, 0x1000, "sb", "x5,-9(x2)", reads={"x2": 0x2000}, pa=0x1FF7)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


# --- WRONG_STORE_DATA --------------------------------------------------------


def test_wrong_store_data_positive():
    i = _instr(0, 0x1000, "sw", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xDEADBEEF},
               pa=0x1000, store=0x12345678)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is not None
    assert v.kind == "WRONG_STORE_DATA"


def test_wrong_store_data_negative():
    i = _instr(0, 0x1000, "sw", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xDEADBEEF},
               pa=0x1000, store=0xDEADBEEF)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_wrong_store_data_width_truncation():
    # sb: only the low byte matters
    i = _instr(0, 0x1000, "sb", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xAABBCCDD},
               pa=0x1000, store=0xDD)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None
    i2 = _instr(0, 0x1000, "sb", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xAABBCCDD},
                pa=0x1000, store=0xEE)
    v2 = isa.check(None, i2, 0x1004, _ctx())
    assert v2 is not None and v2.kind == "WRONG_STORE_DATA"


def test_wrong_store_data_sh_width_truncation():
    # sh: only the low 16 bits matter; high bits differing is not a violation.
    i = _instr(0, 0x1000, "sh", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xAABBCCDD},
               pa=0x1000, store=0x1234CCDD)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None
    i2 = _instr(0, 0x1000, "sh", "x5,0(x6)", reads={"x6": 0x1000, "x5": 0xAABBCCDD},
                pa=0x1000, store=0xEEEE)
    v2 = isa.check(None, i2, 0x1004, _ctx())
    assert v2 is not None and v2.kind == "WRONG_STORE_DATA"


# --- WRONG_LINK ---------------------------------------------------------------


def test_wrong_link_jal_positive():
    i = _instr(0, 0x1000, "jal", "x1,1008", writes={"x1": 0xBAD})
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is not None
    assert v.kind == "WRONG_LINK"


def test_wrong_link_jal_negative():
    i = _instr(0, 0x1000, "jal", "x1,1008", writes={"x1": 0x1004})
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is None


def test_wrong_link_jal_x0_skipped():
    i = _instr(0, 0x1000, "jal", "x0,1008")
    v = isa.check(None, i, 0x1008, _ctx())
    assert v is None


def test_wrong_link_c_jalr():
    i = _instr(0, 0x1000, "c.jalr", "x13", reads={"x13": 0x2000}, writes={"x1": 0xBAD}, size=2)
    v = isa.check(None, i, 0x2000, _ctx())
    assert v is not None
    assert v.kind == "WRONG_LINK"


# --- multiply / divide -------------------------------------------------------


def test_mul_and_mulh():
    i = _instr(0, 0x1000, "mul", "x3,x1,x2", reads={"x1": 6, "x2": 7}, writes={"x3": 42})
    assert isa.check(None, i, 0x1004, _ctx()) is None
    # mulh of two negative numbers -> high bits of a large positive product
    a = 0xFFFFFFFF  # -1
    b = 0xFFFFFFFF  # -1
    # (-1) * (-1) = 1, high 32 bits of the 64-bit signed product = 0
    i2 = _instr(0, 0x1000, "mulh", "x3,x1,x2", reads={"x1": a, "x2": b}, writes={"x3": 0})
    assert isa.check(None, i2, 0x1004, _ctx()) is None


def test_div_divide_by_zero():
    i = _instr(0, 0x1000, "div", "x3,x1,x2", reads={"x1": 7, "x2": 0}, writes={"x3": 0xFFFFFFFF})
    assert isa.check(None, i, 0x1004, _ctx()) is None
    i2 = _instr(0, 0x1000, "divu", "x3,x1,x2", reads={"x1": 7, "x2": 0}, writes={"x3": 0xFFFFFFFF})
    assert isa.check(None, i2, 0x1004, _ctx()) is None
    i3 = _instr(0, 0x1000, "rem", "x3,x1,x2", reads={"x1": 7, "x2": 0}, writes={"x3": 7})
    assert isa.check(None, i3, 0x1004, _ctx()) is None
    i4 = _instr(0, 0x1000, "remu", "x3,x1,x2", reads={"x1": 7, "x2": 0}, writes={"x3": 7})
    assert isa.check(None, i4, 0x1004, _ctx()) is None


def test_div_signed_overflow():
    # INT_MIN / -1 overflows: div = INT_MIN, rem = 0
    i = _instr(0, 0x1000, "div", "x3,x1,x2", reads={"x1": 0x80000000, "x2": 0xFFFFFFFF},
               writes={"x3": 0x80000000})
    assert isa.check(None, i, 0x1004, _ctx()) is None
    i2 = _instr(0, 0x1000, "rem", "x3,x1,x2", reads={"x1": 0x80000000, "x2": 0xFFFFFFFF},
                writes={"x3": 0})
    assert isa.check(None, i2, 0x1004, _ctx()) is None


def test_div_signed_truncation():
    # -7 / 2 = -3 (truncated toward zero, not floored)
    neg7 = 0xFFFFFFF9  # -7
    i = _instr(0, 0x1000, "div", "x3,x1,x2", reads={"x1": neg7, "x2": 2}, writes={"x3": 0xFFFFFFFD})
    assert isa.check(None, i, 0x1004, _ctx()) is None
    i2 = _instr(0, 0x1000, "rem", "x3,x1,x2", reads={"x1": neg7, "x2": 2}, writes={"x3": 0xFFFFFFFF})
    assert isa.check(None, i2, 0x1004, _ctx()) is None


# --- CSR / FENCE / OTHER: never checked --------------------------------------


def test_csr_never_checked():
    i = _instr(0, 0x1000, "csrrw", "x0,misa,x22", reads={"x22": 5}, writes={"x0": 0})
    v = isa.check(None, i, 0x9999_0000, _ctx())
    assert v is None


def test_fence_never_checked():
    i = _instr(0, 0x1000, "fence", "iorw,iorw")
    v = isa.check(None, i, 0x9999_0000, _ctx())
    assert v is None


def test_load_rd_value_never_checked():
    # LOAD's destination register value is not validated (tracer semantics
    # for load: are unreliable, see ARCHITECTURE.md section 3).
    i = _instr(0, 0x1000, "lw", "x5,0(x6)", reads={"x6": 0x1000}, pa=0x1000,
               writes={"x5": 0xDEADBEEF}, load=0x0)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


# --- malformed operand tokens: check() does not raise / check_safe catches ---


def test_jal_empty_operands_no_raise():
    # jal with no operands: _jump_target must not index into an empty list.
    i = _instr(0, 0x1000, "jal", "")
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_c_j_empty_operands_no_raise():
    i = _instr(0, 0x1000, "c.j", "", size=2)
    v = isa.check(None, i, 0x1002, _ctx())
    assert v is None


def test_branch_bad_target_raises_and_check_safe_catches():
    i = _instr(0, 0x1000, "beq", "x1,x2,zz", reads={"x1": 1, "x2": 1})
    try:
        isa.check(None, i, 0x9000, _ctx())
        raised = False
    except ValueError:
        raised = True
    assert raised
    v, err = isa.check_safe(None, i, 0x9000, _ctx())
    assert v is None
    assert err
    assert "ValueError" in err


def test_addi_bad_imm_raises_and_check_safe_catches():
    i = _instr(0, 0x1000, "addi", "x1,x2,abc", reads={"x2": 1}, writes={"x1": 1})
    try:
        isa.check(None, i, 0x1004, _ctx())
        raised = False
    except ValueError:
        raised = True
    assert raised
    v, err = isa.check_safe(None, i, 0x1004, _ctx())
    assert v is None
    assert err


def test_load_non_numeric_offset_does_not_match_mem_regex():
    # "abc(x2)" does not match _MEM_OPERAND_RE (offset must be 0x-hex or
    # decimal digits), so _mem_operands returns None before _parse_imm is
    # ever reached: this is a "not computable" case, not an exception.
    i = _instr(0, 0x1000, "lw", "x1,abc(x2)", reads={"x2": 0x1000}, pa=0x1000)
    v = isa.check(None, i, 0x1004, _ctx())
    assert v is None


def test_andi_bad_imm_raises_and_check_safe_catches():
    i = _instr(0, 0x1000, "andi", "x1,x2,zz", reads={"x2": 1}, writes={"x1": 1})
    try:
        isa.check(None, i, 0x1004, _ctx())
        raised = False
    except ValueError:
        raised = True
    assert raised
    v, err = isa.check_safe(None, i, 0x1004, _ctx())
    assert v is None
    assert err


def test_auipc_bad_imm_raises_and_check_safe_catches():
    i = _instr(0, 0x1000, "auipc", "x1,zz", writes={"x1": 0x1000})
    try:
        isa.check(None, i, 0x1004, _ctx())
        raised = False
    except ValueError:
        raised = True
    assert raised
    v, err = isa.check_safe(None, i, 0x1004, _ctx())
    assert v is None
    assert err


def test_check_safe_returns_none_error_on_valid_instr():
    i = _instr(0, 0x1000, "add", "x3,x1,x2", reads={"x1": 2, "x2": 3}, writes={"x3": 5})
    v, err = isa.check_safe(None, i, 0x1004, _ctx())
    assert v is None
    assert err is None


# --- 2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: shadow x2 & ------
# ALU_OP/viol_group -----------------------------------------------------


def test_c_addi4spn_uses_shadow_when_correct_no_violation():
    ctx = _ctx()
    ctx.sp_shadow = 0x100
    i = _instr(0, 0x1000, "c.addi4spn", "x8,x2,16", writes={"x8": 0x110}, size=2)
    v = isa.check(None, i, 0x1002, ctx)
    assert v is None


def test_c_addi4spn_shadow_mismatch_flags_wrong_result():
    ctx = _ctx()
    ctx.sp_shadow = 0x100
    i = _instr(0, 0x1000, "c.addi4spn", "x8,x2,16", writes={"x8": 0x999}, size=2)
    v = isa.check(None, i, 0x1002, ctx)
    assert v is not None
    assert v.kind == "WRONG_RESULT"
    assert v.mnemonic == "c.addi4spn"


def test_c_addi4spn_shadow_none_skips_check():
    ctx = _ctx()
    assert ctx.sp_shadow is None
    i = _instr(0, 0x1000, "c.addi4spn", "x8,x2,16", writes={"x8": 0x999}, size=2)
    v = isa.check(None, i, 0x1002, ctx)
    assert v is None


def test_c_addi4spn_reads_take_priority_over_shadow():
    ctx = _ctx()
    ctx.sp_shadow = 0x999  # would be wrong if used
    i = _instr(
        0, 0x1000, "c.addi4spn", "x8,x2,16",
        reads={"x2": 0x100}, writes={"x8": 0x110}, size=2,
    )
    v = isa.check(None, i, 0x1002, ctx)
    assert v is None


def test_observe_syncs_shadow_from_reads_then_writes():
    ctx = _ctx()
    # reads x2 -> shadow follows the read value.
    i1 = _instr(0, 0x1000, "sw", "x1,0(x2)", reads={"x2": 0x50})
    isa.observe(ctx, i1)
    assert ctx.sp_shadow == 0x50
    # writes x2 -> shadow follows the (newer) write value, even though a
    # read is also present (reads first, then writes override).
    i2 = _instr(1, 0x1004, "addi", "x2,x2,16", reads={"x2": 0x50}, writes={"x2": 0x60})
    isa.observe(ctx, i2)
    assert ctx.sp_shadow == 0x60
    # an instruction that touches neither x2 read nor write leaves the
    # shadow untouched.
    i3 = _instr(2, 0x1008, "addi", "x3,x0,1", writes={"x3": 1})
    isa.observe(ctx, i3)
    assert ctx.sp_shadow == 0x60


def test_alu_op_groups_sra_and_c_srai_same_op_others_none():
    assert isa.alu_op("sra") == "sra"
    assert isa.alu_op("c.srai") == "sra"
    assert isa.alu_op("bge") is None
    assert isa.alu_op("lw") is None


def test_alu_op_covers_every_addsub_logic_shift_slt_mul_div_mnemonic():
    # ALU_OP must have an entry for every mnemonic mclass places in one of
    # the six ALU-recomputable classes (dynamically enumerated from
    # isa.MCLASS, not a hand-written list -- a future _register() addition
    # to any of these classes must not silently fall through ALU_OP
    # uncovered), and conversely every ALU_OP mnemonic's mclass must be one
    # of those six classes.
    alu_mclasses = {"ADDSUB", "LOGIC", "SHIFT", "SLT", "MUL", "DIV"}
    mclass_alu_mnemonics = {
        mn for mn, cls in isa.MCLASS.items() if cls in alu_mclasses
    }
    assert mclass_alu_mnemonics <= set(isa.ALU_OP)
    for mn in isa.ALU_OP:
        assert isa.mclass(mn) in alu_mclasses


def test_viol_group_wrong_result_uses_mnemonic_for_alu_mclass_otherwise():
    # sra and c.srai are different mnemonics -> different viol_group even
    # though they share an alu_op (hard_key granularity is per-mnemonic).
    assert isa.viol_group("WRONG_RESULT", "sra") == "sra"
    assert isa.viol_group("WRONG_RESULT", "c.srai") == "c.srai"
    assert isa.viol_group("WRONG_RESULT", "sra") != isa.viol_group("WRONG_RESULT", "c.srai")
    # non-ALU mnemonics (branch condition, load) fall back to mclass
    # regardless of kind.
    assert isa.viol_group("BRANCH_WRONGLY_TAKEN", "bge") == isa.mclass("bge") == "BR_GE"
    assert isa.viol_group("WRONG_RESULT", "lw") == isa.mclass("lw") == "LOAD"


def test_viol_group_polarity_only_for_branch_nt_wt_on_beq_bne():
    # polarity is limited to BRANCH_NOT_TAKEN / BRANCH_WRONGLY_TAKEN
    assert isa.viol_group("BRANCH_NOT_TAKEN", "beq") == "BR_EQ-"
    assert isa.viol_group("BRANCH_NOT_TAKEN", "c.bnez") == "BR_EQ+"
    assert isa.viol_group("BRANCH_WRONGLY_TAKEN", "c.beqz") == "BR_EQ+"
    assert isa.viol_group("BRANCH_WRONGLY_TAKEN", "bne") == "BR_EQ-"
    # any other kind on beq/bne stays at mclass
    assert isa.viol_group("TRAP_NOT_TAKEN", "beq") == "BR_BEQ"
    assert isa.viol_group("WRONG_RESULT", "beq") == "BR_BEQ"
    assert isa.viol_group("TRAP_NOT_TAKEN", "bne") == "BR_BNE"
    assert isa.viol_group("WRONG_RESULT", "bne") == "BR_BNE"
    assert isa.viol_group("BRANCH_WRONG_TARGET", "beq") == "BR_BEQ"


# --- READ_MISMATCH (read value vs last written value) -----------------------


def _rm(ctx, index, mnemonic, operands, reads=None, writes=None, next_pc=None, pc=0x1000):
    """check_safe + observe, the way parse_trace drives one instruction."""
    i = _instr(index, pc, mnemonic, operands, reads=reads, writes=writes)
    v, err = isa.check_safe(None, i, next_pc if next_pc is not None else pc + 4, ctx)
    assert err is None
    isa.observe(ctx, i)
    return v


def test_read_mismatch_after_write():
    ctx = _ctx()
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": 0xA}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 0xB}, writes={"x6": 0xB})
    assert v is not None
    assert v.kind == "READ_MISMATCH"
    assert v.mnemonic == "add"
    assert v.detail == {"reg": "x5", "expected": 0xA, "actual": 0xB, "fwd": False}
    assert isa.viol_group("READ_MISMATCH", "add") == "REGREAD"
    assert isa.viol_group("READ_MISMATCH", "beq") == "REGREAD"


def test_read_mismatch_fwd_true_stale_value_next_instruction():
    ctx = _ctx()
    P, A = 0x5, 0xA
    # seed x5 = P by a read, then idx0 writes A, idx1 reads the stale P
    assert _rm(ctx, -1, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P}) is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": A}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P})
    assert v is not None and v.kind == "READ_MISMATCH"
    assert v.detail["fwd"] is True


def test_read_mismatch_fwd_uses_last_of_two_writes():
    # prev / wr_at must track the *last* write: x5 = P (seed), A (idx0),
    # B (idx1); idx2 reading A is the stale value of the idx1 write.
    ctx = _ctx()
    P, A, B = 0x5, 0xA, 0xB
    assert _rm(ctx, -1, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P}) is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": A}) is None
    assert _rm(ctx, 1, "addi", "x5,x0,11", writes={"x5": B}) is None
    v = _rm(ctx, 2, "add", "x6,x5,x0", reads={"x5": A}, writes={"x6": A})
    assert v is not None and v.kind == "READ_MISMATCH"
    assert v.detail["fwd"] is True


def test_read_mismatch_fwd_false_when_distance_two():
    ctx = _ctx()
    P, A = 0x5, 0xA
    assert _rm(ctx, -1, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P}) is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": A}) is None
    assert _rm(ctx, 1, "addi", "x7,x0,1", writes={"x7": 1}) is None
    v = _rm(ctx, 2, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P})
    assert v is not None and v.detail["fwd"] is False


def test_read_mismatch_fwd_false_when_not_stale():
    ctx = _ctx()
    P, A = 0x5, 0xA
    assert _rm(ctx, -1, "add", "x6,x5,x0", reads={"x5": P}, writes={"x6": P}) is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": A}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": A ^ 0x10}, writes={"x6": A ^ 0x10})
    assert v is not None and v.detail["fwd"] is False


def test_read_mismatch_fwd_false_when_prev_unknown():
    ctx = _ctx()
    # x5 never seeded before its first write: prev is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": 0xA}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 0xB}, writes={"x6": 0xB})
    assert v is not None and v.detail["fwd"] is False


def test_viol_group_read_mismatch_fwd():
    assert isa.viol_group("READ_MISMATCH", "add", {"fwd": True}) == "REGREAD_FWD"
    assert isa.viol_group("READ_MISMATCH", "add", {"fwd": False}) == "REGREAD"
    assert isa.viol_group("READ_MISMATCH", "add", None) == "REGREAD"
    assert isa.viol_group("READ_MISMATCH", "add") == "REGREAD"


def test_read_mismatch_first_read_seeds_value():
    ctx = _ctx()
    assert _rm(ctx, 0, "add", "x6,x5,x0", reads={"x5": 7}) is None
    assert _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 7}) is None
    v = _rm(ctx, 2, "add", "x6,x5,x0", reads={"x5": 8})
    assert v is not None and v.kind == "READ_MISMATCH"
    assert v.detail["expected"] == 7 and v.detail["actual"] == 8


def test_read_mismatch_known_value_not_updated_by_reads():
    ctx = _ctx()
    _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": 0xA})
    assert _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 0xB}).kind == "READ_MISMATCH"
    # the wrong value did not become the known value: still reported
    v = _rm(ctx, 2, "add", "x6,x5,x0", reads={"x5": 0xB})
    assert v is not None and v.kind == "READ_MISMATCH"
    assert v.detail["expected"] == 0xA
    assert _rm(ctx, 3, "add", "x6,x5,x0", reads={"x5": 0xA}) is None


def test_read_mismatch_yields_to_branch_violation():
    ctx = _ctx()
    _rm(ctx, 0, "addi", "x1,x0,9", writes={"x1": 9})
    # bge x1,x2: reads x1=5 (!= 9, would be a READ_MISMATCH) and 5 >= 3 but
    # falls through -> BRANCH_NOT_TAKEN wins
    v = _rm(ctx, 1, "bge", "x1,x2,1008", reads={"x1": 5, "x2": 3}, next_pc=0x1004)
    assert v is not None
    assert v.kind == "BRANCH_NOT_TAKEN"


def test_read_mismatch_ignores_x0():
    ctx = _ctx()
    assert _rm(ctx, 0, "add", "x6,x0,x0", reads={"x0": 5}) is None
    assert _rm(ctx, 1, "add", "x6,x0,x0", reads={"x0": 6}) is None
    assert "x0" not in ctx.regs


def test_read_mismatch_same_instr_read_and_write_uses_write():
    ctx = _ctx()
    assert _rm(ctx, 0, "addi", "x5,x5,1", reads={"x5": 4}, writes={"x5": 5}) is None
    assert ctx.regs["x5"] == 5
    assert _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 5}) is None
    assert _rm(ctx, 2, "add", "x6,x5,x0", reads={"x5": 4}).kind == "READ_MISMATCH"


def test_read_mismatch_fwd_prev_seeded_by_same_instruction_read():
    ctx = _ctx()
    assert _rm(ctx, 0, "addi", "x5,x5,1", reads={"x5": 4}, writes={"x5": 5}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 4}, writes={"x6": 4})
    assert v is not None and v.detail["fwd"] is True


def test_read_mismatch_fwd_false_when_prev_unknown_even_if_read_is_zero():
    ctx = _ctx()
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": 0xA}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 0}, writes={"x6": 0})
    assert v is not None and v.detail["fwd"] is False


def test_read_mismatch_fwd_true_when_stale_value_is_zero():
    ctx = _ctx()
    assert _rm(ctx, -1, "add", "x6,x5,x0", reads={"x5": 0}, writes={"x6": 0}) is None
    assert _rm(ctx, 0, "addi", "x5,x0,10", writes={"x5": 0xA}) is None
    v = _rm(ctx, 1, "add", "x6,x5,x0", reads={"x5": 0}, writes={"x6": 0})
    assert v is not None and v.detail["fwd"] is True
