"""Coverage-driven tests for rfb.isa: operand-count / unreadable-register
guards in check()/_arith_operands() and a few helper contracts. Every test
asserts a concrete output so removing the guard it targets turns it red
(most guards would otherwise raise IndexError/TypeError from check()).
"""

import pytest

from rfb import isa
from rfb.isa import ScanContext
from rfb.record import Instr


def _instr(pc, mnemonic, operands, reads=None, writes=None, size=4):
    return Instr(
        index=0, pc=pc, size=size, mnemonic=mnemonic, operands=operands,
        reads=reads or {}, writes=writes or {}, pa=None, store=None, load=None,
    )


def _ctx():
    return ScanContext(trap_entries=set())


# --- _branch_operands: c.beqz / c.bnez with the target operand missing ------
# (isa.py 246-248)


@pytest.mark.parametrize("mn", ["c.beqz", "c.bnez"])
def test_compressed_branch_missing_target_is_not_checked(mn):
    # reads make the branch decidable, but with no target operand check()
    # must return None instead of raising IndexError on ops[1].
    for reads in ({"x1": 0}, {"x1": 7}):
        i = _instr(0x2000, mn, "x1", reads=reads, size=2)
        assert isa.check(None, i, 0x2002, _ctx()) is None
        assert isa.check(None, i, 0x3000, _ctx()) is None


def test_c_beqz_with_target_is_checked_positive_control():
    i = _instr(0x2000, "c.beqz", "x1,2008", reads={"x1": 0}, size=2)
    v = isa.check(None, i, 0x2002, _ctx())
    assert v is not None and v.kind == "BRANCH_NOT_TAKEN"
    assert v.detail == {"target": 0x2008}


# --- _writeval x0 (isa.py 237): unreachable via check() -> direct -----------


def test_writeval_x0_is_always_zero():
    assert isa._writeval({}, "x0") == 0
    assert isa._writeval({"x0": 5}, "x0") == 0
    assert isa._writeval({"x5": 7}, "x5") == 7
    assert isa._writeval({}, "x5") is None


# --- _jump_target: jal / c.j / c.jal without operands (isa.py 274-275) ------


@pytest.mark.parametrize("mn", ["jal", "c.j", "c.jal"])
def test_direct_jump_without_operands_is_not_checked(mn):
    i = _instr(0x1000, mn, "", size=4 if mn == "jal" else 2)
    assert isa._jump_target(mn, [], {}) is None
    assert isa.check(None, i, 0x9999, _ctx()) is None


def test_jal_with_target_is_checked_positive_control():
    i = _instr(0x1000, "jal", "x0,1100")
    v = isa.check(None, i, 0x9999, _ctx())
    assert v is not None and v.kind == "BRANCH_WRONG_TARGET"
    assert v.detail == {"target": 0x1100}


# --- arithmetic recompute: table of (mnemonic, operands, reads, rd, expect) -

_ARITH = [
    # c.* with rd doubling as rs1: rd is in reads, rs2 too.
    ("c.add", "x5,x6", {"x5": 1, "x6": 2}, "x5", 3),
    ("c.sub", "x5,x6", {"x5": 5, "x6": 7}, "x5", 0xFFFFFFFE),
    ("c.and", "x5,x6", {"x5": 0xF0F0, "x6": 0x0FF0}, "x5", 0x00F0),
    ("c.or", "x5,x6", {"x5": 0xF0F0, "x6": 0x0FF0}, "x5", 0xFFF0),
    ("c.xor", "x5,x6", {"x5": 0xF0F0, "x6": 0x0FF0}, "x5", 0xFF00),
    ("c.andi", "x8,-2", {"x8": 0xFF}, "x8", 0xFE),
    ("c.slli", "x8,4", {"x8": 1}, "x8", 0x10),
    ("c.srli", "x8,4", {"x8": 0x80000000}, "x8", 0x08000000),
    ("c.srai", "x8,4", {"x8": 0x80000000}, "x8", 0xF8000000),
    ("c.addi16sp", "x2,-16", {"x2": 0x100}, "x2", 0xF0),
    ("c.mv", "x5,x6", {"x6": 9}, "x5", 9),
    ("c.addi", "x5,4", {"x5": 1}, "x5", 5),
    ("c.li", "x5,-3", {}, "x5", 0xFFFFFFFD),
]


@pytest.mark.parametrize("mn,ops,reads,rd,expect", _ARITH)
def test_compressed_arith_correct_write_is_clean(mn, ops, reads, rd, expect):
    i = _instr(0x1000, mn, ops, reads=reads, writes={rd: expect}, size=2)
    assert isa.check(None, i, 0x1002, _ctx()) is None


@pytest.mark.parametrize("mn,ops,reads,rd,expect", _ARITH)
def test_compressed_arith_wrong_write_reports_expected(mn, ops, reads, rd, expect):
    bad = expect ^ 0x1
    i = _instr(0x1000, mn, ops, reads=reads, writes={rd: bad}, size=2)
    v = isa.check(None, i, 0x1002, _ctx())
    assert v is not None
    assert v.kind == "WRONG_RESULT"
    assert v.mnemonic == mn
    assert v.detail == {"expected": expect, "actual": bad}


# rd (the implicit rs1) missing from reads -> not computable -> no verdict,
# even though the written value is clearly not the "correct" one.
# isa.py 358-359 (c.add..c.xor), 383-384 (c.andi), 388-389 (shifts),
# 416-418 (c.addi16sp).
_UNREADABLE_RD = [
    ("c.add", "x5,x6", {"x6": 2}, "x5"),
    ("c.sub", "x5,x6", {"x6": 7}, "x5"),
    ("c.and", "x5,x6", {"x6": 0x0FF0}, "x5"),
    ("c.or", "x5,x6", {"x6": 0x0FF0}, "x5"),
    ("c.xor", "x5,x6", {"x6": 0x0FF0}, "x5"),
    ("c.andi", "x8,-2", {}, "x8"),
    ("c.slli", "x8,4", {}, "x8"),
    ("c.srli", "x8,4", {}, "x8"),
    ("c.srai", "x8,4", {}, "x8"),
    ("c.addi16sp", "x2,-16", {}, "x2"),
]


@pytest.mark.parametrize("mn,ops,reads,rd", _UNREADABLE_RD)
def test_compressed_arith_unreadable_source_is_not_checked(mn, ops, reads, rd):
    i = _instr(0x1000, mn, ops, reads=reads, writes={rd: 0x12345678}, size=2)
    assert isa._arith_operands(mn, isa._split_operands(ops), reads) is None
    assert isa.check(None, i, 0x1002, _ctx()) is None


# --- lui / c.lui / c.addi16sp / c.addi4spn / auipc operand counts -----------
# isa.py 404-405, 413-414, 422-423, 429-430, 645-646


@pytest.mark.parametrize(
    "mn,ops",
    [
        ("c.addi4spn", "x8,x2"),
        ("c.addi16sp", "x2"),
        ("lui", "x5"),
        ("c.lui", "x5"),
        ("auipc", "x5"),
    ],
)
def test_too_few_operands_is_not_checked(mn, ops):
    i = _instr(0x1000, mn, ops, reads={"x2": 0x100}, writes={"x5": 0x77, "x8": 0x77, "x2": 0x77})
    assert isa.check(None, i, 0x1004, _ctx()) is None


_UPPER = [
    ("lui", "x5,0x12345", 0x12345000),
    ("c.lui", "x5,1", 0x1000),
    ("c.lui", "x5,-1", 0xFFFFF000),
]


@pytest.mark.parametrize("mn,ops,expect", _UPPER)
def test_lui_family_checked_against_shifted_immediate(mn, ops, expect):
    good = _instr(0x1000, mn, ops, writes={"x5": expect})
    assert isa.check(None, good, 0x1004, _ctx()) is None
    bad = _instr(0x1000, mn, ops, writes={"x5": expect ^ 0x1000})
    v = isa.check(None, bad, 0x1004, _ctx())
    assert v is not None and v.kind == "WRONG_RESULT"
    assert v.detail == {"expected": expect, "actual": expect ^ 0x1000}


def test_auipc_adds_pc_and_shifted_immediate():
    good = _instr(0x1000, "auipc", "x5,0x1", writes={"x5": 0x2000})
    assert isa.check(None, good, 0x1004, _ctx()) is None
    bad = _instr(0x1000, "auipc", "x5,0x1", writes={"x5": 0x3000})
    v = isa.check(None, bad, 0x1004, _ctx())
    assert v is not None and v.kind == "WRONG_RESULT"
    assert v.detail == {"expected": 0x2000, "actual": 0x3000}


def test_c_addi4spn_missing_operand_ignores_shadow_too():
    # even with a known shadow sp, 2 operands can't be recomputed
    ctx = _ctx()
    ctx.sp_shadow = 0x100
    i = _instr(0x1000, "c.addi4spn", "x8,x2", writes={"x8": 0x77}, size=2)
    assert isa.check(None, i, 0x1002, ctx) is None


# --- _arith_operands contract for mnemonics handled by the caller (435-438) -


def test_arith_operands_auipc_and_unknown_mnemonic_return_none():
    # auipc needs pc, so the caller handles it; helper must decline.
    assert isa._arith_operands("auipc", ["x5", "1"], {}) is None
    assert isa._arith_operands("bogus", ["x5", "x6", "x7"], {"x6": 1, "x7": 2}) is None


# --- _jump_target: jalr / c.jr with unreadable base (isa.py 287-288, 294-295)


def test_jalr_with_unreadable_base_register_is_not_checked():
    i = _instr(0x1000, "jalr", "x0,4(x5)", reads={})
    assert isa._jump_target("jalr", ["x0", "4(x5)"], {}) is None
    assert isa.check(None, i, 0x9999, _ctx()) is None


def test_jalr_target_clears_bit0_and_is_checked_positive_control():
    i = _instr(0x1000, "jalr", "x0,4(x5)", reads={"x5": 0x2001})
    assert isa._jump_target("jalr", ["x0", "4(x5)"], {"x5": 0x2001}) == 0x2004
    assert isa.check(None, i, 0x2004, _ctx()) is None
    v = isa.check(None, i, 0x9999, _ctx())
    assert v is not None and v.kind == "BRANCH_WRONG_TARGET"
    assert v.detail == {"target": 0x2004}


def test_jump_target_jalr_adds_imm_and_clears_bit0():
    # x5=0x2000、imm=3 -> 0x2003 -> 清 bit0 -> 0x2002
    assert isa._jump_target("jalr", ["x0", "3(x5)"], {"x5": 0x2000}) == 0x2002


def test_jump_target_c_jr_and_c_jalr_clear_bit0():
    assert isa._jump_target("c.jr", ["x5"], {"x5": 0x2001}) == 0x2000
    assert isa._jump_target("c.jalr", ["x5"], {"x5": 0x2003}) == 0x2002


def test_jump_target_c_jr_unreadable_or_missing_register_is_none():
    assert isa._jump_target("c.jr", ["x5"], {}) is None
    assert isa._jump_target("c.jalr", ["x5"], {}) is None
    assert isa._jump_target("c.jr", [], {"x5": 0x2000}) is None


# --- 不變式：表列的 mnemonic 都有對應的 compute 分支（不 raise） -------------


@pytest.mark.parametrize("mn", sorted(isa._REG_REG))
def test_every_reg_reg_mnemonic_is_computable(mn):
    # 刪掉 _compute_reg_reg / _compute_divrem 任一分支會落到 AssertionError
    assert isinstance(isa._compute_reg_reg(mn, 7, 3), int)


@pytest.mark.parametrize("mn", sorted(isa._REG_IMM))
def test_every_reg_imm_mnemonic_is_computable(mn):
    assert isinstance(isa._compute_reg_imm(mn, 7, 3), int)


@pytest.mark.parametrize("mn", ["div", "divu", "rem", "remu"])
def test_every_divrem_mnemonic_is_computable(mn):
    assert isinstance(isa._compute_divrem(mn, 7, 3), int)
    assert isinstance(isa._compute_divrem(mn, 7, 0), int)  # 除以 0 分支


# 分支清單從 MCLASS 的 BR_* 類派生（含 c.beqz、c.bnez）。c.beqz/c.bnez 的 rs2 由
# _branch_operands 定為 x0，check() 再用 _regval 取其值 0；本測項只確認每個分支
# mnemonic 在 _branch_taken 都有規則、不會落到哨兵。
_BRANCH_MNEMONICS = sorted(
    m for m, c in isa.MCLASS.items() if c in ("BR_BEQ", "BR_BNE", "BR_LT", "BR_GE")
)


def test_branch_mnemonic_list_derived_from_mclass_includes_compressed():
    assert {"beq", "bne", "blt", "bltu", "bge", "bgeu", "c.beqz", "c.bnez"} <= set(
        _BRANCH_MNEMONICS
    )


@pytest.mark.parametrize("mn", _BRANCH_MNEMONICS)
def test_every_branch_mnemonic_has_taken_rule(mn):
    assert isinstance(isa._branch_taken(mn, 1, 2), bool)


# --- 不變式：compressed 表內的每個 mnemonic 都能算出結果（不是 None） --------
# ops 依實際格式；reads 給足（rd 兼 rs1、rs2）。


@pytest.mark.parametrize("mn", sorted(isa._COMPR_RD_RS2))
def test_every_compr_rd_rs2_mnemonic_is_computable(mn):
    got = isa._arith_operands(mn, ["x5", "x6"], {"x5": 1, "x6": 2, "x2": 3})
    assert got is not None


@pytest.mark.parametrize("mn", sorted(isa._COMPR_RD_IMM))
def test_every_compr_rd_imm_mnemonic_is_computable(mn):
    got = isa._arith_operands(mn, ["x8", "4"], {"x8": 1, "x2": 3, "x5": 2})
    assert got is not None


# --- 反向：RV32IM 該有的 mnemonic 都必須在表內（表被刪項目時會紅） ------------

_RV32IM_R = ("add sub sll slt sltu xor srl sra or and mul mulh mulhsu mulhu "
             "div divu rem remu").split()
_RV32I_I = "addi slti sltiu xori ori andi slli srli srai".split()


@pytest.mark.parametrize("mn", _RV32IM_R)
def test_rv32im_reg_reg_mnemonic_is_in_table(mn):
    assert mn in isa._REG_REG  # div/rem 也在同一張表，沒有另外的 divrem 表


@pytest.mark.parametrize("mn", _RV32I_I)
def test_rv32i_reg_imm_mnemonic_is_in_table(mn):
    assert mn in isa._REG_IMM


def test_compressed_tables_are_exact_no_unknown_mnemonics():
    # 表被加入沒有對應分支的名字（c.nand / c.xori）時，上面的 computable 測項會紅；
    # 這裡再釘住集合本身。
    assert isa._COMPR_RD_RS2 == {"c.add", "c.sub", "c.and", "c.or", "c.xor", "c.mv"}
    assert isa._COMPR_RD_IMM == {"c.addi", "c.andi", "c.slli", "c.srli", "c.srai"}


# --- jalr 的 32 位元回繞 ------------------------------------------------------


def test_jalr_target_wraps_to_32_bits():
    assert isa._jump_target("jalr", ["x0", "3(x5)"], {"x5": 0xFFFFFFFF}) == 0x2
    assert isa._jump_target("jalr", ["x0", "-4(x5)"], {"x5": 2}) == 0xFFFFFFFE
