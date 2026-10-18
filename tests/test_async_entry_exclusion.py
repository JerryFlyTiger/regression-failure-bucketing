"""Async-interrupt exclusion for BRANCH_WRONG_TARGET: a later xret returning to
the control-flow instruction's correct target/fallthrough means the odd
next_pc was a handler entry (docs/ARCHITECTURE.md 4.3)."""

from rfb import isa
from rfb.isa import ScanContext
from rfb.parse_trace import _pass1_trap_entries, parse_trace
from rfb.record import Instr

_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
_H = 0x80000500  # handler entry, never entered after a non-control-flow insn


def _ln(pc, mn, ops="", regs=""):
    insn = "0001" if mn.startswith("c.") else "00000013"
    return f"1\t1\t{pc:08x}\t{insn}\t{mn}\t{ops}\t{regs}\n"


def _run(tmp_path, lines):
    p = tmp_path / "trace.log"
    p.write_text(_HEADER + "".join(lines))
    return parse_trace(str(p))


def _jal(pc=0x80000100, ops="x0,80000200", regs=""):
    return _ln(pc, "jal", ops, regs)


def _handler(ret_pc):
    return [_ln(_H, "addi", "x0,x0,0", "x0=0x0"), _ln(_H + 4, "mret"),
            _ln(ret_pc, "addi", "x0,x0,0", "x0=0x0")]


def test_a_jal_then_mret_to_target_no_violation(tmp_path):
    info = _run(tmp_path, [_jal()] + _handler(0x80000200))
    assert info.violations == {}


def test_b_no_mret_reports_wrong_target(tmp_path):
    info = _run(tmp_path, [_jal(), _ln(_H, "addi", "x0,x0,0", "x0=0x0")])
    assert info.violations == {"BRANCH_WRONG_TARGET:jal": 1}


def test_c_mret_to_other_address_reports(tmp_path):
    info = _run(tmp_path, [_jal()] + _handler(0x80000900))
    assert info.violations == {"BRANCH_WRONG_TARGET:jal": 1}


def test_d_return_to_target_before_jal_reports(tmp_path):
    lines = [
        _ln(0x80000000, "mret"),                        # idx0 -> returns to target
        _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0"),   # idx1
        _jal(),                                         # idx2 (implicit entry, harmless)
        _ln(_H, "addi", "x0,x0,0", "x0=0x0"),
    ]
    info = _run(tmp_path, lines)
    assert info.violations == {"BRANCH_WRONG_TARGET:jal": 1}


def test_e_taken_branch_return_to_target(tmp_path):
    br = _ln(0x80000100, "beq", "x1,x2,80000200", "x1:0x00000005 x2:0x00000005")
    info = _run(tmp_path, [br] + _handler(0x80000200))
    assert info.violations == {}


def test_e_not_taken_branch_return_to_fallthrough(tmp_path):
    br = _ln(0x80000100, "beq", "x1,x2,80000200", "x1:0x00000005 x2:0x00000006")
    info = _run(tmp_path, [br] + _handler(0x80000104))
    assert info.violations == {}


def test_e_branch_without_return_reports(tmp_path):
    br = _ln(0x80000100, "beq", "x1,x2,80000200", "x1:0x00000005 x2:0x00000005")
    info = _run(tmp_path, [br, _ln(_H, "addi", "x0,x0,0", "x0=0x0")])
    assert info.violations == {"BRANCH_WRONG_TARGET:beq": 1}


def test_f_wrong_link_still_reported_when_target_excluded(tmp_path):
    # correct link would be 0x80000104
    jal = _jal(ops="x1,80000200", regs="x1=0x12345678")
    info = _run(tmp_path, [jal] + _handler(0x80000200))
    assert info.violations == {"WRONG_LINK:jal": 1}


def test_g_default_context_unchanged():
    ctx = ScanContext()
    assert ctx.xret_returns == {}
    i = Instr(index=0, pc=0x80000100, size=4, mnemonic="jal", operands="x0,80000200",
              reads={}, writes={}, pa=None, store=None, load=None)
    v = isa.check(None, i, _H, ctx)
    assert v is not None and v.kind == "BRANCH_WRONG_TARGET"
    assert isa.check(None, i, _H, ScanContext(xret_returns={0x80000200: 1})) is None
    assert isa.check(None, i, _H, ScanContext(xret_returns={0x80000200: 0})) is not None


def test_pass1_xret_index_matches_pass2_with_malformed_lines():
    lines = [
        _HEADER.rstrip("\n"),
        _ln(0x80000000, "addi", "x0,x0,0", "x0=0x0").rstrip("\n"),   # idx0
        "garbage",                                                   # skipped
        "1\t1\tzzzz\t00000013\taddi\tx0,x0,0\tx0=0x0",               # bad pc, skipped
        "",                                                          # skipped
        _ln(0x80000004, "mret").rstrip("\n"),                        # idx1
        "1\t1\t80000099",                                            # skipped
        _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0").rstrip("\n"),   # idx2
        _ln(0x80000204, "mret").rstrip("\n"),                        # idx3
        _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0").rstrip("\n"),   # idx4
    ]
    _, _, xr = _pass1_trap_entries(iter(lines))
    assert xr == {0x80000200: 3}


def test_pass1_index_equals_pass2_instr_index(tmp_path):
    # jal at pass-2 idx 2 (after malformed lines) must see an xret at idx 4:
    # an off-by-malformed pass-1 count would flip the strict ">" comparison.
    lines = [
        _ln(0x80000000, "addi", "x0,x0,0", "x0=0x0"),   # idx0
        "bad line\n",
        _ln(0x80000004, "addi", "x0,x0,0", "x0=0x0"),   # idx1
        "1\t1\tzz\t00000013\taddi\n",
        _jal(0x80000008),                               # idx2
        _ln(_H, "mret"),                                # idx3 (xret, returns to target)
        _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0"),   # idx4
    ]
    p = tmp_path / "trace.log"
    p.write_text(_HEADER + "".join(lines))
    _, _, xr = _pass1_trap_entries(iter((_HEADER + "".join(lines)).split("\n")))
    assert xr == {0x80000200: 3}
    assert parse_trace(str(p)).violations == {}


def test_a_jal_then_dret_to_target_no_violation(tmp_path):
    lines = [_jal(), _ln(_H, "addi", "x0,x0,0", "x0=0x0"), _ln(_H + 4, "dret"),
             _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0")]
    info = _run(tmp_path, lines)
    assert info.violations == {}


def test_non_xret_jump_back_to_target_still_reports(tmp_path):
    # only XRET-class returns count: a legal c.jr back to the target must not suppress it
    lines = [_jal(), _ln(_H, "addi", "x0,x0,0", "x0=0x0"),
             _ln(_H + 4, "c.jr", "x1", "x1:0x80000200"),
             _ln(0x80000200, "addi", "x0,x0,0", "x0=0x0")]
    info = _run(tmp_path, lines)
    assert info.violations.get("BRANCH_WRONG_TARGET:jal") == 1
    assert not any(k.startswith("BRANCH_WRONG_TARGET") and k != "BRANCH_WRONG_TARGET:jal"
                   for k in info.violations)


# --- 2026-10-04 interrupt vector slot exclusion (mtvec base + 4k) -----------

_B = 0x80000800  # mtvec base used below


def _mi(mn, ops, reads=None, idx=0):
    return Instr(index=idx, pc=0x80000010, size=4, mnemonic=mn, operands=ops,
                 reads=reads or {}, writes={}, pa=None, store=None, load=None)


def _obs(ctx, mn, ops, reads=None):
    isa.observe(ctx, _mi(mn, ops, reads))
    return ctx.mtvec_base


def test_mtvec_default_none():
    assert ScanContext().mtvec_base is None


def test_observe_csrrw_masks_low_byte():
    ctx = ScanContext()
    assert _obs(ctx, "csrrw", "x0,mtvec,x22", {"x22": _B | 0x81}) == _B


def test_observe_csrrw_unreadable_rs1_resets_to_unknown():
    # the written value is not visible in the trace: the old base is stale, so unknown (None)
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrw", "x0,mtvec,x22", {}) is None
    assert _obs(ScanContext(), "csrrw", "x0,mtvec,x22", {}) is None
    # an unreadable csrrs/csrrc operand also leaves the old base stale -> unknown (None)
    for mn in ("csrrs", "csrrc"):
        ctx = ScanContext(mtvec_base=_B)
        assert _obs(ctx, mn, "x0,mtvec,x22", {}) is None


def test_observe_csrrw_x0_writes_zero_is_unknown():
    # base 0 would make the exclusion window low addresses (F2): store None
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrw", "x1,mtvec,x0") is None


def test_observe_csrrwi():
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrwi", "x2,mtvec,26") is None      # 26 & ~0xFF = 0 -> unknown


def test_observe_base_zero_is_none_and_excludes_nothing():
    ctx = ScanContext()
    assert _obs(ctx, "csrrw", "x0,mtvec,x5", {"x5": 0x80}) is None
    for off in (0xC, 0x1C, 0x40, 0x7C):
        assert not isa._irq_vector_entry(off, ctx)
    # clearing every set bit also gives 0 -> None; set on unknown stays unknown
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrc", "x1,mtvec,x7", {"x7": _B}) is None
    assert _obs(ctx, "csrrs", "x1,mtvec,x7", {"x7": 0x1000}) is None


def test_observe_numeric_csr_name():
    assert _obs(ScanContext(), "csrrw", "x0,0x305,x5", {"x5": _B}) == _B


def test_observe_csrrs_pure_read_ignored():
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrs", "x1,mtvec,x0") == _B
    assert _obs(ctx, "csrrsi", "x1,mtvec,0") == _B
    assert _obs(ctx, "csrrc", "x1,mtvec,x0") == _B


def test_observe_csrrs_csrrc_known_value():
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrs", "x1,mtvec,x7", {"x7": 0x1000}) == _B | 0x1000
    assert _obs(ctx, "csrrc", "x1,mtvec,x7", {"x7": 0x1000}) == _B
    # csrrsi/csrrci: uimm <= 31 cannot touch bits [31:8], so they never change the base (equivalent)
    assert _obs(ctx, "csrrsi", "x1,mtvec,5") == _B
    assert _obs(ctx, "csrrci", "x1,mtvec,5") == _B


def test_observe_csrrs_csrrc_idempotent_bits():
    # set of an already-set bit / clear of an unset bit changes nothing (kills | -> ^ and & ~v -> ^ v)
    ctx = ScanContext(mtvec_base=_B | 0x1000)
    assert _obs(ctx, "csrrs", "x1,mtvec,x7", {"x7": 0x1000}) == _B | 0x1000
    assert _obs(ctx, "csrrc", "x1,mtvec,x7", {"x7": 0x2000}) == _B | 0x1000
    # mixed: set one new + one existing bit; clear one set + one unset bit
    assert _obs(ctx, "csrrs", "x1,mtvec,x7", {"x7": 0x3000}) == _B | 0x3000
    assert _obs(ctx, "csrrc", "x1,mtvec,x7", {"x7": 0x5000}) == _B | 0x2000


def test_observe_malformed_operands_never_raise_wrong_arity_keeps_base_bad_uimm_resets():
    for ops in ("", "x0,mtvec", "x0", "x0,mtvec,x5,x6"):
        ctx = ScanContext(mtvec_base=_B)
        assert _obs(ctx, "csrrw", ops, {"x5": 0x1234500}) == _B, ops
    # csrrwi with a non-numeric uimm: the ValueError branch, a write of an unknown value -> None
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrwi", "x0,mtvec,zz") is None
    assert _obs(ScanContext(), "csrrwi", "x0,mtvec,zz") is None
    # same for the set/clear i forms: an unparseable uimm leaves the base unknown
    for mn in ("csrrsi", "csrrci"):
        assert _obs(ScanContext(mtvec_base=_B), mn, "x0,mtvec,zz") is None, mn


def test_observe_csrrs_csrrc_unknown_value_not_updated():
    ctx = ScanContext()
    assert _obs(ctx, "csrrs", "x1,mtvec,x7", {"x7": 0x1000}) is None
    assert _obs(ctx, "csrrc", "x1,mtvec,x7", {"x7": 0x1000}) is None


def test_observe_other_csr_untouched():
    ctx = ScanContext(mtvec_base=_B)
    assert _obs(ctx, "csrrw", "x0,mscratch,x5", {"x5": 0x1234500}) == _B
    assert _obs(ctx, "csrrwi", "x2,misa,5") == _B


def _cf(mn):
    if mn == "jal":
        return _instr_cf("jal", "x0,80000200")
    return _instr_cf("beq", "x1,x2,80000200", {"x1": 5, "x2": 5})


def _instr_cf(mn, ops, reads=None):
    return Instr(index=0, pc=0x80000100, size=4, mnemonic=mn, operands=ops,
                 reads=reads or {}, writes={}, pa=None, store=None, load=None)


def test_check_vector_slots_excluded():
    ctx = ScanContext(mtvec_base=_B)
    for mn in ("jal", "beq"):
        for off in (0xC, 0x1C, 0x2C, 0x40, 0x7C):          # k = 3, 7, 11, 16, 31
            assert isa.check(None, _cf(mn), _B + off, ctx) is None, (mn, off)
        # k=0 (base), k=1/2/4..6/8..10/12..15 (no Ibex interrupt uses them), out of range, misaligned
        for off in (0, 4, 8, 0x10, 0x14, 0x18, 0x20, 0x28, 0x30, 0x3C, 0x80, 2, 6, 0x7E, -4):
            v = isa.check(None, _cf(mn), _B + off, ctx)
            assert v is not None and v.kind == "BRANCH_WRONG_TARGET", (mn, off)


def test_irq_slot_set_is_exactly_ibex_ids():
    ctx = ScanContext(mtvec_base=_B)
    hit = {k for k in range(0, 40) if isa._irq_vector_entry(_B + 4 * k, ctx)}
    assert hit == {3, 7, 11, *range(16, 32)} and len(hit) == 19


def test_check_unknown_mtvec_still_violation():
    for mn in ("jal", "beq"):
        v = isa.check(None, _cf(mn), _B + 4, ScanContext())
        assert v is not None and v.kind == "BRANCH_WRONG_TARGET"


def test_check_wrong_link_unaffected_by_vector_slot():
    i = _instr_cf("jal", "x1,80000200")
    i.writes["x1"] = 0x12345678
    v = isa.check(None, i, _B + 0xC, ScanContext(mtvec_base=_B))
    assert v is not None and v.kind == "WRONG_LINK"


def test_e2e_vector_slot_no_violation(tmp_path):
    lines = [
        _ln(0x80000000, "csrrw", "x0,mtvec,x22", f"x22:0x{_B | 1:08x} x0=0x0"),
        _jal(),
        _ln(_B + 0xC, "addi", "x0,x0,0", "x0=0x0"),
    ]
    assert _run(tmp_path, lines).violations == {}


def test_e2e_base_slot_still_violation(tmp_path):
    lines = [
        _ln(0x80000000, "csrrw", "x0,mtvec,x22", f"x22:0x{_B | 1:08x} x0=0x0"),
        _jal(),
        _ln(_B, "addi", "x0,x0,0", "x0=0x0"),
    ]
    assert _run(tmp_path, lines).violations == {"BRANCH_WRONG_TARGET:jal": 1}
