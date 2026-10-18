"""Unit tests for rfb.signature. See docs/ARCHITECTURE.md section 5.3.

2026-09-28 ALU 覆蓋，依 out/arch_alu 實驗（方案 G）: hard_key and
sys_viol_class now use isa.viol_group instead of the coarser isa.mclass for
MISMATCH deviations, and soft gained a wr_sys field. These tests pin down
that behavior.
"""

import pytest

from rfb.isa import mclass as isa_mclass
from rfb.isa import viol_group
from rfb.record import Deviation, FailureRecord, RegrInfo, SimInfo, TraceInfo
from rfb.signature import MIN_SYS_VIOL, _hard_key, _sys_viol, _wr_sys, signature


def _deviation(kind, mnemonic, mclass=None, source="scan", confidence=0.7, detail=None):
    return Deviation(
        kind=kind,
        mnemonic=mnemonic,
        mclass=mclass if mclass is not None else isa_mclass(mnemonic),
        detail=detail if detail is not None else {},
        source=source,
        confidence=confidence,
    )


def _record(family="MISMATCH", deviation=None, violations=None, raw="x"):
    regr = RegrInfo.empty()
    regr.raw = [raw]
    sim = SimInfo.empty()
    trace = TraceInfo.empty()
    trace.violations = dict(violations or {})
    return FailureRecord(
        case_id=1,
        family=family,
        regr=regr,
        sim=sim,
        trace=trace,
        deviation=deviation,
        parse_errors=[],
    )


def test_sra_and_c_srai_mismatch_hard_key_differ():
    rec_sra = _record(deviation=_deviation("WRONG_RESULT", "sra"))
    rec_csrai = _record(deviation=_deviation("WRONG_RESULT", "c.srai"))
    assert _hard_key(rec_sra) != _hard_key(rec_csrai)
    assert _hard_key(rec_sra) == ("MISMATCH", "WRONG_RESULT", "sra")
    assert _hard_key(rec_csrai) == ("MISMATCH", "WRONG_RESULT", "c.srai")


def test_bge_and_bgeu_mismatch_hard_key_same():
    rec_bge = _record(deviation=_deviation("BRANCH_WRONGLY_TAKEN", "bge"))
    rec_bgeu = _record(deviation=_deviation("BRANCH_WRONGLY_TAKEN", "bgeu"))
    assert _hard_key(rec_bge) == _hard_key(rec_bgeu)
    assert _hard_key(rec_bge) == ("MISMATCH", "BRANCH_WRONGLY_TAKEN", "BR_GE")


# 2026-10-01 BR_LT/BR_GE 評估（PLAN.md 4c）: LT/GE keep plain mclass
# grouping on both kinds -- no polarity token, no signed/unsigned split.
# Expected groups are literals, not mclass() calls, so a mclass rename or a
# viol_group special case both fail here.
@pytest.mark.parametrize("kind", ["BRANCH_NOT_TAKEN", "BRANCH_WRONGLY_TAKEN"])
@pytest.mark.parametrize(
    "mn,group", [("bge", "BR_GE"), ("bgeu", "BR_GE"), ("blt", "BR_LT"), ("bltu", "BR_LT")]
)
def test_lt_ge_branch_group_is_mclass(kind, mn, group):
    assert viol_group(kind, mn) == group
    assert _hard_key(_record(deviation=_deviation(kind, mn))) == ("MISMATCH", kind, group)
    # both _sys_viol paths: trace.violations only, then deviation only
    via_viol = _record(family="OTHER", violations={f"{kind}:{mn}": MIN_SYS_VIOL})
    assert _sys_viol(via_viol)[0] == frozenset({(kind, group)})
    via_dev = _record(deviation=_deviation(kind, mn))
    assert _sys_viol(via_dev)[0] == frozenset({(kind, group)})


def test_bge_and_bgeu_not_taken_hard_key_same():
    # bug_107 (set 2) is exactly this: bge and bgeu NOT_TAKEN from one bug.
    rec_bge = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "bge"))
    rec_bgeu = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "bgeu"))
    assert _hard_key(rec_bge) == _hard_key(rec_bgeu)


def test_beq_and_bne_mismatch_hard_key_differ():
    rec_beq = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "beq"))
    rec_bne = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "bne"))
    assert _hard_key(rec_beq) != _hard_key(rec_bne)
    # opposite is_equal error polarity (isa.viol_group)
    assert _hard_key(rec_beq) == ("MISMATCH", "BRANCH_NOT_TAKEN", "BR_EQ-")
    assert _hard_key(rec_bne) == ("MISMATCH", "BRANCH_NOT_TAKEN", "BR_EQ+")


def test_compressed_branches_share_mclass_with_base_form():
    assert isa_mclass("c.beqz") == isa_mclass("beq") == "BR_BEQ"
    assert isa_mclass("c.bnez") == isa_mclass("bne") == "BR_BNE"
    rec_beq = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "beq"))
    rec_cbeqz = _record(deviation=_deviation("BRANCH_NOT_TAKEN", "c.beqz"))
    assert _hard_key(rec_beq) == _hard_key(rec_cbeqz)


def test_viol_group_polarity_beq_nt_equals_bne_wt():
    # beq not taken / bne wrongly taken: is_equal wrongly false
    assert viol_group("BRANCH_NOT_TAKEN", "beq") == "BR_EQ-"
    assert viol_group("BRANCH_WRONGLY_TAKEN", "bne") == "BR_EQ-"


def test_viol_group_polarity_beq_wt_equals_bne_nt():
    # beq wrongly taken / bne not taken: is_equal wrongly true
    assert viol_group("BRANCH_WRONGLY_TAKEN", "beq") == "BR_EQ+"
    assert viol_group("BRANCH_NOT_TAKEN", "bne") == "BR_EQ+"


def test_viol_group_polarity_compressed_forms_match_base():
    assert viol_group("BRANCH_NOT_TAKEN", "c.beqz") == "BR_EQ-"
    assert viol_group("BRANCH_WRONGLY_TAKEN", "c.beqz") == "BR_EQ+"
    assert viol_group("BRANCH_NOT_TAKEN", "c.bnez") == "BR_EQ+"
    assert viol_group("BRANCH_WRONGLY_TAKEN", "c.bnez") == "BR_EQ-"


def test_viol_group_wrong_target_keeps_mclass():
    assert viol_group("BRANCH_WRONG_TARGET", "beq") == "BR_BEQ"
    assert viol_group("BRANCH_WRONG_TARGET", "c.beqz") == "BR_BEQ"
    assert viol_group("BRANCH_WRONG_TARGET", "bne") == "BR_BNE"
    assert viol_group("BRANCH_WRONG_TARGET", "c.bnez") == "BR_BNE"


def test_wr_sys_only_counts_sys_viol_threshold():
    below = _record(violations={"WRONG_RESULT:sra": MIN_SYS_VIOL - 1})
    at = _record(violations={"WRONG_RESULT:sra": MIN_SYS_VIOL})
    assert _wr_sys(below) == frozenset()
    assert _wr_sys(at) == frozenset({"sra"})


def test_wr_sys_excludes_non_alu_mnemonics_and_deviation_only():
    # non-ALU (LOAD) mnemonic never enters wr_sys even above threshold.
    rec = _record(
        violations={"WRONG_RESULT:lw": MIN_SYS_VIOL},
        deviation=_deviation("WRONG_RESULT", "sra"),
    )
    # deviation alone (no scan violations entry) never enters wr_sys either
    # -- wr_sys is scan-only per docs/ARCHITECTURE.md 5.3.
    assert _wr_sys(rec) == frozenset()


def test_pcdelta_lw_wrong_result_still_uses_load_group():
    rec = _record(
        deviation=_deviation("WRONG_RESULT", "lw", mclass="LOAD", source="pcdelta"),
    )
    assert _hard_key(rec) == ("MISMATCH", "WRONG_RESULT", "LOAD")


def test_sys_viol_class_uses_viol_group_for_alu_and_mclass_for_others():
    rec = _record(
        violations={"WRONG_RESULT:sra": MIN_SYS_VIOL, "BRANCH_WRONGLY_TAKEN:bge": MIN_SYS_VIOL},
        deviation=_deviation("WRONG_RESULT", "sra"),
    )
    cls, keys = _sys_viol(rec)
    assert ("WRONG_RESULT", "sra") in cls
    assert ("BRANCH_WRONGLY_TAKEN", "BR_GE") in cls
    assert ("WRONG_RESULT", "sra") in keys
    assert ("BRANCH_WRONGLY_TAKEN", "bge") in keys


def test_signature_soft_has_wr_sys_field():
    rec = _record(violations={"WRONG_RESULT:sra": MIN_SYS_VIOL})
    sig = signature(rec)
    assert sig.soft["wr_sys"] == frozenset({"sra"})


def test_sys_viol_violations_path_uses_polarity_group_for_beq_bne():
    # no deviation: only the trace.violations path contributes
    rec = _record(
        family="OTHER",
        violations={
            "BRANCH_NOT_TAKEN:beq": MIN_SYS_VIOL,
            "BRANCH_WRONGLY_TAKEN:beq": MIN_SYS_VIOL,
            "BRANCH_NOT_TAKEN:bne": MIN_SYS_VIOL,
            "BRANCH_WRONGLY_TAKEN:bne": MIN_SYS_VIOL,
        },
    )
    cls, keys = _sys_viol(rec)
    assert cls == frozenset(
        {
            ("BRANCH_NOT_TAKEN", "BR_EQ-"),
            ("BRANCH_NOT_TAKEN", "BR_EQ+"),
            ("BRANCH_WRONGLY_TAKEN", "BR_EQ+"),
            ("BRANCH_WRONGLY_TAKEN", "BR_EQ-"),
        }
    )
    assert ("BRANCH_NOT_TAKEN", "beq") in keys and ("BRANCH_WRONGLY_TAKEN", "bne") in keys


# --- READ_MISMATCH: scan counts stay out of sys_viol / wr_sys ---------------


def test_read_mismatch_scan_count_not_in_sys_viol_or_wr_sys():
    rec = _record(violations={"READ_MISMATCH:add": MIN_SYS_VIOL + 3})
    assert _sys_viol(rec) == (frozenset(), frozenset())
    # Documentation only, not a defence: _wr_sys never collected anything but WRONG_RESULT.
    assert _wr_sys(rec) == frozenset()
    sig = signature(rec)
    assert sig.soft["sys_viol_class"] == frozenset()
    assert sig.soft["sys_viol_keys"] == frozenset()
    assert sig.soft["wr_sys"] == frozenset()


def test_read_mismatch_deviation_hard_key_and_sys_viol():
    rec = _record(deviation=_deviation("READ_MISMATCH", "add"),
                  violations={"READ_MISMATCH:add": 5})
    assert _hard_key(rec) == ("MISMATCH", "READ_MISMATCH", "REGREAD")
    cls, keys = _sys_viol(rec)
    assert cls == frozenset({("READ_MISMATCH", "REGREAD")})
    assert keys == frozenset({("READ_MISMATCH", "add")})


def test_read_mismatch_fwd_deviation_hard_key_and_sys_viol():
    rec = _record(deviation=_deviation("READ_MISMATCH", "add", detail={"fwd": True}),
                  violations={"READ_MISMATCH:add": 5})
    assert _hard_key(rec) == ("MISMATCH", "READ_MISMATCH", "REGREAD_FWD")
    cls, keys = _sys_viol(rec)
    assert cls == frozenset({("READ_MISMATCH", "REGREAD_FWD")})
    assert keys == frozenset({("READ_MISMATCH", "add")})
    nofwd = _record(deviation=_deviation("READ_MISMATCH", "add", detail={"fwd": False}))
    assert _hard_key(nofwd) == ("MISMATCH", "READ_MISMATCH", "REGREAD")


# --- hknegu: CSR X-origin signature (PLAN.md 4c 公開合併 0.96 根因) -----------


from rfb.signature import _xsig  # noqa: E402


def _xrecord(family, x_origins, templates=()):
    rec = _record(family=family)
    rec.trace.x_origins = dict(x_origins)
    rec.sim.error_templates = list(templates)
    if family == "FATAL":
        rec.sim.fatals = []
    return rec


_X2 = {"mip:.....X..": 2}


def test_xsig_requires_min_sys_viol_occurrences():
    assert _xsig(_xrecord("FATAL", {"mip:.....X..": MIN_SYS_VIOL - 1})) == frozenset()
    assert _xsig(_xrecord("FATAL", {"mip:.....X..": MIN_SYS_VIOL})) == frozenset({"mip:.....X.."})


def test_xsig_count_one_ignored_two_counted():
    assert _xsig(_xrecord("FATAL", {"mip:.....X..": 1})) == frozenset()
    assert _xsig(_xrecord("FATAL", {"mip:.....X..": 2})) == frozenset({"mip:.....X.."})


def test_fatal_hard_key_includes_xsig():
    key = _hard_key(_xrecord("FATAL", _X2))
    assert key[-1] == frozenset({"mip:.....X.."})
    assert key != _hard_key(_xrecord("FATAL", {}))


def test_assert_hard_key_includes_xsig():
    key = _hard_key(_xrecord("ASSERT", _X2))
    assert key[-1] == frozenset({"mip:.....X.."})
    assert key != _hard_key(_xrecord("ASSERT", {}))


def test_other_hard_key_includes_xsig():
    key = _hard_key(_xrecord("OTHER", _X2))
    assert key[-1] == frozenset({"mip:.....X.."})


def test_mismatch_hard_key_unchanged_by_xsig():
    dev = _deviation("WRONG_RESULT", "add")
    a = _record(deviation=dev)
    b = _record(deviation=dev)
    b.trace.x_origins = dict(_X2)
    assert _hard_key(a) == _hard_key(b)


def test_soft_carries_xsig():
    assert signature(_xrecord("FATAL", _X2)).soft["xsig"] == frozenset({"mip:.....X.."})


def test_mismatch_none_kind_hard_key_unchanged_by_xsig():
    a = _record(deviation=_deviation("NONE", None))
    b = _record(deviation=_deviation("NONE", None))
    b.trace.x_origins = dict(_X2)
    assert _xsig(b)
    assert not _xsig(a)
    assert _hard_key(a) == _hard_key(b)
