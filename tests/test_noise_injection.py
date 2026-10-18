"""Small wiring test for tools/noise_eval.py's noise injector (not the scenarios).

Set 2 case 24 (MISMATCH, 2050 instrs, 92 jal) and case 18 (ASSERT, 1758 instrs):
injecting c jal-noise sites must raise the BRANCH_WRONG_TARGET:jal count by exactly c,
put (BRANCH_WRONG_TARGET, JUMP) in sys_viol_class only for c >= 2, and leave the
MISMATCH deviation / hard_key unchanged (inject_case asserts these itself and raises
InjectionError otherwise; they are re-checked here independently).
"""
import os

import pytest

from rfb.signature import signature
from tools import noise_eval as ne

KEY = "BRANCH_WRONG_TARGET:jal"


def _case(set_no, case_id):
    d = os.path.join(ne.PUB, f"benchmark_set_{set_no}", f"case_{case_id}")
    return d, ne.parse_case(case_id, d, os.path.join(d, "trace.log.gz"))


@pytest.mark.parametrize("set_no,case_id,c", [(2, 24, 2), (2, 18, 2), (2, 18, 1), (2, 24, 5)])
def test_inject_counts_and_vcls(tmp_path, set_no, case_id, c):
    d, orig = _case(set_no, case_id)
    assert orig.trace.violations.get(KEY, 0) == 0
    info, _ = ne.inject_case(set_no, case_id, "jal", c, str(tmp_path), "trace.log.gz", orig=orig)
    assert info["n_inj"] == c
    assert len(set(info["h"])) == c
    new = ne.parse_case(case_id, d, str(tmp_path / "trace.log.gz"))
    assert new.trace.violations[KEY] == c
    cls = signature(new).soft["sys_viol_class"]
    assert (("BRANCH_WRONG_TARGET", "JUMP") in cls) == (c >= 2)
    assert new.trace.n_instr == orig.trace.n_instr + c
    if orig.family == "MISMATCH":
        assert (new.deviation.kind, new.deviation.mnemonic) == (orig.deviation.kind, orig.deviation.mnemonic)
        assert signature(new).hard_key == signature(orig).hard_key
        assert all(w in ("after", "before") for _, w, _ in info["placed"])
        # MISMATCH: sites are after the anchor when possible
        assert all(w == "after" for _, w, _ in info["placed"])


def test_h_not_in_original_trace(tmp_path):
    d, orig = _case(2, 24)
    info, _ = ne.inject_case(2, 24, "jal", 2, str(tmp_path), "trace.log.gz", orig=orig)
    scan = ne.Scan(ne.read_trace(os.path.join(d, "trace.log.gz")), os.path.join(d, "regr.log"))
    assert not set(info["h"]) & scan.pcs


def test_returning_injection_lines_and_trap_entries(tmp_path):
    """returning mode: after each site a fake addi at H and an mret at H+4, then the
    original next line; trap_entries unchanged.  Independent of rfb.xret_returns."""
    d, orig = _case(2, 18)
    src = os.path.join(d, "trace.log.gz")
    info, _ = ne.inject_case(2, 18, "jal", 3, str(tmp_path), "trace.log.gz", orig=orig, mode="returning")
    assert info["n_inj"] == 3
    old = ne.Scan(ne.read_trace(src), os.path.join(d, "regr.log"))
    new = ne.Scan(ne.read_trace(str(tmp_path / "trace.log.gz")), os.path.join(d, "regr.log"))
    assert new.traps == old.traps
    assert len(new.entries) == len(old.entries) + 6
    pcs = [ins.pc for _, ins in new.entries]
    for h in info["h"]:
        i = pcs.index(h)
        assert [new.entries[i + k][1].mnemonic for k in (0, 1)] == ["addi", "mret"]
        assert new.entries[i + 1][1].pc == h + 4
        # the line before is the jal site, the line after is its original (correct) next line
        assert new.entries[i - 1][1].mnemonic == "jal"
        assert new.entries[i + 2][1].pc != h + 8
        assert new.entries[i + 2][1].pc in old.pcs
    assert not set(info["h"]) & old.pcs
    assert not {h + 4 for h in info["h"]} & old.pcs
    new_rec = ne.parse_case(18, d, str(tmp_path / "trace.log.gz"))
    assert new_rec.trace.n_instr == orig.trace.n_instr + 6
    others = lambda r: {k: v for k, v in r.trace.violations.items() if k != KEY}
    assert others(new_rec) == others(orig)


def test_injected_coverage_needs_two_golden_bugs_per_step():
    """check_injected_coverage: every step's injected cases must span >= 2 golden bugs
    (one bug alone cannot show an R1 collision); 0-site cases do not count."""
    golden = {"1": "bugA", "2": "bugA", "3": "bugB"}

    def res(*step_inj):
        return ({"step_inj": list(step_inj)}, [])

    plan = {"1": ("jal", 2), "2": ("jal", 2), "3": ("jal", 2)}
    one_bug = {"1": res(2), "2": res(1), "3": res(0)}
    with pytest.raises(ne.InjectionError):
        ne.check_injected_coverage("NX", 1, plan, one_bug, golden)
    two_bugs = {"1": res(2), "2": res(0), "3": res(2)}
    assert ne.check_injected_coverage("NX", 1, plan, two_bugs, golden) == ["2/3"]

    steps = {c: [("returning", "jal", 5), ("nonreturn", "jal", 1)] for c in golden}
    second_step_one_bug = {"1": res(5, 1), "2": res(5, 1), "3": res(5, 0)}
    with pytest.raises(ne.InjectionError):
        ne.check_injected_coverage("NX", 1, steps, second_step_one_bug, golden)
    first_step_one_bug = {"1": res(5, 1), "2": res(5, 0), "3": res(0, 1)}
    with pytest.raises(ne.InjectionError):
        ne.check_injected_coverage("NX", 1, steps, first_step_one_bug, golden)
    assert ne.check_injected_coverage("NX", 1, steps, {"1": res(5, 1), "2": res(5, 0), "3": res(5, 1)},
                                      golden) == ["3/3", "2/3"]

    # c >= 2: a case with a single site cannot reach MIN_SYS_VIOL, so it does not count
    single_site = {"1": res(2), "2": res(2), "3": res(1)}
    with pytest.raises(ne.InjectionError):
        ne.check_injected_coverage("NX", 1, plan, single_site, golden)
    # ...but it still shows in the report string (cases with >= 1 site)
    golden4 = dict(golden, **{"4": "bugB"})
    plan4 = dict(plan, **{"4": ("jal", 2)})
    assert ne.check_injected_coverage("NX", 1, plan4, {"1": res(2), "2": res(1), "3": res(2), "4": res(2)},
                                      golden4) == ["4/4"]
    # c = 5: 2 sites already reach MIN_SYS_VIOL, so a short case counts; 1 site does not
    plan5 = {c: ("jal", 5) for c in golden}
    assert ne.check_injected_coverage("NX", 1, plan5, {"1": res(5), "2": res(5), "3": res(2)}, golden) == ["3/3"]
    with pytest.raises(ne.InjectionError):
        ne.check_injected_coverage("NX", 1, plan5, {"1": res(5), "2": res(5), "3": res(1)}, golden)


def test_inject_case_step_inj_per_step(tmp_path):
    """inject_case records one step_inj entry per step (the input of check_injected_coverage)."""
    _, orig = _case(2, 18)
    info, _ = ne.inject_case(2, 18, None, None, str(tmp_path), "trace.log.gz", orig=orig,
                             steps=[("returning", "jal", 3), ("nonreturn", "jal", 1)])
    assert info["step_inj"] == [3, 1]


def test_scan_xret_ret_matches_rfb_pass1():
    """Scan.xret_ret uses the same coordinate as rfb's pass-1 xret_returns (set 2 case 15 has
    thousands of xrets)."""
    from rfb.parse_trace import _iter_lines, _pass1_trap_entries, _read_all_text

    d = os.path.join(ne.PUB, "benchmark_set_2", "case_15")
    path = os.path.join(d, "trace.log.gz")
    scan = ne.Scan(ne.read_trace(path), os.path.join(d, "regr.log"))
    xr = _pass1_trap_entries(_iter_lines(_read_all_text(path)))[2]
    assert len(xr) > 0
    assert scan.xret_ret == xr


# ---------------------------------------------------------------- vector mode
def _tab_line(pc, mn, ops, regs=""):
    return "\t".join(["1000", "1", f"{pc:08x}", "00000013", mn, ops, regs])


def _mini_scan(rows):
    """Scan over a synthetic trace; rows = (pc, mnemonic, operands, regs)."""
    lines = ["Time\tCycle\tPC\tInsn\tDecoded\tOperands\tRegs"] + [_tab_line(*r) for r in rows]
    return ne.Scan(lines, None)


def test_mtvec_base_parsing():
    rows = [(0x100, "csrrw", "x0,mtvec,x5", " x5:0x80012001  x0=0x00000000"),
            (0x104, "addi", "x1,x1,1", ""),
            (0x108, "csrrwi", "x2,mtvec,26", " x2=0x00000001"),
            (0x10C, "csrrw", "x0,mtvec,x6", " x6:0x8000abcd"),
            (0x110, "csrrw", "x0,mscratch,x6", " x6:0x1234"),
            (0x114, "csrrs", "x0,mtvec,x6", " x6:0x1234")]
    s = _mini_scan(rows)
    assert s.mtvec_writes == [(0, 0x80012000), (2, None), (3, 0x8000AB00)]
    assert s.mtvec_before(0) is None          # the write itself is not "before"
    assert s.mtvec_before(1) == 0x80012000
    assert s.mtvec_before(3) is None          # csrrwi 26 -> base 0 -> unknown (same as rfb.isa)
    assert s.mtvec_before(5) == 0x8000AB00    # mscratch write and csrrs are ignored


def test_vector_slot_choice_and_no_mtvec_no_site():
    jal = lambda pc, t: (pc, "jal", f"x0,{t:#x}", "")
    base = 0x80012000
    rows = [(0x100, "csrrw", "x0,mtvec,x5", f" x5:{base | 1:#010x}")]
    rows += [(0x200 + 4 * i, "addi", "x1,x1,1", "") for i in range(12)]
    # real PCs sitting in vector slots k=3 and k=7: they must not be chosen
    rows += [(base + 12, "addi", "x1,x1,1", ""), (base + 28, "addi", "x1,x1,1", "")]
    for j in range(6):
        pc = 0x300 + 8 * j
        rows += [jal(pc, pc + 4), (pc + 4, "addi", "x1,x1,1", "")]
    rows.append((0x400, "addi", "x1,x1,1", ""))
    s = _mini_scan(rows)
    cand = s.candidates("jal", "vector")
    assert len(cand) == 6
    new, info = ne.inject_trace(s, "jal", 6, "vector")
    assert len(info["h"]) == 6
    for h in info["h"]:
        d = h - base
        assert d % 4 == 0 and d // 4 in ne.VEC_KS and d not in (12, 28) and h not in s.pcs
    assert len(set(info["h"])) == 6           # 17 free slots: no repetition needed
    # no mtvec write -> nothing to inject
    s0 = _mini_scan(rows[1:])
    assert s0.candidates("jal", "vector") == []
    assert ne.inject_trace(s0, "jal", 2, "vector")[1]["n_inj"] == 0
    # a jal before the (only) mtvec write is not a site
    early = [jal(0x500 + 8 * j, 0x508 + 8 * j) for j in range(ne.HEAD_SKIP + 1)]
    early += [(0x500 + 8 * (ne.HEAD_SKIP + 1), "addi", "x1,x1,1", "")]
    s1 = _mini_scan(early + [rows[0]] + rows[1:])
    n_early = len(early)
    cand1 = s1.candidates("jal", "vector")
    assert cand1 and all(i > n_early for i in cand1)   # no site before the first mtvec write
    assert all(s1.mtvec_before(i) is not None for i in cand1)
    # sanity: with the mtvec write first, the same early jals ARE candidates (so the check above bites)
    s1b = _mini_scan([rows[0]] + early)
    assert any(i >= ne.HEAD_SKIP for i in s1b.candidates("jal", "vector"))
    # a base-0 / unreadable write makes the base unknown again: later sites are dropped
    s2 = _mini_scan(rows + [(0x600, "csrrw", "x0,mtvec,x0", ""), jal(0x700, 0x704), (0x704, "addi", "x1,x1,1", "")])
    assert all(s2.entries[i][1].pc != 0x700 for i in s2.candidates("jal", "vector"))


def test_vector_real_case_verified_and_fixed_parser_silent(tmp_path):
    d, orig = _case(2, 18)
    info, notes = ne.inject_case(2, 18, "jal", 5, str(tmp_path), "trace.log.gz", orig=orig, mode="vector")
    assert info["n_inj"] == 5
    assert notes and notes[0].startswith("vector:")
    new = ne.parse_case(18, d, str(tmp_path / "trace.log.gz"))
    # the fix (mtvec_base) suppresses WRONG_TARGET at every injected site
    assert new.trace.violations.get(KEY, 0) == orig.trace.violations.get(KEY, 0)
    assert new.trace.n_instr == orig.trace.n_instr + 5


def test_verify_vector_rejects_h_equal_to_site_target():
    """H is a legal slot (k=16) but equals the site's own target: no violation without the fix,
    so verify_vector must raise."""
    base = 0x80012000
    h = base + 0x40
    rows = [(0x100, "csrrw", "x0,mtvec,x5", f" x5:{base | 1:#010x}")]
    rows += [(0x200 + 4 * i, "addi", "x1,x1,1", "") for i in range(12)]
    rows += [(0x300, "jal", f"x0,{h:#x}", ""), (0x304, "addi", "x1,x1,1", "")]
    i = len(rows) - 2                                   # the jal
    orig_scan = _mini_scan(rows)
    new_rows = rows[: i + 1] + [(h, "addi", "x1,x1,1", "")] + rows[i + 1:]
    new_scan = _mini_scan(new_rows)

    def case(n):
        return type("C", (), {"case_id": 0, "family": "FATAL", "trace": type("T", (), {"n_instr": n, "violations": {}})()})()

    info = {"placed": [(i, "after", False)], "h": [h]}
    with pytest.raises(ne.InjectionError, match="without the vector fix"):
        ne.verify_vector(orig_scan, new_scan, case(len(rows)), case(len(new_rows)), "jal", info)
    # control: a different free slot (k=17) passes the same checks
    h2 = base + 0x44
    new2 = _mini_scan(rows[: i + 1] + [(h2, "addi", "x1,x1,1", "")] + rows[i + 1:])
    info2 = {"placed": [(i, "after", False)], "h": [h2]}
    ne.verify_vector(orig_scan, new2, case(len(rows)), case(len(rows) + 1), "jal", info2)


def test_vec_ks_matches_rfb_irq_slots():
    # the injector must use exactly the slots rfb excludes: a missing slot narrows N9/N10,
    # an extra one injects noise rfb (correctly) still reports
    assert set(ne.VEC_KS) == set(ne.isa._IRQ_KS)
    assert len(ne.VEC_KS) == len(set(ne.VEC_KS))


def test_verify_vector_rejects_non_irq_slot():
    """H = base + 4 (k=1) is a vector-table word but not an Ibex interrupt slot: rfb keeps
    the violation there, so verify_vector must refuse it as a vector site."""
    base = 0x80012000
    h = base + 0x4
    rows = [(0x100, "csrrw", "x0,mtvec,x5", f" x5:{base | 1:#010x}")]
    rows += [(0x200 + 4 * i, "addi", "x1,x1,1", "") for i in range(12)]
    rows += [(0x300, "jal", "x0,0x400", ""), (0x400, "addi", "x1,x1,1", "")]
    i = len(rows) - 2                                   # the jal
    orig_scan = _mini_scan(rows)
    new_scan = _mini_scan(rows[: i + 1] + [(h, "addi", "x1,x1,1", "")] + rows[i + 1:])

    def case(n):
        return type("C", (), {"case_id": 0, "family": "FATAL", "trace": type("T", (), {"n_instr": n, "violations": {}})()})()

    info = {"placed": [(i, "after", False)], "h": [h]}
    with pytest.raises(ne.InjectionError, match="is not a free slot"):
        ne.verify_vector(orig_scan, new_scan, case(len(rows)), case(len(rows) + 1), "jal", info)


def _vec_fixture(h_fn, extra_rows=()):
    """Synthetic trace with one jal site; returns (rows, i, orig_scan, new_scan, h)."""
    base = 0x80012000
    h = h_fn(base)
    rows = [(0x100, "csrrw", "x0,mtvec,x5", f" x5:{base | 1:#010x}")]
    rows += [(0x200 + 4 * i, "addi", "x1,x1,1", "") for i in range(12)]
    rows += list(extra_rows)
    rows += [(0x300, "jal", "x0,0x400", ""), (0x400, "addi", "x1,x1,1", "")]
    i = len(rows) - 2                                   # the jal
    orig_scan = _mini_scan(rows)
    new_scan = _mini_scan(rows[: i + 1] + [(h, "addi", "x1,x1,1", "")] + rows[i + 1:])
    return rows, i, orig_scan, new_scan, h


def _vcase(n, viol):
    return type("C", (), {"case_id": 0, "family": "FATAL",
                          "trace": type("T", (), {"n_instr": n, "violations": viol})()})()


def test_verify_vector_rejects_rfb_still_reporting():
    """When rfb tracks mtvec_base, a vector site whose violation count rises in rfb's own parse
    (noise_eval and rfb disagree on the base, e.g. across a csrrs) must abort."""
    if not ne.rfb_has_mtvec_base():
        pytest.skip("rfb has no mtvec_base")
    rows, i, orig_scan, new_scan, h = _vec_fixture(lambda b: b + 0x40)
    info = {"placed": [(i, "after", False)], "h": [h]}
    orig = _vcase(len(rows), {KEY: 2})
    with pytest.raises(ne.InjectionError, match="rfb still reports"):
        ne.verify_vector(orig_scan, new_scan, orig, _vcase(len(rows) + 1, {KEY: 3}), "jal", info)
    # a drop is refused too (the count must stay unchanged, not merely not rise)
    with pytest.raises(ne.InjectionError, match="rfb still reports"):
        ne.verify_vector(orig_scan, new_scan, orig, _vcase(len(rows) + 1, {KEY: 1}), "jal", info)
    # control: same count passes
    ne.verify_vector(orig_scan, new_scan, orig, _vcase(len(rows) + 1, {KEY: 2}), "jal", info)


def test_verify_vector_rejects_unaligned_h():
    """H = base + 0xE: 0xE >> 2 = 3 is a legal slot number but the address is not 4-aligned."""
    rows, i, orig_scan, new_scan, h = _vec_fixture(lambda b: b + 0xE)
    info = {"placed": [(i, "after", False)], "h": [h]}
    with pytest.raises(ne.InjectionError, match="is not a free slot"):
        ne.verify_vector(orig_scan, new_scan, _vcase(len(rows), {}), _vcase(len(rows) + 1, {}), "jal", info)


def test_verify_vector_rejects_h_already_in_trace():
    """H - base is a legal slot (k=16) but that PC already occurs in the original trace."""
    base = 0x80012000
    rows, i, orig_scan, new_scan, h = _vec_fixture(lambda b: b + 0x40,
                                                    extra_rows=[(base + 0x40, "addi", "x1,x1,1", "")])
    assert h in orig_scan.pcs
    info = {"placed": [(i, "after", False)], "h": [h]}
    with pytest.raises(ne.InjectionError, match="is not a free slot"):
        ne.verify_vector(orig_scan, new_scan, _vcase(len(rows), {}), _vcase(len(rows) + 1, {}), "jal", info)
