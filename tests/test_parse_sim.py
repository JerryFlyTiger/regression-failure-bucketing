from pathlib import Path

from rfb.parse_sim import parse_sim

ROOT = Path(__file__).resolve().parents[1]
SET1 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_1"
SET2 = ROOT / "testcase/2026_08_20_samples/problem/benchmark_set_2"


def test_set2_case14_error_templates():
    info = parse_sim(str(SET2 / "case_14/sim.log.gz"))
    assert len(info.error_templates) == 2
    assert set(info.error_templates.values()) == {5140}
    assert "u_ibex_core.cs_registers_i.IbexCsrWdataIntKnown" in info.error_templates
    assert "u_ibex_core.load_store_unit_i.IbexDataOffsetKnown" in info.error_templates
    assert info.counts["UVM_ERROR"] == 10280
    assert info.counts["UVM_FATAL"] == 1
    assert info.counts["UVM_INFO"] == 2944
    assert len(info.fatals) == 1


def test_set2_case14_first_errors_capped_at_five():
    info = parse_sim(str(SET2 / "case_14/sim.log.gz"))
    assert len(info.first_errors) == 5


def test_fatal_case_fields():
    info = parse_sim(str(SET1 / "case_1/sim.log.gz"))
    assert info.verdict == "FAILED"
    assert len(info.fatals) == 1
    f = info.fatals[0]
    assert f.file == "core_ibex_base_test.sv"
    assert f.id == "uvm_test_top"
    assert f.template == (
        "Did not receive core_status HANDLING_IRQ within <N> cycle timeout period"
    )
    assert info.error_templates == {}
    assert info.counts == {
        "UVM_INFO": 9,
        "UVM_WARNING": 0,
        "UVM_ERROR": 0,
        "UVM_FATAL": 1,
    }


def test_unparseable_uvm_error_line_still_counted(tmp_path):
    content = (
        "+UVM_TESTNAME=test1\n"
        "UVM_ERROR this is not the expected format at all\n"
        "UVM_ERROR /path/file.sv(10) @ 100: scope [id] some message\n"
    )
    path = tmp_path / "sim.log"
    path.write_text(content)
    info = parse_sim(str(path))
    # both UVM_ERROR lines must be counted, even the one that doesn't match
    # the strict regex (previously silently dropped).
    assert info.counts["UVM_ERROR"] == 2
    # but only the well-formed line contributes a template/first_error.
    assert info.error_templates == {"some message": 1}
    assert len(info.first_errors) == 1


def test_report_summary_matches_counts_across_public_set():
    for base in (SET1, SET2):
        for case_dir in sorted(base.glob("case_*")):
            info = parse_sim(str(case_dir / "sim.log.gz"))
            if info.counts:
                assert info.counts.get("UVM_ERROR", 0) == sum(
                    info.error_templates.values()
                ), case_dir
                assert info.counts.get("UVM_FATAL", 0) == len(info.fatals), case_dir
