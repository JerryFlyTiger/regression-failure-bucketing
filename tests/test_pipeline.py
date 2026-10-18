"""ARCHITECTURE 5.1 family classification order, plus parse_all's error
handling for missing/malformed logs (ARCHITECTURE 4.6's "don't interrupt the
stream" requirement)."""

import csv

from rfb.pipeline import classify_family, parse_all
from rfb.record import RegrInfo, SimInfo, TraceInfo, UvmMsg

_TRACE_HEADER = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"


def _regr(kind: str) -> RegrInfo:
    r = RegrInfo.empty()
    r.kind = kind
    return r


def _sim(fatals=None, error_templates=None) -> SimInfo:
    s = SimInfo.empty()
    if fatals:
        s.fatals = fatals
    if error_templates:
        s.error_templates = error_templates
    return s


def _fatal_msg() -> UvmMsg:
    return UvmMsg(
        severity="UVM_FATAL",
        file="x.sv",
        line=1,
        scope="scope",
        id="id",
        template="tmpl",
        raw="raw",
    )


def test_classify_family_mismatch():
    assert classify_family(_regr("MISMATCH"), _sim()) == "MISMATCH"


def test_classify_family_fatal():
    assert classify_family(_regr("UNKNOWN"), _sim(fatals=[_fatal_msg()])) == "FATAL"


def test_classify_family_assert():
    assert classify_family(_regr("UNKNOWN"), _sim(error_templates={"t": 1})) == "ASSERT"


def test_classify_family_other():
    assert classify_family(_regr("UNKNOWN"), _sim()) == "OTHER"


def test_classify_family_mismatch_takes_priority_over_fatal():
    regr = _regr("MISMATCH")
    sim = _sim(fatals=[_fatal_msg()], error_templates={"t": 1})
    assert classify_family(regr, sim) == "MISMATCH"


def test_classify_family_fatal_takes_priority_over_assert():
    regr = _regr("UNKNOWN")
    sim = _sim(fatals=[_fatal_msg()], error_templates={"t": 1})
    assert classify_family(regr, sim) == "FATAL"


def _write_case(case_dir, trace_lines=None):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "regr.log").write_text("")
    (case_dir / "sim.log").write_text("")
    if trace_lines is not None:
        (case_dir / "trace.log").write_text(_TRACE_HEADER + "".join(trace_lines))


def test_parse_all_missing_trace_file_yields_empty_shell_not_none(tmp_path):
    _write_case(tmp_path / "case_1", trace_lines=None)  # no trace.log written at all

    csv_path = tmp_path / "input.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        w.writerow([1, "case_1/regr.log", "case_1/sim.log", "case_1/trace.log"])

    records = parse_all(str(csv_path))
    assert len(records) == 1
    record = records[0]

    assert record.trace == TraceInfo.empty()
    assert any("parse_trace" in e for e in record.parse_errors)


def test_parse_all_aggregates_line_level_trace_parse_errors(tmp_path):
    # a trailing line with too few columns is a line-level error that
    # parse_trace collects without raising; parse_all must surface it in
    # the record's parse_errors (not just top-level exceptions).
    _write_case(tmp_path / "case_2", trace_lines=["1\t1\t80000000\n"])

    csv_path = tmp_path / "input.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        w.writerow([2, "case_2/regr.log", "case_2/sim.log", "case_2/trace.log"])

    records = parse_all(str(csv_path))
    assert len(records) == 1
    record = records[0]

    assert any("not enough columns" in e for e in record.parse_errors)


def _write_input_csv(tmp_path, case_id, case_name):
    csv_path = tmp_path / "input.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        w.writerow([case_id, f"{case_name}/regr.log", f"{case_name}/sim.log", f"{case_name}/trace.log"])
    return csv_path


def test_parse_all_missing_regr_file_yields_empty_shell(tmp_path):
    case_dir = tmp_path / "case_3"
    _write_case(case_dir, trace_lines=[])
    (case_dir / "regr.log").unlink()  # regr.log missing -> parse_regr raises

    records = parse_all(str(_write_input_csv(tmp_path, 3, "case_3")))
    assert len(records) == 1
    record = records[0]

    assert record.regr == RegrInfo.empty()
    assert record.regr.kind == "UNKNOWN"
    assert record.family == "OTHER"
    assert any("parse_regr" in e for e in record.parse_errors)
    # trace still parsed with the empty regr shell: no anchor, no crash
    assert record.trace.anchor_hits == 0


def test_parse_all_missing_sim_file_yields_empty_shell(tmp_path):
    case_dir = tmp_path / "case_4"
    _write_case(case_dir, trace_lines=[])
    (case_dir / "sim.log").unlink()  # sim.log missing -> parse_sim raises

    records = parse_all(str(_write_input_csv(tmp_path, 4, "case_4")))
    assert len(records) == 1
    record = records[0]

    assert record.sim == SimInfo.empty()
    assert record.sim.fatals == [] and record.sim.error_templates == {}
    assert record.family == "OTHER"
    assert any("parse_sim" in e for e in record.parse_errors)
