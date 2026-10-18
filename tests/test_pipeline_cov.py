"""rfb.pipeline 補測：deviation 後備、parse_all 表頭錯誤、run()。"""

import csv

import pytest

import rfb.diverge
from rfb.pipeline import _derive_deviation, _fallback_deviation, _parse_one_case, parse_all, run
from rfb.record import FailureRecord, RegrInfo, SimInfo, TraceInfo


def _record():
    return FailureRecord(case_id=1, family="OTHER", regr=RegrInfo.empty(),
                         sim=SimInfo.empty(), trace=TraceInfo.empty(),
                         deviation=None, parse_errors=[])


def _write_csv(tmp_path, header, rows):
    p = tmp_path / "input.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return p


def _write_case(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "regr.log").write_text("")
    (d / "sim.log").write_text("")
    (d / "trace.log").write_text("")


def test_fallback_deviation_is_none_shell():
    # 覆蓋 pipeline.py:36
    dev = _fallback_deviation()
    assert dev.kind == "NONE"
    assert dev.mnemonic is None and dev.mclass is None
    assert dev.detail == {}
    assert dev.source == "none"
    assert dev.confidence == 0.0


def test_derive_deviation_exception_recorded(monkeypatch):
    # 覆蓋 pipeline.py:47-48
    def boom(record):
        raise ValueError("kaboom")

    monkeypatch.setattr(rfb.diverge, "derive", boom)
    dev, errors = _derive_deviation(_record())
    assert dev.kind == "NONE" and dev.source == "none"
    assert errors == ["diverge.derive: kaboom"]


def test_derive_deviation_success_returns_derive_result(monkeypatch):
    sentinel = _fallback_deviation()
    monkeypatch.setattr(rfb.diverge, "derive", lambda r: sentinel)
    dev, errors = _derive_deviation(_record())
    assert dev is sentinel
    assert errors == []


def test_parse_one_case_records_derive_error(tmp_path, monkeypatch):
    # 覆蓋 pipeline.py（dev_errors 併入 record.parse_errors）
    _write_case(tmp_path, "c")

    def boom(record):
        raise RuntimeError("bad derive")

    monkeypatch.setattr(rfb.diverge, "derive", boom)
    rec = _parse_one_case(7, str(tmp_path / "c/regr.log"), str(tmp_path / "c/sim.log"),
                          str(tmp_path / "c/trace.log"))
    assert rec.case_id == 7
    assert rec.deviation.kind == "NONE"
    assert "diverge.derive: bad derive" in rec.parse_errors


def test_parse_all_missing_column_raises_keyerror(tmp_path):
    # 覆蓋 pipeline.py:117->115 與 119（表頭缺 Trace Log）
    _write_case(tmp_path, "case_1")
    csv_path = _write_csv(tmp_path, ["Case", "Regr Log", "Sim Log"],
                          [[1, "case_1/regr.log", "case_1/sim.log"]])
    with pytest.raises(KeyError) as ei:
        parse_all(str(csv_path))
    assert "trace log" in str(ei.value)


def test_parse_all_empty_csv_returns_empty_list(tmp_path):
    # 覆蓋 pipeline.py:131-132
    csv_path = _write_csv(tmp_path, ["Case", "Regr Log", "Sim Log", "Trace Log"], [])
    assert parse_all(str(csv_path)) == []


def test_parse_all_header_is_case_and_whitespace_insensitive(tmp_path):
    _write_case(tmp_path, "case_5")
    csv_path = _write_csv(tmp_path, [" CASE ", "regr LOG", "Sim log ", "TRACE log"],
                          [[5, "case_5/regr.log", "case_5/sim.log", "case_5/trace.log"]])
    recs = parse_all(str(csv_path))
    assert [r.case_id for r in recs] == [5]


def test_run_writes_sorted_output_and_creates_out_dir(tmp_path):
    # 覆蓋 run()；輸入順序 3,1,2，輸出需依 case 遞增，且會建立巢狀輸出目錄
    rows = []
    for cid in (3, 1, 2):
        _write_case(tmp_path, f"case_{cid}")
        rows.append([cid, f"case_{cid}/regr.log", f"case_{cid}/sim.log", f"case_{cid}/trace.log"])
    csv_path = _write_csv(tmp_path, ["Case", "Regr Log", "Sim Log", "Trace Log"], rows)
    out = tmp_path / "nested" / "deeper" / "output.csv"
    run(str(csv_path), str(out), 1)
    with open(out, newline="") as f:
        lines = list(csv.reader(f))
    assert lines[0] == ["Case", "bucket"]
    assert [r[0] for r in lines[1:]] == ["1", "2", "3"]
    # k=1：全部同一桶
    assert len({r[1] for r in lines[1:]}) == 1


def test_run_sorts_case_ids_numerically_and_labels_by_min_case_id(tmp_path):
    # 輸入順序 10,30,9,11；10、11 的 sim.log 有 UVM_FATAL（同 hard_key 為一桶），
    # 9、30 無。k=2。min(9,30)=9 < min(10,11)=10，所以 {9,30} 是 bucket_1。
    # FATAL 桶給 10、11，使原子索引序（依 hard_key 排序）與 min 序相反，才殺得掉 order=range。
    # 殺變異：排序 key 改 str(case_id)；label 寫死 "bucket_1"；
    # cluster.py 的 key=min(bucket_ids[i]) 改 max（30>11 會對調）或 str(min)（"10"<"9" 會對調）。
    fatal = "UVM_FATAL /w/a.sv(1) @ 10: reporter [X] boom\n"
    rows = []
    for cid in (10, 30, 9, 11):
        _write_case(tmp_path, f"case_{cid}")
        if cid in (10, 11):
            (tmp_path / f"case_{cid}" / "sim.log").write_text(fatal)
        rows.append([cid, f"case_{cid}/regr.log", f"case_{cid}/sim.log", f"case_{cid}/trace.log"])
    csv_path = _write_csv(tmp_path, ["Case", "Regr Log", "Sim Log", "Trace Log"], rows)
    out = tmp_path / "o" / "output.csv"
    run(str(csv_path), str(out), 2)
    with open(out, newline="") as f:
        got = list(csv.reader(f))
    assert got == [["Case", "bucket"], ["9", "bucket_1"], ["10", "bucket_2"],
                   ["11", "bucket_2"], ["30", "bucket_1"]]
