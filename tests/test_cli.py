"""rfb.cli：參數解析與 main() 轉呼叫 pipeline.run。"""

import runpy

import pytest

from rfb import cli, pipeline


def test_parser_parses_all_required_args():
    args = cli.build_parser().parse_args(["--input", "a.csv", "--output", "b.csv", "--k", "4"])
    assert args.input == "a.csv"
    assert args.output == "b.csv"
    assert args.k == 4 and isinstance(args.k, int)


@pytest.mark.parametrize("argv", [
    ["--output", "b", "--k", "1"],
    ["--input", "a", "--k", "1"],
    ["--input", "a", "--output", "b"],
    ["--input", "a", "--output", "b", "--k", "x"],
])
def test_parser_rejects_missing_or_bad_args(argv):
    with pytest.raises(SystemExit) as ei:
        cli.build_parser().parse_args(argv)
    assert ei.value.code == 2


def test_main_calls_pipeline_run_and_returns_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda i, o, k: calls.append((i, o, k)))
    rc = cli.main(["--input", "in.csv", "--output", "out.csv", "--k", "3"])
    assert rc == 0
    assert calls == [("in.csv", "out.csv", 3)]


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_entrypoint_exits_with_main_return(monkeypatch, tmp_path):
    # 覆蓋 cli.py:24-25（__main__ 區塊）；以 runpy 執行，run 換成 stub
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda i, o, k: calls.append((i, o, k)))
    monkeypatch.setattr("sys.argv", ["regr_fail_bucketing", "--input", "i", "--output", "o", "--k", "2"])
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("rfb.cli", run_name="__main__")
    assert ei.value.code == 0
    assert calls == [("i", "o", 2)]
