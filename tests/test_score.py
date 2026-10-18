import pytest

from tools.score import _read_case_label_csv, balanced_accuracy, main


def test_golden_matches_itself():
    golden = {"1": "a", "2": "a", "3": "b", "4": "b", "5": "c"}
    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, golden)
    assert ba == 1.0
    assert fp == 0
    assert fn == 0


def test_all_same_bucket_gives_half():
    golden = {"1": "a", "2": "a", "3": "b", "4": "b"}
    output = {"1": "x", "2": "x", "3": "x", "4": "x"}
    _, _, _, _, _, _, ba = balanced_accuracy(golden, output)
    assert ba == 0.5


def test_all_different_buckets_gives_half():
    golden = {"1": "a", "2": "a", "3": "b", "4": "b"}
    output = {"1": "w", "2": "x", "3": "y", "4": "z"}
    _, _, _, _, _, _, ba = balanced_accuracy(golden, output)
    assert ba == 0.5


def test_hand_calculated_example_ba_half():
    # golden {1:a,2:a,3:b,4:b}, output {1:x,2:x,3:x,4:y}
    # pairs: (1,2) same-golden same-output -> TP
    #        (1,3) diff-golden same-output -> FP
    #        (1,4) diff-golden diff-output -> TN
    #        (2,3) diff-golden same-output -> FP
    #        (2,4) diff-golden diff-output -> TN
    #        (3,4) same-golden diff-output -> FN
    # TP=1 FP=2 TN=2 FN=1 -> TPR=1/2=0.5 TNR=2/4=0.5 -> BA=0.5
    golden = {"1": "a", "2": "a", "3": "b", "4": "b"}
    output = {"1": "x", "2": "x", "3": "x", "4": "y"}
    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, output)
    assert (tp, fp, tn, fn) == (1, 2, 2, 1)
    assert tpr == 0.5
    assert tnr == 0.5
    assert ba == 0.5


def test_hand_calculated_example_ba_not_half():
    # golden {1:a,2:a,3:a,4:a,5:b}, output {1:x,2:x,3:x,4:y,5:y}
    # same-golden pairs: (1,2)(1,3)(1,4)(2,3)(2,4)(3,4) = 6
    #   (1,2) same-output -> TP
    #   (1,3) same-output -> TP
    #   (1,4) diff-output(x,y) -> FN
    #   (2,3) same-output -> TP
    #   (2,4) diff-output -> FN
    #   (3,4) diff-output -> FN
    # diff-golden pairs (with 5): (1,5)(2,5)(3,5)(4,5) = 4
    #   (1,5) diff-output(x,y) -> TN
    #   (2,5) diff-output -> TN
    #   (3,5) diff-output -> TN
    #   (4,5) same-output(y,y) -> FP
    # TP=3 FN=3 FP=1 TN=3 -> TPR=3/6=0.5 TNR=3/4=0.75 -> BA=0.625
    golden = {"1": "a", "2": "a", "3": "a", "4": "a", "5": "b"}
    output = {"1": "x", "2": "x", "3": "x", "4": "y", "5": "y"}
    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, output)
    assert (tp, fp, tn, fn) == (3, 1, 3, 3)
    assert tpr == 0.5
    assert tnr == 0.75
    assert ba == 0.625


def test_n_less_than_2_gives_one():
    assert balanced_accuracy({}, {}) == (0, 0, 0, 0, 1.0, 1.0, 1.0)
    assert balanced_accuracy({"1": "a"}, {"1": "a"}) == (0, 0, 0, 0, 1.0, 1.0, 1.0)


def test_missing_case_in_output_becomes_singleton():
    golden = {"1": "a", "2": "a", "3": "b"}
    output = {"1": "x", "2": "x"}  # case "3" missing from output
    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, output)
    # (1,2): same-golden, same-output -> TP
    # (1,3): diff-golden, diff-output (3 is a unique singleton) -> TN
    # (2,3): diff-golden, diff-output -> TN
    assert (tp, fp, tn, fn) == (1, 0, 2, 0)
    assert ba == 1.0


def test_read_case_label_csv_header_case_insensitive(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("CASE,Bucket\n1,a\n2,b\n")
    result = _read_case_label_csv(str(path), ("bucket",))
    assert result == {"1": "a", "2": "b"}


def test_read_case_label_csv_duplicate_case_raises(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Case,Bucket\n1,a\n1,b\n")
    with pytest.raises(ValueError):
        _read_case_label_csv(str(path), ("bucket",))


def test_main_prints_ba_line(tmp_path, capsys):
    golden_path = tmp_path / "golden.csv"
    output_path = tmp_path / "output.csv"
    golden_path.write_text("Case,Bug\n1,a\n2,a\n3,b\n4,b\n")
    output_path.write_text("Case,bucket\n1,x\n2,x\n3,y\n4,y\n")

    rc = main([str(golden_path), str(output_path)])
    out = capsys.readouterr().out
    assert rc == 0
    last_line = out.strip().splitlines()[-1]
    assert last_line == "BA=1.000000"


def test_main_missing_file_returns_1_with_clean_message(tmp_path, capsys):
    output_path = tmp_path / "output.csv"
    output_path.write_text("Case,bucket\n1,x\n")

    rc = main([str(tmp_path / "nonexistent.csv"), str(output_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("score.py:")


def test_main_empty_csv_returns_1_with_clean_message(tmp_path, capsys):
    golden_path = tmp_path / "golden.csv"
    output_path = tmp_path / "output.csv"
    golden_path.write_text("")  # zero bytes: no header row at all
    output_path.write_text("Case,bucket\n1,x\n")

    rc = main([str(golden_path), str(output_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("score.py:")
    assert "no header row" in err
