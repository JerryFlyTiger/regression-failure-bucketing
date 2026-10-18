#!/usr/bin/env python3
"""Pairwise balanced accuracy scorer.

Usage: score.py <golden.csv> <output.csv>

Golden and output CSV column names are matched case-insensitively; the
golden file uses "Case"/"Bug", the output file uses "Case"/"bucket".
"""

import csv
import itertools
import sys


def _read_case_label_csv(path: str, label_names: tuple[str, ...]) -> dict[str, str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise KeyError(f"empty CSV (no header row): {path}")
        header_map = {h.strip().lower(): h for h in reader.fieldnames}
        case_key = header_map.get("case")
        label_key = None
        for name in label_names:
            if name in header_map:
                label_key = header_map[name]
                break
        if case_key is None or label_key is None:
            raise KeyError(
                f"expected columns 'case' and one of {label_names} in {reader.fieldnames}"
            )
        result = {}
        for row in reader:
            case = str(row[case_key]).strip()
            label = str(row[label_key]).strip()
            if case in result:
                raise ValueError(f"duplicate case id {case!r} in {path}")
            result[case] = label
        return result


def balanced_accuracy(
    golden: dict[str, str], output: dict[str, str]
) -> tuple[int, int, int, int, float, float, float]:
    """Compute pairwise TP/FP/TN/FN/TPR/TNR/BA.

    golden: case -> bug label (ground truth grouping).
    output: case -> bucket label (predicted grouping). Cases present in
    golden but missing from output are treated as being in their own
    singleton bucket (a warning is expected to be printed by the caller).
    """
    cases = list(golden.keys())
    n = len(cases)
    if n < 2:
        return 0, 0, 0, 0, 1.0, 1.0, 1.0

    tp = fp = tn = fn = 0
    for a, b in itertools.combinations(cases, 2):
        same_golden = golden[a] == golden[b]
        # missing cases: unique placeholder per case -> never equal to any
        # other case's bucket (including another missing case's).
        oa = output.get(a, f"__missing__:{a}")
        ob = output.get(b, f"__missing__:{b}")
        same_output = oa == ob
        if same_golden and same_output:
            tp += 1
        elif same_golden and not same_output:
            fn += 1
        elif not same_golden and same_output:
            fp += 1
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    ba = (tpr + tnr) / 2
    return tp, fp, tn, fn, tpr, tnr, ba


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: score.py <golden.csv> <output.csv>", file=sys.stderr)
        return 1

    golden_path, output_path = argv
    try:
        golden = _read_case_label_csv(golden_path, ("bug",))
        output = _read_case_label_csv(output_path, ("bucket",))
    except (ValueError, KeyError, OSError) as e:
        print(f"score.py: {e}", file=sys.stderr)
        return 1

    missing = [c for c in golden if c not in output]
    if missing:
        print(
            f"warning: {len(missing)} case(s) missing from output, "
            f"treated as singleton buckets: {missing}",
            file=sys.stderr,
        )

    tp, fp, tn, fn, tpr, tnr, ba = balanced_accuracy(golden, output)

    print("TP\tFP\tTN\tFN\tTPR\tTNR\tBA")
    print(f"{tp}\t{fp}\t{tn}\t{fn}\t{tpr:.6f}\t{tnr:.6f}\t{ba:.6f}")
    print(f"BA={ba:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
