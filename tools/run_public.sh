#!/usr/bin/env bash
# Run the bucketing program against both public benchmark sets, in the
# original case order and in a fixed-seed shuffled order, and print a
# balanced-accuracy score table.
#
# Usage: run_public.sh [<program path>]
#   <program path> defaults to bin/regr_fail_bucketing, run with
#   .venv/bin/python. If a program path is given, it is executed directly
#   (it is assumed to be an external executable with its own shebang).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
DEFAULT_PROGRAM="$REPO_ROOT/bin/regr_fail_bucketing"
SHUFFLE_SEED=20260928

PROGRAM="${1:-}"
if [ -n "$PROGRAM" ]; then
    PROGRAM="$(cd "$(dirname "$PROGRAM")" && pwd)/$(basename "$PROGRAM")"
fi
if [ -z "$PROGRAM" ]; then
    RUN_CMD=("$PY" "$DEFAULT_PROGRAM")
elif head -1 "$PROGRAM" 2>/dev/null | grep -qi python; then
    # a python script without a working shebang venv: run explicitly under
    # our venv interpreter instead of relying on `env python3`.
    RUN_CMD=("$PY" "$PROGRAM")
else
    RUN_CMD=("$PROGRAM")
fi

OUT_ROOT="$REPO_ROOT/out"
mkdir -p "$OUT_ROOT"

make_shuffled_input() {
    # $1 = original input.csv, $2 = destination directory
    local src="$1"
    local dest_dir="$2"
    mkdir -p "$dest_dir"
    "$PY" - "$src" "$dest_dir/input.csv" "$SHUFFLE_SEED" << 'PYEOF'
import csv
import os
import random
import sys

src, dest, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
base_dir = os.path.dirname(os.path.abspath(src))

with open(src, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

rng = random.Random(seed)
rng.shuffle(rows)

# resolve the three log path columns (indices 1, 2, 3) to absolute paths
for row in rows:
    for i in (1, 2, 3):
        row[i] = os.path.join(base_dir, row[i])

with open(dest, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
PYEOF
}

declare -a RESULTS=()
FAILED=0

run_one() {
    local set_name="$1"
    local order="$2"
    local input_csv="$3"
    local k="$4"

    local out_dir="$OUT_ROOT/${set_name}_${order}"
    mkdir -p "$out_dir"
    local output_csv="$out_dir/output.csv"

    # Run with cwd set to the directory containing input.csv: the log
    # paths inside input.csv are relative to that directory (see
    # docs/ARCHITECTURE.md section 1), and the reference sample programs
    # resolve them relative to the process cwd rather than the CSV path.
    local input_dir
    input_dir="$(dirname "$input_csv")"

    local start end wall ba
    start=$(date +%s.%N)
    (cd "$input_dir" && "${RUN_CMD[@]}" --input "$input_csv" --output "$output_csv" --k "$k") \
        > "$out_dir/run.log" 2>&1
    local rc=$?
    end=$(date +%s.%N)
    wall=$("$PY" -c "print(f'{$end - $start:.3f}')")

    if [ $rc -ne 0 ]; then
        ba="ERR"
    else
        local golden_csv
        golden_csv="$(dirname "$input_csv")/golden.csv"
        # shuffled input dirs don't have golden.csv alongside; use the
        # original benchmark's golden.csv instead.
        if [ ! -f "$golden_csv" ]; then
            golden_csv="$REPO_ROOT/testcase/2026_08_20_samples/problem/${set_name}/golden.csv"
        fi
        local score_out
        score_out=$("$PY" "$REPO_ROOT/tools/score.py" "$golden_csv" "$output_csv" 2> "$out_dir/score.err")
        if [ $? -ne 0 ]; then
            ba="ERR"
        else
            ba=$(echo "$score_out" | grep '^BA=' | cut -d= -f2)
        fi
    fi

    RESULTS+=("$set_name|$order|$ba|$wall")
}

for set_info in "benchmark_set_1:2" "benchmark_set_2:4"; do
    set_name="${set_info%%:*}"
    k="${set_info##*:}"
    orig_dir="$REPO_ROOT/testcase/2026_08_20_samples/problem/$set_name"
    orig_input="$orig_dir/input.csv"

    run_one "$set_name" "orig" "$orig_input" "$k"

    shuffled_dir="$OUT_ROOT/${set_name}_shuffled"
    make_shuffled_input "$orig_input" "$shuffled_dir"
    run_one "$set_name" "shuffled" "$shuffled_dir/input.csv" "$k"
done

echo "set|order|BA|wall(s)"
for r in "${RESULTS[@]}"; do
    echo "$r"
done

for set_info in "benchmark_set_1:2" "benchmark_set_2:4"; do
    set_name="${set_info%%:*}"
    ba_orig=""
    ba_shuffled=""
    for r in "${RESULTS[@]}"; do
        IFS='|' read -r rset rorder rba rwall <<< "$r"
        if [ "$rset" = "$set_name" ] && [ "$rorder" = "orig" ]; then
            ba_orig="$rba"
        elif [ "$rset" = "$set_name" ] && [ "$rorder" = "shuffled" ]; then
            ba_shuffled="$rba"
        fi
    done
    if [ "$ba_orig" != "$ba_shuffled" ]; then
        echo "UNSTABLE $set_name"
    fi
done
