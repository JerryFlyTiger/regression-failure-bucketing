#!/bin/bash
# usage: mut.sh <name> <file> <python-replace-expr old> <new> <pytest target>
# exit code: 0 = 有判決 (SURVIVED/KILLED), 2 = ABORTED (old 字串找不到), 3 = TIMEOUT/ERROR (無判決)
set -u
mkdir -p out
name="$1"; f="$2"; old="$3"; new="$4"; target="$5"
cp "$f" "out/mut_backup_$(basename $f)"
touch "out/MUTATION_ACTIVE_$name"
trap 'cp "out/mut_backup_$(basename $f)" "$f"; rm -f "out/MUTATION_ACTIVE_$name"' EXIT INT TERM
python3 - "$f" "$old" "$new" <<'PY'
import sys; f,old,new=sys.argv[1:4]; s=open(f).read(); assert old in s, "old not found"; open(f,'w').write(s.replace(old,new,1))
PY
if [ $? -ne 0 ]; then echo "$name: ABORTED (old string not found, file untouched)"; exit 2; fi
PYTHONDONTWRITEBYTECODE=1 timeout 300 .venv/bin/python -B -m pytest -q -x -p no:cacheprovider "$target" > "out/mut_$name.log" 2>&1
rc=$?
# 只有 rc=1（有測試失敗）算 KILLED；2–5 是 pytest 中斷、內部錯誤、用法錯誤、沒收集到測試，124 是 timeout，都不是判決
case $rc in
  0) echo "$name: SURVIVED (tests green)" ;;
  1) echo "$name: KILLED (tests red, rc=1)" ;;
  124) echo "$name: TIMEOUT (300s, no verdict; see out/mut_$name.log)"; exit 3 ;;
  *) echo "$name: ERROR (pytest rc=$rc, no verdict; see out/mut_$name.log)"; exit 3 ;;
esac
