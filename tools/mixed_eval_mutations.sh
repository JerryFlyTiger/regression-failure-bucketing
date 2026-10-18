#!/bin/bash
# tools/mixed_eval.py 的 mutation 清單（對應 tests/test_mixed_eval.py），供冷讀重跑。
# 跑法：bash tools/mixed_eval_mutations.sh [名稱...]   不帶參數跑全部，帶參數只跑那幾條。
# 每條預期都是 KILLED；SURVIVED 代表測試出現缺口。每條只跑單一 node id（整檔約 250 秒，mut.sh 上限 300 秒）。
# mut.sh 會還原 tools/mixed_eval.py；跑完 out/ 不應留下 MUTATION_ACTIVE_*。
cd "$(dirname "$0")/.."
F=tools/mixed_eval.py; T=tests/test_mixed_eval.py
SEL=("$@")
m(){
  if [ ${#SEL[@]} -gt 0 ]; then
    local hit=0 n
    for n in "${SEL[@]}"; do [ "$n" = "$1" ] && hit=1; done
    [ $hit -eq 1 ] || return 0
  fi
  bash tools/mut.sh "$1" "$F" "$2" "$3" "$T::$4"
}
m m1 $'_check_bugs(rf_bugs, br_bugs)\n    base.prepare_out_dir(work)\n    work = os.path.abspath(work)\n    gen_rf' $'pass\n    base.prepare_out_dir(work)\n    work = os.path.abspath(work)\n    gen_rf' test_build_bench_checks_bug_names_itself
m m2 'if not os.path.exists(p):' 'if False:' test_build_bench_missing_path_is_input_error
m m3 'res["worse"] = find_worse(res["diag"], label_family)' 'res["worse"] = []' test_evaluate_worse_family_and_notes
m m4 'label_family = {c["label"]: c["family"] for c in cases.values()}' 'label_family = {c["label"]: "public" for c in cases.values()}' test_evaluate_worse_family_and_notes
m m5 'w["notes"] = _notes(w, res["cases"], res["diag"][w["bench"]])' 'w["notes"] = []' test_evaluate_worse_family_and_notes
m m5b $'    return notes\n' $'    return []\n' test_evaluate_worse_family_and_notes
m m29 $'    for w in res["worse"]:\n        w["notes"] = _notes(w, res["cases"], res["diag"][w["bench"]])' $'    for w in res["worse"]:\n        w["notes"] = []' test_evaluate_worse_family_and_notes
m m6 'tag(p["split"] if p else False)' 'tag(True if p else False)' test_render_split_not_in_public_only_is_not_marked
m m8 'known_exc=_known_exc(c))' 'known_exc=False)' test_evaluate_item_flags_and_dev_mismatch
m m9 'balanced_accuracy(g, bucket)[-1]' 'balanced_accuracy(g, bucket)[0]' test_evaluate_cluster_k_and_ba_per_bar
m m10 'cluster(build_signatures(records), k)' 'cluster(build_signatures(records), k + 1)' test_evaluate_cluster_k_and_ba_per_bar
m m12 'res["bars"][-1]["name"] += " UNSTABLE"' 'pass' test_evaluate_flags_unstable_when_partition_differs
m m14 'len(rows) > ID_SPAN' 'len(rows) >= ID_SPAN' test_id_span_exact_fit_passes
m m14b 'ID_SPAN = 199' 'ID_SPAN = 198' test_id_span_constant_matches_bases
m m26 'res["dev_dist"][lab].get(kind, 0) + 1' 'res["dev_dist"][lab].get(kind, 0) + 0' test_evaluate_dev_dist_counts
m m27 $'        if not ok:\n' $'        if ok:\n' test_evaluate_item_flags_and_dev_mismatch
m m28 'res["dev_mismatch"].append(item)' 'pass' test_evaluate_item_flags_and_dev_mismatch
m N1 'res["diag"][w["bench"]])' 'res["diag"]["all mixed"])' test_evaluate_notes_use_diag_of_the_worse_bench
m N8 'm["expected"].split(":")[0] == kind' 'm["expected"].startswith(kind)' test_dev_expect_ok_table
m N7 'if w.get("notes")' 'if w["notes"]' test_render_worse_line_without_notes_key_and_full_text
# 第四批（冷讀第三輪缺口）
m A1 'return c["family"] == "regfile" and (c["gen_meta"]' 'return (c["gen_meta"]' test_known_exc_table
m A2 'kind, mn = d.kind, d.mnemonic' 'kind, mn = d.mclass, d.mnemonic' test_evaluate_unpatched_item_fields_pinned
m A3 'f"{k}={v}"' 'f"{k}~{v}"' test_render_deviation_sections_verbatim
m A4 "got {m['kind']}:{m['mnemonic']}" "got {m['mnemonic']}:{m['kind']}" test_render_deviation_sections_verbatim
m A4b "(f\", expected {m['expected']}\" if \"expected\" in m else \"\")" '""' test_render_deviation_sections_verbatim
m A5 'bucket_mixed=b_all[str(c["case"])]' 'bucket_mixed=b_sh[str(c["case"])]' test_evaluate_item_bucket_mixed_comes_from_the_all_mixed_run
m A6 'item["expected"] = c["gen_meta"]["expected"]' 'pass' test_evaluate_unpatched_item_fields_pinned
m D1 'if unstable:' 'if True:' test_evaluate_rows_and_diag
m D2 'res["diag"][w["bench"]])' 'res["diag"][w["ref"]])' test_evaluate_worse_family_and_notes
m D3 'if w.get("notes")' 'if "notes" in w' test_render_worse_line_without_notes_key_and_full_text
echo DONE
