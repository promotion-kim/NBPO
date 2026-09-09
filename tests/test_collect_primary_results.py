"""Synthetic CPU fixtures only: no generated fixture number is a paper result."""
import copy
import json
import sys

import numpy as np
import pytest

from scripts.experiments.nbpo_repair_20260909 import collect_primary_results as c
from scripts.experiments.nbpo_repair_20260909 import evaluate_responses as ev


def examples():
    return [{"uid":"a","status":"ok","correct":1,"parse_failure":0},
            {"uid":"b","status":"ok","correct":0,"parse_failure":1},
            {"uid":"c","status":"ok","correct":0,"parse_failure":0}]


def test_independent_scalar_and_micro_instruction_bootstrap_match_reference():
    rows=examples()
    saved=ev.metric_summary(rows,"correct")
    assert c.verify_metric(saved,rows,"correct")["estimate"] == 1/3
    weighted=[{"uid":"a","status":"ok","n":3,"correct":2},
              {"uid":"b","status":"ok","n":1,"correct":1}]
    saved=ev.metric_summary(weighted,"correct","n")
    assert c.verify_metric(saved,weighted,"correct","n")["estimate"] == .75


@pytest.mark.parametrize("field,value", [("estimate",.9),("n_failed",2),("n_scored",1),("complete",False),
    ("n_prompts",1),("denominator",1),("bootstrap_seed",123),("ci95",[.7,.8])])
def test_summary_tampering_does_not_pass_reaggregation(field,value):
    rows=examples(); saved=ev.metric_summary(rows,"correct"); saved[field]=value
    with pytest.raises(c.Invalid):
        c.verify_metric(saved,rows,"correct")


def test_parse_failure_remains_in_em_denominator_not_missing():
    rows=examples(); summary=ev.metric_summary(rows,"correct")
    assert c.verify_metric(summary,rows,"correct")["n_scored"] == 3
    excluded=copy.deepcopy(rows); excluded[1]["status"]="missing_response"
    with pytest.raises(c.Invalid):
        c.verify_metric(summary,excluded,"correct")


def test_paired_bootstrap_joins_uid_and_preserves_direction_and_missing_pairs():
    base=examples(); candidate=copy.deepcopy(base)[::-1]
    candidate[0]["correct"]=1
    candidate[1]["status"]="evaluator_failure"
    summary=ev.paired_delta(candidate,base,"correct")
    got=c.verify_metric(summary,candidate,"correct",paired_base=base)
    assert got["estimate"] == .5 and got["n_paired"] == 2 and got["n_unpaired"] == 1
    candidate.append(candidate[0])
    with pytest.raises(c.Invalid,match="Duplicate"):
        c.verify_metric(summary,candidate,"correct",paired_base=base)


def test_vector_ci_matches_frozen_paired_prompt_bootstrap():
    from scripts.experiments.nbpo_repair_20260909.evaluate_saferlhf import paired_prompt_bootstrap
    values=np.array([[.1,-.2],[.3,.2],[-.1,.15]])
    disagreement=np.array([[.05,-.1],[.15,.1],[-.05,.1]])
    saved=paired_prompt_bootstrap(values,disagreement)
    for key,array in (("V",values),("d_original_reference",disagreement),("s_original_reference",values-disagreement)):
        got=c.bootstrap(array)
        c.close(got["estimate"],saved["metrics"][key]["estimate"],key)
        c.close(got["ci95"],saved["metrics"][key]["ci95"],key)


def test_hash_ledger_rejects_modified_and_mutating_sources(tmp_path):
    target=tmp_path/"input.json"; target.write_text('{"x":1}')
    collector=c.Collector(tmp_path); assert collector.json("input.json") == {"x":1}
    assert str(target) in collector.ledger
    target.write_text('{"x":2}')
    with pytest.raises(c.Invalid,match="changed during"):
        collector.json("input.json")


def test_jsonl_preserves_unicode_line_separators_inside_response_strings(tmp_path):
    rows=[{"uid":"a","output":"before\u2028inside\u2029after"},{"uid":"b","output":"normal"}]
    path=tmp_path/"responses.jsonl"
    path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
    assert c.Collector(tmp_path).jsonl(path) == rows


def test_missing_is_separate_from_invalid_and_never_numeric_zero(tmp_path):
    collector=c.Collector(tmp_path)
    assert collector.attempt("missing",lambda:collector.json("absent.json")) is None
    assert collector.missing and not collector.errors
    assert collector.attempt("bad",lambda:c.require(False,"tampered metric")) is None
    assert collector.errors
    assert c.point(None) == "not available"
    with pytest.raises(c.Invalid):
        c.close(None,0,"missing is not zero")


def test_full_raw_loader_checks_real_manifest_prompt_model_and_token_bindings(tmp_path,monkeypatch):
    data=tmp_path/"data"; data.mkdir()
    c.write_json(data/"unused.json",{})
    (data/"gsm8k_test.jsonl").write_text(json.dumps({"id":1,"question":"q","answer":"#### 2"})+"\n")
    from scripts.experiments.nbpo_repair_20260909.generate_eval import items_for
    planned=items_for("gsm8k",tmp_path)
    folder=tmp_path/"responses/base"; folder.mkdir(parents=True)
    settings={"model_weights":{"model.safetensors":"a"},"terminal_ids":[2],"max_new_tokens":3}
    c.write_json(folder/"settings.json",settings)
    uid,prompt,_=planned[0]
    row={"uid":uid,"prompt":prompt,"prompt_sha256":c.digest(prompt),"model_sha256":c.object_hash(settings["model_weights"]),
        "output":"The answer is 2.","output_token_ids":[7,2],"n_output_tokens":2,"finish_reason":"stop"}
    path=folder/"gsm8k.jsonl"; path.write_text(json.dumps(row)+"\n")
    manifest={"settings_sha256":c.file_hash(folder/"settings.json"),"response_file_sha256":c.file_hash(path),
        "prompt_sha256":c.object_hash([(uid,prompt)]),"n_planned":1,"n_generated":1,"n_failed":0}
    c.write_json(folder/"gsm8k.manifest.json",manifest)
    monkeypatch.setitem(c.COUNTS,"gsm8k",1)
    collector=c.Collector(tmp_path); collector.load_raw("base","gsm8k")
    assert collector.provenance["base","gsm8k"]["n_generated"] == 1
    row["prompt_sha256"]="changed"; path.write_text(json.dumps(row)+"\n")
    manifest["response_file_sha256"]=c.file_hash(path)
    (folder/"gsm8k.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(c.Invalid,match="prompt binding"):
        c.Collector(tmp_path).load_raw("base","gsm8k")


def test_original_reference_values_recomputed_from_reference_probabilities(tmp_path):
    folder=tmp_path/"scores/shard0"; folder.mkdir(parents=True)
    ref=np.full((3,2,2,8,8),.5); ref[:,0,0]=.75; ref[:,1,1]=.25
    np.savez(folder/"probabilities.npz",reference=ref)
    manifest={"prompt_ids":["p0","p1"],"probabilities_sha256":c.file_hash(folder/"probabilities.npz"),
        "objectives":["helpfulness","harmlessness"],"seeds":[41,42,43]}
    c.write_json(folder/"manifest.json",manifest)
    fixed={"source_artifacts":[{"shard":0,"score_manifest_sha256":c.file_hash(folder/"manifest.json"),
        "probabilities_sha256":manifest["probabilities_sha256"]}],"comparator_occurrences_sha256":"test",
        "reference_ensemble_probabilities_sha256":"test","prompt_order_sha256":"test","teacher":{}}
    fixed["sha256"]=c.object_hash(fixed)
    fixed.update(beta=[.25,.25],comparator_occurrence_mass=[.125]*8)
    result=c.Collector(tmp_path).original_disagreement(fixed)
    np.testing.assert_allclose(result["p0"],[.25,0],atol=1e-15)
    np.testing.assert_allclose(result["p1"],[0,-.25],atol=1e-15)


def test_full_saferlhf1500_schema_reaggregates_arrays_vectors_and_split_cis(tmp_path):
    from scripts.experiments.nbpo_repair_20260909.evaluate_saferlhf import summarize_label
    n=1500
    prompts=[{"prompt_id":str(i),"split":"dev" if i<500 else "test"} for i in range(n)]
    raw=[{"uid":"saferlhf:"+p["prompt_id"],"output":"hello","n_output_tokens":1,"finish_reason":"stop"} for p in prompts]
    probabilities=np.full((3,2,n,8),.55)
    d=np.zeros((n,2))
    records,reports=summarize_label("base",prompts,raw,probabilities,d,probabilities)
    score=tmp_path/"scores/shard0"; score.mkdir(parents=True)
    np.savez(score/"probabilities.npz",reference=np.full((3,2,n,8,8),.5))
    manifest={"prompt_ids":[p["prompt_id"] for p in prompts],"probabilities_sha256":c.file_hash(score/"probabilities.npz"),
        "objectives":["helpfulness","harmlessness"],"seeds":[41,42,43]}
    c.write_json(score/"manifest.json",manifest)
    fixed={"source_artifacts":[{"shard":0,"score_manifest_sha256":c.file_hash(score/"manifest.json"),
        "probabilities_sha256":manifest["probabilities_sha256"]}],"comparator_occurrences_sha256":"fixture",
        "reference_ensemble_probabilities_sha256":"fixture","prompt_order_sha256":"fixture","teacher":{}}
    fixed["sha256"]=c.object_hash(fixed)
    fixed.update(beta=[.25,.25],comparator_occurrence_mass=[.125]*8)
    directory=tmp_path/"evaluations/saferlhf_primary_v1"; folder=directory/"base"; folder.mkdir(parents=True)
    c.write_json(directory/"fixed_game.json",fixed)
    (directory/"teacher_truncation.jsonl").write_text("")
    (folder/"per_prompt.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n")
    np.savez(folder/"probabilities.npz",probabilities=probabilities)
    provenance={"fixture":"synthetic-test-only"}
    c.write_json(folder/"scoring_manifest.json",{"prompt_ids":[p["prompt_id"] for p in prompts],
        "fixed_game_sha256":fixed["sha256"],"response_provenance":provenance,
        "probabilities_sha256":c.file_hash(folder/"probabilities.npz"),
        "objectives":["helpfulness","harmlessness"],"seeds":[41,42,43]})
    summary={"splits":reports,"label":"base","objectives":["helpfulness","harmlessness"],
        "response_provenance":provenance,"fixed_game_sha256":fixed["sha256"],
        "teacher_truncation":{"n_evaluated":0,"prompt_truncated_fraction":None,"response_truncated_fraction":None},
        "response_length":{"mean_tokens":1.,"median_tokens":1.,"max_length_hits":0},
        "reference_interpretation":"fixture","scope":"fixture","test_exposure":"fixture"}
    c.write_json(folder/"report.json",summary)
    collector=c.Collector(tmp_path)
    collector.raw["base","saferlhf"]={r["uid"]:r for r in raw}
    collector.expected["saferlhf"]={r["uid"]:("prompt",{"split":p["split"]}) for r,p in zip(raw,prompts)}
    collector.provenance["base","saferlhf"]=provenance
    result=collector.saferlhf("base")
    assert result["independently_recomputed_probability_values"]
    assert result["splits"]["dev"]["n_scored"]==500
    assert result["splits"]["test"]["n_paired_vs_base"]==1000
    assert not collector.missing


def incomplete_report():
    return {"status":"INCOMPLETE","data_complete":False,"benchmarks":{b:{l:None for l in c.LABELS} for b in c.COUNTS},
        "diagnostics_only":{},"teacher_rm":None,"missing":[{"section":"primary","source":"pending","reason":"absent"}],"errors":[]}


def test_incomplete_blocks_tex_and_markdown_is_explicit():
    report=incomplete_report()
    with pytest.raises(c.Invalid,match="Incomplete"):
        c.paper_fragments(report)
    rendered=c.markdown(report)
    assert "INCOMPLETE" in rendered and "not available" in rendered and "NOT official" in rendered


@pytest.mark.parametrize("allow,exit_code",[(False,2),(True,0)])
def test_cli_missing_required_sources_fails_default_and_never_writes_tex(tmp_path,monkeypatch,allow,exit_code):
    out=tmp_path/"report"
    monkeypatch.setattr(c.Collector,"collect",lambda self:incomplete_report())
    args=["collect","--root",str(tmp_path),"--output",str(out)]
    if allow: args.append("--allow-incomplete")
    else: args.append("--paper-fragments")
    monkeypatch.setattr(sys,"argv",args)
    with pytest.raises(SystemExit) as exc:
        c.main()
    assert exc.value.code == exit_code
    assert not list(out.glob("*.tex"))
    assert json.loads((out/"primary_results.json").read_text())["status"] == "INCOMPLETE"


def test_allow_incomplete_flag_cannot_authorize_paper_output(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,"argv",["collect","--root",str(tmp_path),"--output",str(tmp_path/"r"),
        "--allow-incomplete","--paper-fragments"])
    with pytest.raises(c.Invalid,match="never writes"):
        c.main()


def test_complete_synthetic_report_renders_three_rows_two_panels_and_appendix():
    report=incomplete_report(); report.update(status="COMPLETE",data_complete=True,missing=[])
    metric=ev.metric_summary(examples(),"correct")
    diag=c.diagnostics([{"output":"fixture","n_output_tokens":1,"finish_reason":"stop"}])
    for label in c.LABELS:
        for bench in ("ifeval","gsm8k","harmbench"):
            report["benchmarks"][bench][label]={"metrics":{name:copy.deepcopy(metric) for name in c.DEFINITIONS[bench]}}
        for bench,names in (("alpaca_eval",["all805"]),("arena_hard",["hard500","creative250"])):
            report["benchmarks"][bench][label]={"subsets":{name:{**metric,"candidate_diagnostics":diag} for name in names}}
        report["benchmarks"]["saferlhf"][label]={"splits":{}}
        report["benchmarks"]["xstest"][label]={"status":"UNADJUDICATED"}
    report["teacher_rm"]={"metrics":{"fixture_only":{**metric,"n_prompts":3}}}
    main,appendix=c.paper_fragments(report)
    assert "NBPO--WBC" in main and "NBPO--MSE" in main and "Base" in main
    assert "A. Accuracy" in main and "B. Local scalar-RM" in main
    assert "UNADJUDICATED" not in main and "unadjudicated" in appendix
    assert "Keywords" in appendix and "Failed" in appendix
    assert main.count(r"\begin{tabular}")==main.count(r"\end{tabular}") == 2
    assert "not official AlpacaEval" in main
