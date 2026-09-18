from pathlib import Path
import json,hashlib,html,datetime,platform,zipfile,collections
import numpy,scipy,torch,pytest
O=Path('/mnt/data/nbpo_reaudit_f0efbb5');R=Path('/mnt/data/_reaudit_work/current/NBPO-main');S='analysis/sub_20260914/code_snapshot_20260917_union/'
evidence=[
 ('E01',S+'nbpo_local_gate.py',[(1,137)],'New gate: contract, imports, CLI defaults'),
 ('E02',S+'nbpo_local_gate.py',[(138,250)],'Occurrence reweighting and one-representation-per-prompt evaluation'),
 ('E03',S+'nbpo_local_gate.py',[(252,334)],'Fit predicate, decision, fingerprint/promotion and policy result'),
 ('E04',S+'solve_pros4_targets_uw1c.py',[(212,303)],'Score schema, alias equality and over-restrictive A_LL equality rejection'),
 ('E05',S+'solve_pros4_targets_uw1c.py',[(93,209)],'Panel label, content digest, row digest verification, serializer'),
 ('E06',S+'solve_pros4_pw_nbpo_uw1c.py',[(63,88),(132,147),(198,243),(254,286)],'Actual target-worker and manifests; pool digest check placement'),
 ('E07',S+'README.md',[(109,151)],'New README assertions and explicit historical ungated status'),
 ('E08',S+'make_pros_train_jobs.py',[(126,190)],'Campaign training job construction'),
 ('E09',S+'panel_stage2.py',[(309,375)],'Panel training stage construction'),
 ('E10','mnpo_scripts/run_mnpo.py',[(300,410)],'Training and checkpoint saving'),
 ('E11','analysis/uf4_20260910/code_snapshot_20260917/diag_neural_pools.py',[(1,73)],'The actual helper dependency, pool tokens and log-probabilities'),
 ('E12',S+'nbpo_local_gate_selftest.py',[(1,101)],'Hard-coded pod selftest; fit check omitted and textual pass/fail reporting'),
 ('E13',S+'union_score_panel.py',[(140,278)],'Correct LL/LR/RR scorer retained'),
 ('E14',S+'build_panel_solvers.py',[(30,82)],'Generator panel-name propagation'),
 ('E15','scripts/nbpo/eval_game_value.py',[(1,100)],'Global evaluator intentionally retained for the Global Nash control'),
]
manifest=[];md=['# f0efbb5 original-source evidence\n'];parts=[]
for eid,path,ranges,title in evidence:
 p=R/path;txt=p.read_text();lines=txt.splitlines();sha=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append(dict(id=eid,path=path,sha256=sha,ranges=ranges))
 md+=['\n## '+eid+' — '+title+'\n','Path: `'+path+'`\nSHA256: `'+sha+'`\n']
 content=[]
 for a,b in ranges:
  block='\n'.join(f'{i:5d}  {lines[i-1]}' for i in range(a,min(b,len(lines))+1))
  md+=['```text\n'+block+'\n```\n'];content.append('<pre>'+html.escape(block)+'</pre>')
 parts.append(f'<section id="{eid}"><h2>{eid} — {html.escape(title)}</h2><p><code>{html.escape(path)}</code><br>SHA256: <code>{sha}</code></p>'+''.join(content)+'</section>')
(O/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));(O/'code_evidence.md').write_text('\n'.join(md))
nav=' | '.join(f'<a href="#{x[0]}">{x[0]}</a>' for x in evidence)
(O/'code_evidence.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>NBPO f0efbb5 source evidence</title><style>body{font:16px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 24px;color:#18222b}pre{background:#f3f5f7;padding:15px;overflow:auto;font:12px/1.55 ui-monospace,monospace}section{margin:50px 0}code{overflow-wrap:anywhere}h2{font-size:22px}nav{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ddd}</style><h1>NBPO f0efbb5: original source and line numbers</h1><p>Read-only evidence from the uploaded ZIP. No source modifications. Tests and scope are in the accompanying audit report.</p><nav>'+nav+'</nav>'+''.join(parts)+'</html>')
# Establish unchanged integration files and actual references, without searching archived audit copies.
critical=[S+'make_pros_train_jobs.py',S+'panel_stage2.py','mnpo_scripts/run_mnpo.py','scripts/nbpo/run_nbpo_stage.py','mnpo_scripts/nbpo_generic.py','mnpo_scripts/mnpo_trainer.py']
old=Path('/mnt/data/_reaudit_work/previous/NBPO-main');changes=[]
for path in critical:changes.append({'path':path,'unchanged_from_97e03c0':(R/path).read_bytes()==(old/path).read_bytes(),'sha256':hashlib.sha256((R/path).read_bytes()).hexdigest()})
(O/'unchanged_critical_paths.json').write_text(json.dumps(changes,indent=2))
refs=[]
for p in list((R/S).glob('*.py'))+list((R/'scripts').rglob('*.py'))+list((R/'mnpo_scripts').rglob('*.py')):
 for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
  if 'nbpo_local_gate' in line:refs.append({'path':str(p.relative_to(R)),'line':n,'text':line})
(O/'gate_callsite_search.json').write_text(json.dumps(refs,indent=2))
counts={}
for name,key in [('new_checks_results.json','checks'),('regression_results.json','checks'),('prior_regression_results.json','results')]:
 d=json.loads((O/name).read_text());counts[name]=dict(collections.Counter(x['status'] for x in d[key]))
man={'runtime_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current':json.loads((O/'current_identity.json').read_text()),'previous':json.loads((O/'previous_identity.json').read_text()),'paper_sha256':hashlib.sha256(Path('/mnt/data/main.tex').read_bytes()).hexdigest(),'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'pytest':pytest.__version__},'counts':counts,'source_integrity':json.loads((O/'source_integrity.json').read_text()),'scope':'Uploaded ZIP only; CPU synthetic fixtures; cached log-probability gate replay; no live Git checkout, model downloads, real LLM training/forward, GPU, bf16/DDP, actual benchmark outputs or historical checkpoint provenance.'}
(O/'audit_manifest.json').write_text(json.dumps(man,indent=2))
