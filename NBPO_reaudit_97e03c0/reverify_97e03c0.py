#!/usr/bin/env python3
"""Offline, read-only source re-audit. No model/API calls or repository changes.
python reverify_97e03c0.py --repo /path/NBPO-main --out result.json
Requires numpy/scipy/torch/PyYAML. fixture_support.py must be beside this file.
Full source modules are imported unless a test's scope explicitly says otherwise.
"""
from __future__ import annotations
import argparse, copy, contextlib, hashlib, importlib, inspect, io, itertools, json, math, os, subprocess, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import torch
import fixture_support as fs
RESULTS=[]
def rec(name,status,**d):
 r=dict(name=name,status=status,**d);RESULTS.append(r);print(json.dumps(r,ensure_ascii=False,default=str),flush=True)
def check(name,fn):
 try:fn()
 except Exception as e:rec(name,'HARNESS_ERROR',error=repr(e),traceback=traceback.format_exc())
def objsha(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def module(n):return importlib.import_module(n)
def cli(mod,args):
 argv=sys.argv;sys.argv=[mod.__file__]+args
 try:
  with contextlib.redirect_stdout(io.StringIO()):mod.main()
 finally:sys.argv=argv

def scorers(repo,snap):
 for K in (2,4):
  with tempfile.TemporaryDirectory(prefix='audit97_roles_') as tmp:
   sc=fs.load(snap/'union_score_panel.py',f'audit97_scorer{K}')
   expected,verdicts,jp=fs.fixture(sc,Path(tmp),K=K)
   out=fs.run_score(sc,'full',K);path=out/'shard0/chunk0000.npz'
   with np.load(path) as z:
    arrays={k:z[k].copy() for k in z.files}
   errors={}
   for name,pos in [('A_LL',0),('A_LR',1),('A_RR',2),('A_policy',1),('A_ref',2)]:
    ex=np.stack([expected[str(pid)][pos] for pid in arrays['prompt_ids']],axis=1)
    errors[name]=float(abs(ex-arrays[name]).max())
   complete=out/'shard0/complete_shard0.json';cm=json.loads(complete.read_text())
   rec(f'scorer_roles_K{K}','PASS' if max(errors.values())==0 else 'OPEN',errors=errors,schema=cm.get('tensor_role_schema'),scope='actual scorer main on synthetic JSONL; I/O roots redirected to temporary directory')
   # each registered loader must accept the new scorer output and reject stale schema
   suffixes=['us1','ut1'] if K==2 else ['uw1','uw1c','uw3']
   for suffix in suffixes:
    base=module('solve_pros4_targets_'+suffix);scores,_=base.load_scores(out,1)
    err=max(float(abs(scores[p][0]-expected[p][1]).max()) for p in scores)
    stale=dict(cm);stale.pop('tensor_role_schema');complete.write_text(json.dumps(stale))
    try:base.load_scores(out,1);rejected=False;error=None
    except ValueError as e:rejected=True;error=str(e)
    complete.write_text(json.dumps(cm))
    rec('score_loader_'+suffix,'PASS' if rejected and err==0 else 'OPEN',new_schema_maxerr=err,stale_schema_rejected=rejected,exception=error,scope='actual complete module load_scores, not AST')
   # absence of RR edge in exactly one order rejects only the affected prompt
   vs=[v for v in verdicts if not(v['prompt_id']=='p0' and v['rubric']=='item0' and v['role_i']=='comparator' and v['i']==0 and v['j']==1 and v['order']==1)]
   jp.write_text(''.join(json.dumps(v)+'\n' for v in vs));out2=fs.run_score(sc,'one_missing_rr',K)
   z=np.load(out2/'shard0/chunk0000.npz');kept=list(map(str,z['prompt_ids']));z.close()
   rec(f'RR_one_order_K{K}','PASS' if kept==['p1','p2'] else 'OPEN',retained=kept)
   # v2 metadata alone cannot certify the aliases or declared banks.
   base=module('solve_pros4_targets_'+suffixes[0]);altered={**arrays,'A_policy':arrays['A_LL'].copy()}
   np.savez_compressed(path,**altered);mp=path.with_suffix('.manifest.json');meta=json.loads(mp.read_text());meta['sha256']=fs.sha(path);mp.write_text(json.dumps(meta))
   try:
    got,_=base.load_scores(out,1);accepted=True;aliaserr=float(abs(got['p0'][0]-expected['p0'][1]).max())
   except ValueError as e:accepted=False;aliaserr=None
   rec(f'schema_tag_not_alias_validation_K{K}','OPEN' if accepted else 'PASS',accepted_mislabeled_v2_shard=accepted,policy_tensor_error=aliaserr,scope='deliberately inconsistent test artifact with updated content hash; not evidence this happened in a run')
 p=subprocess.run([sys.executable,str(snap/'union_score_uw.py'),'--help'],capture_output=True,text=True,env=os.environ,timeout=15)
 rec('legacy_scorer_default_refusal','PASS' if p.returncode!=0 and 'superseded' in p.stderr.lower() else 'OPEN',returncode=p.returncode,stderr=p.stderr[:600])

def imports_and_serializers(repo,snap):
 from mnpo_scripts.nbpo_neural import validate_canonical_pair_dataset
 from mnpo_scripts.pair_tokenization import pair_from_candidate_events
 for suffix in ('uw1','uw1c','uw3','us1','ut1'):
  for role in ('pw_nbpo','pw_fixedref','prosper'):
   name=f'solve_pros4_{role}_{suffix}'
   try:module(name);rec('import_'+name,'PASS')
   except Exception as e:rec('import_'+name,'OPEN',error=repr(e))
 for suffix in ('','_v2','_uw1','_uw1c','_uw3','_us1','_ut1'):
  base=module('solve_pros4_targets'+suffix);quant=base.quantize_canonical_row
  probs=[1e-12,1.49e-10,.05,.1,.15,.2,.22,0.];probs[-1]=1.-sum(probs[:-1]);rows=[];maxerr=0.
  for i,j in itertools.combinations(range(8),2):
   r=pair_from_candidate_events([1,2],{'response_token_ids':[10+i,3]},{'response_token_ids':[10+j,3]})
   target=math.log(probs[i]/.125)-math.log(probs[j]/.125)
   r.update(prompt_id='p',chosen_response_id=f'c{i}',rejected_response_id=f'c{j}',nbpo_weight_a=probs[i],nbpo_weight_b=probs[j],nbpo_center_a=.125,nbpo_center_b=.125,nbpo_logratio_target=target,nbpo_num_candidates=8,target_mode='canonical_logratio',target_units='final_logratio_change',eta_already_included=True,solver_artifact_sha256='fixture')
   v=json.loads(json.dumps(quant(r)));rows.append(v);maxerr=max(maxerr,abs(v['nbpo_logratio_target']-target))
  got=validate_canonical_pair_dataset(rows)
  rec('serializer'+(suffix or '_base'),'PASS' if maxerr<1e-9 else 'OPEN',max_target_error=maxerr,min_mass=min(r['nbpo_weight_a'] for r in rows),validator=got,scope='actual imported serializer + JSON round-trip + actual canonical pair validator, no Arrow')

def pipeline(repo,snap):
 from mnpo_scripts.nbpo_core import uniform_policy
 from mnpo_scripts.nbpo_generic import solve_finite_pool,validate_finite_pool_solution
 from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
 from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
 from scripts.nbpo.build_nbpo_pairs import load_canonical_artifact,build_rows,flip_pair_row
 from mnpo_scripts.nbpo_neural import validate_canonical_pair_dataset
 # use the real scorer and loader, and both actual US/UW workers, with immutable synthetic tokens
 for K,suffix in [(2,'us1'),(4,'uw3')]:
  with tempfile.TemporaryDirectory(prefix='audit97_flow_') as tmp:
   root=Path(tmp);sc=fs.load(snap/'union_score_panel.py',f'audit97_flow_scorer{K}');_,vs,jp=fs.fixture(sc,root,K=K,X=2)
   for v in vs:
    if v['role_i']=='learner' and v['role_j']=='comparator':
     v['value_for_i']=.55+.25*(7-v['i'])/7 + .008*int(v['prompt_id'][1:])+.004*int(v['rubric'][4:])
    else:v['value_for_i']=.5
   jp.write_text(''.join(json.dumps(v)+'\n' for v in vs));score_dir=fs.run_score(sc,'flow',K)
   base=module('solve_pros4_targets_'+suffix);scored,_=base.load_scores(score_dir,1);pids=sorted(scored)
   A=np.stack([scored[p][0] for p in pids],axis=1);Ar=np.stack([scored[p][1] for p in pids],axis=1)
   worker=module('solve_pros4_pw_nbpo_'+suffix);worker._init(A,Ar,.25,.7,None,160,1e-12)
   solved=[worker._solve_one(x) for x in range(2)];saved=A[:,1].copy();worker._SHARED['A'][:,1]*=.8;again=worker._solve_one(0)
   independence=float(abs(again[1]-solved[0][1]).max());A[:,1]=saved
   rows=[];residuals=[]
   for x,pid in enumerate(pids):
    s=solved[x];pi,nu,w,g=s[1],s[2],s[3],s[4]
    learners={str(i):{pid:dict(prompt=f'Prompt {pid}',generated_text=f'Answer {pid} {i}',prompt_token_ids=[1,2],response_token_ids=[10+i,3])} for i in range(8)}
    meta={'policy_learner_ids':[f'policy:{i}' for i in range(8)],'comparator_ids':[f'ref:{i}' for i in range(8)]}
    rr=build_rows([pid],list(worker.OBJECTIVES),A[:,x:x+1],nu,w,np.full(K,.25),learners,None,np.random.default_rng(42),'canonical_logratio',meta,{'solver_artifact_sha256':'fixture-'+pid},canonical_data={'p_star':pi,'p_t':np.full_like(pi,.125),'g':g})
    for r in rr:base.quantize_canonical_row(r)
    maxerror=max(abs(r['nbpo_logratio_target']-(g[0,r['chosen_candidate_index']]-g[0,r['rejected_candidate_index']])) for r in rr)
    residuals.append(maxerror);rows+=rr
    # separate true single-prompt core result -> generic writer -> verified loader
    rep=AdaptiveGameRepresentation(torch.from_numpy(A[:,x:x+1].copy()),torch.from_numpy(Ar[:,x:x+1]),uniform_policy(1,8),torch.full((K,),.25,dtype=torch.float64),reference_construction='independent_samples')
    sol=solve_finite_pool(rep,'nash',eta=.7,inner_solver='exact',dual_solver='root',M=160,dual_tol=1e-10)
    sd=root/f'solution_{pid}';md={'objectives':list(worker.OBJECTIVES),'prompt_ids':[pid]}
    manifest=write_generic_solution_artifact(sd,sol,md,{},root/'synthetic_tensors',0,False,{})
    artifact=load_canonical_artifact(sd,manifest,expected_prompt_ids=[pid],expected_representation='adaptive_game',expected_aggregation='nash')
    rec(f'writer_loader_K{K}_{pid}','PASS',serialized_target_error=float(abs(artifact['g']-sol.target_log_ratio.numpy()).max()),certificate=manifest['certificate']['certified'])
    # tampering with serialized NPZ without updating manifest is rejected
    f=sd/'target_log_ratio.npz';old=f.read_bytes();np.savez_compressed(f,h=artifact['g']+.1)
    try:load_canonical_artifact(sd,manifest);rejected=False
    except ValueError:rejected=True
    f.write_bytes(old);rec(f'canonical_artifact_tamper_K{K}_{pid}','PASS' if rejected else 'OPEN',rejected=rejected)
   pair_path=root/'pairs.jsonl';pair_path.write_text(''.join(json.dumps(r)+'\n' for r in rows));loaded=[json.loads(l) for l in pair_path.read_text().splitlines()]
   report=validate_canonical_pair_dataset(loaded)
   flipped=[flip_pair_row(r) for r in loaded];flipreport=validate_canonical_pair_dataset(flipped)
   rec(f'scorer_to_canonical_validator_K{K}','PASS' if max(residuals)<1e-9 and independence==0 else 'OPEN',report=report,flip_report=flipreport,local_independence_error=independence,max_serialized_target_error=max(residuals),all_local_certified=all(s[6] for s in solved),scope='actual scorer main, score loader, imported local _init/_solve_one, imported canonical builder/serializer, JSON I/O, imported validator; no fake Trainer, no model, no tokenizer metadata lookup or Arrow')

def digests_and_gate(repo,snap):
 from scripts.nbpo.eval_game_value import evaluate_game_value
 from scripts.nbpo.run_nbpo_stage import apply_gate
 base=module('solve_pros4_targets_uw3')
 pool={f'p{x}':{role:{i:dict(candidate_id=f'{x}-{role}-{i}',response=f'text {x} {role} {i}',response_sha256=hashlib.sha256(f'text {x} {role} {i}'.encode()).hexdigest()) for i in range(8)} for role in ('learner','comparator')} for x in range(2)}
 digest=base.split_pool_digest;initial=digest(pool,['p0']);dev=digest(pool,['p1']);changed=copy.deepcopy(pool)
 changed['p1']['learner'][0]['response_sha256']='new_dev_hash';outside=digest(changed,['p0'])
 changed=copy.deepcopy(pool);changed['p0']['learner'][0]['response']+='!';changed['p0']['learner'][0]['response_sha256']=hashlib.sha256(changed['p0']['learner'][0]['response'].encode()).hexdigest();inside=digest(changed,['p0'])
 rec('split_pool_digest_scoping','PASS' if initial!=dev and initial==outside and inside!=initial else 'OPEN',different_split_digests=initial!=dev,unrelated_split_unchanged=initial==outside,consistent_content_update_changes_digest=inside!=initial)
 changed=copy.deepcopy(pool);changed['p0']['learner'][0]['response']+='!'
 stale=digest(changed,['p0']);rec('pool_digest_cached_hash_consistency','OPEN' if stale==initial else 'PASS',text_changes_with_stale_cached_hash_not_detected=stale==initial,scope='counterexample to byte-binding; not evidence existing row hashes are actually stale')
 for value in [float('nan'),float('inf'),-float('inf'),0.,-.1,.1]:
  with tempfile.TemporaryDirectory(prefix='audit97_gate_') as tmp:
   root=Path(tmp);parent=root/'parent';candidate=root/'candidate';parent.mkdir();candidate.mkdir();(candidate/'MARKER').write_text('toy')
   got=apply_gate(value,parent,candidate,root/'accepted',fingerprint='synthetic')
   should=math.isfinite(value) and value>0
   rec('gate_'+str(value),'PASS' if got['accepted']==should else 'OPEN',accepted=got['accepted'],promotion_link=(root/'accepted').is_symlink(),parent_retained=got['promoted_path']==str(parent),scope='actual legacy gate including filesystem promotion for positive input; not local-gate orchestration')
 A=torch.zeros(2,2,3,3,dtype=torch.float64);A[:,0]=.2;A[:,1]=-.1;Ar=torch.zeros_like(A);beta=torch.full((2,),.25,dtype=torch.float64);ev=evaluate_game_value(A,Ar,beta)
 with tempfile.TemporaryDirectory(prefix='audit97_global_') as tmp:
  root=Path(tmp);(root/'parent').mkdir();(root/'candidate').mkdir();(root/'candidate/MARKER').write_text('toy')
  got=apply_gate(ev['min_surplus'],root/'parent',root/'candidate',root/'accepted',fingerprint='toy')
  rec('legacy_global_evaluator_gate_not_local','OPEN' if got['accepted'] else 'PASS',locals=[.2,-.1],evaluator_min_surplus=ev['min_surplus'],accepted=got['accepted'],scope='actual evaluator and filesystem gate; Global Nash semantics, not evidence newer jobs invoked this gate')

def styles(repo,snap):
 sc=module('ahv2_style_control');pa=module('ahv2_sc_paired');text='def add(a, b):\n    return a + b';counter,meta=sc.make_length('words',None)
 with tempfile.TemporaryDirectory(prefix='audit97_style_') as tmp:
  root=Path(tmp);jd=root/'judge';jd.mkdir();(root/'arm.jsonl').write_text(json.dumps(dict(prompt_id='p',response=text,n_tokens=18))+'\n');(root/'base.jsonl').write_text(json.dumps(dict(uid='u',response=text))+'\n');(root/'panel.jsonl').write_text(json.dumps(dict(prompt_id='p',uid='u'))+'\n');(jd/'complete.json').write_text(json.dumps(dict(arm='A',judge='fixture',kind='hard',baseline='fixture-base',WIN_RATE_VS_BASELINE=.5,arm_responses=str(root/'arm.jsonl'))))
  path=jd/'verdicts.jsonl';vs=[dict(prompt_id='p',value_for_arm=.5,status='ok',order=o) for o in (0,1)];path.write_text(''.join(json.dumps(v)+'\n' for v in vs))
  _,rows=pa.load(jd,root/'base.jsonl',root/'panel.jsonl',counter);err=float(abs(rows[0][2]).max());rec('style_identical_text','PASS' if err==0 else 'OPEN',difference=rows[0][2].tolist(),length_counter=meta,scope='actual paired loader using supported word-count mode; AutoTokenizer path not run')
  path.write_text(json.dumps(vs[0])+'\n');dropped={};_,rows=pa.load(jd,root/'base.jsonl',root/'panel.jsonl',counter,dropped);rec('style_requires_both_orders','PASS' if not rows and dropped.get('single_order_prompts')==1 else 'OPEN',rows=len(rows),dropped=dropped)
  path.write_text(''.join(json.dumps(v)+'\n' for v in vs));cli(sc,['--judged',str(jd),'--baseline',str(root/'base.jsonl'),'--panel',str(root/'panel.jsonl'),'--length-unit','words','--replicates','10','--out',str(root/'single.json')]);single=json.loads((root/'single.json').read_text())
  cli(pa,['--a',str(jd),'--b',str(jd),'--baseline',str(root/'base.jsonl'),'--panel',str(root/'panel.jsonl'),'--length-unit','words','--replicates','10','--out',str(root/'paired.json')]);pair=json.loads((root/'paired.json').read_text());rec('style_point_and_paired_cli','PASS' if pair['difference']==0 and single['point_fit']['converged'] else 'OPEN',point=single.get('style_controlled_win_rate'),pair_difference=pair['difference'],paired_bootstrap=pair['bootstrap'],scope='actual two CLI main functions with identical-text synthetic fixture and --length-unit words')
 X=np.ones((4,1));y=np.ones(4);out=sc.fit(y,X,iters=1);rec('style_iteration_exhaustion','PASS' if out is not None and out[1]['converged'] is False else 'OPEN',info=None if out is None else out[1])
 out=sc.fit(np.array([.1,.3,.7,.9]),X);rec('style_converged_fit','PASS' if out and out[1]['converged'] else 'OPEN',beta=None if out is None else out[0].tolist(),info=None if out is None else out[1])

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();repo=args.repo.resolve();snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union';sys.path[:0]=[str(repo),str(snap)];torch.set_num_threads(1)
 for name,fn in [('scorers',lambda:scorers(repo,snap)),('imports_serializers',lambda:imports_and_serializers(repo,snap)),('pipeline',lambda:pipeline(repo,snap)),('digests_gate',lambda:digests_and_gate(repo,snap)),('styles',lambda:styles(repo,snap))]:check(name,fn)
 args.out.write_text(json.dumps({'repo':str(repo),'checks':RESULTS},indent=2,ensure_ascii=False,default=str)+'\n');return int(any(r['status']=='HARNESS_ERROR' for r in RESULTS))
if __name__=='__main__':raise SystemExit(main())
