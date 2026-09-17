#!/usr/bin/env python3
"""Read-only, CPU follow-up audit. Source extraction is explicitly marked; no model downloads.
Run: python reverify_b949919.py --repo /path/NBPO-main --out expanded_results.json
Requires numpy, scipy, torch. Does not change audited source files.
"""
from __future__ import annotations
import argparse, ast, contextlib, copy, hashlib, importlib.util, io, itertools
import json, math, os, subprocess, sys, tempfile, traceback
from pathlib import Path
import numpy as np

RESULTS=[]
def record(name,status,**details):
    row=dict(name=name,status=status,**details); RESULTS.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def isolated(path,name,env):
    tree=ast.parse(path.read_text()); fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    fn.decorator_list=[]
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),env)
    return env[name]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def fixture(mod,root,K=4,X=3):
    mod.UF=root/'uf'; mod.SUB=root/'sub'
    pd=mod.UF/'pools/toy/shard0';pd.mkdir(parents=True)
    pool=[]; verdicts=[]; matrices={}
    rng=np.random.default_rng(984)
    for x in range(X):
        pid=f'p{x}'
        pool += [dict(prompt_id=pid,role=r,sample_index=i,response_sha256=f'{pid}_{r}_{i}') for r in ('learner','comparator') for i in range(8)]
        LL=np.zeros((K,8,8)); LR=np.zeros_like(LL); RR=np.zeros_like(LL)
        for k in range(K):
            for r1,r2,pairs,M in [('learner','learner',itertools.combinations(range(8),2),LL),('learner','comparator',itertools.product(range(8),repeat=2),LR),('comparator','comparator',itertools.combinations(range(8),2),RR)]:
                for i,j in pairs:
                    v0,v1=rng.integers(1,8,size=2)/8
                    p=(v0+v1)/2;M[k,i,j]=p-.5
                    if r1==r2:M[k,j,i]=.5-p
                    for order,v in enumerate([v0,v1]):
                        verdicts.append(dict(prompt_id=pid,rubric=f'item{k}',role_i=r1,i=i,role_j=r2,j=j,order=order,value_for_i=float(v),status='ok'))
        matrices[pid]=(LL,LR,RR)
    path=pd/'chunk0000.jsonl';path.write_text(''.join(json.dumps(v)+'\n' for v in pool))
    (pd/'chunk0000.manifest.json').write_text(json.dumps({'sha256':sha(path)}));(pd/'settings.json').write_text('{}')
    jd=mod.SUB/'pool_judgments/toy/shard0';jd.mkdir(parents=True);(jd/'complete.json').write_text('{}')
    jp=jd/'chunk0000.jsonl';jp.write_text(''.join(json.dumps(v)+'\n' for v in verdicts))
    return matrices,verdicts,jp

def run_score(mod,out,K=None):
    prev=sys.argv
    sys.argv=['score','--tag','toy','--pool','toy','--pool-shards','1','--judge-shards','1','--shards','1','--out',out]
    if K is not None:sys.argv+=['--objectives',str(K)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):mod.main()
    finally:sys.argv=prev
    return mod.UF/'scores'/out

def test_scorers(repo,snap):
    from mnpo_scripts.nbpo_core import uniform_policy
    from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
    import torch
    for K in (2,4):
        with tempfile.TemporaryDirectory(prefix='nbpo_roles_') as d:
            mod=load(snap/'union_score_panel.py',f'audit_new_scorer_{K}')
            expected,verdicts,jp=fixture(mod,Path(d),K=K)
            out=run_score(mod,'corrected',K)
            z=np.load(out/'shard0/chunk0000.npz')
            errors={}
            for field,num in [('A_LL',0),('A_LR',1),('A_RR',2),('A_policy',1),('A_ref',2)]:
                e=np.stack([expected[str(pid)][num] for pid in z['prompt_ids']],axis=1)
                errors[field]=float(np.max(np.abs(e-z[field])))
            meta=json.loads((out/'shard0/chunk0000.manifest.json').read_text())
            record(f'five_tensor_fields_K{K}','PASS' if max(errors.values())==0 else 'FAIL',max_abs_errors=errors,roles=meta['roles'],prompts=len(z['prompt_ids']))
            # Load actual baseline-label function, not the unavailable datasets package.
            get=isolated(snap/'build_panel_softlabels.py','load_policy_block',{'Path':Path,'np':np})
            labs=get(out,1)
            error=max(float(np.max(np.abs(labs[p]-expected[p][0]))) for p in labs)
            record(f'softlabels_use_LL_K{K}','PASS' if error==0 else 'FAIL',max_error=error,scope='original loader function AST; DatasetDict processing not run')
            # One missing presentation order of one RR edge, not every RR edge.
            kept=[v for v in verdicts if not (v['prompt_id']=='p1' and v['role_i']=='comparator' and v['rubric']=='item0' and v['i']==0 and v['j']==1 and v['order']==1)]
            jp.write_text(''.join(json.dumps(v)+'\n' for v in kept))
            dropped=run_score(mod,'one_missing',K)
            dz=np.load(dropped/'shard0/chunk0000.npz');ids=dz['prompt_ids'].tolist()
            record(f'partial_RR_missing_drops_whole_prompt_K{K}','PASS' if ids==['p0','p2'] else 'FAIL',retained=ids)
    # Old scorer still available in same current directory, not merely an archived old date.
    with tempfile.TemporaryDirectory(prefix='nbpo_oldroles_') as d:
        mod=load(snap/'union_score_uw.py','audit_unchanged_uw_scorer')
        expected,_,_=fixture(mod,Path(d),K=4,X=1);out=run_score(mod,'old')
        z=np.load(out/'shard0/chunk0000.npz');e=expected['p0']
        err_lr=float(np.max(abs(z['A_policy'][:,0]-e[1])))
        err_rr=float(np.max(abs(z['A_ref'][:,0]-e[2])))
        record('old_uw_scorer_still_miswired','OPEN' if err_lr>0 or err_rr>0 else 'PASS',LR_error=err_lr,RR_error=err_rr,scope='executed unchanged union_score_uw.py')
        get=isolated(snap/'build_panel_softlabels.py','load_policy_block',{'Path':Path,'np':np})
        try:get(out,1);record('softlabel_loader_rejects_old_schema','FAIL')
        except SystemExit as ex:record('softlabel_loader_rejects_old_schema','PASS',error=str(ex))
        scoreget=isolated(snap/'solve_pros4_targets_uw1c.py','load_scores',{'Path':Path,'np':np,'json':json,'file_hash':sha,'object_hash':objsha,'OBJECTIVES':[f'item{k}' for k in range(4)],'POOL':8})
        try:
            got,_=scoreget(out,1)
            record('corrected_solver_loader_rejects_old_schema','OPEN',accepted_obsolete_score=True,scope='original load_scores AST; import barrier not bypassed in any full run',policy_tensor_still_LL=bool(np.allclose(got['p0'][0],e[0])))
        except (ValueError,SystemExit,KeyError) as ex:record('corrected_solver_loader_rejects_old_schema','PASS',error=str(ex))

def test_imports(repo,snap):
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(repo),str(snap)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for suffix in ('uw1c','uw3'):
        for arm in ('pw_nbpo','pw_fixedref','prosper'):
            name=f'solve_pros4_{arm}_{suffix}'
            p=subprocess.run([sys.executable,'-c',f'import {name}'],env=env,cwd=repo,capture_output=True,text=True,timeout=25)
            record(f'import_{name}','PASS' if p.returncode==0 else 'BLOCKED',returncode=p.returncode,stderr=p.stderr.strip())

def test_serializers(repo,snap):
    from mnpo_scripts.nbpo_neural import validate_canonical_pair_dataset
    from mnpo_scripts.pair_tokenization import pair_from_candidate_events
    for suffix in ('uw1','uw1c','uw3','us1','ut1'):
        path=snap/f'solve_pros4_targets_{suffix}.py'
        quant=isolated(path,'quantize_canonical_row',{'POOL':8,'CANONICAL_DECIMALS':10,'CANONICAL_DECIMALS_LEGACY':10})
        row=dict(target_mode='canonical_logratio',nbpo_weight_a=1.49e-10,nbpo_weight_b=.2,nbpo_center_a=.125,nbpo_center_b=.125)
        orig=math.log(row['nbpo_weight_a']/.125)-math.log(row['nbpo_weight_b']/.125)
        q=json.loads(json.dumps(quant(dict(row))))
        drift=abs(q['nbpo_logratio_target']-orig)
        # 8-candidate/all-28-pair fixture checked by the actual lightweight validator.
        probs=[1e-12,1.49e-10,.05,.1,.15,.2,.22,0.]
        probs[-1]=1.-sum(probs[:-1]);rows=[];maxerr=0.
        for i,j in itertools.combinations(range(8),2):
            r=pair_from_candidate_events([1,2],{'response_token_ids':[10+i,3]},{'response_token_ids':[10+j,3]})
            target=math.log(probs[i]/.125)-math.log(probs[j]/.125)
            r.update(prompt_id='fixture',chosen_response_id=f'c{i}',rejected_response_id=f'c{j}',nbpo_weight_a=probs[i],nbpo_weight_b=probs[j],nbpo_center_a=.125,nbpo_center_b=.125,nbpo_logratio_target=target,nbpo_num_candidates=8,target_mode='canonical_logratio',target_units='final_logratio_change',eta_already_included=True,solver_artifact_sha256='synthetic-target-hash')
            result=json.loads(json.dumps(quant(r)));rows.append(result)
            maxerr=max(maxerr,abs(target-result['nbpo_logratio_target']))
        try:
            report=validate_canonical_pair_dataset(rows);accepted=True;error=None
        except Exception as ex:accepted=False;error=repr(ex);report=None
        record(f'serializer_{suffix}','PASS' if drift<=1e-9 and accepted else 'OPEN',counterexample_drift=drift,min_serialized_mass=min(r['nbpo_weight_a'] for r in rows),all_28_validator_accepted=accepted,error=error,max_28_target_drift=maxerr,report=report,scope='unmodified serializer function AST, json.dumps/loads and actual validate_canonical_pair_dataset; not Arrow materialization')

def test_styles(snap):
    sc=load(snap/'ahv2_style_control.py','ahv2_style_control')
    paired=load(snap/'ahv2_sc_paired.py','reaudit_sc_paired')
    text='def add(a, b):\n    return a + b'
    # Exercise the actual paired file loader using its asymmetric feature constructors.
    with tempfile.TemporaryDirectory(prefix='nbpo_style_') as d:
        root=Path(d);jd=root/'judged';jd.mkdir()
        (root/'arm.jsonl').write_text(json.dumps(dict(prompt_id='p0',response=text,n_tokens=18))+'\n')
        (root/'base.jsonl').write_text(json.dumps(dict(uid='u0',messages=[{'role':'assistant','content':text}]))+'\n')
        (root/'panel.jsonl').write_text(json.dumps(dict(prompt_id='p0',uid='u0'))+'\n')
        (jd/'complete.json').write_text(json.dumps({'arm':'A','arm_responses':str(root/'arm.jsonl')}))
        (jd/'verdicts.jsonl').write_text(''.join(json.dumps(dict(prompt_id='p0',value_for_arm=.5,status='ok',order=o))+'\n' for o in (0,1)))
        _,rows=paired.load(jd,root/'base.jsonl',root/'panel.jsonl')
        err=float(np.max(np.abs(rows[0][2])))
        record('style_identical_text_invariant','OPEN' if err>0 else 'PASS',same_response=True,stored_arm_n_tokens=18,baseline_whitespace_words=len(text.split()),difference=rows[0][2].tolist(),scope='actual ahv2_sc_paired.load; illustrative token count is fixture, not a tokenizer measurement')
        # Check missing order: the same loader accepts one valid order.
        (jd/'verdicts.jsonl').write_text(json.dumps(dict(prompt_id='p0',value_for_arm=1.,status='ok',order=0))+'\n')
        _,rows=paired.load(jd,root/'base.jsonl',root/'panel.jsonl')
        record('style_incomplete_order_contract','OBSERVATION',retains_one_order_prompt=bool(rows),scope='paired style loader retains available orders, unlike declared both-order comparison')
    X=np.ones((4,1));y=np.ones(4)
    b=sc.fit(y,X,iters=1)
    grad=float(np.abs(X.T@(y-1/(1+np.exp(-(X@b))))).max())
    record('style_fit_exhaustion_signaled','OPEN' if b is not None and grad>1e-4 else 'PASS',returned_coefficients=b.tolist() if b is not None else None,gradient_inf=grad,max_iterations=1,scope='actual fit returns coefficients even when stopping budget is exhausted')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    repo=args.repo.resolve();snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union';sys.path[:0]=[str(repo),str(snap)]
    import torch;torch.set_num_threads(1)
    for name,fn in [('scorers',lambda:test_scorers(repo,snap)),('imports',lambda:test_imports(repo,snap)),('serializers',lambda:test_serializers(repo,snap)),('styles',lambda:test_styles(snap))]:
        try:fn()
        except Exception as ex:record(name,'HARNESS_ERROR',error=repr(ex),traceback=traceback.format_exc())
    args.out.write_text(json.dumps({'repo':str(repo),'checks':RESULTS},indent=2,ensure_ascii=False)+'\n')
    return int(any(r['status']=='HARNESS_ERROR' for r in RESULTS))
if __name__=='__main__':raise SystemExit(main())
