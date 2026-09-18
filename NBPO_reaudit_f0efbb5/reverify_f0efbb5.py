#!/usr/bin/env python3
"""Read-only f0efbb5 source audit. Actual CPU modules and CLI subprocesses.
No model downloads, LLM training, or source patches. Requires numpy/scipy/torch,
fixture_support.py. --out-dir receives fixtures and raw logs for reproducibility.
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, importlib, io, json, os, subprocess, sys, traceback
from pathlib import Path
import numpy as np
import torch
import fixture_support as fs

RESULTS=[]
def record(name,status,**details):
    r={'name':name,'status':status,**details};RESULTS.append(r)
    print(json.dumps(r,ensure_ascii=False,default=str),flush=True)
def call_main(mod,argv):
    old=sys.argv;sys.argv=[mod.__file__]+argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):return mod.main()
    finally:sys.argv=old

def make_scored(repo,root,K=4,kind='standard'):
    snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union'
    sc=fs.load(snap/'union_score_panel.py','f0_scorer_'+root.name)
    expected,vs,jp=fs.fixture(sc,root,K=K,X=4)
    levels=np.linspace(-.25,.25,8)
    if kind=='equal':
        levels=np.repeat([-.1875,-.0625,.0625,.1875],2)
    for pid in expected:
        if kind=='equal':
            mat=levels[:,None]-levels[None,:]
            LL=LR=RR=np.repeat(mat[None],K,axis=0)
        else:
            LL=np.repeat((levels[:,None]-levels[None,:])[None],K,axis=0)
            RR=np.zeros((K,8,8))
            LR=np.repeat(np.broadcast_to(levels[:,None],(8,8))[None],K,axis=0).copy()
            if kind=='beta':LR+=np.array([-.25]+[.125]*7)[None,None,:]
            if kind=='tiny':LR[:]=5e-9
        expected[pid]=(LL.copy(),LR.copy(),RR.copy())
    for v in vs:
        ll,lr,rr=expected[v['prompt_id']];k=int(v['rubric'][4:]);i,j=v['i'],v['j']
        mat=rr if v['role_i']=='comparator' else (lr if v['role_j']=='comparator' else ll)
        v['value_for_i']=float(.5+mat[k,i,j])
    jp.write_text(''.join(json.dumps(v)+'\n' for v in vs))
    call_main(sc,['--tag','toy','--pool','toy','--pool-shards','1','--judge-shards','1','--shards','4','--out','scores','--objectives',str(K)])
    return sc.UF/'scores/scores',expected

def load_array_file(path):
    with np.load(path,allow_pickle=True) as z:return {k:z[k].copy() for k in z.files}
def update_npz(path,arrays):
    np.savez_compressed(path,**arrays)
    mp=path.with_suffix('.manifest.json');meta=json.loads(mp.read_text());meta['sha256']=fs.sha(path);mp.write_text(json.dumps(meta))

def loader_checks(repo,root):
    from mnpo_scripts.nbpo_core import uniform_policy
    from mnpo_scripts.nbpo_generic import solve_finite_pool,validate_finite_pool_solution
    from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
    for K,suffixes in [(2,['us1','ut1']),(4,['uw1','uw1c','uw3'])]:
        sd,expected=make_scored(repo,root/f'equal_K{K}',K,'equal')
        LL,A,B=expected['p0'];rep=AdaptiveGameRepresentation(torch.from_numpy(A[:,None]),torch.from_numpy(B[:,None]),uniform_policy(1,8),torch.full((K,),.25,dtype=torch.float64),reference_construction='independent_samples')
        sol=solve_finite_pool(rep,'nash',eta=.7,inner_solver='exact',dual_solver='root',M=160,dual_tol=1e-10)
        cert=validate_finite_pool_solution(sol)
        for suffix in suffixes:
            mod=importlib.import_module('solve_pros4_targets_'+suffix)
            try:mod.load_scores(sd,4);rej=False;error=None
            except ValueError as e:rej=True;error=str(e)
            record('valid_equal_semantic_arrays_'+suffix,'OPEN' if rej else 'PASS',rejected=rej,error=error,core_certified=cert['certified'],core_min_surplus=float(sol.surplus.min()),scope='actual generalized scorer and actual loader; independent banks may yield identical matrices; core solve shows positive feasible improvement')
    sd,_=make_scored(repo,root/'validation_K4',4)
    path=sd/'shard0/chunk0000.npz';original=load_array_file(path);mod=importlib.import_module('solve_pros4_targets_uw1c')
    for name,modify,expect_reject in [
      ('wrong_alias',lambda d:d.update(A_policy=d['A_LL'].copy()),True),
      ('missing_semantics',lambda d:(d.pop('A_LR'),d.pop('A_RR'),d.update(A_policy=d['A_LL'].copy())),True),
    ]:
        arrays=copy.deepcopy(original);modify(arrays);update_npz(path,arrays)
        try:mod.load_scores(sd,4);rej=False;error=None
        except ValueError as e:rej=True;error=str(e)
        record('loader_'+name,'PASS' if rej==expect_reject else 'OPEN',rejected=rej,error=error,scope='synthetic altered artifact, not evidence production artifacts are corrupt')
        update_npz(path,original)
    mp=path.with_suffix('.manifest.json');om=json.loads(mp.read_text());bad=copy.deepcopy(om);bad['bank_ids']['A_ref']=['learner','learner'];mp.write_text(json.dumps(bad))
    try:mod.load_scores(sd,4);rej=False;error=None
    except ValueError as e:rej=True;error=str(e)
    mp.write_text(json.dumps(om));record('loader_wrong_bank_roles','PASS' if rej else 'OPEN',rejected=rej,error=error)

def digest_checks():
    for suffix in ['uw1','uw1c','uw3','us1','ut1']:
        mod=importlib.import_module('solve_pros4_targets_'+suffix)
        pool={'p':{r:{i:{'candidate_id':f'p:{r}:{i}','response':f'text {r} {i}','response_sha256':hashlib.sha256(f'text {r} {i}'.encode()).hexdigest()} for i in range(8)} for r in ('learner','comparator')}}
        h=mod.split_pool_digest(pool,['p']);n=mod.verify_pool_row_digests(pool,['p']);bad=copy.deepcopy(pool);bad['p']['learner'][0]['response']+='!';h2=mod.split_pool_digest(bad,['p'])
        try:mod.verify_pool_row_digests(bad,['p']);rejected=False
        except ValueError:rejected=True
        record('digest_'+suffix,'PASS' if h!=h2 and rejected and n==16 else 'OPEN',checked=n,content_edit_changes_digest=h!=h2,stale_sha_rejected=rejected,panel_label=mod.PANEL_LABEL)

class GateFixture:
    def __init__(self,repo,root,K=4,kind='standard',target_beta=.25):
        self.repo,self.root,self.K=repo,root,K;self.snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union'
        self.helper=repo/'analysis/uf4_20260910/code_snapshot_20260917';self.suffix='uw1c' if K==4 else 'us1'
        self.scores,self.expected=make_scored(repo,root,K,kind);self.pids=sorted(self.expected)
        self.parent=root/'parent';self.candidate=root/'candidate'
        for p in [self.parent,self.candidate]:
            p.mkdir();(p/'config.json').write_text(json.dumps({'fixture':True,'role':p.name}));(p/'model.safetensors').write_bytes(b'NOT REAL MODEL WEIGHTS; AUDIT FINGERPRINT FIXTURE')
        self.targets=root/'targets';self.targets.mkdir();self.kind=kind;self.beta=target_beta
        A=np.stack([self.expected[p][1] for p in self.pids],axis=1);B=np.stack([self.expected[p][2] for p in self.pids],axis=1)
        if kind!='tiny':
            worker=importlib.import_module('solve_pros4_pw_nbpo_'+self.suffix);worker._init(A,B,target_beta,.7,None,160,1e-12)
            sols=[worker._solve_one(x) for x in range(4)]
            self.g=np.stack([s[4][0] for s in sols]);pis=np.stack([s[1][0] for s in sols]);mins=np.array([s[7] for s in sols]);cert=all(s[6] for s in sols)
        else:self.g=np.zeros((4,8));pis=np.full((4,8),.125);mins=np.full(4,5e-9);cert=False
        self.npz=self.targets/'dev_per_prompt.npz';np.savez_compressed(self.npz,prompt_ids=np.array(self.pids),pi=pis,g=self.g,min_surplus=mins,identity_residual=np.zeros(4))
        md=self.targets/'dev/solver';md.mkdir(parents=True);(md/'solution.json').write_text(json.dumps({'beta':target_beta,'eta':.7,'all_certified':cert,'per_prompt_artifact_sha256':fs.sha(self.npz),'per_prompt_artifact':str(self.npz),'representation':'adaptive_game','aggregation':'prompt_wise_nash','n_prompts':4}))
        self.env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,[repo,self.snap,self.helper])),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    def run(self,name,mode='good',fit=.5,coverage=None,beta=None,missing_candidate=False,drop_lp=False):
        d=self.root/name;d.mkdir();lp={'candidate':{},'parent':{}}
        for x,pid in enumerate(self.pids):
            delta=self.g[x].copy()
            if mode=='bad' or (mode=='mixed' and x>=2):delta=-delta
            if mode=='uniform':delta[:]=0
            for i in range(8):
                cid=f'{pid}:learner:{i}';lp['parent'][cid]=-30.;lp['candidate'][cid]=float(-30.+delta[i])
        if mode=='nan':lp['candidate']['p0:learner:0']=float('nan')
        if mode=='inf':lp['candidate']['p0:learner:0']=float('inf')
        if drop_lp:del lp['candidate']['p0:learner:0']
        lpfile=d/'logprobs.json';lpfile.write_text(json.dumps(lp));out=d/'decision.json';link=d/'accepted'
        cmd=[sys.executable,str(self.snap/'nbpo_local_gate.py'),'--candidate',str(self.root/'nonexistent' if missing_candidate else self.candidate),'--parent',str(self.parent),'--targets',str(self.targets),'--pool',str(self.root/'pool_not_read_in_cached_mode'),'--scores',str(self.scores),'--solver-module',str(self.snap/('solve_pros4_targets_'+self.suffix+'.py')),'--logprobs',str(lpfile),'--promote-to',str(link),'--out',str(out)]
        if coverage is not None:cmd+=['--coverage-min',str(coverage)]
        if beta is not None:cmd+=['--beta',str(beta)]
        if fit!='omit':
            state={'log_history':[]} if fit=='missing' else {'log_history':[{'eval_nbpo/nmse':fit}]}
            sf=d/'trainer_state.json';sf.write_text(json.dumps(state));cmd+=['--nmse-from',str(sf)]
        cp=subprocess.run(cmd,cwd=self.repo,env=self.env,capture_output=True,text=True,timeout=30)
        (d/'command.json').write_text(json.dumps(cmd,indent=2));(d/'stdout.log').write_text(cp.stdout);(d/'stderr.log').write_text(cp.stderr)
        dec=json.loads(out.read_text()) if out.exists() else None
        got={'returncode':cp.returncode,'decision_exists':dec is not None,'stderr':cp.stderr[-1000:]}
        if dec:
            got.update(accepted=dec['accepted'],checks=dec['checks'],coverage=dec['measured']['coverage'],nonfinite=dec['measured']['nonfinite_prompts'],nmse=dec['measured']['held_out_nmse'],parent_retained=dec['policy_after_this_stage']==str(self.parent),promoted=bool(dec['promotion']),local_surpluses=[r['min_surplus'] for r in dec['per_prompt']],promotion_contents=sorted(p.name for p in link.resolve().iterdir()) if link.exists() else [],fingerprint=None if not dec['promotion'] else dec['promotion']['candidate_fingerprint'])
        return got

def gate_checks(repo,root):
    snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union'
    env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,[repo,snap])),'PYTHONDONTWRITEBYTECODE':'1'}
    cp=subprocess.run([sys.executable,str(snap/'nbpo_local_gate.py'),'--help'],env=env,cwd=repo,capture_output=True,text=True,timeout=15)
    record('gate_root_current_import','OPEN' if cp.returncode else 'PASS',returncode=cp.returncode,error=cp.stderr[-800:],scope='root and current snapshot on PYTHONPATH; no /work overlay')
    f=GateFixture(repo,root/'gate_standard')
    for name,kw,should in [
      ('positive',{},True),('negative',{'mode':'bad'},False),
      ('nonfinite_logp_nan',{'mode':'nan'},False),('nonfinite_logp_inf',{'mode':'inf'},False),
      ('fit_above_one',{'fit':1.2},False),('fit_nan',{'fit':float('nan')},False),('fit_posinf',{'fit':float('inf')},False),
      ('mixed_default',{'mode':'mixed'},False),('mixed_full_coverage',{'mode':'mixed','coverage':1.0},False),
      ('nmse_requested_but_missing',{'fit':'missing'},False),('nmse_neginf',{'fit':-float('inf')},False),
      ('nonexistent_checkpoint',{'missing_candidate':True},False)
    ]:
        v=f.run(name,**kw);passed=v.get('accepted')==should and v.get('decision_exists')
        record('gate_'+name,'PASS' if passed else 'OPEN',expected_accepted_under_paper_contract=should,**v,scope='actual CLI subprocess and filesystem with synthetic checkpoint sentinel, cached log-probabilities; no model forward')
    v=f.run('fit_omitted',fit='omit');record('gate_fit_omitted','OBSERVATION',**v,note='documented optional bypass, but unlike manuscript all-check contract; not claimed an accidental missing file')
    v=f.run('missing_logp',drop_lp=True);record('gate_missing_logp','PASS' if v['returncode']!=0 and not v['decision_exists'] else 'OPEN',**v,note='fails closed by terminating; no decision record produced')
    # A genuine second panel, 2 objectives, same 4-shard graph and local inference.
    f2=GateFixture(repo,root/'gate_two_objectives',K=2)
    v=f2.run('positive');record('gate_two_objectives_positive','PASS' if v.get('accepted') else 'OPEN',**v)
    tiny=GateFixture(repo,root/'gate_tiny',kind='tiny');v=tiny.run('positive_below_epsilon',mode='uniform',coverage=1.0)
    record('gate_surplus_epsilon','OPEN' if v.get('accepted') else 'PASS',**v,required_paper_margin=1e-8)
    # Certified beta=.05 target, but gate defaults to beta=.25 and never reads solution.json.
    b=GateFixture(repo,root/'gate_beta',kind='beta',target_beta=.05)
    wrong=b.run('default_beta',mode='uniform',fit=1.,coverage=1.)
    correct=b.run('correct_beta',mode='uniform',fit=1.,coverage=1.,beta=.05)
    record('gate_beta_binding','OPEN' if wrong.get('accepted') and not correct.get('accepted') else 'PASS',target_beta=.05,default_beta_result=wrong,matching_beta_result=correct,note='same certified target and candidate LP; only runtime beta changed; metadata mismatch not rejected')
    # Explicit mutation of target artifact after a matching hash was recorded.
    z=load_array_file(f.npz);z['g']+=7.;update_hash_before=fs.sha(f.npz);np.savez_compressed(f.npz,**z)
    v=f.run('tampered_target');record('gate_target_manifest_binding','OPEN' if v.get('accepted') else 'PASS',accepted=v.get('accepted'),target_changed=fs.sha(f.npz)!=update_hash_before,note='supplied target hash no longer matches solution.json; gate does not inspect that manifest')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union';sys.path[:0]=[str(repo),str(snap)];torch.set_num_threads(1)
    for name,fn in [('loaders',lambda:loader_checks(repo,out/'fixtures')),('digests',digest_checks),('local_gate',lambda:gate_checks(repo,out/'fixtures'))]:
        try:fn()
        except Exception as e:record(name,'HARNESS_ERROR',error=repr(e),traceback=traceback.format_exc())
    (out/'new_checks_results.json').write_text(json.dumps({'repo':str(repo),'checks':RESULTS},indent=2,ensure_ascii=False,default=str)+'\n')
    return int(any(x['status']=='HARNESS_ERROR' for x in RESULTS))
if __name__=='__main__':raise SystemExit(main())
