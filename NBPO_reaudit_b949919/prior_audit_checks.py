#!/usr/bin/env python3
"""Offline audit of the uploaded NBPO source. Never loads model weights or writes the repository.

Run: python audit_checks.py --repo /path/to/NBPO-main --out /path/to/audit_results.json
Requires numpy, scipy, torch. Heavy Trainer dependencies are NOT substituted: one
unmodified method AST is executed separately and reported as an isolated unit test.
"""
from __future__ import annotations
import argparse, ast, contextlib, copy, hashlib, importlib, importlib.util, inspect
import io, itertools, json, math, sys, tempfile, time, traceback, typing
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import logsumexp

RESULTS=[]

def record(name, status, **details):
    item=dict(name=name,status=status,**details); RESULTS.append(item)
    print(json.dumps(item,ensure_ascii=False),flush=True)

def extract_function(path, name, global_values, class_name=None):
    tree=ast.parse(Path(path).read_text())
    nodes=tree.body if class_name is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name).body
    fun=copy.deepcopy(next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name))
    fun.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fun],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),global_values)
    return global_values[name]

def import_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod);return mod

def run(repo):
    sys.path.insert(0,str(repo)); snap=repo/'analysis/sub_20260914/code_snapshot_20260917_union'
    sys.path.insert(0,str(snap)); torch.set_num_threads(1)
    from mnpo_scripts.nbpo_core import compute_regularized_game_value,compute_regularized_opponent,compute_margins,uniform_policy
    from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
    from mnpo_scripts.nbpo_generic import solve_finite_pool,validate_finite_pool_solution,solve_proximal_exact
    from scripts.nbpo.build_nbpo_pairs import build_rows
    from mnpo_scripts.nbpo_neural import nbpo_regression_target
    from mnpo_scripts.response_logps import response_logps

    # Real import failures, independent of missing Transformers/Accelerate.
    try:
        importlib.import_module('solve_pros4_pw_nbpo_uw1')
        record('campaign_solver_import','PASS')
    except Exception as e:
        record('campaign_solver_import','REPRODUCED_FAILURE',error=repr(e))
    module=importlib.import_module('scripts.nbpo.solve_nbpo_dual')
    record('generic_artifact_writer_available','PASS' if hasattr(module,'write_generic_solution_artifact') else 'REPRODUCED_FAILURE',export='write_generic_solution_artifact')
    try:
        build_rows([],[],None,None,None,None,None,None,None,'canonical_logratio',{}, {}, canonical_data={})
        record('canonical_builder_signature','PASS')
    except TypeError as e:
        record('canonical_builder_signature','REPRODUCED_FAILURE',error=repr(e),signature=str(inspect.signature(build_rows)))

    # Independent concave primal solve versus the real nested dual-root solver.
    outcomes=[]
    for seed in range(6):
        rng=np.random.default_rng(seed); K=2+seed%2; I=3+seed%3; J=4
        A=np.linspace(.35,-.10,I)[None,None,:,None]+rng.uniform(-.05,.05,(K,1,I,J))
        B=rng.uniform(-.05,.05,(K,1,J,J)); B=(B-B.swapaxes(-1,-2))/2
        beta=torch.tensor([.2+.05*k for k in range(K)],dtype=torch.float64)
        rep=AdaptiveGameRepresentation(torch.from_numpy(A),torch.from_numpy(B),uniform_policy(1,J),beta)
        eta=.4+seed*.3
        res=solve_finite_pool(rep,'nash',eta=eta,inner_solver='exact',dual_solver='root',dual_tol=1e-9,M=160)
        cert=validate_finite_pool_solution(res)
        center=np.ones(I)/I
        b_np=beta.numpy()
        # Independent analytic value; defined off-simplex so SLSQP numerical
        # differentiation does not trigger the repository's simplex validator.
        ref_margins=B[:,0].mean(axis=1)
        d_np=-b_np*logsumexp(-math.log(J)-ref_margins/b_np[:,None],axis=-1)
        def s(p):
            margins=np.einsum('i,kij->kj',p,A[:,0])
            return -b_np*logsumexp(-math.log(J)-margins/b_np[:,None],axis=-1)-d_np
        def objective(p):
            surplus=s(p)
            return 1e10 if np.any(surplus<=0) else -float(np.log(surplus).sum()-np.sum(p*np.log(p/center))/eta)
        independent=minimize(objective,center,method='SLSQP',bounds=[(1e-12,1)]*I,
            constraints=[{'type':'eq','fun':lambda p:p.sum()-1},{'type':'ineq','fun':lambda p:s(p)-1e-10}],
            options={'maxiter':800,'ftol':1e-12})
        gap=abs(objective(res.pi.numpy()[0])-independent.fun)
        outcomes.append(dict(seed=seed,certified=cert['certified'],independent_success=bool(independent.success),
             policy_max_abs=float(np.max(abs(res.pi.numpy()[0]-independent.x))),objective_gap=float(gap),
             dual_residual=cert['dual_unprojected_residual'],complementarity=cert['nash_complementarity_inf'],
             target_identity=res.target_log_ratio_check()))
    record('six_exact_dual_vs_independent_primal','PASS' if all(v['certified'] and v['independent_success'] and v['objective_gap']<1e-8 for v in outcomes) else 'CHECK',games=outcomes)

    # RPS stress test: uniform empirical comparator, nonuniform proximal center.
    from mnpo_scripts.nbpo_solver import solve_weighted_policy
    rps=torch.tensor([[0.,.5,-.5],[-.5,0.,.5],[.5,-.5,0.]],dtype=torch.float64)[None,None]
    anchor=torch.tensor([[.34,.33,.33]],dtype=torch.float64)
    mu_rps=uniform_policy(1,3)
    rep_rps=AdaptiveGameRepresentation(rps,rps,mu_rps,torch.tensor([.25],dtype=torch.float64))
    w_rps=torch.tensor([4.],dtype=torch.float64)
    exact=solve_proximal_exact(rep_rps,anchor,w_rps,1.)
    legacy=solve_weighted_policy(rps,mu_rps,anchor,w_rps,rep_rps.beta,1.,100)
    record('rps_direct_vs_undamped_legacy','PASS' if exact.fixed_point_residual<1e-8 and float(torch.linalg.vector_norm(legacy.pi-exact.pi))>.01 else 'CHECK',exact_policy=exact.pi.tolist(),
        exact_fixedpoint_residual=exact.fixed_point_residual,
        legacy_after_100_policy_distance=float(torch.linalg.vector_norm(legacy.pi-exact.pi)),
        legacy_extra_map_residual=legacy.extra_map_residual,
        scope='Fixed-multiplier RPS inner solve with uniform reference and nonuniform center, not a full Nash-dual/neural test or exact copy of paper example')

    # Execute the actual per-prompt worker body, isolated from missing package imports.
    local_rng=np.random.default_rng(94); ak=np.zeros((2,2,8,8))
    for x in range(2):
        for k in range(2):
            ak[k,x]=np.linspace(.25+.05*(x+k),-.10,8)[:,None]+local_rng.uniform(-.03,.03,(8,8))
    bk=local_rng.uniform(-.03,.03,(2,2,8,8));bk=(bk-bk.swapaxes(-1,-2))/2
    shared={'A':ak,'Aref':bk,'beta':.25,'eta':.7,'M':160,'floor':1e-12}
    worker=extract_function(snap/'solve_pros4_pw_nbpo_uw1.py','_solve_one',
        {'torch':torch,'np':np,'_SHARED':shared,'POOL':8,'OBJECTIVES':['item0','item1'],
         'AdaptiveGameRepresentation':AdaptiveGameRepresentation,'uniform_policy':uniform_policy,
         'solve_finite_pool':solve_finite_pool,'validate_finite_pool_solution':validate_finite_pool_solution})
    lx0,lx1=worker(0),worker(1)
    shared['A'][:,1]*=.4;changed_other=worker(0)
    err=float(np.max(abs(lx0[1]-changed_other[1])))
    record('actual_local_worker_independence','PASS' if err==0 and lx0[6] and lx1[6] else 'CHECK',
        other_prompt_change_error=err,prompt0_weights=lx0[3].tolist(),prompt1_weights=lx1[3].tolist(),
        scope='Unmodified _solve_one AST; full module import remains broken')

    # Nontransitive (cycle) game and analytic Hessian checked on tangent directions.
    rng=np.random.default_rng(121); K,I,J=2,4,3
    A=rng.uniform(-.4,.4,(K,1,I,J)); B=rng.uniform(-.2,.2,(K,1,J,J));B=(B-B.swapaxes(-1,-2))/2
    rep=AdaptiveGameRepresentation(torch.from_numpy(A),torch.from_numpy(B),uniform_policy(1,J),torch.tensor([.25,.4],dtype=torch.float64))
    p=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([2.4,3.2],dtype=torch.float64);eta=.7
    fun=lambda q: (w*rep.game_values(q)).sum()-(q*(q/uniform_policy(1,I)).log()).sum()/eta
    grad=torch.autograd.grad(fun(p),p,create_graph=True)[0]
    nu,q=rep.opponent_and_gradient(p)
    analytic_grad=torch.einsum('k,kxi->xi',w,q)-(torch.log(p/uniform_policy(1,I))+1)/eta
    hessian=torch.autograd.functional.hessian(fun,p).reshape(I,I)
    ax=torch.from_numpy(A[:,0]);nk=nu[:,0]; qk=q[:,0]
    covariance=torch.einsum('kj,kij,klj->kil',nk,ax,ax)-torch.einsum('ki,kl->kil',qk,qk)
    ah=-torch.einsum('k,kil->il',w/rep.beta,covariance)-torch.diag(1/(eta*p[0]))
    record('adaptive_gradient_and_hessian','PASS' if torch.allclose(grad,analytic_grad,atol=1e-10) and torch.allclose(hessian,ah,atol=1e-10) else 'CHECK',gradient_error=float((grad-analytic_grad).abs().max().detach()),hessian_error=float((hessian-ah).abs().max().detach()))

    # Execute the actual scorer main on one fully observed synthetic prompt.
    with tempfile.TemporaryDirectory(prefix='nbpo_audit_') as d:
        root=Path(d); scorer=import_file(snap/'union_score_panel.py','audit_union_score_panel')
        scorer.UF=root/'uf';scorer.SUB=root/'sub';pd=scorer.UF/'pools/toy/shard0';pd.mkdir(parents=True)
        entries=[dict(prompt_id='p0',role=role,sample_index=i,response_sha256=f'{role}{i}') for role in ['learner','comparator'] for i in range(8)]
        poolpath=pd/'chunk0000.jsonl';poolpath.write_text(''.join(json.dumps(v)+'\n' for v in entries))
        (pd/'chunk0000.manifest.json').write_text(json.dumps({'sha256':hashlib.sha256(poolpath.read_bytes()).hexdigest()}))
        (pd/'settings.json').write_text('{}')
        jd=scorer.SUB/'pool_judgments/toy/shard0';jd.mkdir(parents=True);(jd/'complete.json').write_text('{}')
        verdicts=[]
        for k in range(2):
            for r1,r2,pairs,value in [('learner','learner',list(itertools.combinations(range(8),2)),.5),('learner','comparator',list(itertools.product(range(8),repeat=2)),.75),('comparator','comparator',list(itertools.combinations(range(8),2)),.5)]:
                for i,j in pairs:
                    for order in (0,1):
                        verdicts.append(dict(prompt_id='p0',rubric=f'item{k}',role_i=r1,i=i,role_j=r2,j=j,order=order,value_for_i=value,status='ok'))
        (jd/'chunk0000.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in verdicts))
        argv=sys.argv
        sys.argv=['union_score_panel.py','--objectives','2','--tag','toy','--pool','toy','--pool-shards','1','--judge-shards','1','--shards','1','--out','full']
        with contextlib.redirect_stdout(io.StringIO()): scorer.main()
        actual=np.load(scorer.UF/'scores/full/shard0/chunk0000.npz')
        ap=actual['A_policy']; ar=actual['A_ref']; expected_a=np.full_like(ap,.25);expected_b=np.zeros_like(ar)
        badrep=AdaptiveGameRepresentation(torch.from_numpy(ap),torch.from_numpy(ar),uniform_policy(1,8),torch.full((2,),.25,dtype=torch.float64),reference_construction='independent_samples')
        goodrep=AdaptiveGameRepresentation(torch.from_numpy(expected_a),torch.from_numpy(expected_b),uniform_policy(1,8),torch.full((2,),.25,dtype=torch.float64))
        record('scorer_tensor_role_mismatch','REPRODUCED_FAILURE' if not np.allclose(ap,expected_a) or not np.allclose(ar,expected_b) else 'PASS',A_policy_error=float(np.max(abs(ap-expected_a))),A_ref_error=float(np.max(abs(ar-expected_b))),
            serialized_surplus=badrep.surplus(uniform_policy(1,8)).tolist(),paper_surplus=goodrep.surplus(uniform_policy(1,8)).tolist(),
            example='All learner responses tie one another; all references tie; every learner beats every reference with p=.75.')
        verdicts=[v for v in verdicts if v['role_i']!='comparator']
        (jd/'chunk0000.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in verdicts))
        sys.argv[-1]='missing_rr'
        try:
            with contextlib.redirect_stdout(io.StringIO()): scorer.main()
            report=json.loads((scorer.UF/'scores/missing_rr/score_report.json').read_text())
            record('missing_reference_triangle_retained','REPRODUCED_FAILURE' if report['prompts_complete'] else 'PASS',prompts_complete=report['prompts_complete'],reference_diagnostic=report['reference_disagreement'])
        except SystemExit as exc:
            record('missing_reference_triangle_retained','PASS',rejected=True,message=str(exc))
        finally:
            sys.argv=argv

    # Isolation of Trainer source method: no mock full Trainer run is claimed.
    globals_={'torch':torch,'F':F,'nbpo_regression_target':nbpo_regression_target}
    lossfn=extract_function(repo/'mnpo_scripts/mnpo_trainer.py','mnpo_loss',globals_,'MNPOTrainer')
    fake=SimpleNamespace(accelerator=SimpleNamespace(device=torch.device('cpu')),loss_type='nbpo',eta=2.5,beta=1.,nbpo_target_mode='canonical_logratio')
    pc=torch.tensor([1.,-1.],requires_grad=True);pr=torch.zeros(2);prev=torch.zeros(2);target=torch.zeros(2)
    losses,_,_=lossfn(fake,pc,pr,prev,prev,[(prev,prev)],nbpo_target=target)
    record('trainer_mean_of_squared_residuals','PASS' if losses.mean().item()==1. else 'CHECK',MSE=float(losses.mean().detach()),squared_mean_residual=float(pc.mean().square().detach()),scope='unmodified mnpo_loss AST only')
    target=torch.tensor([.2,.4]); pc=torch.tensor([.2,.4])
    losses,_,_=lossfn(fake,pc,pr,prev,prev,[(prev,prev)],nbpo_target=target)
    record('canonical_target_eta_not_reapplied','PASS' if losses.max().item()==0 else 'CHECK',eta=fake.eta,max_loss=float(losses.max()))
    torch.manual_seed(9);logits=torch.randn(2,7,11,requires_grad=True);labels=torch.tensor([[-100,-100,2,3,4,5,-100],[-100,2,4,7,8,-100,-100]])
    actual=response_logps(logits,labels,False,chunk_size=2)
    labs=labels[:,1:];mask=labs!=-100;safe=labs.masked_fill(~mask,0)
    reference=logits[:,:-1].log_softmax(-1).gather(-1,safe.unsqueeze(-1)).squeeze(-1).masked_fill(~mask,0).sum(-1)
    g1=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0];g2=torch.autograd.grad(reference.sum(),logits)[0]
    record('response_logps_forward_and_backward','PASS' if torch.allclose(actual,reference,atol=2e-6) and torch.allclose(g1,g2,atol=2e-6) else 'CHECK',forward_error=float((actual-reference).abs().max().detach()),gradient_error=float((g1-g2).abs().max()))

    # Demonstrate target drift introduced by the actual quantization function.
    quant=extract_function(snap/'solve_pros4_targets_uw1.py','quantize_canonical_row',{'CANONICAL_DECIMALS':10,'POOL':8})
    row={'target_mode':'canonical_logratio','nbpo_weight_a':1.49e-10,'nbpo_weight_b':.2,'nbpo_center_a':.125,'nbpo_center_b':.125}
    original=math.log(row['nbpo_weight_a']/row['nbpo_center_a'])-math.log(row['nbpo_weight_b']/row['nbpo_center_b'])
    changed=quant(dict(row))
    record('quantized_target_drift','REPRODUCED_FAILURE' if abs(original-changed['nbpo_logratio_target'])>1e-9 else 'PASS',original_target=original,quantized_target=changed['nbpo_logratio_target'],absolute_drift=abs(original-changed['nbpo_logratio_target']),original_mass_above_filter=row['nbpo_weight_a']>=1e-10)

    from scripts.nbpo.eval_game_value import evaluate_game_value
    A=torch.zeros(2,2,3,3,dtype=torch.float64); A[:,0]=.2;A[:,1]=-.1;B=torch.zeros_like(A);beta=torch.full((2,),.25,dtype=torch.float64)
    summary=evaluate_game_value(A,B,beta)
    local=compute_regularized_game_value(compute_margins(A,uniform_policy(2,3)),uniform_policy(2,3),beta,per_prompt=True)
    record('legacy_gate_uses_global_not_local_surplus','REPRODUCED_MISMATCH' if summary['min_surplus']>0 and float(local.min())<0 else 'CHECK',reported_min_surplus=summary['min_surplus'],true_min_local_surplus=float(local.min()),scope='legacy evaluator used by run_nbpo_stage; not the campaign trainer')

    # A narrow stress test of the actual gate source: replace only file-promotion effects.
    effects=[]
    gate=extract_function(repo/'scripts/nbpo/run_nbpo_stage.py','apply_gate',{'checkpoint_fingerprint':lambda p:'fake','promote_candidate':lambda *a: (effects.append(a) or {'versioned_dir':'candidate'}),'Path':Path})
    result=gate(float('nan'),Path('parent'),Path('candidate'),Path('next'),fingerprint='fake')
    record('legacy_gate_nan','REPRODUCED_FAILURE' if result['accepted'] else 'PASS',accepted=result['accepted'],promotion_called=bool(effects),scope='isolated function with promotion side effect replaced; no claim upstream currently emits NaN')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    start=time.time()
    try:run(args.repo.resolve())
    except Exception as e:
        record('audit_runner','ERROR',error=repr(e),traceback=traceback.format_exc())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'elapsed_s':time.time()-start,'repo':str(args.repo.resolve()),'results':RESULTS},indent=2,ensure_ascii=False)+'\n')
    if any(r['status']=='ERROR' for r in RESULTS): sys.exit(2)
