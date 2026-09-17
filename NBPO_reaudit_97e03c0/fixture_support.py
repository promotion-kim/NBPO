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

