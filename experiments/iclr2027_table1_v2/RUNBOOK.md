# RUNBOOK — ICLR-2027 Table-1 rebuild

Everything here is reproducible from the repository plus the cluster pod. Paths
are absolute where they cross the machine boundary.

## Where things live

| what | where |
|---|---|
| branch | `exp/iclr27-table1-v2` |
| experiment root (local) | `/home/sjkim/MNPO/experiments/iclr2027_table1_v2` |
| campaign root (cluster) | `/work/iclr27_table1_v2` on pod `nbpo-judge`, namespace `p-aipr` |
| code on the cluster | `/work/iclr27_table1_v2/code` (`PYTHONPATH` points here) |
| logs on the cluster | `/work/iclr27_table1_v2/logs` |
| kubeconfig | `~/.kube/aipr-kubeconfig.yaml` |

Shell prelude for every cluster command:

```bash
export KUBECONFIG=~/.kube/aipr-kubeconfig.yaml
K="kubectl -n p-aipr exec nbpo-judge -c main -- bash -lc"
```

## GPUs — read this before launching anything

The brief specified four H200s on pod `nbpo-judge2`. That pod requests **one**
GPU and has never scheduled: both usable nodes in zone `private-h200-aipr-0` are
fully allocated and the zone's other two nodes are cordoned. The work therefore
runs on the **three idle H200s inside the already-running `nbpo-judge` pod**,
which is what the operator chose. Consequences, all deliberate:

* judge phases run **TP=2**, not TP=4 (a 32B in bf16 is ~64 GB, a 70B ~140 GB;
  both fit, and the prior campaign ran the 70B at TP=2 in this same pod);
* at most three single-GPU jobs run at once, so the four-way policy fan-out of
  the brief becomes three-way plus one queued.

`nbpo-judge` also still runs the **v3 campaign's `supervisor.sh`**, which
restarts a lane whenever it finds one down with pending work. It last logged on
2026-09-05 and holds no GPU leases, so it is quiescent — but before a long GPU
phase, confirm it has not woken up:

```bash
$K 'tail -3 /work/v3_20260904/logs/supervisor.log; ls -d /work/v3_20260904/locks/gpu* 2>/dev/null'
```

Its lanes take a lease by `mkdir /work/v3_20260904/locks/gpu<N>`. If it ever
starts competing, take those leases first and `rmdir` them when done — that is
the same mechanism its own lanes use, and it is reversible. Do **not** delete
the pod: it is the operator's, and deleting it also destroys the only free GPU
capacity in the namespace.

Disk: `/work` is a Lustre mount whose `df` reports the whole 80 T filesystem at
100 %. That number is not the PVC quota — a 2 GiB write probe succeeded. Space is
still the tightest resource (existing campaigns hold ~8.5 TB), so checkpoints are
kept only for the fixed final iterate and deleted **after** their hashes and
evaluation artifacts are complete, never before.

## The matched-step protocol (the easiest thing to get wrong)

NBPO's multipliers are raw: `lambda_k ~ 1/s_k`, so `||lambda||_1` is of order
`sum_k 1/s_k` — tens to hundreds, not one. A control solved with weights on the
simplex takes a step two orders of magnitude smaller and loses on step size
alone. So:

1. solve the Nash dual first and keep its lambda **raw** (the non-negotiable);
2. pass its `||lambda||_1` as `--weight-l1` to every other aggregation *of the
   same representation*.

Each representation matches to **its own** Nash norm. The game rows and the BT
rows sit on different scales (centered preferences live in [-1/2, 1/2], z-scored
rewards have unit variance), and forcing one norm across both would be matching
the wrong thing. `--match-weight-l1-to-nash` does step 1 and 2 in one call and
records the number in the artifact.

Measured on the synthetic verification pool: BT-RM-utilitarian at BT-RM-Nash's
norm gives KL 1.006 against Nash's 0.996; at the *game's* norm it gives 1.343 —
a 35 % larger step that would have been read as a method effect.

## Pipeline, in order

```bash
# 0. split (local, CPU, network) -- already built and committed
python scripts/experiments/iclr2027_table1_v2/build_ultrafeedback_split.py \
    --out-dir experiments/iclr2027_table1_v2

# 1. response pool: 4 learner seeds (101-104) + 4 comparator seeds (201-204)
$K 'cd /work/iclr27_table1_v2/code && CUDA_VISIBLE_DEVICES=0 nohup bash smoke_gen.sh \
      > /work/iclr27_table1_v2/logs/smoke_gen.log 2>&1 &'

# 2. judgment bank: Qwen3-32B, TRAINING role, both presentation orders
$K 'cd /work/iclr27_table1_v2/code && CUDA_VISIBLE_DEVICES=0,1 nohup bash smoke_judge.sh \
      > /work/iclr27_table1_v2/logs/smoke_judge.log 2>&1 &'

# 3. tensors
$K 'cd /work/iclr27_table1_v2/code && PYTHONPATH=. python3 -m scripts.nbpo.build_preference_tensor ...'

# 4. solve -- one call per Table-1 row (see below)
# 5. pair targets -> precompute -> train -> decode -> independent evaluation
```

### The six matched solver calls

```bash
T=/work/iclr27_table1_v2/smoke/tensor          # the shared tensor artifact
S=/work/iclr27_table1_v2/smoke/solve
C="python3 -m scripts.nbpo.solve_nbpo_dual --tensor-dir $T --beta 0.25 --eta 1.0 \
     --gamma 0.5 -M 40000 -R 3"

# 2. NBPO                     -- raw lambda, nothing matched to it
$C --out-dir $S/nbpo
# 3. Fixed-reference Nash     -- same tensors, comparator frozen at mu
$C --out-dir $S/fixref     --representation fixed_reference
# 4. BT-RM-Nash               -- same judged pairs, scalar reward heads
$C --out-dir $S/btnash     --representation bt_reward --reward-table $S/rewards.npz
# 5. Game-utilitarian         -- matched to NBPO's norm
$C --out-dir $S/gameutil   --aggregation utilitarian        --match-weight-l1-to-nash
# 6. Game-KS                  -- matched to NBPO's norm
$C --out-dir $S/gameks     --aggregation kalai_smorodinsky  --match-weight-l1-to-nash \
                           --ks-unregularized-diagnostics
# 7. BT-RM-utilitarian        -- matched to BT-RM-Nash's OWN norm
$C --out-dir $S/btutil     --representation bt_reward --reward-table $S/rewards.npz \
                           --aggregation utilitarian --match-weight-l1-to-nash
```

`--legacy-solver` reruns an adaptive-game row through the original
`solve_nbpo_dual` code path. It is bitwise identical (asserted in
`tests/test_nbpo_generic_solver.py`) and exists so a published artifact can be
regenerated by the exact code that produced it.

Every solve writes `solution.json` with the raw weights, `V`, `d`, the raw
surpluses, every residual, and
`target_log_ratio_identity_residual` — the check that
`[log pi* - log pi_t](y) - [log pi* - log pi_t](y') == eta * sum_k w_k (q_k(y) - q_k(y'))`.
If that residual exceeds 1e-9 the solver **refuses to write a usable artifact**,
because the Eq. (26) pair target would not be the target the solve optimized.

Game-KS can legitimately have no answer: if no policy in the proximal family
lifts every objective above its disagreement point, the bargaining set above `d`
is empty and the Nash program is infeasible on the same pool. That writes
`solution_blocked.json` with diagnostics and exits nonzero. It is a property of
the stage, not a bug — do not work around it by clamping surpluses.

## Judge protocols: v2 retired, v3 inadmissible, v4 under test

Read `judge_protocol_v3/PROTOCOL_AMENDMENT_001.md` and
`judge_protocol_v4/PROTOCOL_AMENDMENT_002.md` before interpreting any judge
number. In short:

* **v2** (hard verdict, 16-token budget) — retired. Swap consistency 0.59-0.70.
* **v3 / P2_deliberative** — **inadmissible**. It is the only protocol that
  detects deterministic dishonesty edits (1.000) but its 96-token budget left
  4-21% of pairs with no verdict at all, five times the gate even at
  calibration. Its artifacts are quarantined.
* **v4** — the same protocol with the budget repaired (512 tokens,
  stop-on-marker with the marker retained, one deterministic 1024-token retry),
  plus a balanced-analysis-order variant. Under test.

**A larger token budget is not by itself sufficient.** v3 had two defects; the
budget repair addresses only the first. Nothing may be described as validated
until the fresh holdout and the downstream target-stability gates both pass.

### v4 prompt splits — do not reuse

`splits_judge_v4/` holds three immutable groups cut from prompts used by nothing
else: `judge_v4_dev` (100), `judge_v4_holdout` (200) and
`judge_v4_backup_holdout` (200). **The backup is sealed** — open it only if v4
itself needs revision, so a second attempt does not have to reuse the first
attempt's holdout. Hashes and the six zero-overlap checks are in
`judge_v4_split_manifest.json`.

### Quarantined artifacts

`QUARANTINED_poolsize100_exploratory_v3_invalid_protocol/` on the pod, mirrored
by `quarantine/` in the repo. It may not select 4+4 versus 8+8, enter the paper,
or be combined with v4 measurements. The pool-size comparison restarts from
scratch under a passing judge.

### Running v4

```bash
$K 'cd /work/iclr27_table1_v2/code && setsid nohup bash run_v4_dev.sh \
      > /work/iclr27_table1_v2/logs/v4_dev.log 2>&1 < /dev/null &'
python -m scripts.experiments.iclr2027_table1_v2.analyze_v4 \
   --protocol P2_long=<dev>.jsonl,<controls>.jsonl,<manifest>.json \
   --protocol P2_balanced=<dev>.jsonl,<controls>.jsonl,<manifest>.json \
   --out-dir experiments/iclr2027_table1_v2/judge_protocol_v4 --label development
python -m scripts.experiments.iclr2027_table1_v2.target_stability \
   --results <holdout results>.jsonl --pairs <pairs>.jsonl \
   --out-dir experiments/iclr2027_table1_v2/judge_audit_v4_holdout
```

### A GPU trap that has cost two runs

`nvidia-smi` inside this pod under-reports: it shows 0 MiB while a hung engine
holds 130 GB, and a decode then dies with *"Free memory on device (18.18/139.8
GiB) ... less than desired GPU memory utilization"*. Check for live engines
instead:

```bash
$K 'ps -eo pid,ppid,etime,pcpu,cmd | grep VLLM::EngineCore | grep -v defunct'
```

An engine whose **parent is 1** is orphaned. Three of them (started Sep 6) belong
to the v3 campaign and are left alone. Killing a `run_judge_protocol` parent does
**not** kill its engine child — reclaim that explicitly or the next run has one
fewer GPU.

## The judge audit

```bash
$K 'cd /work/iclr27_table1_v2/code && python3 judge_audit.py \
      --verdicts /work/iclr27_table1_v2/smoke/verdicts.jsonl \
      --out-dir  /work/iclr27_table1_v2/smoke/judge_audit'
```

Exits nonzero when a gate fails, and writes `report.json` + `report.md`. The
gates live in `GATES` at the top of the file so that changing one is a visible
diff rather than a flag someone passed once.

Read `swap_consistent_winner_rate` and nothing else as *the* consistency number:
it is agreement over the pairs **both presentation orders decided**. The two
neighbours are there to stop the wrong one being quoted —
`exact_agreement_rate` counts both-tie agreement and so flatters a judge that
ties everything, and `one_order_tie_rate` is partial agreement rather than
contradiction. `contradiction_rate` is the one that indicts a judge.

Two diagnostics exist for when it fails, and both are on the pod:

```bash
# same pool, same judge, only the rubric changes
$K 'cd /work/iclr27_table1_v2/code && CUDA_VISIBLE_DEVICES=0,1 bash diag_judge.sh'
# same pool, same rubric, only the judge changes (Llama-3.3-70B, TP=2)
$K 'cd /work/iclr27_table1_v2/code && CUDA_VISIBLE_DEVICES=0,1 bash diag_70b.sh'
# compact per-objective swap-consistency table for any bank
$K 'python3 /work/iclr27_table1_v2/code/swapdiag.py <verdicts.jsonl>'
```

**Do not run the v1 rubric at a 16-token judge budget.** It is 86.5 %
unparseable on the first pass and leaves 6684 of 8800 cells invalid after two
retries, at which point the judging CLI refuses to write a matrix with holes in
it. v1 was always run at 512 tokens; v2 is what the 16-token budget was designed
for.

## The launcher

```bash
L="python scripts/experiments/iclr2027_table1_v2/launcher.py"
$L dry-run  --plan experiments/iclr2027_table1_v2/plans/<plan>.json
$L launch   --plan experiments/iclr2027_table1_v2/plans/<plan>.json
$L status
$L resume   --plan experiments/iclr2027_table1_v2/plans/<plan>.json   # same as launch
$L validate                                   # re-hash every completed run's outputs
$L aggregate                                  # registry -> run_registry.csv
$L cancel --run-id <id>
```

`status` re-derives liveness from the PID/job id and the exit sentinel, so a run
whose process died without writing one is reported **failed**, not "running".
`launch` skips a completed run only when its config hash still matches; a changed
config makes a new run rather than overwriting the old artifacts.

## Tests

```bash
python -m pytest tests/ -q                       # whole suite
python -m pytest tests/test_nbpo_ks.py -q        # Kalai-Smorodinsky properties
python -m pytest tests/test_uf_split_v2.py -q    # split disjointness + decontamination
```

The two skips in `tests/test_nbpo_bt_reward.py` are intentional and documented:
at a hard label (`p_hat` 0 or 1) the soft-BT optimum is at infinity, so there is
no finite margin to check the gradient at.
