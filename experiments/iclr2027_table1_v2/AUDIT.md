# ICLR-2027 Table-1 rebuild — initial audit

Written before any code change, from commands run on 2026-09-07 (KST).
Everything below is measured, not assumed.

## Source

| item | value |
|---|---|
| repository | `/home/sjkim/MNPO` (checkout of `promotion-kim/NBPO`) |
| branch created | `exp/iclr27-table1-v2` |
| base commit | `2b4c48cac19bf3509f3a9df24745885f7de90096` — *Split both responses at one prompt boundary, and name the real nu source* |
| remotes | `nbpo` → `https://github.com/promotion-kim/NBPO.git`; `origin` → `https://github.com/smiles724/MNPO.git` |
| active manuscript | `nbpo_iclr/main_v5.tex` (untracked; `main.tex` … `main_v4.tex` also present) |

### Working tree at branch creation — preserved verbatim

`git status --short` reported **887 deletions, 61 untracked, 11 modified**, all of
them the operator's uncommitted work. Nothing was reset, cleaned, stashed or
restored; `git checkout -b` leaves the tree byte-identical and the same counts
were re-measured after the branch was created.

The uncommitted change is a directory rename in progress — `game_nbpo_iclr/`
deleted, `nbpo_iclr/` created with `main_v2..v5.tex` untracked — plus the removal
of ~880 old result artifacts under `nbpo_iclr/results/` and `analysis/*`. New
work is therefore committed **file by file**, never with `git commit -a`.

## Hardware — deviates from the brief, and this is a blocker

The brief says "four NVIDIA H200 GPUs" on this machine. Measured:

**Local machine** — 3 × NVIDIA A100 80GB PCIe (not H200, not four), driver
570.211.01, CUDA 12.8. All three are **fully occupied by another user**
(`/home/tsyou/anaconda3/envs/polyedit`, 68 GB each of 80 GB, 96–99 % utilization).
Unusable for this project.

**MLXP cluster** — kubeconfig `~/.kube/aipr-kubeconfig.yaml`, namespace `p-aipr`,
context `…@mlx-kpb4r/p-aipr`. The operator directed this work at pod
`nbpo-judge2`:

| pod | status | GPUs | note |
|---|---|---|---|
| `nbpo-judge2` | **Pending, 96 min, never scheduled** | requests **1**, not 4 | `FailedScheduling` ×3704: *"0/317 nodes are available: 2 Insufficient nvidia.com/gpu, 2 node(s) were unschedulable, 313 didn't match node affinity"* |
| `nbpo-judge` | Running, 3 d | **3 × H200 143 GB**, **all idle: 0 MiB used, 0 % util, no GPU processes** | holds the zone's free GPUs; three `VLLM::EngineCore` processes spin at 99 % CPU with no GPU memory (hung), plus a `supervisor.sh` chain from the finished `/work/v3_20260904` campaign |

So the four requested GPUs do not exist in the reachable zone: it has 4 matching
nodes, 2 cordoned and 2 with no free GPU, and the 3 H200s that are allocated sit
in `nbpo-judge` doing nothing. `nbpo-judge2` cannot schedule until they are
released. Node listing is forbidden to this service account, so the totals above
come from the scheduler's own message. RBAC does allow `create`/`patch`/`delete`
on pods in `p-aipr`.

**This is the one genuine blocker to the GPU phases.** Freeing `nbpo-judge` would
kill a running (if idle) pod belonging to the operator, so it is not done
unilaterally. Everything that does not need a GPU proceeds meanwhile.

## Storage

| mount | size | avail | note |
|---|---|---|---|
| `/` (`/dev/nvme1n1p3`) | 3.4 T | **26 G — 100 % full** | unusable for artifacts |
| `/home` (NFS) | 6.7 T | 516 G | repo and HF cache live here |
| `/ext_hdd2` | 7.3 T | 550 G | |
| cluster PVC `sjkim` mounted at `/work` in the pods | — | — | campaign roots (`/work/v3_20260904`) |

## Scheduler

**No Slurm** — `sbatch`, `squeue`, `sinfo`, `srun`, `scontrol` all absent and
`/etc/slurm` does not exist. Execution is direct shell plus `tmux` locally, and
`kubectl exec` into MLXP pods. The launcher therefore targets explicit
`CUDA_VISIBLE_DEVICES` + PID files, and detects Slurm at run time only to stay
portable.

## Software

Tests and CPU solvers run under the base interpreter; the training and inference
environments are separate and older.

| env | python | torch | transformers | vllm | accelerate | trl | datasets | CUDA / NCCL |
|---|---|---|---|---|---|---|---|---|
| base (`~/anaconda3`) | 3.12.2 | 2.7.1+cu126 | — | — | — | 3.6.0 | — |
| `mnpo_train` | 3.10.20 | 2.3.0+cu121 | 4.44.2 | — | 0.29.2 | 0.9.6 | 2.18.0 | 12.1 / (2,20,5) |
| `mnpo_infer` | 3.10.20 | 2.3.0+cu121 | 4.53.3 | 0.5.1 | — | — | 2.18.0 | 12.1 / (2,20,5) |
| `vllm` | 3.10.20 | 2.10.0+cu128 | 5.7.0 | 0.19.1 | 1.13.0 | — | 4.8.5 | 12.8 / (2,27,5) |

`scipy`, `cvxpy` and `datasketch` are absent from **every** environment. The
consequences are recorded rather than worked around:

* the finite-pool and Kalai–Smorodinsky solvers are pure `torch` float64 and add
  no dependency;
* the KS verification uses an **independent brute-force grid search** over the
  weight simplex rather than CVXPY — which the brief permits only in CPU toy
  tests anyway, and which is not installed;
* MinHash for decontamination is implemented in-repo (character 13-grams, 128
  permutations, 32 bands) with an exact-Jaccard recomputation on every LSH
  candidate, so the LSH only ever proposes.

## Models and data

Local HF cache `~/.cache/huggingface/hub` holds
`meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-8B`, `roberta-base` and a few
small models — **none of the checkpoints this protocol needs**
(`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-32B`,
`meta-llama/Llama-3.3-70B-Instruct`, `microsoft/phi-4`). The prior campaign kept
them on the cluster PVC, which is where the real runs must read them from.
`huggingface.co` is reachable and a token is present.

`datasets--HuggingFaceH4--ultrafeedback_binarized` exists in the cache but holds
only `README.md`; the split builder re-fetches it and records the fingerprint.

## Baseline test state

`python -m pytest tests/ -q` on the base commit: **219 passed, 0 failed** in
144.8 s. That is the number every later run is compared against.

## Decisions taken here, recorded rather than asked about

1. **Branch, not worktree.** `/` is 100 % full; a second worktree of a repo this
   size is a needless copy. A branch leaves the dirty tree exactly where it is.
2. **Commit file by file.** With 887 uncommitted deletions in the tree, `-a`
   would sweep the operator's in-progress rename into a code commit.
3. **CPU-first ordering.** Since the GPU blocker is external, the whole
   implementation, test, split, decontamination and launcher programme runs
   first, so that the moment GPUs free up the pipeline is ready to launch rather
   than starting from scratch.
