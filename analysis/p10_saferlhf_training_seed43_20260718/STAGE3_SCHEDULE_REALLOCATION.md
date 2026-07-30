# Stage-3 GPU schedule reallocation

## 2026-07-18 15:05 KST

Three already-running, authorized MMLU evaluations occupy B200 host-1 GPU 1 and
host-2 GPUs 0--1. They are not preempted. To avoid leaving the remaining B200
capacity idle while those read-only evaluations finish, the pending seed-43
`inpo_avg` and `ht_mnpo_helpfulness` Stage-3 queues move from host-1 GPU 1 to
host-1 GPU 3 after its already queued `simpo` run completes.

This is a scheduling-only change. The parent checkpoints, data, target columns,
900-step budget, seed, optimizer recipe, stability gate, W&B configuration, and
all outcome-blind locks remain unchanged. The previous GPU-1 queues had not
started a pool or training process when they were replaced.

## 2026-07-18 15:20 KST

The host-1 MMLU worker released GPU 1. Three read-only samples showed no compute
process and zero allocated memory. The still-unstarted `inpo_avg` Stage-3 pool,
training, and stability-gate queues therefore move back from GPU 3 to GPU 1.
This prevents an authorized B200 from idling and shortens the critical path.
The later `ht_mnpo_helpfulness` queue remains on GPU 3. No experimental setting
or artifact was changed.
