#!/usr/bin/env python3
"""Append a fixed, audit-ready P10 correction note."""

from pathlib import Path


def main() -> None:
    path = Path("/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p10_saferlhf_training_seed43_20260718/fix_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = ["""## 2026-07-18T09:50:00+09:00 — RONPO-OS smoke status false-negative

The corrected RONPO-OS 20-step smoke returned code 0 with finite metrics, but its append-only arm log retained a traceback from the earlier no-history attempt. The status checker searched the entire appended log and marked the corrected smoke failed, so the sequential wrapper stopped before the full run. Original logs and the failed status are preserved. `train_stage1_arm.py` now writes one immutable log per invocation and checks only that attempt log. The successful smoke remains evidence; the full run will be launched separately after read-only GPU samples.

""", """## 2026-07-18T10:06:00+09:00 — queued Stage-1 wave launcher permission error

The first queued-wave launcher failed before any training process started because the newly synced shell script did not have its executable bit. The three launcher logs are preserved. The script will be marked executable and the same locked jobs re-queued; their own three read-only idle samples occur immediately before launch.

""", """## 2026-07-18T10:18:00+09:00 — Stage-2 parent-status compatibility repair

The legacy IPO, DPO, and INPO Stage-1 launchers can retain a false failed status because their shared append-only logs include an earlier, corrected attempt. Their original status JSON and complete logs are preserved, while `recover_legacy_full_status.py` creates a separate evidence-backed `job_status_repaired.json` only when the final training invocation has code 0, finite metrics, and no terminal error. `wait_for_idle_then_prepare_stage2.sh` now recognizes that repaired evidence using the same status order as the fixed-panel evaluator. This changes scheduling only; it does not alter a training result, a metric, or a checkpoint.

""", """## 2026-07-18T10:27:00+09:00 — Stage-2 GPU-pair queue deconfliction

Two P10-owned DPO and SimPO pool wait-queues were stopped before they acquired a GPU. Their trigger was an IPO pool completion, which could have raced an IPO Stage-2 training queue for the same two GPUs. No pool build or training process had started. The later pools will be re-queued after the preceding Stage-2 training releases the assigned GPU, preserving one process per GPU and the stage-matched protocol.

""", """## 2026-07-18T10:32:00+09:00 — evaluation queue executable-bit repair

Three IPO/DPO/SimPO fixed-panel evaluation queues exited before any decode because the helper had not been marked executable on the second container. Their launcher outputs were redirected and remain empty; no model process was started. The helper is now executable and the identical locked-panel queues are relaunched through `bash`, retaining the original manifest, decode configuration, and gate thresholds.

""", """## 2026-07-18T10:46:00+09:00 — Stage-2 runner executable-bit repair

The RONPO-OS Stage-2 queue verified its completed parent, prepared target dataset, and three idle samples, then exited before a smoke run because `run_stage2_arm_sequence.sh` lacked its executable bit. No Stage-2 training process or checkpoint was created. The completed pool and all preparation logs remain unchanged. The runner is marked executable and the same locked OS queue will be re-registered; its independent idle checks run again before launch.

""", """## 2026-07-18T10:47:00+09:00 — stale Stage-2 pool completion marker quarantined

While investigating the launcher repair, a `PREPARED` marker and target dataset timestamped before a still-running P10-owned pool scorer were found in the same OS pool directory. That marker could have allowed training to read an earlier pool attempt while the current scorer rewrote shards. Only the marker is removed; all pool files and logs are preserved. The running scorer is allowed to finish, and only its final `PREPARED` marker plus an artifact-count audit will permit the re-queued OS Stage-2 smoke.

""", """## 2026-07-18T10:50:00+09:00 — stale-marker Stage-2 OS launch stopped and audit strengthened

The re-registered OS queue observed a second stale `PREPARED` marker before the active P10-owned OS pool builder had finished and started its full Stage-2 run. The exact P10 OS process tree was verified by command line and terminated before it could be considered for evaluation or upload. Its partial configuration and logs are retained under `audit/stage2_ronpo_os_stale_pool_attempt/`; it is invalid evidence. The pool audit now fail-closes if any required artifact is newer than `PREPARED`, and queues invalidate a stale `POOL_AUDITED.json` before launch. The active pool builder is not modified and will emit a fresh marker when it completes.

""", """## 2026-07-18T13:19:00+09:00 — P10 Stage-2 evaluation launcher directory repair

The first five-arm Stage-2 evaluation launch attempt failed before any decode process started because `logs/stage2_eval/` did not yet exist for shell redirection. No generation, score, gate, model, or training artifact was created or modified. The directory was created, three fresh read-only idle samples were taken on both authorized containers, and the identical frozen 1,000-prompt decode-and-gate commands were re-launched.

""", """## 2026-07-18T13:22:00+09:00 — HT-MNPO Stage-2 pool vLLM reservation repair

The queued HT-MNPO harmless Stage-2 pool stopped during parent-response decode. The logs show vLLM engine initialization/generation OOMs caused by a 0.88 GPU-cache reservation while two parent engines were initializing, not a training loss, target, seed, decode sample, or reward result. The pool has no `PREPARED` marker and cannot be used. The P10 pool queue now sets only the vLLM cache reservation to 0.55, leaving temperature, top-p, seeds, max tokens, models, pairs, optimization recipe, and scoring unchanged. The shared pool helper keeps its 0.88 default for every non-P10 invocation. The failed logs remain preserved; the same P10 HT-MNPO pool will be relaunched after the current Stage-2 evaluation frees the authorized GPUs.

""", """## 2026-07-18T13:25:00+09:00 — P10 Stage-2 evaluation-root reconciliation and duplicate OS decode

The P10 Stage-2 decode helper writes to `stage2_eval_p8_locked_panel`, while the newly prepared score scripts initially expected `stage2_eval`. All five original frozen decodes and corrected-gate JSONs were already complete in the former directory. Before recognizing the root mismatch, one diagnostic `bash -x` invocation re-decoded RONPO-OS with the identical locked checkpoint, P8 manifest, seed, sampling settings, and gate; it completed before any reward score was read. This is a duplicate measurement on an already-open, non-fresh P8 panel, not a new split or a model-selection event. Its output overwrote the same OS generation path and is retained with the debug log. The scorer and aggregation scripts now consume the existing `stage2_eval_p8_locked_panel` artifacts; no further decode will be launched for this evaluation.

""", """## 2026-07-18T13:45:00+09:00 — Stage-2 evaluation display-label correction

The shared aggregation helper’s default labels describe historical Stage-1 inputs. P10 passed Stage-2 model IDs without a suffix, which would have displayed correctly measured Stage-2 rows as “prior Stage-1.” An optional JSON display-name map now changes only report labels to accurate Stage-2 names. It does not alter any response, scorer output, normalization, bootstrap sample, ranking, or comparison.

""", """## 2026-07-18T13:38:00+09:00 — RONPO top-mass target-column lock correction

The P10 Stage-2 lock fixes the top-mass arm to `target_topmass_k0p1`. A newly registered wait queue used `target_topmass_k0p05`; it was still waiting for its pool and had not launched a smoke or full training process. That P10-owned queue was stopped, its pre-start log preserved under a distinct name, and a replacement queue was registered with the locked `target_topmass_k0p1` value. No checkpoint, W&B run, reward score, or gate artifact was created by the incorrect queue.

"""]
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            if entry not in existing:
                handle.write(entry)


if __name__ == "__main__":
    main()
