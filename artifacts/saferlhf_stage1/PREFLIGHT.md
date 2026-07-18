# Preflight: scorer calibration and shared-pool trade-off

**Status: PASS.** This report is rendered only from the four JSON artifacts cited below. The human-label calibration predates every preference-training run in this experiment; no trained-policy reward outcome was consulted for this gate.

## Scorer calibration on PKU dual-preference conflict rows

- Beaver cost, reported as harmlessness (-cost), selected the human `safer_response_id` on 482/696 rows: **69.25%** (threshold 65.00%).
- Beaver reward selected the human `better_response_id` on 429/696 rows: **61.64%** (threshold 60.00%).
- The heads are pinned to distinct revisions: reward `375cd6a9f0d7e339d2199b05ba129a4a8906596d` and cost `c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28`. Their same-response Spearman correlation is -0.4528.

Non-conflict rows are descriptive only: Beaver cost safer-response accuracy was 65.73% (1072/1631); Beaver reward better-response accuracy was 52.79% (861/1631).

## Trade-off in the shared base response pool

The 2,500-prompt shared pool contains four base responses per prompt. Median within-prompt reward/cost Spearman is 0.3333 (gate ≤ 0.5); the mean is 0.1853. The reward argmax and harmlessness argmax differ for 1488/2500 prompts (59.52%).
Mean within-prompt ranges are 6.4367 for helpfulness and 5.7106 for harmlessness. 20 prompts have a constant objective and therefore undefined Spearman; they remain in the shared data and are not silently dropped.

## Data provenance and limitation

Training uses 2500 conflict prompts (SHA-256 `6296a9efa506e6b5fde1786f6c6c58d9df2d35e913f61e2f8bf671bbadad5f39`). The prompt-disjoint test-conflict validation panel has only 49 rows (SHA-256 `8b11f4c7afc9fe6d1cc978e0c4ab63a0b8b2a257911e51266309aded4246e843`), because 646 of 696 raw test conflict rows overlap earlier held-out manifests. This is an explicit power limitation, not a reason to pad or replace the panel.

## Source artifacts

- conflict calibration: `results/p4_8b_saferlhf_kappa_imbalance_20260717/preflight/calibration_summary.json`
- non-conflict calibration: `results/p4_8b_saferlhf_kappa_imbalance_20260717/preflight/nonconflict_calibration_summary.json`
- shared-pool trade-off gate: `results/p4_8b_saferlhf_table4_20260717/tradeoff_pool_gate.json`
- dataset manifest: `results/p4_8b_saferlhf_table4_20260717/dataset_manifest/data_manifest.json`
