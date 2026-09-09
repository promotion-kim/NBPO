# NBPO overnight repair, 2026-09-09

Reading order.

| file | what it answers |
|---|---|
| `morning_report.md` | Everything, in the order it happened. Start here. |
| `audit_findings.json` | The eight code defects: status, code pointer, the test that holds each, and what was measured about its scope in the released artifacts. |
| `correctness_test_report.md` | What the tests cover, CPU unit tests kept separate from real-8B GPU probes, plus what the external evaluators do and do not guarantee. |
| `blind_audit_alpaca100.json` | Why the pre-repair checkpoints lost usefulness: blinded read of 100 Alpaca prompts, and what the teacher actually prefers. |
| `aggregation_control.json` | Why the Nash-versus-utilitarian arm was prepared and not run. |
| `v6_revision_memo.md` | Every change made to `nbpo_iclr/main_v6.tex`, and what still waits. |
| `artifacts/` | The run artifacts every number is read from, pulled off the cluster. |

The run itself lives at `/work/nbpo_repair_20260909` on pod `nbpo-judge`
(namespace `p-aipr`), with `result_index.json` there listing every backing file
with its sha256.

Code is under `scripts/experiments/nbpo_repair_20260909/`. The source that
actually produced the arms is the commit that snapshots the cluster tree; the
controllers, recovery scripts and table generators written to supervise the run
are separate commits after it.

**Paid judge API calls: 0.**
