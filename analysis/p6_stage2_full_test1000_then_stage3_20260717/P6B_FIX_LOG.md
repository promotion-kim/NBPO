# P6b fix log

## 2026-07-17T21:07+09:00: shared-filesystem executable-bit failure

The first decode queue attempted to execute
`analysis/p6_stage2_full_test1000_then_stage3_20260717/decode_and_gate_stage2_model.sh`
directly from the shared Lustre project path.  Every queue exited before vLLM
started with `Permission denied`; GPU memory remained at zero and no generation
artifact was created.  The shared mount permits reading the script but rejects
direct executable invocation.

Fix: invoke the unchanged script through `bash <script>`, preserving the
locked P6b manifest, model map, decode configuration, and stability gate.  The
original queue logs are retained as evidence.
