# Session 5 report — 2026-09-08

Branch `exp/iclr27-table1-v2`, pushed to `nbpo` (promotion-kim/NBPO).
Full test suite: **422 passed, 2 skipped, 0 failed** (was 404 + 2; +18 tests).

Nothing was launched that the gates did not release: no policy training, no
reward-model training, no 7 500-prompt bank, no holdout opened, no seed 42/43/44.

## The three results

**1. The prompted-judge route is closed, and the reason is specific.**
The judge's slot bias is **lexical**, not positional. Under A/B rendering the
physical slot contributes +0.012 (helpfulness) and +0.026 (instruction
following); the letter "A" contributes **+0.237** and **+0.147**. It is strong
enough to override a known answer — deterministic-degradation accuracy in the
factorial's A/B condition is 0.480/0.680/0.980/0.640, and 1.000/1.000/1.000/0.940
on the same controls under opaque identifiers.

Opaque identifiers **relocate** the bias rather than removing it. The total
first-and-"A" advantage is conserved (0.249 → 0.236, 0.173 → 0.153) as the mass
moves from the lexical channel to the physical one (+0.168, +0.158). Both schemes
fail confident swap (0.448–0.750 vs 0.85) and order split-half Spearman
(0.008–0.465 vs 0.75). **Decision B**, applied without adjustment.

**2. The controlled-nontransitivity negative was a solver failure.**
`rho* > 0` on all 25 instances, certified to ≤ 3.1e-7, so every instance is
inside Assumption 1. The exact solutions hold **87–94 %** of `rho*` at every
alpha. The deployed `R`-step map reaches **−0.185** at alpha = 1 with a
fixed-point residual of **1.000** and a weight norm of 608 — it is oscillating,
not converging. Fixed-reference Nash and BT-RM-Nash have residual **exactly 0**
by construction, so the comparison was between one diverging iteration and two
that cannot diverge.

The subproblem is concave *and separates across prompts*, so it is X small
independent programs. Solved that way, the same outer loop reaches **+0.014** at
alpha = 1 with residual 1.9e-6, within TV 0.027 of the exact proximal policy, on
a twelfth of the outer budget.

At a **constant** feasible margin (`rho*` held at 0.030000, sd 6e-8, while BT
deviance rises 0.0003 → 0.350) the exact solutions are **flat** — 0.920, 0.919,
0.913, 0.914, 0.918 of `rho*` — and only the solver collapses.

"Roughly two thirds is step size" is **retracted**: 2.2 %, 78.1 %, 82.8 %,
46.9 %, 23.8 % across alpha, and the decomposition is void anyway.

**3. SafeRLHF cannot carry a cyclicity claim; it can carry a conflict claim.**
The released comparison graph is a near-perfect matching — 73 906 distinct edges
over 143 668 responses, 1 369 of degree ≥ 2, **zero triangles**, verified under
two definitions of response identity. Observable three-cycles are 0 by
construction of the annotation protocol.

A matched-budget anti-symmetric GPM (126.5M vs BT's 126.4M parameters, 419 s vs
420 s) beats a constant predictor decisively and beats BT by **nothing reliable**:
run 1's one significant result (helpfulness ΔNLL −0.0039 [−0.0073, −0.0005]) did
not replicate in run 2 (−0.0024 [−0.0059, +0.0012]).

The sharper measurement: the GPM's cyclic logit residual on **real** held-out
encodings is **2.414** against BT's exactly 0 — it is genuinely free to be
intransitive on these inputs — and it predicts **0 cycles in 3 392 triples**. A
model that can be intransitive learns a transitive preference here. That is a
statement about the model on this data, not about people.

What SafeRLHF does carry: **24.33 %** of held-out rows prefer different responses
on helpfulness and on harmlessness, from independent human judgements of the same
pair.

## Section 4 go/no-go: NO-GO

Conditions 1–3 pass (feasibility positive; exact NBPO correct; a practical solver
at residual < 1e-4 and close to the exact proximal policy). Condition 4 passes
only on its weaker clause: the GPM clears the constant-predictor floor, but the
clause that decides the route — a benefit over BT on an observable nontransitive
subset — **cannot be evaluated**, because that subset is empty.

## Where to read the detail

| document | contents |
|---|---|
| `judge_protocol_v4/DECISION_V4_FINAL.md` | v4 selected-but-inadmissible, and the control-pairing bug |
| `judge_protocol_v5/DECISION_V5_FINAL.md` | the opaque-ID factorial and Decision B |
| `NONTRANSITIVITY_AUDIT.md` | `rho*`, the certificate, the four solvers, the retraction, v2 |
| `SAFERLHF_SUPERVISION.md` | splits, the zero-triangle finding, the conflict rate |
| `GPM_VS_BT.md` | matched-budget comparison, exactness, observed vs predicted cycles |
| `GO_NO_GO.md` | the four conditions, and what would change the answer |
| `TABLE_CONTRACT.md` | what is fillable, what is blocked, what was removed |
| `RUNBOOK.md` | exact resume commands for all three tracks |
