# main_v6.tex revision memo — 2026-09-09

## Applied

1. **Empirical importance distribution (§4.2).** `p_t` was called "the current
   learner mass" without saying what kind of object it is. It is now defined as
   the pool's occurrence measure, uniform at $1/n_x$ with duplicates kept, and
   explicitly distinguished from the model's conditional law, with the pool
   mapping named as the only bridge. The paragraph states that reading
   $\mathrm{softmax}_i\log\pi_\theta$ as the pool distribution reverses measured
   surpluses, because that error cost this project a wrong conclusion once.
2. **Scope of the boxed dual certificate (A.5).** A new paragraph says the
   reported residual is that of the *box-constrained* dual and certifies the
   original problem only when no bound is active; bound activity is now reported
   with every residual, and a run with an active bound is called uncertified
   rather than converged. The SafeRLHF multipliers $\lambda=(6.33,6.19)$ in a
   $[10^{-3},10^{3}]^2$ box are interior by at least two decades.
3. **One geometry in the convergence proof (A.6).** $h$ carried the prompt
   expectation and was then paired with $\langle\cdot,\cdot\rangle_\D$, counting
   $\D$ twice. $h$ is now the prompt-wise negative entropy and the expectation
   enters once, through the pairing. Monotonicity was already stated for $t\ge1$.
4. **Second projection (Eq. 28).** Weighted behaviour cloning on the same
   $p^\star$, defined next to Eq. (26), framed as an empirical forward-KL
   approximation on the sampled support that inherits none of the finite-pool
   exactness. The per-pair identity that makes it share one dataloader with the
   regression is stated.
5. **Regression gate is a diagnostic (§5.3).** Rewritten to separate four things
   that are easy to conflate — solver self-consistency, fit on training pairs,
   transfer to unseen prompts, and downstream usefulness — and to say that no
   checkpoint, horizon or hyperparameter is selected by the regression numbers.
   Safety gains arriving with shorter answers are called a trade-off, not a
   Pareto improvement.
6. **Table 13 caption.** The two false sentences ("No arm meets either", "every
   arm sits below it") are gone; verified absent from the current source.

Compiles at 24 pages, 0 undefined references, 0 overfull boxes.

## Applied after the arms finished

7. **Abstract and third contribution bullet.** Both said the neural realization
   "remains open". They now say it is realizable with the implementation
   corrected, quote Pearson $0.52$ and $69\%$ sign agreement on held-out prompts,
   and name the remaining error as magnitude rather than direction.
8. **New main-body table `tab:repaired-realization`**, generated from the arms'
   own `trainer_state.json` with an `nMSE$^\ast$` column that separates what the
   projection learned about the target's direction from how far it then moved.
   The pre-repair diagnostics table keeps its place and its caption now names
   its scope and points here.
9. **`tab:saferlhf-main` replaced.** It asked for win rates over three seeds
   against six aggregation baselines and every cell was pending. It now reports
   the objective-wise surplus the run measured on $1{,}000$ held-out prompts,
   with the finite-pool target as the ceiling and the two declared short-horizon
   arms pending. Its caption declines the Algorithm-1 acceptance certificate on
   the same grounds the run's own record declines it.
10. **`tab:general-capability` filled**, generated from the scoring summaries,
    including the reference-win columns that invert the training teacher's
    ordering, and a median-response-tokens column.
11. **"Three evaluators, three orderings"** added to §5.3: the training teacher,
    the official benchmarks and an independent scalar reward model rank the two
    projections three different ways, which is stated as part of the result.

Compiles at 26 pages, 0 undefined references, 0 overfull boxes.

## Still pending

12. The two short-horizon rows in both result tables, and their generation and
    scoring, are running.
13. Held-out pool-drift diagnostics with prompt-clustered bootstrap intervals for
    all four checkpoints, with the reference against itself as the zero control.
14. Aggregation-rule controls (fixed-reference Nash, BT-RM--Nash,
    game-utilitarian, game-KS) and additional training seeds. Named in the
    SafeRLHF caption as next steps rather than implied.

## Earlier pending items, now resolved above

7. **Abstract and third contribution bullet.** Both currently say the neural
   realization "remains open" and that we "report its realization diagnostics
   rather than a passing result". At 250 updates the repaired MSE arm meets all
   four pre-registered regression criteria at once. The sentence must state the
   horizon it holds at, and must not be rewritten until the pre-registered
   1750-update result is in, so that both are reported.
8. **New main-body table `tab:repaired-realization`.** Generated from the arms'
   own `trainer_state.json` by
   `scripts/experiments/nbpo_repair_20260909/export_neural_realization_table.py`;
   nothing typed by hand. The appendix's pre-repair diagnostics tables stay as
   the record of what the old pipeline did.
9. **`tab:general-capability`.** Every cell is `\pending`, and its columns
   (AlpacaEval-2 LC, TruthfulQA MC2) are not what this protocol measures. It
   needs the API-free column set actually being run — IFEval strict, GSM8K EM
   with parse-failure rate, HarmBench harmful rate, XSTest, local reward-model
   reference win on the AlpacaEval and Arena-Hard prompt sets, and median
   response tokens — with the local proxy never named as an official score.
10. **Response length as a reported column.** The blinded audit shows the
    pre-repair failure is compression, not refusal or truncation, and that the
    teacher did not ask for it. Length belongs in the capability table rather
    than in prose.
