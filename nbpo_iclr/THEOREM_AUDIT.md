# Game-NBPO Theorem Audit

This document records the mathematical checks applied while revising the manuscript. It is not a substitute for independent author verification or formal peer review.

| Claim | Status in revision | Main assumptions / scope | Audit note |
|---|---|---|---|
| Closed-form soft adversary | Included and checked | Finite response set; \(\mu\) has full support; \(\beta_k>0\) | Gibbs minimizer follows from the KL-regularized linear inner problem. |
| Concavity and continuity of \(V_{k,\beta_k}\) | Included and checked | Finite policy simplex | Pointwise infimum of affine functions is concave; compactness gives continuity. |
| Danskin gradient | Included and checked | Unique KL-regularized inner minimizer | Gradient is the payoff against the soft adversary. |
| Anchor approximation \(0\le a_k-V_{k,\beta_k}\le1/(8\beta_k)\) | Added and checked | Payoffs in \([-1/2,1/2]\) | Jensen gives the lower inequality; Hoeffding's lemma gives the constant. |
| Uniform surplus error \(\le1/(8\beta_k)\) | Added and checked | Same as above; skew-symmetric oracle | Uses \(a_k(\mu)=0\) and \(V_{k,\beta_k}(\mu)\le0\). |
| \(\beta\to\infty\) optimizer consistency | Added and checked | Jointly positive anchor surplus; compact finite policy space | Uniform surplus convergence and strict positivity around anchor-optimal solutions yield argmax consistency. |
| \(\beta\downarrow0\) security limit | Included and checked | Finite response set; full support | Standard soft-min limit. |
| Large-\(\beta\) variance correction | Added | Fixed finite response set; expansion away from degeneracies | Cumulant expansion of the log moment-generating function. Authors should retain the stated asymptotic scope. |
| Convex comprehensive game-value hull | Included and checked | Criterion game values concave; policy randomization allowed | Concavity ensures a mixed policy weakly dominates the convex combination of guaranteed utility vectors. |
| Strict individual rationality | Included and checked | Joint positive game-value surplus | Log welfare tends to \(-\infty\) at the positive-domain boundary. |
| Pareto optimality of every Nash optimizer | Included and checked | Same | Strict monotonicity of log surplus. Claim is only in game-value coordinates. |
| Unique optimal utility vector | Included and checked | Convex comprehensive hull | Strict concavity in the surplus vector; policy itself can remain nonunique. |
| Proportional fairness | Included and checked | Convex surplus set; positive Nash solution | First-order optimality of \(\sum_k\log s_k\). |
| Positive-affine utility covariance | Included and checked | Criterion-wise \(c_k>0\) | Transformed objective differs by a policy-independent constant. |
| Raw-payoff scaling identity | Added and checked | \(\lambda_k>0\) | \(V_{\beta}^{\lambda A}=\lambda V_{\beta/\lambda}^{A}\). Fixed-temperature label scaling is therefore not exact invariance. |
| Utilitarian counterexample | Checked | Finite skew-symmetric pairwise games | Joint improvement exists while equal-weight utilitarian selection can put one criterion below disagreement. |
| Absolute-maxmin counterexample | Checked | Finite skew-symmetric pairwise games | Absolute levels can override fallback-relative improvement. |
| Surplus-maxmin dominated tie | Replaced and checked | Finite skew-symmetric pairwise games | The inherited construction was invalid; the revision uses a valid explicit construction. |
| Regression recovers mirror step | Included and checked | Full-support current policy; exact population squared loss | Conditional-mean decomposition plus equality of pairwise log differences gives uniqueness. |
| Relative-smoothness constant | Included and checked | Iterates in \(\Pi_\varepsilon\); \(\beta_k>0\) | Curvature combines the soft-value Hessian and the log-surplus outer Hessian. |
| Exact last-iterate \(O(1/T)\) welfare convergence | Included and checked | Exact mirror steps; positive-surplus region; suitable step size; comparator optimum in region | Telescoping mirror-descent inequality plus monotonic gaps. |
| Utility-vector rate | Included and checked | Unique utility vector; bounded positive surpluses | Strong concavity of log welfare over the attainable surplus range. |
| Inexact last-iterate bound | Added and checked | Bounded gradient error, mirror residual, monotonic acceptance | Error terms accumulate explicitly. Without monotonic acceptance, only a best-iterate statement is retained. |
| Nested finite-sample game-value intervals | Added and checked conservatively | Independent or sealed preallocated checks; bounded pairwise labels; finite predeclared number of checks | Separates pairwise-label error, comparator Monte Carlo error, and prompt sampling error; uses a simultaneous union bound. |
| Surplus confidence interval | Added and checked | Valid intervals for candidate and reference game values | Lower candidate minus upper disagreement; upper candidate minus lower disagreement. |

## Claims intentionally not made

- No theorem says that Game-NBPO is normatively superior to every aggregation rule.
- No theorem says that improvement in game value is a per-prompt safety guarantee.
- No theorem says that every pairwise comparison improves.
- No finite-temperature theorem gives exact invariance to coin-flip label contamination when \(\beta_k\) is held fixed.
- No neural-policy theorem hides adversary, regression, or finite-sample error.
- No empirical claim is made for finite-temperature Game-NBPO before the matched runs are completed.

## Author verification priorities

1. Re-derive the anchor approximation and optimizer-consistency proof independently.
2. Check the nested confidence constants against the exact implementation and sampling dependence.
3. Match the implementation's regression residual to the variational residual used in the theorem.
4. Confirm that all baseline definitions, especially PROSPER/MaxEntBW and the two maxmin variants, match their cited sources.
5. Regenerate every finite-game figure and numerical coordinate from code committed with the submission.
6. Verify every citation and update any 2026 papers from preprint to final proceedings metadata when available.
