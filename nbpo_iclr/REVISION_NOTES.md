# Game-NBPO ICLR 2027 Revision Notes

## Revision objective

This revision preserves the fixed-reference NBPO manuscript wherever possible while repairing the methodological mismatch between the paper's general-pairwise motivation and its former fixed-anchor utility. The resulting main method is **Game-Theoretic Nash Bargaining Preference Optimization (Game-NBPO)**.

The manuscript is theory-complete but **experiment-pending**: all finite-temperature LLM result cells remain explicitly marked `TBD`. Existing fixed-anchor experiments are retained only as an appendix audit trail and are not used as evidence for finite-temperature Game-NBPO.

## What was preserved from the NBPO manuscript

- One deployed policy for a fixed panel of preference criteria.
- The deployed reference policy as the bargaining fallback.
- Pure Nash welfare as the policy-selection objective; no objective-level reference KL penalty.
- Two-phase optimization: positive-surplus feasibility followed by bargaining optimization.
- Normalized inverse-surplus weights.
- Entropy-mirror policy updates.
- Partition-free binary pairwise log-ratio regression.
- HelpSteer2 and SafeRLHF experimental panels and the existing judge protocol.
- Existing fixed-anchor numerical results, moved to an appendix and labeled as precursor evidence only.

## Minimal methodological change

The former anchored utility

\[
P_k(\pi\succ\mu)-\tfrac12
\]

is replaced by a criterion-wise entropy-regularized security value

\[
V_{k,\beta_k}(\pi)
=
\min_\nu
\left\{
 g_k(\pi,\nu)
 +\beta_k\,\mathbb E_x\operatorname{KL}(\nu(\cdot\mid x)\|\mu(\cdot\mid x))
\right\}.
\]

The bargaining disagreement and surplus become

\[
d_k=V_{k,\beta_k}(\mu),
\qquad
s_k(\pi)=V_{k,\beta_k}(\pi)-d_k,
\]

and the outer objective remains

\[
F(\pi)=\sum_{k=1}^K\log s_k(\pi).
\]

This preserves the original bargaining architecture while allowing the full pairwise comparison matrix, rather than only its projection against a fixed anchor, to affect policy selection.

## Main theoretical additions and repairs

### 1. Anchor-to-security interpolation

The paper proves

\[
0\le g_k(\pi,\mu)-V_{k,\beta_k}(\pi)\le \frac{1}{8\beta_k}
\]

and the uniform surplus bound

\[
\sup_\pi
\left|
[V_{k,\beta_k}(\pi)-V_{k,\beta_k}(\mu)]-g_k(\pi,\mu)
\right|
\le \frac{1}{8\beta_k}.
\]

Hence fixed-anchor NBPO is the \(\beta_k\to\infty\) endpoint, while \(\beta_k\downarrow0\) recovers global worst-comparator security.

### 2. Optimizer consistency

An added corollary establishes that accumulation points of finite-temperature Nash optimizers converge to the fixed-anchor NBPO optimizer set as all temperatures diverge. Their anchor-surplus vectors converge to the unique fixed-anchor bargaining outcome.

### 3. Game-value bargaining geometry

The comprehensive game-value hull is shown to be compact, convex, comprehensive, and nondegenerate under joint positive surplus. This yields:

- strict individual rationality in game-value coordinates;
- Pareto optimality of every Nash optimizer in those coordinates;
- uniqueness of the selected game-utility vector, though not necessarily of the policy;
- proportional fairness;
- covariance to independent positive affine transformations of criterion utilities.

### 4. Exact statement of competing rules

The comparison avoids claiming that utilitarian or maxmin aggregation lacks every desirable property:

- positive-weight utilitarian maximizers are Pareto optimal but can fall below disagreement;
- absolute maxmin can violate disagreement-based individual rationality;
- surplus maxmin is strictly individually rational under joint feasibility, but plain maxmin can contain Pareto-dominated tied maximizers;
- Nash obtains strict individual rationality, Pareto optimality for every optimizer, a unique utility outcome, and independent scale covariance simultaneously.

### 5. Raw-label scaling correction

The manuscript distinguishes utility-scale covariance from fixed-temperature label-noise invariance. At the oracle level,

\[
V_{\beta_k}^{\lambda_k A_k}(\pi)
=
\lambda_k V_{\beta_k/\lambda_k}^{A_k}(\pi).
\]

Thus scaling \(A_k\) with fixed \(\beta_k\) changes the effective comparator neighborhood. Exact covariance is recovered by jointly scaling \((A_k,\beta_k)\), or in the unregularized \(\beta_k=0\) limit. The paper does not overclaim exact finite-temperature invariance to coin-flip contamination.

### 6. Valid finite-sample certification

The former one-level confidence interval is not applied to the nested soft game value. The revision supplies a nested estimator using:

- independent prompts;
- reference comparators per prompt;
- independent learner comparisons per comparator;
- simultaneous confidence intervals over all criteria and predeclared checks.

The surplus interval correctly combines a lower bound for the candidate value with an upper bound for the disagreement value. An uncertified warm start is explicitly distinguished from a population infeasibility proof.

### 7. Inexact neural-update theorem

The exact population convergence result is retained and extended. The inexact theorem makes two approximation terms explicit:

- gradient/adversary estimation error \(\xi_t\);
- mirror variational residual \(\delta_t\) from neural regression.

With a monotonic acceptance gate, the last-iterate welfare bound includes their accumulated contribution. Without the gate, the corresponding guarantee is stated for the best iterate rather than silently claimed for the last iterate.

### 8. Counterexample repair

The inherited surplus-maxmin dominated-tie construction was mathematically invalid. It was replaced by an explicit valid pair of skew-symmetric finite preference games in which two policies attain the same optimal minimum surplus while one Pareto dominates the other.

## Positioning relative to prior work

The revised novelty claim is deliberately narrow. It does not claim to introduce game utilities, entropy-regularized comparators, nonlinear multi-criteria aggregation, or Nash welfare individually. It claims the following combination:

1. preserve one criterion-wise security utility for each general pairwise judge;
2. define disagreement through the deployed reference in each security coordinate;
3. apply fallback-relative Nash bargaining across those security improvements;
4. connect the method formally to fixed-anchor NBPO and global security through temperature limits;
5. provide bargaining-specific certification and exact/inexact convergence analysis.

## Formatting and compilation status

- Main text: 9 pages.
- References begin on page 10.
- Appendices follow the bibliography.
- Total compiled length: 22 pages.
- The paper is anonymous.
- The AI-use statement is included.
- The PDF was rendered and visually inspected; no clipping, overlap, missing glyph, or broken figure was observed.
- All currently used fonts are embedded.

## Important submission caveat

This draft is **not ready for submission until the matched finite-temperature experiments are completed**. The abstract and conclusion intentionally contain no empirical result claim, and the main tables contain red `TBD` cells. Filling those cells without actually rerunning the finite-temperature objective would be scientifically invalid.
