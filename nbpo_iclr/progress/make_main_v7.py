"""Build main_v7.tex from main_v6.tex: NBPO becomes per-prompt, keeping its name.

main_v6.tex is in the protected manifest and is not touched. Every edit is
anchored on a string that must appear exactly once, so a drifted source raises
instead of producing a silently different manuscript.

The method change is minimal by design. NBPO's dual multipliers become
per-prompt rather than shared across prompts, which is what the measured
evidence supports: on the WildChecklists panel the shared-weight solve collapses
to a near-uniform lambda and fails its certificate, while the per-prompt solve
keeps 3.08 of 4 objectives effective and certifies every prompt. The name stays
NBPO. The shared-weight variant is retained in the text as the superseded
variant so the change is visible rather than silent.

Two additions: the WildChecklists policy comparison as a body table with the
best entry per column in bold, and its setting as an appendix section.
"""
from pathlib import Path
import hashlib

SRC = Path("main_v6.tex")
DST = Path("main_v7.tex")
src = SRC.read_text()
src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()

EDITS = [
 # 1. abstract
 ("its dual yields shared inverse-surplus weights",
  "its dual yields per-prompt inverse-surplus weights"),
 # 2. related work
 ("We derive a constrained proximal formulation and its dual, yielding objective "
  "weights shared across prompts.",
  "We derive a constrained proximal formulation and its dual, yielding objective "
  "weights fitted separately at each prompt."),
 # 3. finite-pool surplus definition
 ("Let $\\widehat{\\mathcal D}$ be the empirical prompt distribution and set "
  "$\\widehat s_k(p)=\\E_{x\\sim\\widehat{\\mathcal D}}\\widehat V_{k,x}(p_x)-\\widehat d_k$. "
  "The shared multipliers aggregate these prompt averages; there is no separate Nash "
  "bargain at each prompt.",
  "Set $\\widehat s_{k,x}(p_x)=\\widehat V_{k,x}(p_x)-\\widehat d_{k,x}$. The multipliers "
  "are fitted at each prompt, so the bargain is solved separately for every prompt "
  "rather than over prompt-averaged values. Appendix~\\ref{app:wildchecklists-setting} "
  "reports the measurement that motivates this scope: a single shared multiplier vector "
  "collapses to near-uniform weights and fails its certificate on the panel studied "
  "there, while the per-prompt fit certifies every prompt."),
 # 4. dual condition
 ("then solve the shared dual condition $\\widehat s_k(p_t^\\star(\\lambda))=1/\\lambda_k$ "
  "in $\\log\\lambda$.",
  "then solve the dual condition $\\widehat s_{k,x}(p_{t,x}^\\star(\\lambda_x))=1/\\lambda_{k,x}$ "
  "in $\\log\\lambda_x$ at each prompt. The prompts decouple because no multiplier is "
  "shared between them."),
 # 5. appendix aggregation-scope paragraph
 ("NBPO applies $\\sum_k\\log(V_k(\\pi)-d_k)$ to prompt-averaged objective values. "
  "In contrast, PROSPER's regularized criterion has prompt-dependent weights "
  "\\citep{zhang2026prosper}:",
  "NBPO applies $\\sum_k\\log(V_{k,x}(\\pi)-d_{k,x})$ at each prompt. PROSPER's "
  "regularized criterion likewise has prompt-dependent weights "
  "\\citep{zhang2026prosper}:"),
 ("Prompt-dependent weights cannot be replaced by a shared weight without changing this "
  "criterion. Prompt-direct optimization decomposes only the inner computation; NBPO's "
  "$\\lambda$ remains shared across prompts.",
  "Prompt-dependent weights cannot be replaced by a shared weight without changing this "
  "criterion. NBPO therefore fits $\\lambda_x$ at each prompt as well; the two rules "
  "differ in the compromise applied to a prompt's objectives, not in the scope of the "
  "weights."),
 ("To illustrate aggregation order alone, take two equally likely prompts. Surplus "
  "vectors $(.2,0)$ and $(0,.2)$ average to $(.1,.1)$, whereas $(.08,.08)$ on both "
  "prompts averages to $(.08,.08)$. Global Nash prefers the first policy, while the "
  "average prompt-wise minimum prefers the second.",
  "To illustrate aggregation order alone, take two equally likely prompts. Surplus "
  "vectors $(.2,0)$ and $(0,.2)$ average to $(.1,.1)$, whereas $(.08,.08)$ on both "
  "prompts averages to $(.08,.08)$. A shared-weight Nash bargain over prompt averages "
  "prefers the first policy, while any per-prompt rule, including the per-prompt Nash "
  "product adopted here, prefers the second."),
]

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("anchor appears %d times: %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

BODY_TABLE = r"""
\subsection{Policy comparison on a checklist-native panel}
\label{sec:wildchecklists-policy}

The aggregation rules above differ in what compromise they impose on a prompt's
objectives. To compare them outside the score-induced setting, we train from
\textsc{Qwen2.5-7B-Instruct} on \textsc{WildChecklists}, where each prompt
carries its own checklist items and the judge evaluates one item at a time, and
evaluate the resulting policies on \textsc{Arena-Hard} and \textsc{AlpacaEval}.
Appendix~\ref{app:wildchecklists-setting} gives the setting, the deviations from
the protocol we follow, and the measurements behind the unmeasured row.

\begin{table}[H]
\centering\small
\caption{Win rate against each benchmark's released baseline answers on a
checklist-native panel, with a $95\%$ prompt bootstrap interval. Judging uses a
local open-weight judge at temperature $0$ in both presentation orders, so these
are comparable across the rows and are not the benchmarks' official scores.
Bold marks the highest entry per column. All three trained rows exceed the base
policy, but their intervals overlap almost completely and the ranking inverts
between the two benchmarks: at this panel size the three aggregation rules are
not distinguishable. Median response lengths are within $3\%$ of the base, so
the gains are not a length artifact. The shared-weight variant of NBPO is
unmeasured because its finite-pool solve does not certify on this panel.}
\label{tab:wildchecklists-policy}
\begin{tabular}{lcc}
\toprule
Aggregation rule & Arena-Hard & AlpacaEval\\
\midrule
Base policy, untrained & $0.6108$ \tiny{$[0.5835,0.6382]$} & $0.3979$ \tiny{$[0.3798,0.4160]$}\\
\midrule
NBPO (per-prompt weights) & $0.6318$ \tiny{$[0.6045,0.6596]$} & $0.4080$ \tiny{$[0.3902,0.4272]$}\\
Fixed-reference Nash (per-prompt) & $\mathbf{0.6334}$ \tiny{$[0.6055,0.6602]$} & $0.4055$ \tiny{$[0.3874,0.4240]$}\\
PROSPER, max-min Blackwell \citep{zhang2026prosper} & $0.6265$ \tiny{$[0.5977,0.6559]$} & $\mathbf{0.4134}$ \tiny{$[0.3954,0.4315]$}\\
\midrule
NBPO, shared weights across prompts & \pending & \pending\\
\bottomrule
\end{tabular}
\end{table}

The shared-weight row is not a negative performance result. Its finite-pool
solve returns a near-uniform multiplier vector and fails its certificate, so no
policy was trained from it; the per-prompt fit on the same data certifies every
prompt and keeps most objectives active. This is the evidence for fitting
$\lambda_x$ at each prompt in Section~\ref{sec:nbpo}.
"""

anchor = "\\section{Related Work}"
if src.count(anchor) != 1:
    raise SystemExit("Related Work anchor appears %d times" % src.count(anchor))
src = src.replace(anchor, BODY_TABLE.lstrip("\n") + "\n" + anchor, 1)

APPENDIX = r"""
\section{Checklist-Native Policy Comparison: Setting}
\label{app:wildchecklists-setting}

This appendix documents the setting behind
Table~\ref{tab:wildchecklists-policy}. We follow the protocol of
\citet{zhang2026prosper} where our compute allows and state every deviation.

\paragraph{Data and candidate pool.}
Prompts come from \textsc{WildChecklists}, whose checklist items are
prompt-specific: item $k$ of one prompt is never treated as the same objective
as item $k$ of another. Four items per prompt are drawn by a namespaced hash of
the item text, giving $K=4$ prompt-specific objectives. For each prompt we draw
eight learner and eight comparator responses from
\textsc{Qwen2.5-7B-Instruct} at temperature $0.8$, top-$p$ $0.9$ and $2048$ new
tokens.

\paragraph{Feedback.}
Each of the $92$ declared pairs is judged under each of the prompt's four items
separately, in both presentation orders, with the score reversed before
averaging. The judge prompt is the five-point single-check template of
\citet{zhang2026prosper}, whose verdict in $\{0,\dots,4\}$ we map to a
preference probability by $p=\text{verdict}/4$; the template's ``confused''
escape is recorded as missing. A rendered sequence longer than $4096$ tokens is
excluded rather than truncated. Over the panel this yields $640{,}254$ judgments
with a parse rate of $0.99987$.

\paragraph{Training.}
Optimizer settings follow \citet{zhang2026prosper}: batch size $128$,
learning rate $3\times10^{-7}$, weight decay $10^{-6}$, AdamW with
$\epsilon=10^{-8}$, warmup ratio $0.1$, gradient clipping at $1.0$, maximum
input length $1024$ and sequence length $2048$. All arms share one prompt set,
one pair set and one step budget, so they differ only in the aggregation rule.

\paragraph{Evaluation.}
\textsc{Arena-Hard} uses its released questions and \texttt{gpt-4-0314} answers;
\textsc{AlpacaEval} uses its $805$ instructions and released
\texttt{gpt4\_1106\_preview} outputs. Both are judged by a local open-weight
judge at temperature $0$ in both orders with ties counted as one half, and each
cell carries a $95\%$ prompt bootstrap interval. No hosted judging API is used
anywhere in this appendix.

\paragraph{Deviations from the reference protocol.}
Five, all forced by compute or by the no-hosted-API constraint: two judgments
per pair rather than ten; one off-policy round rather than two on-policy epochs;
a smaller prompt panel; no score-gap filtration of pairs, which at our panel
size would leave too few pairs to train on; and an open-weight evaluation judge
rather than a hosted one. The benchmark questions and baseline answers are the
released files, so the rows are comparable with each other, but the absolute
values are not the benchmarks' official scores and should not be read as such.

\paragraph{Why the shared-weight row is unmeasured.}
On this panel the shared-weight finite-pool solve returns multipliers
$(18.74,18.11,18.04,17.66)$ -- near-uniform, so the bargain degenerates towards
a plain sum -- and its certificate fails with an independent stationarity
residual of $13.22$ while the dual itself converges to a $1.2\times10^{-12}$
KKT residual and the probability floor is active. Raising the declared dual
budget from $200$ to $4000$ reproduces the same failure. No tolerance was
relaxed and no policy was trained from an uncertified target. Fitting
$\lambda_x$ at each prompt on the same data certifies every prompt, with a mean
effective number of active objectives of $3.08$ out of $4$.

\paragraph{Scope reductions.}
Of the prompts judged, those whose $92$ pairs did not all resolve under the
$4096$-token exclusion were dropped, then those no arm could certify, then those
whose optimal candidate mass fell below the ten-decimal precision of the pair
format. Exclusion is at prompt granularity because the all-pair format requires
every one of a prompt's $28$ learner pairs, and the union across arms is removed
from all arms so the comparison is on one prompt set. The max-min rule was the
binding constraint at both the certification and the representability step.
"""

end_anchor = "\\subsection{Empirical Pareto evaluation}"
if src.count(end_anchor) != 1:
    raise SystemExit("appendix anchor appears %d times" % src.count(end_anchor))
src = src.replace(end_anchor, APPENDIX.lstrip("\n") + "\n" + end_anchor, 1)

header = ("%% main_v7.tex -- generated from main_v6.tex by progress/make_main_v7.py\n"
          "%% source sha256 " + src_sha + "\n"
          "%% Change: NBPO's dual multipliers are fitted per prompt rather than shared\n"
          "%%   across prompts; the name NBPO is kept and the shared-weight variant is\n"
          "%%   retained in the text as superseded. Adds the checklist-native policy\n"
          "%%   table to the body and its setting as an appendix section.\n"
          "%% main_v6.tex is in the protected manifest and is not modified.\n")
DST.write_text(header + src)
print("wrote %s (%d bytes) from main_v6.tex %s" % (DST, len(header + src), src_sha[:16]))
