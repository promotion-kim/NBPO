# Game-NBPO ICLR 2027 Source Package

## Build

From the source directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Package contents

- `main.tex`: revised manuscript.
- `iclr2027_conference.bib`: bibliography database.
- `iclr2027_conference.sty`: user-supplied ICLR 2027 style file.
- `iclr2027_conference.bst`: ICLR-compatible fallback bibliography style used for local compilation.
- `math_commands.tex`: original math command file retained in the package.
- `figures/cyclic_bargaining_geometry.pdf`: vector figure used by the paper.
- `REVISION_NOTES.md`: detailed methodological changes.
- `EXPERIMENTS_REQUIRED.md`: experiments required before submission.
- `THEOREM_AUDIT.md`: mathematical audit and theorem scope.

## Submission status

The compiled manuscript has 9 pages of main text, followed by references and appendices. It is anonymous and includes the required AI-use statement.

The source contains no unresolved `TBD` marker. Current neural results are
reported conservatively and do not establish a performance advantage. Do not
submit the paper until the frozen neural-realization, matched-selector,
independent-judge, and final multi-seed gates have been resolved.

## Style-file caveat

The uploaded materials contained the official-looking 2027 `.sty` file but did not include a 2027 `.bst` bibliography file. The package therefore contains an ICLR-compatible fallback `.bst` solely to make the local source compile. Before submission, replace it with the bibliography style distributed in the official ICLR 2027 style bundle if the official bundle contains a different file.

## Final verification checklist

- Recompile with the complete official ICLR 2027 style bundle.
- Confirm the main text remains at or below the submission page limit after inserting results.
- Update the abstract, conclusion, and result tables only from gate-passing matched runs.
- Independently verify all AI-assisted mathematical claims, proofs, citations, and code.
- Update the AI-use statement to match the actual final research workflow.
- Check anonymity in the PDF, source, metadata, code repository, and supplementary files.
- Run PDF preflight and visually inspect every page after the final compilation.
