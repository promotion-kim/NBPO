# P6 initial Alpaca3-8B-test panel: aborted before decode

The initial 1,000-prompt panel was locked from the previously used
`Alpaca3-8B/test.jsonl` after excluding all known training and evaluation
panels.  The score-blind manifest audit showed that all 693 conflict prompts
had already been used by the recorded retrospective diagnostic, leaving a
selected panel with zero dual-preference-conflict prompts.

No model generation, reward scoring, ranking, checkpoint selection, upload,
or training was performed from this lock.  It is retained for audit, but is
not a valid test of a conflicting-objective robustness claim.  The successor
protocol uses the full official PKU-SafeRLHF default test configuration at the
same pinned dataset revision and remains prompt-disjoint from all earlier
panels.
