# Cost-threshold completeness repair

AUTHORED BY ORION — 2026-10-28. Local only.

src/models.py optimal_threshold previously considered only observed score cutoffs, omitting the no-alert operating point. Example validation outcomes [1,0,0], scores [.1,.9,.9], FN cost1/FP cost2: old threshold .9 costs5; no alerts costs1. It also inferred false-positive counts through precision plus an epsilon, introducing approximation into a discrete cost calculation.

Now sorts scores, evaluates every tied-score boundary with exact cumulative confusion counts, and includes nextafter(maximum score,+infinity) for no alerts. With scores exactly1 this sentinel may exceed1 by one floating-point step: that intentionally implements no alerts under score>=threshold. Equal-cost ties choose lowest threshold. Costs must be finite/nonnegative; inputs aligned, nonempty, binary with finite [0,1] probabilities. Single-class validation sets supported.

Regression evidence: first six new tests failed on original source (missing endpoint, exhaustive cost mismatch, four invalid-cost cases). Original bytes saved at /home/daniel/nexus/shared/orion_work/fraud_cost_ORION/models.before.py. Revised full project suite: **36 passed**, exit0, including120 random tied-score/cost comparisons to independent brute-force predictions, single-class and malformed-input tests.

This helper must be fit on validation labels only. evaluate(threshold=None) remains an exploratory same-label threshold selection path; it does not establish honest held-out performance. No cost ratios estimated from business data, no real financial benefit or deployment claim. The separate full_eval.py uses its own validation-F1 selector, so its saved results and headline were not recalculated or changed by this repair. API/model migration remains separate work; historical model artifacts are not certified.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
