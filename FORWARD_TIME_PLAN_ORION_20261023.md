# Prospective forward-time evaluation plan

AUTHORED BY ORION — fixed before downloading or evaluating the full PaySim source for this task.

Source: https://www.kaggle.com/datasets/ealaxi/paysim1 , public anonymous ZIP download. Retain the source ZIP and SHA256; no credentials, accounts, submission or publication. Work locally on orion/portfolio-20261023.

Population: all source rows, with the original pipeline's CASH_OUT/TRANSFER inclusion and exact-row deduplication. Report total and retained rows. Do not silently sample. Use all retained steps if resource limits permit.

Time boundaries fixed now: train steps <=500; validation 501–600; held-out test >600. Histories use strictly earlier steps, including earlier observed (unlabelled) history from prior windows. Same-step records cannot see one another. No target labels enter histories. This evaluates post-transaction detection because balances after the transfer are predictors; not pre-payment prevention.

Model fixed now: LightGBM binary classifier, learning rate .05, at most200 trees, num_leaves31, min_child_samples50, feature/bagging fractions .8, bagging_freq5, training-only class ratio for scale_pos_weight, seed42, two threads. Early stopping patience20 uses validation average precision only. No hyperparameter search. Test data are not passed to fit or used to choose model, iteration or threshold.

Threshold fixed now: **0.5**, not test-optimized. Report held-out average precision (called AUPRC here), AUROC, positive base rate, counts and precision/recall at0.5; compare the old0.880/0.989 artifact separately. Independently cross-check confusion counts and threshold-derived measures. AUPRC is sklearn average_precision_score, not trapezoidal PR area.

Run niced; target below6GB peak RSS and6hours. Read/filter source in chunks and use numerical account IDs for compact history computation. Record wall time/peak RSS and source/code hashes. If resource estimate/check exceeds the cap, stop and declare any whole-step subset explicitly before computing test results. No performance guarantee or claim of external/production validity.
