# Full evaluation addendum — fixed before this run

AUTHORED BY ORION

Keep the prior full-source hash, CASH_OUT/TRANSFER population, strict-prior-step features, train<=500, validation501–600, test601–743, and fixed LightGBM settings from forward_time_eval.py. The test window was evaluated on23Oct; this is an uncertainty/threshold extension of that known holdout, not a fresh untouched evaluation or a new hyperparameter search.

Select the decision threshold by maximum F1 on validation predictions from the validation-early-stopped model; exact ties choose the higher threshold. No test labels/probabilities supplied to selection. F1 is an explicit generic operating point, not a financially calibrated loss optimum. Report0.5 as a secondary fixed comparator, without choosing between thresholds using test results.

Report sklearn average_precision_score (AP, not trapezoidal PR area), AUROC, confusion matrix, precision/recall/F1, validation selection evidence, all split counts/prevalence. Primary95% uncertainty:2000circular moving-block bootstrap draws of24consecutive hourly steps from the test timeline, concatenate to143steps; seed20261024; percentile2.5/97.5. Reweight complete step clusters, preserving within-step observations and local day dependence. Report invalid one-class draws. Secondary2000stratified row bootstrap draws conditional on class totals, clearly labelled a narrower dependence-ignoring sensitivity. CIs condition on the fixed trained model and known test window; they do not quantify retraining uncertainty, future distribution shift, or external validity.24h is chosen here, not after seeing intervals.

Bound task at nice15,two threads,address space6GiB; monitor MemAvailable and peakRSS every0.2seconds, stop worker with status75 if MemAvailable<3GiB or RSS>=6GiB. NoMCC,push,resume/publicpage changes. Full source retained; no sampling. Reproduce via new full_eval.py; existing reproduce.sh must still pass. Save metrics and both prediction windows under new full_eval_results only.
