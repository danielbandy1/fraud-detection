# JARVIS II review: HELIOS full PaySim retrain (branch helios/full-retrain-20261102, c8856cf)

## Re-checked independently (2026-11-02)
- Split counts recounted from the raw zip with my own code (not HELIOS's): eligible TRANSFER + CASH_OUT rows 2,770,409
  (all 8,213 frauds are those two types); steps 1-500: 2,645,779 rows / 5,561 fraud; 501-600: 81,080 / 1,052;
  601-743: 43,550 / 1,600. All match results/full_retrain_20261102.json.
- Metrics recomputed from the saved test predictions with scikit-learn: AP 0.88027, AUROC 0.99557, precision 0.87812,
  recall 0.99063 at threshold 0.99568 (chosen on validation only). All match.

## Say these in an interview (they are true and they matter)
1. **AP is not comparable with the old 0.880.** The test block's fraud rate is 3.7% (train 0.21%): late PaySim steps
   have few legitimate transactions (a simulator property). AP depends on the base rate, so the matching "0.880" is a
   coincidence, not a reproduction. Report AP with the test base rate next to it.
2. **Post-transaction balances are used** (newbalanceorig, newbalancedest, the deltas, dest_no_change). That is fine for
   scoring a transaction after it posts; for blocking it before authorisation it would be leakage. PaySim frauds also
   leave tell-tale balance patterns (e.g. destination balances that never change), so part of the skill is learning
   the simulator. A pre-authorisation variant without post-balance features is the honest next experiment.

## Verdict
Reproducible and correctly evaluated on a forward-time split. Resume-safe wording, e.g.: "Fraud detection on PaySim
(6.36M simulated transactions): LightGBM with leak-free history features, forward-time split; AP 0.88 at a 3.7%
fraud rate, AUROC 0.996 on held-out later days." Not merged to main; Daniel reviews first.

AUTHORED BY JARVIS II
