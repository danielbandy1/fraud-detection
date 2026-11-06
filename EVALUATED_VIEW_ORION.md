# Verified forward-time evaluation view

AUTHORED BY ORION — 2026-10-28. Local educational view; no deployment.

New evaluated_app_ORION.py preserves historical app.py intact. The latter's missing logistic/OOF artifacts and full-dataset SHAP branch are not silently substituted. New view uses verified model/report and saved validation/test predictions only; no raw account records displayed, no fitting or feature rebuild. Exposes frozen testAP.8894915913/AUROC.9946323897, prevalence3.673938%,43550rows, validation-F1 threshold.9974552616 and report's24-hour block bootstrap intervals/caveats. Chart labels identify fixed test curves; AP is average precision, not trapezoidal PR area.

src/evaluation_artifacts.py checks model/feature provenance via the existing loader, binds the same report hash, verifies prediction bytes/schema/population/label/probability bounds, recomputes both AP/AUROC and fixed-threshold confusion matrices for both partitions, and rejects overlapping retained-row positions. Source data collection completeness, authentic external provenance and independence beyond the declared partitions are not established by these checks.

Interactive cost ratios are hypothetical; threshold selection and displayed cost counts use validation predictions only. Frozen test metrics are never recomputed at the interactive threshold. This avoids presenting test-optimised costs as held-out evidence. Validation reuse still creates selection uncertainty; no dollar savings or deployment operating policy claimed. The full report caveats remain visible, including prior exposure to this test window and bootstrap conditionality.

Verification:58project tests PASS. Seven new artifact-view tests cover valid synthetic partitions, checksum corruption, overlap, temporal-range violations, out-of-bounds probabilities, mismatched metric and exact no-alert cost. Real saved-artifact AppTest default run: correct43550rows/AP, two Plotly elements, no errors/exceptions. After raising missed-fraud cost from10to50, test metric widgets remain identical. This verifies runtime/control/chart serialization, not browser pixel rendering; no chart screenshot or deployed endpoint claimed. Existing Streamlit deprecation warnings remain.

Run locally with existing environment:

```bash
.venv/bin/streamlit run evaluated_app_ORION.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
```

Missing or inconsistent artifacts display an error and stop. Each rerun checks artifacts anew rather than caching away file changes. No historical predictions/logistic weights fabricated; original SHAP figures/code untouched. No public server started during verification, no source-data rewrite, model search, Kaggle/MCC action or push.
