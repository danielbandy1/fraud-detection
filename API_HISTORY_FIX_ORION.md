# Fraud API history-indicator honesty

AUTHORED BY ORION — 2026-10-28.

api/serve.py formerly set graph_features_available whenever len(rows)>1. It now derives the flag from the selected target's actual engineered orig_tx_count or dest_recv_count being positive. Thus tied-hour, future and unrelated rows cannot claim history availability. Earlier transactions sharing either originator or destination count as partial context. Flag means some supplied context contributes, never complete feature-store history or validated accuracy. Updated endpoint/OpenAPI descriptions remove accurate-scoring promises and document strictly earlier steps.

Original behavior fails four of seven initial fake-model cases; revised full suite **44 passed**, exit0 (one Starlette dependency deprecation warning). Cases cover singleton, future, same-step, unrelated, origin-only, destination-only and non-last target. Additional in-process FastAPI HTTP test checks /score/window and /score serialization/status/flag semantics. Tests stub loading/model/explainer, use synthetic transactions, and do not load historical model files or launch a network server. Source original at /home/daniel/nexus/shared/orion_work/fraud_api_history_ORION/serve.before.py.

No prediction feature formula, model, threshold, SHAP routine, forward-time result or public deployment changed. This does not migrate the API's historical model artifact to the newer forward-time benchmark model, nor certify risk-tier thresholds or input-domain validation. Those remain separate integration concerns. Local tests only.

Run OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests.
