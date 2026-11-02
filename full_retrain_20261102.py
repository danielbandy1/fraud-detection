"""Full-source public PaySim retrain on the HELIOS branch: leak-free strict
prior-step features, time-ordered train/validation/test split, LightGBM with
the prespecified parameters of FORWARD_TIME_PLAN_ORION_20261023.md, and a
cost-based threshold (src.models.optimal_threshold, ORION cost repair) selected
on VALIDATION labels only.

AUTHORED BY HELIOS, 2026-11-02. Train on the full local public source only:
data/raw/paysim_source.zip (do not re-download). Report test-block metrics only.

Run (gentle on a shared 16 GiB host):
  nice -n 15 bash -c 'ulimit -v 8000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python full_retrain_20261102.py'
"""
import os
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "2"
import resource
import threading
import time
import json
import gc
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
SPLIT_STEPS = [500, 600]  # train <= 500, validation 501-600, test > 600
# Repo-wide operating costs from src/models.py evaluate(): FN=1.0, FP=0.1.
# Documented in COST_THRESHOLD_FIX_ORION.md as the example cost basis.
COST_FN, COST_FP = 1.0, 0.1
samples = []


def available():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text()
                    .splitlines() if x.startswith("MemAvailable:"))) * 1024


def watch():
    while True:
        av, rss = available(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        samples.append((av, rss))
        if av < 2 * 1024**3:
            print("STOP: system free memory below 2 GiB", av, rss, flush=True)
            os._exit(75)
        time.sleep(0.25)


def hfile(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.nice(0) >= 15, "Run with nice -n 15"
    threading.Thread(target=watch, daemon=True).start()
    started = time.monotonic()

    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

    from src.features import clean, build_features, FEATURE_COLS
    from src.models import optimal_threshold
    from forward_time_eval import SHA, MODEL_PARAMS

    archive = ROOT / "data/raw/paysim_source.zip"
    assert hfile(archive) == SHA, "Local source ZIP hash mismatch; not re-downloading."
    types = {"step": "int16", "type": "category", "amount": "float64",
             "nameOrig": "string", "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
             "nameDest": "string", "oldbalanceDest": "float64", "newbalanceDest": "float64",
             "isFraud": "int8", "isFlaggedFraud": "int8"}
    parts, raw, rawfraud = [], 0, 0
    with zipfile.ZipFile(archive) as z, z.open("PS_20174392719_1491204439457_log.csv") as f:
        for chunk in pd.read_csv(f, dtype=types, chunksize=200000):
            raw += len(chunk)
            rawfraud += int(chunk.isFraud.sum())
            s = chunk[chunk.type.isin(["CASH_OUT", "TRANSFER"])].copy()
            s["type"] = s["type"].astype(pd.CategoricalDtype(["CASH_OUT", "TRANSFER"]))
            parts.append(s)
    del chunk, s
    d = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    d.columns = [c.lower() for c in d.columns]
    before = len(d)
    d = clean(d)  # exact-row dedup; retains CASH_OUT/TRANSFER only
    assert raw == 6362620 and int(d.isfraud.sum()) == 8213

    y = d.isfraud.to_numpy(dtype=np.int8)
    steps = d.step.to_numpy(copy=True)
    idx = {"train": np.flatnonzero(steps <= SPLIT_STEPS[0]),
           "valid": np.flatnonzero((steps >= SPLIT_STEPS[0] + 1) & (steps <= SPLIT_STEPS[1])),
           "test": np.flatnonzero(steps > SPLIT_STEPS[1])}
    assert idx["valid"].size and idx["test"].size and y[idx["train"]].sum() > 0

    for c in ["nameorig", "namedest"]:
        d[c] = pd.factorize(d[c], sort=False)[0].astype(np.int32)
    eng = build_features(d)
    del d
    gc.collect()
    X = eng[FEATURE_COLS].astype("float32")
    del eng
    gc.collect()
    assert np.isfinite(X.to_numpy()).all()

    tr, va, te = (idx[k] for k in ("train", "valid", "test"))
    params = {**MODEL_PARAMS, "seed": SEED,
              "scale_pos_weight": float(len(tr) / y[tr].sum() - 1)}
    train_ds = lgb.Dataset(X.iloc[tr], label=y[tr], free_raw_data=True)
    valid_ds = lgb.Dataset(X.iloc[va], label=y[va], reference=train_ds, free_raw_data=True)
    model = lgb.train(params, train_ds, num_boost_round=200, valid_sets=[valid_ds],
                      valid_names=["validation"],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                 lgb.log_evaluation(period=50)])
    best = int(model.best_iteration or model.current_iteration())
    pv = model.predict(X.iloc[va], num_iteration=best)
    assert np.isfinite(pv).all() and ((pv >= 0) & (pv <= 1)).all()
    threshold = optimal_threshold(y[va], pv, cost_fn=COST_FN, cost_fp=COST_FP)
    # Frozen before any test prediction.
    pt = model.predict(X.iloc[te], num_iteration=best)
    assert np.isfinite(pt).all() and ((pt >= 0) & (pt <= 1)).all()

    def measures(y_true, p, thr):
        tn, fp, fn, tp = map(int, confusion_matrix(y_true, np.asarray(p) >= thr,
                                                   labels=[0, 1]).ravel())
        return {"threshold": float(thr), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
                "rows": int(len(y_true)), "fraud": int(y_true.sum())}

    test = measures(y[te], pt, threshold)
    ap = float(average_precision_score(y[te], pt))
    auroc = float(roc_auc_score(y[te], pt))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    artifacts = out / "full_retrain_20261102_artifacts"
    artifacts.mkdir(exist_ok=True)
    for name, index, p in (("validation", va, pv), ("test", te, pt)):
        pd.DataFrame({"retained_row_position": index, "step": steps[index],
                      "isfraud": y[index], "probability": p}
                     ).to_parquet(artifacts / f"{name}_predictions.parquet", index=False)
    model.save_model(str(artifacts / "model.txt"), num_iteration=best)

    result = {
        "authored_by": "HELIOS",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "ealaxi/paysim1",
        "license": "CC BY-SA 4.0 (source metadata)",
        "source_zip_sha256": SHA,
        "raw_rows": raw,
        "raw_fraud": rawfraud,
        "eligible_rows": int(len(X)),
        "duplicates_removed": int(before - len(X)),
        "seed": SEED,
        "split_steps": SPLIT_STEPS,
        "splits": {k: {"rows": int(len(v)), "fraud": int(y[v].sum()),
                       "base_rate": float(y[v].mean()),
                       "min_step": int(steps[v].min()), "max_step": int(steps[v].max())}
                   for k, v in idx.items()},
        "rows_train": int(len(tr)),
        "rows_valid": int(len(va)),
        "rows_test": int(len(te)),
        "params": params,
        "best_iteration": best,
        "threshold_basis": ("cost-based src.models.optimal_threshold fit on validation "
                            f"labels only: cost_fn={COST_FN}, cost_fp={COST_FP} "
                            "(COST_THRESHOLD_FIX_ORION.md; repo evaluate() costs); "
                            "equal-cost ties choose the lowest threshold; no-alert "
                            "operating point included"),
        "threshold": test["threshold"],
        "ap": ap,
        "auroc": auroc,
        "precision": test["precision"],
        "recall": test["recall"],
        "test_at_cost_threshold": test,
        "test_at_fixed_0_5": measures(y[te], pt, 0.5),
        "feature_cols": FEATURE_COLS,
        "features_sha256": hfile(ROOT / "src/features.py"),
        "cost_threshold_sha256": hfile(ROOT / "src/models.py"),
        "script_sha256": hfile(__file__),
        "caveats": [
            "Test window (steps > 600) was already scored on 2026-10-23/24; scores are "
            "bound by the same data, not a pristine future holdout.",
            "Post-transaction detection: balance-after-transfer features cannot support "
            "pre-payment prevention claims.",
            "Population limited to CASH_OUT/TRANSFER rows (fraud types in PaySim).",
            "AP is sklearn average_precision_score; legacy 0.880/0.989 came from random "
            "CV with leaky full-window histories and thresholds tuned on evaluated "
            "labels, so it is not comparable evidence.",
            "Cost costs mirror the repo defaults, not an estimated business loss model.",
        ],
        "resource": {
            "actual_nice": os.nice(0),
            "threads": int(os.environ["OMP_NUM_THREADS"]),
            "address_space_limit_bytes": 8000000 * 1024,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "minimum_system_available_gib": min(x[0] for x in samples) / 1024**3,
            "monitor_samples": len(samples),
            "float32_features": True,
        },
        "runtime_s": time.monotonic() - started,
        "files": {p.name: hfile(p) for p in sorted(artifacts.glob("*")) if p.is_file()},
    }
    assert result["resource"]["minimum_system_available_gib"] >= 2
    (out / "full_retrain_20261102.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "rows_train", "rows_valid", "rows_test", "ap", "auroc", "precision",
        "recall", "threshold", "seed", "runtime_s")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
