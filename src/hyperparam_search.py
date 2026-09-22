"""
Hyperparameter search (grid search) on N24.
Configs : learning_rate in [1e-6, 5e-6, 1e-5, 5e-5]  x  num_epochs in [5,10,20]  x  lambda_XtoC in [0.5,1,2,5]
Total   : 48 runs
Output  : results/hyperparam_search.json

Usage:
    python hyperparam_search.py

The search is resumable: configs already present in the output file are skipped.
"""

import itertools, json, os, time, traceback
from datetime import datetime
from CBM_pipeline import run_CBM

# ── Fixed config ──────────────────────────────────────────────────────────────
FIXED_CONFIG = dict(
    dataset="N24", dataset_type="CBLLM", combine_type="combine",
    backbone="clip-large", concept_representation="importance",
    load=False, plot=False,
    leakage_loss=True, leakage_loss_activation="up", kan_layer=True,
)

# ── Search space ──────────────────────────────────────────────────────────────
SEARCH_SPACE = dict(
    learning_rate = [1e-6, 5e-6, 1e-5, 5e-5],
    num_epochs    = [5, 10, 20],
    lambda_XtoC   = [0.5, 1.0, 2.0, 5.0],
)

OUTPUT_PATH = "results/hyperparam_search.json"
METRIC      = "test_f1"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(path):
    return json.load(open(path)) if os.path.exists(path) else []

def _save(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(results, open(path, "w"), indent=2, default=str)

def _metrics(history):
    keys = ["test_acc", "test_f1", "test_concept_acc", "test_concept_f1", "test_concept_leakage"]
    return {k: history.get(k) for k in keys}

def _summary(results, metric):
    valid = sorted([r for r in results if r["status"]=="ok" and r["metrics"].get(metric)],
                   key=lambda r: r["metrics"][metric], reverse=True)
    errors = [r for r in results if r["status"]=="error"]

    print(f"\n{'='*60}")
    print(f"Top 10 by {metric}  ({len(valid)} successful / {len(errors)} errors)")
    print(f"{'='*60}")
    print(f"{'#':<4} {metric:<8} {'acc':<8} {'lr':<10} {'epochs':<8} lambda")
    print("-"*60)
    for rank, r in enumerate(valid[:10], 1):
        m, c = r["metrics"], r["config"]
        print(f"{rank:<4} {m.get(metric,0):<8.4f} {m.get('test_acc',0):<8.4f} "
              f"{c['learning_rate']:<10} {c['num_epochs']:<8} {c['lambda_XtoC']}")

    if valid:
        best = valid[0]
        print(f"\n{'='*60}")
        print(f"BEST CONFIG  ({metric} = {best['metrics'][metric]:.4f})")
        print(f"  learning_rate : {best['config']['learning_rate']}")
        print(f"  num_epochs    : {best['config']['num_epochs']}")
        print(f"  lambda_XtoC   : {best['config']['lambda_XtoC']}")
        print(f"  test_acc      : {best['metrics'].get('test_acc')}")
        print(f"  test_f1       : {best['metrics'].get('test_f1')}")
        print(f"  concept_acc   : {best['metrics'].get('test_concept_acc')}")
        print(f"  leakage       : {best['metrics'].get('test_concept_leakage')}")

    if errors:
        print(f"\n{len(errors)} failed run(s):")
        for r in errors:
            print(f"  config={r['config']}  error={r['error'].splitlines()[-1] if r['error'] else '?'}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_hyperparameter_search(output_path=OUTPUT_PATH, metric=METRIC):
    results = _load(output_path)
    done    = {json.dumps(r["config"], sort_keys=True) for r in results}
    combos  = list(itertools.product(
        SEARCH_SPACE["learning_rate"],
        SEARCH_SPACE["num_epochs"],
        SEARCH_SPACE["lambda_XtoC"],
    ))
    total = len(combos)
    print(f"{total} configs  |  already done: {len(done)}  |  remaining: {total-len(done)}\n")

    for i, (lr, epochs, lxc) in enumerate(combos, 1):
        cfg     = dict(learning_rate=lr, num_epochs=epochs, lambda_XtoC=lxc)
        cfg_key = json.dumps(cfg, sort_keys=True)
        if cfg_key in done:
            print(f"[{i:>2}/{total}] SKIP  lr={lr}  epochs={epochs}  lambda={lxc}")
            continue
        print(f"\n[{i:>2}/{total}] lr={lr}  epochs={epochs}  lambda={lxc}")
        t0 = time.time()
        try:
            history, _ = run_CBM(**{**FIXED_CONFIG, **cfg})
            m, status, error = _metrics(history), "ok", None
            print(f"  -> f1={m.get('test_f1'):.4f}  acc={m.get('test_acc'):.4f}")
        except Exception:
            m, status, error = {}, "error", traceback.format_exc()
            print(f"  -> ERROR: {error.splitlines()[-1]}")

        results.append({"timestamp": datetime.now().isoformat(),
                        "config": cfg, "fixed_config": FIXED_CONFIG,
                        "metrics": m, "status": status, "error": error,
                        "elapsed_s": round(time.time()-t0, 1)})
        _save(results, output_path)

    _summary(results, metric)
    return results

if __name__ == "__main__":
    run_hyperparameter_search()
