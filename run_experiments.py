"""
run_experiments.py -- train and evaluate every model/graph combination over several seeds.

    python run_experiments.py            # all experiments, seeds 0 1 2 (about 80 min on 2 CPU cores)
    python run_experiments.py --quick    # smoke test: 4 models, 1 seed, few epochs (a few minutes)
    python run_experiments.py --only "R-GCN" --seeds 0 1 2 3 4 5 6 7 8 9

Every finished run is written to the results file immediately; runs already in it are
skipped, so an interrupted run can simply be started again.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

import tep_utils as U
from models import MLP, SensorGNN


def make_loader(X, y, shuffle, batch_size=64):
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def prepare(data_dir, budget):
    """Load the data once and build loaders and graphs for both settings."""
    train, test = U.load_tep(U.download_tep(data_dir))

    # Setting 1: all training windows. Standardise with training statistics only.
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = U.build_datasets(train, test, window=20)
    scaler = U.Standardizer().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    full = (make_loader(X_train, y_train, True), make_loader(X_val, y_val, False),
            make_loader(X_test, y_test, False))

    n_edges = U.structure_graph().shape[1] // 2
    graphs = {
        "knowledge": U.structure_graph(),
        "correlation": U.correlation_graph(X_train[y_train == 0], n_edges),
        "random": U.random_graph(n_edges, seed=0),
        "empty": U.empty_graph(),
    }

    # Setting 2: only `budget` training windows per class, spread evenly over each run.
    (Xb, yb), (Xv, yv), (Xt, yt) = U.build_datasets(train, test, window=20)
    keep = np.concatenate([np.where(yb == k)[0][np.linspace(0, (yb == k).sum() - 1, budget).astype(int)]
                           for k in range(U.N_CLASSES)])
    Xb, yb = Xb[keep], yb[keep]
    scaler_b = U.Standardizer().fit(Xb)          # the scaler only sees the small training set
    small = (make_loader(scaler_b.transform(Xb), yb, True), make_loader(scaler_b.transform(Xv), yv, False),
             make_loader(scaler_b.transform(Xt), yt, False))
    graphs_b = dict(graphs, correlation=U.correlation_graph(scaler_b.transform(Xb)[yb == 0], n_edges))
    return full, small, graphs, graphs_b


def experiments(full, small, graphs, graphs_b, budget, epochs, budget_epochs):
    """(name, loaders, epochs, model factory) for every configuration, in reporting order."""
    typed_ei, typed_et = U.typed_structure_graph()
    one_type = np.zeros_like(typed_et)
    shuffled = U.shuffled_edge_types(typed_et, seed=0)
    E = [("MLP", full, epochs, lambda: MLP())]
    for g in graphs:
        E.append((f"GCN | {g}", full, epochs, lambda g=g: SensorGNN(graphs[g], conv="gcn")))
    for g in graphs:
        E.append((f"GCN+res | {g}", full, epochs, lambda g=g: SensorGNN(graphs[g], conv="gcn", residual=True)))
    E += [
        ("GAT+res | knowledge", full, epochs,
         lambda: SensorGNN(graphs["knowledge"], conv="gat", residual=True)),
        ("R-GCN+res | knowledge (typed)", full, epochs,
         lambda: SensorGNN(typed_ei, conv="rgcn", residual=True, edge_type=typed_et)),
        ("R-GCN+res | knowledge (one type)", full, epochs,
         lambda: SensorGNN(typed_ei, conv="rgcn", residual=True, edge_type=one_type)),
        ("R-GCN+res | knowledge (shuffled types)", full, epochs,
         lambda: SensorGNN(typed_ei, conv="rgcn", residual=True, edge_type=shuffled)),
        ("GCN+res | knowledge, mean readout", full, epochs,
         lambda: SensorGNN(graphs["knowledge"], conv="gcn", residual=True, readout="mean")),
        (f"MLP | {budget}/class", small, budget_epochs, lambda: MLP()),
    ]
    for g in graphs_b:
        E.append((f"GCN+res | {g} | {budget}/class", small, budget_epochs,
                  lambda g=g: SensorGNN(graphs_b[g], conv="gcn", residual=True)))
    E += [
        (f"R-GCN+res | knowledge (typed) | {budget}/class", small, budget_epochs,
         lambda: SensorGNN(typed_ei, conv="rgcn", residual=True, edge_type=typed_et)),
        (f"R-GCN+res | knowledge (one type) | {budget}/class", small, budget_epochs,
         lambda: SensorGNN(typed_ei, conv="rgcn", residual=True, edge_type=one_type)),
    ]
    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--budget", type=int, default=30, help="training windows per class in the small-data setting")
    ap.add_argument("--budget-epochs", type=int, default=200)
    ap.add_argument("--only", default=None, help="run only experiments whose name contains this text")
    ap.add_argument("--quick", action="store_true", help="smoke test: few models, one seed, few epochs")
    ap.add_argument("--out", default=None, help="results file (default results/results.json)")
    ap.add_argument("--data-dir", default="tep_data")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.budget_epochs = [0], 3, 10
    out = args.out or ("results/quick.json" if args.quick else "results/results.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    results = json.load(open(out)) if os.path.exists(out) else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   torch {torch.__version__}   results file: {out}")
    full, small, graphs, graphs_b = prepare(args.data_dir, args.budget)
    todo = experiments(full, small, graphs, graphs_b, args.budget, args.epochs, args.budget_epochs)
    if args.quick:
        quick_names = {"MLP", "GCN | knowledge", "GCN+res | knowledge", "R-GCN+res | knowledge (typed)"}
        todo = [e for e in todo if e[0] in quick_names]
    if args.only:
        todo = [e for e in todo if args.only in e[0]]

    for name, (train_loader, val_loader, test_loader), n_epochs, make_model in todo:
        for seed in args.seeds:
            key = f"{name} | seed {seed}"
            if key in results:
                print(f"{key:55s} test macro-F1 {results[key]['f1']:.3f}   (already in {out})")
                continue
            U.set_seed(seed)
            model = make_model()
            t0 = time.time()
            U.fit(model, train_loader, val_loader, epochs=n_epochs, device=device, verbose=False)
            f1_all, P, Y = U.evaluate(model, test_loader, device)
            results[key] = {"name": name, "seed": seed, "f1": f1_all,
                            "f1_no_hard": U.macro_f1_without_hard(Y, P),
                            "per_class": f1_score(Y, P, average=None).tolist(),
                            "epochs": n_epochs, "device": device,
                            "minutes": (time.time() - t0) / 60}
            with open(out, "w") as f:
                json.dump(results, f, indent=1)
            print(f"{key:55s} test macro-F1 {f1_all:.3f}   ({results[key]['minutes']:.1f} min)", flush=True)

    print("\nmean ± std over seeds (test macro-F1, 22 classes):")
    for name, *_ in todo:
        f = np.array([v["f1"] for v in results.values() if v["name"] == name])
        if len(f):
            print(f"  {name:50s} {f.mean():.3f} ± {f.std():.3f}   (n = {len(f)})")


if __name__ == "__main__":
    main()
