"""
tep_utils.py -- data, leakage-free splits, process-knowledge graphs and training
helpers for GNN fault diagnosis on the Tennessee Eastman process (TEP).

Sections:
  1. Data: download and load the Braatz TEP training/test runs
  2. Windows, splits and scaling (split by run first, then cut windows)
  3. Graphs over the 52 variables: process knowledge, correlation, random, empty
  4. Training and evaluation (Adam, early stopping on validation macro-F1)
  5. Reading the control loops from the simulator code
"""
import os
import subprocess
import numpy as np

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
DATA_URL = "https://github.com/camaramm/tennessee-eastman-profBraatz.git"
N_VARS = 52           # 41 measurements (XMEAS) + 11 manipulated variables (XMV)
N_CLASSES = 22        # class 0 = normal, classes 1..21 = fault IDV(1)..IDV(21)
TEST_ONSET = 160      # in the *_te.dat files the fault starts at sample 160 (8 h)

VAR_NAMES = (
    [f"XMEAS{i}" for i in range(1, 42)] + [f"XMV{i}" for i in range(1, 12)]
)

VAR_DESCRIPTIONS = {
    "XMEAS1": "A feed (stream 1)", "XMEAS2": "D feed (stream 2)",
    "XMEAS3": "E feed (stream 3)", "XMEAS4": "A and C feed (stream 4)",
    "XMEAS5": "Recycle flow (stream 8)", "XMEAS6": "Reactor feed rate (stream 6)",
    "XMEAS7": "Reactor pressure", "XMEAS8": "Reactor level",
    "XMEAS9": "Reactor temperature", "XMEAS10": "Purge rate (stream 9)",
    "XMEAS11": "Separator temperature", "XMEAS12": "Separator level",
    "XMEAS13": "Separator pressure", "XMEAS14": "Separator underflow (stream 10)",
    "XMEAS15": "Stripper level", "XMEAS16": "Stripper pressure",
    "XMEAS17": "Stripper underflow (stream 11)", "XMEAS18": "Stripper temperature",
    "XMEAS19": "Stripper steam flow", "XMEAS20": "Compressor work",
    "XMEAS21": "Reactor cooling water outlet temp",
    "XMEAS22": "Separator cooling water outlet temp",
    **{f"XMEAS{22 + i}": f"Reactor feed analysis, component {c}"
       for i, c in enumerate("ABCDEF", start=1)},
    **{f"XMEAS{28 + i}": f"Purge gas analysis, component {c}"
       for i, c in enumerate("ABCDEFGH", start=1)},
    **{f"XMEAS{36 + i}": f"Product analysis, component {c}"
       for i, c in enumerate("DEFGH", start=1)},
    "XMV1": "D feed flow valve (stream 2)", "XMV2": "E feed flow valve (stream 3)",
    "XMV3": "A feed flow valve (stream 1)", "XMV4": "A and C feed flow valve (stream 4)",
    "XMV5": "Compressor recycle valve", "XMV6": "Purge valve (stream 9)",
    "XMV7": "Separator pot liquid flow valve (stream 10)",
    "XMV8": "Stripper liquid product flow valve (stream 11)",
    "XMV9": "Stripper steam valve", "XMV10": "Reactor cooling water flow",
    "XMV11": "Condenser cooling water flow",
}


ZIP_URL = ("https://github.com/camaramm/tennessee-eastman-profBraatz/"
           "archive/refs/heads/master.zip")


def download_tep(data_dir="tep_data"):
    """Fetch the classic Braatz TEP dataset (about 30 MB) if it is not there yet.

    Uses git if it is installed, otherwise downloads the repository as a zip file.
    """
    target = os.path.join(data_dir, "braatz")
    if os.path.exists(os.path.join(target, "d00.dat")):
        return target
    os.makedirs(data_dir, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", DATA_URL, target], check=True)
    except (OSError, subprocess.CalledProcessError):
        import io, shutil, urllib.request, zipfile
        print("git not available - downloading the zip archive instead ...")
        with urllib.request.urlopen(ZIP_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(data_dir)
        shutil.move(os.path.join(data_dir, "tennessee-eastman-profBraatz-master"), target)
    return target


def load_tep(folder):
    """Return two dicts {class_id: array of shape (time, 52)} for train and test files."""
    train, test = {}, {}
    for k in range(N_CLASSES):
        tr = np.loadtxt(os.path.join(folder, f"d{k:02d}.dat"))
        if tr.shape[0] == N_VARS:          # d00.dat is stored transposed (52 x 500)
            tr = tr.T
        train[k] = tr
        test[k] = np.loadtxt(os.path.join(folder, f"d{k:02d}_te.dat"))
    return train, test


# ---------------------------------------------------------------------------
# 2. Windows, splits, scaling
# ---------------------------------------------------------------------------
def make_windows(series, window, stride=1):
    """Cut a (time, vars) array into overlapping windows -> (n_windows, window, vars)."""
    starts = range(0, series.shape[0] - window + 1, stride)
    return np.stack([series[s:s + window] for s in starts])


def build_datasets(train, test, window=20, val_fraction=0.2, stride=1):
    """
    Train/validation come from the training files, test from the *_te files.

    - Validation = the LAST val_fraction of every training run, separated from
      the training part by a gap of `window` samples so no window overlaps both.
    - Test: d00_te is all normal; in d01_te..d21_te the fault starts at sample 160,
      so we keep only windows that start at or after sample 160.
    """
    Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []
    for k in range(N_CLASSES):
        run = train[k]
        cut = int(len(run) * (1 - val_fraction))
        w_tr = make_windows(run[:cut], window, stride)
        w_va = make_windows(run[cut + window:], window, stride)
        Xtr.append(w_tr); ytr.append(np.full(len(w_tr), k))
        Xva.append(w_va); yva.append(np.full(len(w_va), k))

        te_run = test[k] if k == 0 else test[k][TEST_ONSET:]
        w_te = make_windows(te_run, window, stride)
        Xte.append(w_te); yte.append(np.full(len(w_te), k))
    cat = np.concatenate
    return (cat(Xtr), cat(ytr)), (cat(Xva), cat(yva)), (cat(Xte), cat(yte))


class Standardizer:
    """Per-variable z-score. Fit on training data only, then apply everywhere."""
    def fit(self, X):                       # X: (n, window, vars)
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Graphs
# ---------------------------------------------------------------------------
def _idx(name):
    return VAR_NAMES.index(name)


# Process units of the Tennessee Eastman flowsheet (Downs & Vogel 1993).
UNITS = {
    "feed":       ["XMEAS1", "XMEAS2", "XMEAS3", "XMV1", "XMV2", "XMV3"],
    "reactor":    ["XMEAS6", "XMEAS7", "XMEAS8", "XMEAS9", "XMEAS21",
                   "XMEAS23", "XMEAS24", "XMEAS25", "XMEAS26", "XMEAS27", "XMEAS28",
                   "XMV10"],
    "separator":  ["XMEAS11", "XMEAS12", "XMEAS13", "XMEAS14", "XMEAS22", "XMV7", "XMV11"],
    "compressor": ["XMEAS5", "XMEAS20", "XMV5"],
    "purge":      ["XMEAS10"] + [f"XMEAS{i}" for i in range(29, 37)] + ["XMV6"],
    "stripper":   ["XMEAS4", "XMEAS15", "XMEAS16", "XMEAS17", "XMEAS18", "XMEAS19",
                   "XMV4", "XMV8", "XMV9"],
    "product":    [f"XMEAS{i}" for i in range(37, 42)],
}

# Links between units along the material streams, written as variable pairs.
STREAM_LINKS = [
    ("XMEAS1", "XMEAS6"), ("XMEAS2", "XMEAS6"), ("XMEAS3", "XMEAS6"),  # feeds -> reactor feed (s6)
    ("XMEAS5", "XMEAS6"),                       # recycle (s8) -> reactor feed (s6)
    ("XMEAS4", "XMEAS6"),                       # stream 4 reaches s6 via stripper overhead (s5)
    ("XMEAS7", "XMEAS13"), ("XMEAS9", "XMEAS11"),  # reactor -> condenser/separator (s7)
    ("XMEAS13", "XMEAS5"), ("XMEAS13", "XMEAS20"),  # separator vapour -> compressor
    ("XMEAS13", "XMEAS10"),                     # separator vapour -> purge (s9)
    ("XMEAS14", "XMEAS15"), ("XMV7", "XMEAS15"),  # separator liquid (s10) -> stripper
] + [("XMEAS17", f"XMEAS{i}") for i in range(37, 42)]  # stripper bottoms (s11) -> product analysis

# Control loops, read from temain_mod.f (the closed-loop code distributed with the data).
# Each pair = (variable the controller watches, valve it finally moves).
CONTROL_LINKS = [
    ("XMEAS2", "XMV1"), ("XMEAS3", "XMV2"), ("XMEAS1", "XMV3"), ("XMEAS4", "XMV4"),
    ("XMEAS5", "XMV5"), ("XMEAS10", "XMV6"), ("XMEAS13", "XMV6"), ("XMEAS12", "XMV7"),
    ("XMEAS15", "XMV8"), ("XMEAS19", "XMV9"), ("XMEAS21", "XMV10"), ("XMEAS17", "XMV11"),
    # cascade loops: outer measurement -> valve of the inner loop
    ("XMEAS18", "XMV9"), ("XMEAS8", "XMV4"), ("XMEAS9", "XMV10"), ("XMEAS23", "XMV3"),
    ("XMEAS26", "XMV1"), ("XMEAS27", "XMV2"), ("XMEAS30", "XMV6"), ("XMEAS38", "XMV9"),
]


def edges_to_index(pairs):
    """Undirected pairs of names -> PyG-style edge_index array of shape (2, 2*E)."""
    src, dst = [], []
    for a, b in sorted(set(tuple(sorted((_idx(a), _idx(b)))) for a, b in pairs)):
        src += [a, b]; dst += [b, a]
    return np.array([src, dst], dtype=np.int64)


def unit_pairs():
    pairs = []
    for members in UNITS.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs.append((a, b))
    return pairs


def structure_graph(include=("unit", "stream", "control")):
    """Hand-authored process-knowledge graph."""
    pairs = []
    if "unit" in include:
        pairs += unit_pairs()
    if "stream" in include:
        pairs += STREAM_LINKS
    if "control" in include:
        pairs += CONTROL_LINKS
    return edges_to_index(pairs)


def correlation_graph(X_normal, n_edges):
    """Data-driven graph: the n_edges variable pairs with the largest |correlation|."""
    flat = X_normal.reshape(-1, X_normal.shape[-1])
    C = np.abs(np.corrcoef(flat.T))
    np.fill_diagonal(C, 0)
    iu = np.triu_indices(C.shape[0], k=1)
    order = np.argsort(C[iu])[::-1][:n_edges]
    pairs = [(VAR_NAMES[iu[0][o]], VAR_NAMES[iu[1][o]]) for o in order]
    return edges_to_index(pairs)


def random_graph(n_edges, seed=0):
    """Control: same number of edges, placed at random."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N_VARS, k=1)
    pick = rng.choice(len(iu[0]), size=n_edges, replace=False)
    pairs = [(VAR_NAMES[iu[0][p]], VAR_NAMES[iu[1][p]]) for p in pick]
    return edges_to_index(pairs)


def empty_graph():
    """No edges at all: every sensor is processed on its own."""
    return np.zeros((2, 0), dtype=np.int64)


EDGE_TYPES = ("unit", "stream", "control")


def typed_structure_graph():
    """The knowledge graph with one relation type per edge: 0 = unit, 1 = stream, 2 = control.

    A pair that is both a unit and a control edge appears once per type.
    Returns (edge_index, edge_type).
    """
    parts = [structure_graph((t,)) for t in EDGE_TYPES]
    edge_index = np.concatenate(parts, axis=1)
    edge_type = np.concatenate([np.full(p.shape[1], i) for i, p in enumerate(parts)])
    return edge_index, edge_type


def shuffled_edge_types(edge_type, seed=0):
    """Control for typed edges: the same edges, their types permuted at random.

    Each undirected edge is stored as two consecutive directed edges, so the
    permutation is done per undirected edge and both directions keep one type.
    """
    pair_types = edge_type[::2].copy()
    np.random.default_rng(seed).shuffle(pair_types)
    return np.repeat(pair_types, 2)


# ---------------------------------------------------------------------------
# 4. Training and evaluation
# ---------------------------------------------------------------------------
HARD_FAULTS = (3, 9, 15)   # practically unobservable in this dataset


def _unpack(batch, device):
    """Works for plain (x, y) batches and for PyTorch Geometric batches."""
    if isinstance(batch, (list, tuple)):
        x, y = batch
        return x.to(device), y.to(device)
    batch = batch.to(device)
    return batch, batch.y


def evaluate(model, loader, device="cpu"):
    """Return (macro-F1 over all 22 classes, predictions, true labels)."""
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = _unpack(batch, device)
            preds.append(model(x).argmax(dim=1).cpu())
            labels.append(y.cpu())
    P, Y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return f1_score(Y, P, average="macro"), P, Y


def macro_f1_without_hard(Y, P):
    """Macro-F1 averaged over the 19 classes other than faults 3, 9 and 15."""
    from sklearn.metrics import f1_score
    keep = [c for c in range(N_CLASSES) if c not in HARD_FAULTS]
    return f1_score(Y, P, labels=keep, average="macro")


def fit(model, train_loader, val_loader, epochs=40, lr=1e-3, weight_decay=1e-4,
        device="cpu", verbose=True):
    """Train with Adam; keep the weights of the epoch with the best validation macro-F1."""
    import copy
    import torch
    import torch.nn.functional as F
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = _unpack(batch, device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y); n += len(y)
        val_f1 = evaluate(model, val_loader, device)[0]
        history.append((epoch, total / n, val_f1))
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"epoch {epoch:3d}   train loss {total / n:.3f}   val macro-F1 {val_f1:.3f}")
    model.load_state_dict(best_state)
    return history


def set_seed(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 5. Reading the control loops from the simulator code
# ---------------------------------------------------------------------------
def extract_control_loops(fortran_path):
    """Read the PI loops from temain_mod.f (the closed-loop code shipped with the data).

    Finds every controller subroutine that is actually called, the measurement it
    watches (``ERR = SETPT - XMEAS(m)``) and what it moves (``XMV(j)`` directly, or
    the setpoint of another controller in a cascade), and follows cascades down to
    the valve. Returns a dict {controller number: ("XMEASm", "XMVj")}.

    Override logic written as IF statements (the purge valve is forced open on high
    separator pressure inside CONTRL6) is not found by this pattern; CONTROL_LINKS
    adds it by hand.
    """
    import re
    src = open(fortran_path).read().splitlines()
    code = "\n".join(line for line in src if line and line[0] not in "Cc*")   # drop comment lines
    called = {int(n) for n in re.findall(r"CALL\s+CONTRL(\d+)", code)}
    loops = {}
    for block in re.split(r"SUBROUTINE\s+CONTRL", code)[1:]:
        n = int(re.match(r"(\d+)", block).group(1))
        if n not in called:
            continue
        watched = re.search(r"ERR\d+\s*=\s*\(?\s*SETPT\(\d+\)\s*-\s*XMEAS\((\d+)\)", block)
        moved = re.search(r"\n\s*(XMV|SETPT)\((\d+)\)\s*=\s*(?:XMV|SETPT)\(\d+\)\s*\+\s*DXMV", block)
        loops[n] = (int(watched.group(1)), moved.group(1), int(moved.group(2)))

    def final_valve(n):
        _, kind, target = loops[n]
        return target if kind == "XMV" else final_valve(target)

    return {n: (f"XMEAS{loops[n][0]}", f"XMV{final_valve(n)}") for n in sorted(loops)}
