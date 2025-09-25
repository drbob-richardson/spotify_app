#!/usr/bin/env python3
# build_spotify_models.py — v3 (6 families × Low/Medium/High)
from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional XGBoost
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

import skops.io as skio

# ------------------------ Paths (hard-coded to your project) ------------------------
HERE = Path("/Users/robertrichardson/Documents/Spotify_App").resolve()
DATA_CANDIDATES = [HERE / "dataset.csv", Path("/mnt/data/dataset.csv")]
OUT_DIR = (HERE / "artifacts" / "spotify_v3").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[trainer] Writing artifacts to: {OUT_DIR}")

# ------------------------ Data schema ------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.20
DRY_RUN_N    = None

DROP_COLS = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]
NUM_COLS  = ["duration_ms", "danceability", "energy", "loudness", "speechiness",
             "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
CAT_COLS  = ["explicit", "key", "mode", "time_signature", "track_genre"]

CLASS_NAMES = ["Low","Medium","High"]

# ------------------------ IO helpers ------------------------
def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists(): return p
    raise FileNotFoundError(f"dataset.csv not found in: {[str(p) for p in DATA_CANDIDATES]}")

def rollup_rare_categories(df: pd.DataFrame, min_freq: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    for col, thr in min_freq.items():
        if col in df.columns:
            vc = df[col].value_counts()
            rare = set(vc[vc < thr].index)
            if len(rare): df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DROP_COLS:
        if c in df.columns: df = df.drop(columns=c)
    df = rollup_rare_categories(df, {"track_genre": 200, "key": 50, "time_signature": 50})
    missing = [c for c in NUM_COLS + CAT_COLS + ["popularity"] if c not in df.columns]
    if missing: raise ValueError(f"Missing expected columns: {missing}")
    return df

def make_labels(df: pd.DataFrame) -> pd.Series:
    q1, q2 = df["popularity"].quantile([0.3333, 0.6666])
    def to_pop3(p):
        if p <= q1: return "Low"
        if p <= q2: return "Medium"
        return "High"
    return df["popularity"].apply(to_pop3)

def split_train_val_test(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, stratify=y_train, random_state=RANDOM_STATE
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test

# ------------------------ Preprocessors ------------------------
def _ohe_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def pre_poly_dense() -> ColumnTransformer:
    # For linear/NN: squares + interactions + scale
    num = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler(with_mean=True))])
    cat = _ohe_dense()
    return ColumnTransformer(
        [("cat", cat, CAT_COLS), ("num", num, NUM_COLS)],
        remainder="drop", sparse_threshold=0.0
    )

def pre_tree() -> ColumnTransformer:
    # For tree/rf/xgb: no scaling, no poly; OHE for cats
    cat = _ohe_dense()
    return ColumnTransformer(
        [("cat", cat, CAT_COLS), ("num", "passthrough", NUM_COLS)],
        remainder="drop", sparse_threshold=0.0
    )

def pre_kmeans() -> ColumnTransformer:
    # For kmeans: scale numeric; OHE cats
    cat = _ohe_dense()
    num = Pipeline([("scaler", StandardScaler(with_mean=True))])
    return ColumnTransformer(
        [("cat", cat, CAT_COLS), ("num", num, NUM_COLS)],
        remainder="drop", sparse_threshold=0.0
    )

# ------------------------ Metrics ------------------------
def metrics_multi(y_true, y_pred, labels=CLASS_NAMES):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "n_test": int(len(y_true)),
        "class_balance": pd.Series(y_true).value_counts(normalize=True).to_dict(),
    }

def save_model(model, out_dir: Path, metrics: Dict[str,Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    skio.dump(model, file=out_dir / "model.skops")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {out_dir}/model.skops")

# ------------------------ Simple KMeans classifier ------------------------
class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """KMeans with C clusters (flexibility) mapped to class distributions."""
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c:i for i,c in enumerate(self.classes_)}
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        cl = self.kmeans_.fit_predict(X)
        counts = np.zeros((self.n_clusters, len(self.classes_)), dtype=float)
        for ci, yi in zip(cl, y):
            counts[ci, self.class_to_idx_[yi]] += 1.0
        counts += 1e-6
        self.cluster_proba_ = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, X):
        ci = self.kmeans_.predict(X)
        return self.cluster_proba_[ci]

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array([self.classes_[i] for i in np.argmax(proba, axis=1)])

# ------------------------ Family trainers (Low/Medium/High) ------------------------
def train_logreg_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    pre = pre_poly_dense()
    Cs = np.array([0.5, 1.0, 2.0, 4.0])

    # LOW: top-10 with L2
    pipe_low = Pipeline([
        ("pre", pre),
        ("sel", ("passthrough",)),  # placeholder replaced below if needed
    ])
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    pipe_low = Pipeline([
        ("pre", pre),
        ("sel", SelectKBest(mutual_info_classif, k=10)),
        ("clf", LogisticRegressionCV(Cs=Cs, multi_class="multinomial",
                                     solver="lbfgs", max_iter=3000, cv=5)),
    ])
    pipe_low.fit(X_tr, y_tr); pred = pipe_low.predict(X_te)
    save_model(pipe_low, task_dir / "logreg_low", metrics_multi(y_te, pred))

    # MEDIUM: L1 sparse subset
    pipe_med = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegressionCV(Cs=Cs, penalty="l1", solver="saga",
                                     multi_class="multinomial", max_iter=3000, cv=5, n_jobs=-1)),
    ])
    pipe_med.fit(X_tr, y_tr); pred = pipe_med.predict(X_te)
    save_model(pipe_med, task_dir / "logreg_medium", metrics_multi(y_te, pred))

    # HIGH: L2 all features
    pipe_high = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegressionCV(Cs=Cs, penalty="l2", solver="lbfgs",
                                     multi_class="multinomial", max_iter=3000, cv=5, n_jobs=-1)),
    ])
    pipe_high.fit(X_tr, y_tr); pred = pipe_high.predict(X_te)
    save_model(pipe_high, task_dir / "logreg_high", metrics_multi(y_te, pred))

def train_tree_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    pre = pre_tree()

    # Build pruning alphas from train
    Xtr_proc = pre.fit_transform(X_tr)
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    base.fit(Xtr_proc, y_tr)
    path = base.cost_complexity_pruning_path(Xtr_proc, y_tr)
    alphas = np.unique(path.ccp_alphas)
    alphas = alphas[alphas >= 0.0]
    if len(alphas) > 60:
        idx = np.linspace(0, len(alphas)-1, 60, dtype=int)
        alphas = alphas[idx]

    # Evaluate on val to pick best
    def score_alpha(a: float):
        pipe = Pipeline([("pre", pre),
                         ("clf", DecisionTreeClassifier(ccp_alpha=float(a), random_state=RANDOM_STATE))])
        pipe.fit(X_tr, y_tr)
        v_pred = pipe.predict(X_val)
        return f1_score(y_val, v_pred, average="macro"), pipe

    scores = []
    for a in alphas:
        s, pipe = score_alpha(float(a))
        scores.append((s, a, pipe))
    scores.sort(key=lambda t: t[0], reverse=True)
    best_alpha = float(scores[0][1])

    # LOW: heavy prune (90th percentile)
    a_low = float(np.quantile(alphas, 0.90))
    pipe_low = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(ccp_alpha=a_low, random_state=RANDOM_STATE))])
    pipe_low.fit(X_tr, y_tr); pred = pipe_low.predict(X_te)
    save_model(pipe_low, task_dir / "tree_low", metrics_multi(y_te, pred))

    # MEDIUM: best α
    pipe_med = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=RANDOM_STATE))])
    pipe_med.fit(X_tr, y_tr); pred = pipe_med.predict(X_te)
    save_model(pipe_med, task_dir / "tree_medium", metrics_multi(y_te, pred))

    # HIGH: unpruned
    pipe_high = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(ccp_alpha=0.0, random_state=RANDOM_STATE))])
    pipe_high.fit(X_tr, y_tr); pred = pipe_high.predict(X_te)
    save_model(pipe_high, task_dir / "tree_high", metrics_multi(y_te, pred))

def train_mlp_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    from sklearn.neural_network import MLPClassifier
    pre = pre_poly_dense()

    candidates: List[Tuple[Tuple[int,...], float]] = [
        ((64,),            1e-3),
        ((64,64),          5e-4),
        ((64,64,64),       5e-4),
        ((128,64,32),      1e-3),
        ((128,128,128),    5e-4),
        ((128,128,128,64), 1e-3),
        ((128,128,128,128,128), 1e-3),
    ]
    trials = []
    for hidden, alpha in candidates:
        clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                            alpha=alpha, solver="adam", max_iter=800,
                            learning_rate_init=0.003, early_stopping=True,
                            validation_fraction=0.15, n_iter_no_change=25,
                            batch_size=256, random_state=RANDOM_STATE)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        v_pred = pipe.predict(X_val); v_f1 = f1_score(y_val, v_pred, average="macro")
        t_pred = pipe.predict(X_te);  t_met = metrics_multi(y_te, t_pred)
        trials.append({"hidden": hidden, "alpha": alpha, "val_f1": float(v_f1),
                       "pipe": pipe, "metrics": t_met,
                       "complexity": sum(hidden) + 10*len(hidden)})

    mid = max(trials, key=lambda r: r["val_f1"])                      # MEDIUM
    low = min(trials, key=lambda r: (r["complexity"], -r["val_f1"]))  # LOW
    high = max(trials, key=lambda r: (r["complexity"], r["val_f1"]))  # HIGH

    save_model(low["pipe"],  task_dir / "mlp_low",    low["metrics"])
    save_model(mid["pipe"],  task_dir / "mlp_medium", mid["metrics"])
    save_model(high["pipe"], task_dir / "mlp_high",   high["metrics"])

def train_rf_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    pre = pre_tree()

    # LOW: simpler forest
    pipe_low = Pipeline([("pre", pre),
                         ("clf", RandomForestClassifier(
                             n_estimators=120, max_depth=6, min_samples_leaf=5,
                             max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE
                         ))])
    pipe_low.fit(X_tr, y_tr); pred = pipe_low.predict(X_te)
    save_model(pipe_low, task_dir / "rf_low", metrics_multi(y_te, pred))

    # MEDIUM: small grid (pick best on val)
    grid = [
        dict(n_estimators=250, max_depth=None, min_samples_leaf=2, max_features="sqrt"),
        dict(n_estimators=350, max_depth=12,   min_samples_leaf=2, max_features="sqrt"),
        dict(n_estimators=350, max_depth=20,   min_samples_leaf=1, max_features="log2"),
    ]
    best = None
    best_score = -1.0
    for params in grid:
        pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **params))])
        pipe.fit(X_tr, y_tr)
        v_pred = pipe.predict(X_val); v_f1 = f1_score(y_val, v_pred, average="macro")
        if v_f1 > best_score:
            best_score, best = v_f1, (pipe, params)
    pipe_med, _ = best
    pred = pipe_med.predict(X_te)
    save_model(pipe_med, task_dir / "rf_medium", metrics_multi(y_te, pred))

    # HIGH: unnecessarily flexible
    pipe_high = Pipeline([("pre", pre),
                          ("clf", RandomForestClassifier(
                              n_estimators=700, max_depth=None, min_samples_leaf=1,
                              max_features=None, n_jobs=-1, random_state=RANDOM_STATE
                          ))])
    pipe_high.fit(X_tr, y_tr); pred = pipe_high.predict(X_te)
    save_model(pipe_high, task_dir / "rf_high", metrics_multi(y_te, pred))

def train_xgb_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    if not HAVE_XGB:
        return
    pre = pre_tree()

    def make_clf(n_estimators, max_depth, lr, subsample=1.0, colsample=1.0, reg=0.0):
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            subsample=subsample, colsample_bytree=colsample,
            reg_lambda=reg,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1
        )

    # LOW: small, shallow
    pipe_low = Pipeline([("pre", pre), ("clf", make_clf(200, 3, 0.2, 0.9, 0.9, 1.0))])
    pipe_low.fit(X_tr, y_tr); pred = pipe_low.predict(X_te)
    save_model(pipe_low, task_dir / "xgb_low", metrics_multi(y_te, pred))

    # MEDIUM: tuned candidates (select on val)
    cands = [
        make_clf(300, 4, 0.15, 0.9, 0.9, 1.0),
        make_clf(400, 5, 0.10, 0.9, 0.8, 1.0),
        make_clf(450, 6, 0.07, 0.85, 0.8, 1.0),
    ]
    best_pipe, best_score = None, -1.0
    for clf in cands:
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        v_pred = pipe.predict(X_val); v_f1 = f1_score(y_val, v_pred, average="macro")
        if v_f1 > best_score:
            best_score, best_pipe = v_f1, pipe
    pred = best_pipe.predict(X_te)
    save_model(best_pipe, task_dir / "xgb_medium", metrics_multi(y_te, pred))

    # HIGH: deep & many trees
    pipe_high = Pipeline([("pre", pre), ("clf", make_clf(800, 8, 0.05, 1.0, 1.0, 0.0))])
    pipe_high.fit(X_tr, y_tr); pred = pipe_high.predict(X_te)
    save_model(pipe_high, task_dir / "xgb_high", metrics_multi(y_te, pred))

def train_kmeans_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir: Path):
    pre = pre_kmeans()

    def eval_k(k: int):
        pipe = Pipeline([("pre", pre), ("clf", KMeansClassifier(n_clusters=k, random_state=RANDOM_STATE))])
        pipe.fit(X_tr, y_tr)
        v_pred = pipe.predict(X_val); v_f1 = f1_score(y_val, v_pred, average="macro")
        t_pred = pipe.predict(X_te)
        return v_f1, pipe, t_pred

    # LOW/MED/HIGH k (#clusters controls flexibility)
    low_v,  low_p,  low_pred  = eval_k(3)
    med_v,  med_p,  med_pred  = eval_k(6)
    high_v, high_p, high_pred = eval_k(12)

    save_model(low_p,  task_dir / "kmeans_low",  metrics_multi(y_te, low_pred))
    save_model(med_p,  task_dir / "kmeans_medium", metrics_multi(y_te, med_pred))
    save_model(high_p, task_dir / "kmeans_high", metrics_multi(y_te, high_pred))

# ------------------------ Orchestration ------------------------
def run(df: pd.DataFrame) -> Dict[str, Any]:
    y_all = make_labels(df)
    X_all = df.drop(columns=["popularity"])

    if DRY_RUN_N and len(df) > DRY_RUN_N:
        idxs = []
        for _, grp in X_all.assign(_y=y_all).groupby("_y"):
            k = min(len(grp), DRY_RUN_N // 3)
            idxs.extend(grp.sample(k, random_state=RANDOM_STATE).index.tolist())
        X_all, y_all = X_all.loc[idxs], y_all.loc[idxs]

    X_tr, X_val, X_te, y_tr, y_val, y_te = split_train_val_test(X_all, y_all)
    task_dir = OUT_DIR / "pop3"
    task_dir.mkdir(parents=True, exist_ok=True)

    ds_info = {
        "task": "pop3",
        "rows": {"total": int(len(X_all)), "train": int(len(X_tr)), "val": int(len(X_val)), "test": int(len(X_te))},
        "numeric_features": [c for c in NUM_COLS if c in X_all.columns],
        "categorical_features": [c for c in CAT_COLS if c in X_all.columns],
        "class_names": CLASS_NAMES,
    }
    (task_dir / "dataset_info.json").write_text(json.dumps(ds_info, indent=2))

    # Train all six families (each: low/medium/high)
    train_logreg_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)
    train_tree_tiers(  X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)
    train_mlp_tiers(   X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)
    train_rf_tiers(    X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)
    if HAVE_XGB:
        train_xgb_tiers( X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)
    train_kmeans_tiers(X_tr, X_val, X_te, y_tr, y_val, y_te, task_dir)

    # Manifest
    dirs = sorted([p.name for p in task_dir.iterdir() if p.is_dir()])
    families = {
        "logreg": [d for d in dirs if d.startswith("logreg_")],
        "tree":   [d for d in dirs if d.startswith("tree_")],
        "mlp":    [d for d in dirs if d.startswith("mlp_")],
        "rf":     [d for d in dirs if d.startswith("rf_")],
        "kmeans": [d for d in dirs if d.startswith("kmeans_")],
    }
    if HAVE_XGB:
        families["xgb"] = [d for d in dirs if d.startswith("xgb_")]

    manifest = {
        "base_dir": str(OUT_DIR),
        "tasks": {
            "pop3": {
                "path": str(task_dir),
                "dataset_info": ds_info,
                "families": families
            }
        }
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def main():
    print("Loading data…")
    df = load_df(find_data())
    run(df)
    print(f"Done. Artifacts at: {OUT_DIR}")

if __name__ == "__main__":
    main()
