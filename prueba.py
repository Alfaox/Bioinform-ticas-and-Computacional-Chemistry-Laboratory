import os
import json
import math
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score, confusion_matrix, make_scorer

# =========================
# CONFIG
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Input dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "diversified_master.xlsx")
SMILES_COL = "SMILES"
CLASS_COL  = "CLASS"

OUTDIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTDIR, exist_ok=True)

FP_RADIUS = 2
FP_NBITS  = 2048  # Aumentado a 2048 para mejor colisión/resolución

# Model parameters (Random Forest)
N_ESTIMATORS = 200
N_JOBS = -1

# DoA
DOA_K = 5
DOA_PERCENTILE = 5
DOA_TRAIN_SUBSAMPLE = 20000

# Split
TEST_SIZE = 0.20

# Thresholds default
THR_DEFAULT = 0.50

# =========================
# Helpers
# =========================
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)

def scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return "NO_SCAFFOLD"
        s = Chem.MolToSmiles(scaf)
        return s if s else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"

def morgan_fp(mol: Chem.Mol, radius: int, nBits: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def fps_to_csr(fps: List, nBits: int) -> csr_matrix:
    rows, cols, data = [], [], []
    for i, fp in enumerate(fps):
        onbits = list(fp.GetOnBits())
        rows.extend([i] * len(onbits))
        cols.extend(onbits)
        data.extend([1] * len(onbits))
    return csr_matrix((data, (rows, cols)), shape=(len(fps), nBits), dtype=np.uint8)

def knn_mean_similarity(fp_query, fps_train: List, k: int = 5) -> float:
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, fps_train)
    if not sims:
        return 0.0
    topk = np.sort(np.asarray(sims))[-k:]
    return float(np.mean(topk))

def doa_threshold_from_train(fps_train: List, k: int, percentile: float, sample_n: int = 1200, seed: int = 13) -> float:
    rng = np.random.RandomState(seed)
    n = len(fps_train)
    if n == 0:
        return 0.0
    sample_n = min(sample_n, n)
    idx = rng.choice(np.arange(n), size=sample_n, replace=False)
    vals = [knn_mean_similarity(fps_train[i], fps_train, k=k) for i in idx]
    return float(np.percentile(vals, percentile))


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def make_binary_labels(y: np.ndarray, positive_class: str) -> np.ndarray:
    return (y == positive_class).astype(int)

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"MCC": mcc, "Recall": rec, "Precision": prec, "F1": f1}

def train_optimize_rf(X_train, y_train, X_test, y_test, label_name: str, groups_train=None):
    """
    Entrena un Random Forest usando RandomizedSearchCV para buscar mejores hiperparámetros.
    Optimiza para MCC.
    """
    print(f"\n--- Optimizando modelo para: {label_name} ---")
    
    # Grid básico para buscar equilibrio
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS)

    # Usar GroupKFold si se proveen grupos, sino StratifiedKFold
    cv = 3
    if groups_train is not None:
         # StratifiedGroupKFold es ideal para splits químicos por scaffold
         cv = StratifiedGroupKFold(n_splits=3)
    
    # Buscamos maximizar MCC
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring=mcc_scorer, 
        cv=cv, 
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1
    )
    
    if groups_train is not None:
        search.fit(X_train, y_train, groups=groups_train)
    else:
        search.fit(X_train, y_train)
        
    best_clf = search.best_estimator_
    print(f"Mejores params ({label_name}): {search.best_params_}")
    
    # Validación en Test Set (Hold-out independiente)
    y_prob_test = best_clf.predict_proba(X_test)[:, 1]
    
    # Calcular métricas con umbral 0.5 por defecto
    metrics = evaluate_metrics(y_test, y_prob_test, threshold=0.5)
    
    print(f"Resultados en TEST ({label_name}) [Th=0.5]:")
    print(f"  MCC:       {metrics['MCC']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    
    return best_clf

def main():
    print("Cargando datos...")
    df = load_table(DATA_PATH)
    df = df[[SMILES_COL, CLASS_COL]].dropna().copy()
    df[SMILES_COL] = df[SMILES_COL].astype(str)
    df[CLASS_COL]  = df[CLASS_COL].astype(str)

    print(f"Total registros: {len(df)}")

    # Parse mols
    mols, smiles_can, scaffs, y, fps = [], [], [], [], []
    print("Generando fingerprints y scaffolds...")
    for s, c in zip(df[SMILES_COL].values, df[CLASS_COL].values):
        m = smiles_to_mol(s)
        if m is None:
            continue
        mols.append(m)
        smiles_can.append(canonical_smiles(m))
        scaffs.append(scaffold_smiles(m))
        y.append(c)
        fps.append(morgan_fp(m, FP_RADIUS, FP_NBITS))

    df2 = pd.DataFrame({
        "SMILES": smiles_can,
        "CLASS": y,
        "Scaffold": scaffs,
    })
    y = np.asarray(y)
    X = fps_to_csr(fps, FP_NBITS)

    # Split por scaffold (GroupShuffleSplit) - Esto es CRÍTICO para evitar sobreajuste "químico"
    # El test set será químicamente distinto al train set.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    tr, te = next(splitter.split(df2, y=y, groups=df2["Scaffold"].values))

    X_train, X_test = X[tr], X[te]
    y_train_full, y_test_full = y[tr], y[te]
    groups_train = df2["Scaffold"].iloc[tr].values

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Entrenar 3 binarios optimizados
    # DSSTOX
    y_train_dsstox = make_binary_labels(y_train_full, "DSSTOX")
    y_test_dsstox  = make_binary_labels(y_test_full, "DSSTOX")
    clf_dsstox = train_optimize_rf(X_train, y_train_dsstox, X_test, y_test_dsstox, "DSSTOX-like", groups_train)

    # CLUE
    y_train_clue = make_binary_labels(y_train_full, "CLUE")
    y_test_clue  = make_binary_labels(y_test_full, "CLUE")
    clf_clue = train_optimize_rf(X_train, y_train_clue, X_test, y_test_clue, "CLUE-like", groups_train)

    # UIREF
    y_train_uiref = make_binary_labels(y_train_full, "UIREF")
    y_test_uiref  = make_binary_labels(y_test_full, "UIREF")
    clf_uiref = train_optimize_rf(X_train, y_train_uiref, X_test, y_test_uiref, "UIREF-like", groups_train)

    # DoA: guarda subset de fps de train
    fps_train = [fps[i] for i in tr]
    if len(fps_train) > DOA_TRAIN_SUBSAMPLE:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(np.arange(len(fps_train)), size=DOA_TRAIN_SUBSAMPLE, replace=False)
        fps_train_sub = [fps_train[i] for i in idx]
    else:
        fps_train_sub = fps_train

    print("\nCalculando DoA threshold...")
    doa_thr = doa_threshold_from_train(
        fps_train=fps_train_sub,
        k=DOA_K,
        percentile=DOA_PERCENTILE,
        sample_n=min(1200, len(fps_train_sub)),
        seed=RANDOM_SEED
    )

    # Guardar
    print("\nGuardando modelos...")
    joblib.dump(clf_dsstox, os.path.join(OUTDIR, "clf_stage1_dsstoxlike.joblib"))
    joblib.dump(clf_clue,   os.path.join(OUTDIR, "clf_stage2_cluelike.joblib"))
    joblib.dump(clf_uiref,  os.path.join(OUTDIR, "clf_stage3_uireflike.joblib"))

    # Guardar fps_train_sub para DoA
    joblib.dump(fps_train_sub, os.path.join(OUTDIR, "doa_train_fps_sub.joblib"))

    config = {
        "FP_RADIUS": FP_RADIUS,
        "FP_NBITS": FP_NBITS,
        "DOA_K": DOA_K,
        "DOA_PERCENTILE": DOA_PERCENTILE,
        "DOA_THRESHOLD": doa_thr,
        "THR_DSSTOX": THR_DEFAULT,
        "THR_CLUE": THR_DEFAULT,
        "THR_UIREF": THR_DEFAULT,
        "LABELS": ["DSSTOX", "CLUE", "UIREF", "COCONUT"],
        "NOTE": "Modelos Random Forest Optimizados (MCC/Recall). Validado con Scaffold Split."
    }
    with open(os.path.join(OUTDIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("OK. Modelos actualizados en:", OUTDIR)
    print("DoA threshold:", doa_thr)


if __name__ == "__main__":
    main()
