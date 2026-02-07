import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class CascadePredictor:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

        with open(os.path.join(models_dir, "config.json"), "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.FP_RADIUS = int(self.cfg["FP_RADIUS"])
        self.FP_NBITS  = int(self.cfg["FP_NBITS"])

        self.THR_DSSTOX = float(self.cfg["THR_DSSTOX"])
        self.THR_CLUE   = float(self.cfg["THR_CLUE"])
        self.THR_UIREF  = float(self.cfg["THR_UIREF"])

        self.DOA_K = int(self.cfg["DOA_K"])
        self.DOA_THRESHOLD = float(self.cfg["DOA_THRESHOLD"])

        self.clf_dsstox = joblib.load(os.path.join(models_dir, "clf_stage1_dsstoxlike.joblib"))
        self.clf_clue   = joblib.load(os.path.join(models_dir, "clf_stage2_cluelike.joblib"))
        self.clf_uiref  = joblib.load(os.path.join(models_dir, "clf_stage3_uireflike.joblib"))

        self.fps_train_sub = joblib.load(os.path.join(models_dir, "doa_train_fps_sub.joblib"))

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    def _fp(self, mol: Chem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_RADIUS, nBits=self.FP_NBITS)

    def _fp_to_sparse_row(self, fp):
        # CSR 1xN
        onbits = list(fp.GetOnBits())
        if not onbits:
            from scipy.sparse import csr_matrix
            return csr_matrix((1, self.FP_NBITS), dtype=np.uint8)
        from scipy.sparse import csr_matrix
        data = np.ones(len(onbits), dtype=np.uint8)
        rows = np.zeros(len(onbits), dtype=np.int32)
        cols = np.array(onbits, dtype=np.int32)
        return csr_matrix((data, (rows, cols)), shape=(1, self.FP_NBITS), dtype=np.uint8)

    def _knn_mean_similarity(self, fp_query, k: int = 5) -> float:
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, self.fps_train_sub)
        if not sims:
            return 0.0
        topk = np.sort(np.asarray(sims))[-k:]
        return float(np.mean(topk))

    def predict(self, smiles: str) -> Dict[str, Any]:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return {"ok": False, "error": "SMILES inválido o no parseable."}

        smiles_can = Chem.MolToSmiles(mol, canonical=True)
        fp = self._fp(mol)
        X = self._fp_to_sparse_row(fp)

        # DoA
        knn_sim = self._knn_mean_similarity(fp, k=self.DOA_K)
        in_domain = bool(knn_sim >= self.DOA_THRESHOLD)

        # Stage 1: DSSTOX-like
        p_dsstox = float(self.clf_dsstox.predict_proba(X)[0, 1])
        stage1_alert = bool(p_dsstox >= self.THR_DSSTOX)

        # Stage 2: CLUE-like
        p_clue = float(self.clf_clue.predict_proba(X)[0, 1])
        stage2_pass = bool(p_clue >= self.THR_CLUE)

        # Stage 3: UIREF-like
        p_uiref = float(self.clf_uiref.predict_proba(X)[0, 1])
        stage3_candidate = bool(p_uiref >= self.THR_UIREF)

        # Determinar etiqueta final basada en las 3 etapas
        if stage1_alert:
            final_label = "ALERTA_TOXICIDAD"
        elif not stage2_pass:
            final_label = "NO_CONCLUYENTE_BIOACTIVIDAD"
        elif stage3_candidate:
            final_label = "CANDIDATO_UIREF_LIKE"
        else:
            final_label = "CLUE_LIKE_NO_UIREF"

        return {
            "ok": True,
            "smiles_input": smiles,
            "smiles_canonical": smiles_can,
            "doa": {
                "kNN_mean_sim": knn_sim,
                "threshold": self.DOA_THRESHOLD,
                "in_domain": in_domain
            },
            "stage1_dsstox_like": {
                "p": p_dsstox,
                "threshold": self.THR_DSSTOX,
                "alert": stage1_alert
            },
            "stage2_clue_like": {
                "p": p_clue,
                "threshold": self.THR_CLUE,
                "pass": stage2_pass
            },
            "stage3_uiref_like": {
                "p": p_uiref,
                "threshold": self.THR_UIREF,
                "candidate": stage3_candidate
            },
            "final_label": final_label,
            "note": self.cfg.get("NOTE", "")
        }
