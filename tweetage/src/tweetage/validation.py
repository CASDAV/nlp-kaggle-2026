"""Partición train/validación sin fuga de textos repetidos.

Grupo = texto normalizado (strip + casefold): todas las copias de un mismo texto
quedan del mismo lado. Estratificado por clase. Un fold de 5 (≈20 %) como validación.

StratifiedGroupKFold tarda ~25 s con ~168 mil grupos, así que los índices se
guardan en disco (misma carpeta de caché) y se reutilizan.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from tweetage.cache import CACHE_DIR, _fingerprint

N_SPLITS = 5


def groups_of(X) -> np.ndarray:
    return np.array([t.strip().casefold() for t in X])


def split_indices(X, y, seed: int, use_cache: bool = True):
    """Índices (train, valid) del primer fold para la semilla dada."""
    path = CACHE_DIR / f"split__seed{seed}__{_fingerprint(N_SPLITS, X, y)}.npz"
    if use_cache and path.exists():
        data = np.load(path)
        return data["train"], data["valid"]
    groups = groups_of(X)
    folds = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    idx_tr, idx_va = next(folds.split(X, y, groups=groups))
    overlap = len(set(groups[idx_tr]) & set(groups[idx_va]))
    if overlap:
        raise ValueError(f"{overlap} textos aparecen en train y validación")
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, train=idx_tr, valid=idx_va)
    return idx_tr, idx_va


def split(X, y, seed: int, use_cache: bool = True):
    """Devuelve (X_train, y_train, X_valid, y_valid)."""
    y = np.asarray(y)
    idx_tr, idx_va = split_indices(X, y, seed, use_cache)
    return [X[i] for i in idx_tr], y[idx_tr], [X[i] for i in idx_va], y[idx_va]
