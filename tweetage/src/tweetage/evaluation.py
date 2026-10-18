"""Evaluación v2: F1 macro rápido, Δ pareado, bootstrap y regla de decisión.

La regla para considerar que un experimento "pasa" (y así vale la pena enviarlo o
combinarlo) es:

* Δ F1 > 0 en la semilla de ajuste y en la de confirmación, y
* el intervalo bootstrap al 95 % del Δ en la semilla de confirmación tiene límite
  inferior > 0.

"fuerte" = pasa y además el Δ medio de las dos semillas es ≥ 0.002.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tweetage.pipeline import CLASES

K = len(CLASES)
_IDX = {c: i for i, c in enumerate(CLASES)}


def to_idx(y) -> np.ndarray:
    return np.fromiter((_IDX[c] for c in y), dtype=np.int64, count=len(y))


def as_pred(p) -> np.ndarray:
    """Acepta probabilidades/puntajes (n, 6) o predicciones ya hechas (n,)."""
    p = np.asarray(p)
    return p.argmax(axis=1) if p.ndim == 2 else p


def _f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    pred_tot, true_tot = cm.sum(axis=0), cm.sum(axis=1)
    prec = np.divide(tp, pred_tot, out=np.zeros(K), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros(K), where=true_tot > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros(K), where=den > 0)
    return float(f1.mean())


def f1_macro_idx(y_idx: np.ndarray, pred_idx: np.ndarray) -> float:
    cm = np.bincount(y_idx * K + pred_idx, minlength=K * K).reshape(K, K)
    return _f1_from_cm(cm)


def f1_macro(y_idx, p) -> float:
    return f1_macro_idx(y_idx, as_pred(p))


def accuracy(y_idx, p) -> float:
    return float((as_pred(p) == y_idx).mean())


def f1_per_class(y_idx, p) -> pd.Series:
    pred = as_pred(p)
    out = {}
    for k, c in enumerate(CLASES):
        tp = np.sum((pred == k) & (y_idx == k))
        fp = np.sum((pred == k) & (y_idx != k))
        fn = np.sum((pred != k) & (y_idx == k))
        out[c] = 2 * tp / max(1, 2 * tp + fp + fn)
    return pd.Series(out)


def paired_bootstrap(y_idx, pred_a, pred_b, n_boot: int = 1000, seed: int = 0):
    """Intervalo al 95 % de F1(b) - F1(a), remuestreando las mismas filas para ambos."""
    pred_a, pred_b = as_pred(pred_a), as_pred(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_idx)
    code_a, code_b = y_idx * K + pred_a, y_idx * K + pred_b
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cm_a = np.bincount(code_a[idx], minlength=K * K).reshape(K, K)
        cm_b = np.bincount(code_b[idx], minlength=K * K).reshape(K, K)
        deltas[i] = _f1_from_cm(cm_b) - _f1_from_cm(cm_a)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def compare(nombre, y_aj, base_aj, cand_aj, y_cf, base_cf, cand_cf, n_boot: int = 1000) -> dict:
    """Fila de resultados de un experimento frente a la mezcla vigente."""
    f_ba, f_ca = f1_macro(y_aj, base_aj), f1_macro(y_aj, cand_aj)
    f_bc, f_cc = f1_macro(y_cf, base_cf), f1_macro(y_cf, cand_cf)
    lo, hi = paired_bootstrap(y_cf, base_cf, cand_cf, n_boot=n_boot)
    d_aj, d_cf = f_ca - f_ba, f_cc - f_bc
    pasa = d_aj > 0 and d_cf > 0 and lo > 0
    return {
        "experimento": nombre,
        "f1_ajuste": f_ca,
        "f1_confirmacion": f_cc,
        "acc_ajuste": accuracy(y_aj, cand_aj),
        "acc_confirmacion": accuracy(y_cf, cand_cf),
        "delta_ajuste": d_aj,
        "delta_confirmacion": d_cf,
        "delta_medio": (d_aj + d_cf) / 2,
        "ic95_bajo": lo,
        "ic95_alto": hi,
        "pred_cambiadas": float(np.mean(as_pred(base_cf) != as_pred(cand_cf))),
        "pasa": pasa,
        "fuerte": pasa and (d_aj + d_cf) / 2 >= 0.002,
    }


def pred_distribution(p) -> pd.Series:
    pred = as_pred(p)
    return pd.Series(np.bincount(pred, minlength=K), index=CLASES)
