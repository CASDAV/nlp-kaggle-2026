"""Mezcla v2: sesgos por clase, temperaturas y pesos ajustados de forma conjunta.

Todo se ajusta sobre las probabilidades de validación ya cacheadas (no se reentrena):

* ``fit_bias`` / ``apply_bias``: 6 desplazamientos en log-probabilidad elegidos por
  búsqueda de coordenadas para maximizar F1 macro (corrige el desbalance de predicciones).
* ``LinearBlend``: p = Σ_k w_k·g_k·softmax(log p_k / T_k) / Σ_k w_k·g_k (interpolación
  lineal de LMs). g_k es la compuerta del canal (1 si el tweet tiene su evidencia).
  Primero se calibra la temperatura de cada canal (log-loss propio) y luego los pesos
  minimizando el log-loss de la mezcla; opcionalmente se refinan ambos juntos.
* ``LogLinearBlend``: p ∝ exp(Σ_k w_k·g_k·log p_k) (interpolación log-lineal / producto
  de expertos, un peso no negativo por canal).
* ``add_channel`` / ``grid_add`` / ``specialist_apply``: protocolo del 02 (agregar un
  canal con una grilla de temperatura × peso sobre una base).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from tweetage import evaluation as E
from tweetage import models as M
from tweetage.pipeline import CLASES

# Rango de temperaturas: los NB multinomiales dan posteriores casi 0/1 y necesitan T alta.
T_MIN, T_MAX = 0.005, 200.0


def logp(p, floor=1e-300):
    return np.log(np.clip(p, floor, 1.0))


def with_temperature(p, temperature):
    return M.softmax(logp(p) / temperature)


def _nll(prob, y_idx):
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y_idx)), y_idx], 1e-300, 1.0))))


# ---------------------------------------------------------------- sesgos por clase


def apply_bias(p, bias):
    """Puntajes log p + b (sirven para argmax). ``bias=None`` deja p igual."""
    return logp(p) if bias is None else logp(p) + np.asarray(bias)


def fit_bias(p, y_idx, grid=np.linspace(-0.6, 0.6, 25), n_rounds=3):
    """Búsqueda de coordenadas de 6 sesgos que maximizan F1 macro."""
    L = logp(p)
    b = np.zeros(L.shape[1])
    best = E.f1_macro(y_idx, L)
    for _ in range(n_rounds):
        improved = False
        for k in range(len(b)):
            for d in grid:
                cand = b.copy()
                cand[k] = d
                f = E.f1_macro(y_idx, L + cand)
                if f > best + 1e-9:
                    best, b, improved = f, cand, True
        if not improved:
            break
    return b - b.mean()


# ---------------------------------------------------------------- mezclas conjuntas


def _gate_matrix(names, gate_spec, masks, n):
    G = np.ones((len(names), n), dtype=np.float64)
    for i, name in enumerate(names):
        key = (gate_spec or {}).get(name)
        if key is not None:
            G[i] = masks[key].astype(np.float64)
    return G


def calibrate_temperature(p, y_idx, mask=None):
    """Temperatura que minimiza el log-loss del canal (solo en las filas con evidencia)."""
    L = logp(p)
    if mask is not None:
        L, y_idx = L[mask], y_idx[mask]
    if len(y_idx) == 0:
        return 1.0
    res = minimize_scalar(
        lambda lt: _nll(M.softmax(L / np.exp(lt)), y_idx), bounds=(np.log(T_MIN), np.log(T_MAX)), method="bounded"
    )
    return float(np.exp(res.x))


class LinearBlend:
    kind = "linear"

    def __init__(self, names, gate_spec=None, joint=False, l2=1e-3):
        self.names, self.gate_spec, self.joint, self.l2 = list(names), dict(gate_spec or {}), joint, l2

    def _Q(self, probs, temps):
        return np.stack([with_temperature(probs[n], t) for n, t in zip(self.names, temps, strict=True)])

    @staticmethod
    def _mix(Q, G, w):
        num = np.einsum("k,kn,knc->nc", w, G, Q)
        den = np.einsum("k,kn->n", w, G)[:, None]
        return num / den

    def fit(self, probs, y_idx, masks):
        n = len(y_idx)
        G = _gate_matrix(self.names, self.gate_spec, masks, n)
        temps = np.array([calibrate_temperature(probs[nm], y_idx, G[i] > 0) for i, nm in enumerate(self.names)])
        Q = self._Q(probs, temps)
        Qy = Q[:, np.arange(n), y_idx]  # (K, n)

        def loss_grad(a):
            w = np.exp(a)
            num = (w[:, None] * G * Qy).sum(0)
            den = (w[:, None] * G).sum(0)
            loss = -np.mean(np.log(np.clip(num / den, 1e-300, None))) + self.l2 * np.sum(a**2)
            g = (-(G * Qy) / num + G / den).mean(axis=1) * w + 2 * self.l2 * a
            return loss, g

        res = minimize(loss_grad, np.zeros(len(self.names)), jac=True, method="L-BFGS-B")
        a = res.x
        if self.joint:
            L = np.stack([logp(probs[nm]) for nm in self.names])

            def loss_joint(theta):
                aa, lt = theta[: len(a)], theta[len(a) :]
                Qj = np.stack([M.softmax(L[i] / np.exp(lt[i])) for i in range(len(a))])
                p = self._mix(Qj, G, np.exp(aa))
                return _nll(p, y_idx) + self.l2 * np.sum(aa**2)

            theta0 = np.concatenate([a, np.log(temps)])
            bounds = [(None, None)] * len(a) + [(np.log(T_MIN), np.log(T_MAX))] * len(a)
            res = minimize(loss_joint, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
            a, temps = res.x[: len(a)], np.exp(res.x[len(a) :])
        w = np.exp(a)
        self.weights_ = dict(zip(self.names, (w / w.sum()).tolist(), strict=True))
        self.temps_ = dict(zip(self.names, temps.tolist(), strict=True))
        return self

    def predict_proba(self, probs, masks):
        n = len(next(iter(probs.values())))
        G = _gate_matrix(self.names, self.gate_spec, masks, n)
        Q = self._Q(probs, [self.temps_[nm] for nm in self.names])
        return self._mix(Q, G, np.array([self.weights_[nm] for nm in self.names]))

    def to_dict(self):
        return {"kind": self.kind, "names": self.names, "gate_spec": self.gate_spec,
                "weights": self.weights_, "temps": self.temps_}


class LogLinearBlend:
    kind = "loglinear"

    def __init__(self, names, gate_spec=None, l2=1e-3, floor=1e-30):
        self.names, self.gate_spec, self.l2, self.floor = list(names), dict(gate_spec or {}), l2, floor

    def _scores(self, probs, G, w):
        L = np.stack([logp(probs[nm], self.floor) for nm in self.names])  # (K, n, C)
        return np.einsum("k,kn,knc->nc", w, G, L), L

    def fit(self, probs, y_idx, masks):
        n = len(y_idx)
        G = _gate_matrix(self.names, self.gate_spec, masks, n)
        L = np.stack([logp(probs[nm], self.floor) for nm in self.names])
        GL = G[:, :, None] * L  # (K, n, C)
        GLy = GL[:, np.arange(n), y_idx]  # (K, n)

        def loss_grad(a):
            w = np.exp(a)
            S = np.einsum("k,knc->nc", w, GL)
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(np.clip(P[np.arange(n), y_idx], 1e-300, None))) + self.l2 * np.sum(a**2)
            expected = np.einsum("nc,knc->kn", P, GL)
            g = (expected - GLy).mean(axis=1) * w + 2 * self.l2 * a
            return loss, g

        res = minimize(loss_grad, np.full(len(self.names), np.log(1.0 / len(self.names))), jac=True, method="L-BFGS-B")
        self.weights_ = dict(zip(self.names, np.exp(res.x).tolist(), strict=True))
        return self

    def predict_proba(self, probs, masks):
        n = len(next(iter(probs.values())))
        G = _gate_matrix(self.names, self.gate_spec, masks, n)
        S, _ = self._scores(probs, G, np.array([self.weights_[nm] for nm in self.names]))
        return M.softmax(S)

    def to_dict(self):
        return {"kind": self.kind, "names": self.names, "gate_spec": self.gate_spec, "weights": self.weights_}


def blend_from_dict(d):
    """Reconstruye una mezcla guardada con ``to_dict`` (para el notebook de submission)."""
    if d["kind"] == "linear":
        b = LinearBlend(d["names"], d["gate_spec"])
        b.weights_, b.temps_ = d["weights"], d["temps"]
    else:
        b = LogLinearBlend(d["names"], d["gate_spec"])
        b.weights_ = d["weights"]
    return b


# ---------------------------------------------------------------- agregar un canal


def add_channel(base, q, temperature, weight):
    return (1 - weight) * base + weight * with_temperature(q, temperature)


def grid_add(base, q, y_idx, temps=(0.05, 0.1, 0.2, 0.5, 1.0), weights=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """Grilla temperatura × peso para agregar el canal ``q`` a ``base``; devuelve la tabla."""
    rows = [{"temperatura": np.nan, "peso": 0.0, "f1_macro": E.f1_macro(y_idx, base)}]
    for t in temps:
        qt = with_temperature(q, t)
        for w in weights:
            rows.append({"temperatura": t, "peso": w, "f1_macro": E.f1_macro(y_idx, (1 - w) * base + w * qt)})
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


PAIR = (CLASES.index("18-24"), CLASES.index("25-34"))


def specialist_apply(base, q_spec, temperature, weight, pair=PAIR):
    """Cuando las dos clases del par son el top-2 de ``base``, reparte su masa conjunta
    mezclando la proporción de ``base`` con la del especialista."""
    i, j = pair
    out = base.copy()
    top2 = np.argsort(-base, axis=1)[:, :2]
    rows = np.flatnonzero(((top2[:, 0] == i) & (top2[:, 1] == j)) | ((top2[:, 0] == j) & (top2[:, 1] == i)))
    if len(rows) == 0:
        return out
    qs = with_temperature(q_spec, temperature)[rows]
    ratio_spec = qs[:, i] / np.clip(qs[:, i] + qs[:, j], 1e-300, None)
    mass = base[rows, i] + base[rows, j]
    ratio = (1 - weight) * base[rows, i] / mass + weight * ratio_spec
    out[rows, i] = mass * ratio
    out[rows, j] = mass * (1 - ratio)
    return out


def grid_specialist(base, q_spec, y_idx, temps=(0.1, 0.2, 0.5, 1.0), weights=(0.1, 0.2, 0.3, 0.5)):
    rows = [{"temperatura": np.nan, "peso": 0.0, "f1_macro": E.f1_macro(y_idx, base)}]
    for t in temps:
        for w in weights:
            rows.append({"temperatura": t, "peso": w, "f1_macro": E.f1_macro(y_idx, specialist_apply(base, q_spec, t, w))})
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
