"""Tanda v3: búsqueda greedy de movimientos sobre una spec (agregar, reemplazar o cambiar canales).

Protocolo (el mismo del 02):
* cada movimiento elige sus parámetros (temperatura, peso) solo en la semilla de ajuste;
* en cada ronda se toma el movimiento con mayor F1 de ajuste y se acepta solo si mejora
  frente a la spec actual en **ambas** semillas;
* se repite hasta que ningún movimiento mejore.

Movimientos:
* ``add``: agrega un canal encima, (1 − w)·mezcla + w·canal^(1/T).
* ``replace_add``: cambia un canal agregado por otro (con nueva T y w).
* ``swap``: reemplaza un canal de la mezcla vigente por otro con su propia temperatura,
  con el mismo peso y en la misma posición (``swap_temps`` en la spec).
"""

from __future__ import annotations

import copy
import itertools

import pandas as pd

from tweetage import evaluation as E

TEMPS_ADD = (0.05, 0.1, 0.2, 0.5, 1.0)
PESOS_ADD = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
TEMPS_SWAP = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)


def f1(spec, m) -> float:
    return E.f1_macro(m.y_idx, m.predict(spec))


def _add(spec, name, t, w):
    s = copy.deepcopy(spec)
    s["adds"] = list(s.get("adds") or []) + [{"name": name, "temperature": t, "weight": w}]
    return s


def _replace_add(spec, i, name, t, w):
    s = copy.deepcopy(spec)
    s["adds"][i] = {"name": name, "temperature": t, "weight": w}
    return s


def _swap(spec, old, new, t):
    s = copy.deepcopy(spec)
    s["swaps"] = {**(s.get("swaps") or {}), old: new}
    s["swap_temps"] = {**(s.get("swap_temps") or {}), old: t}
    return s


def candidates(spec, add_names, swap_options: dict):
    """Genera (descripción, spec) de todos los movimientos con su grilla de parámetros."""
    used = {a["name"] for a in spec.get("adds") or []}
    for name in add_names:
        if name in used:
            continue
        for t, w in itertools.product(TEMPS_ADD, PESOS_ADD):
            yield ("add", name, None), f"+ {name} (T={t}, w={w})", _add(spec, name, t, w)
        for i, a in enumerate(spec.get("adds") or []):
            for t, w in itertools.product(TEMPS_ADD, PESOS_ADD):
                yield ("replace_add", name, a["name"]), f"{a['name']} → {name} (T={t}, w={w})", _replace_add(spec, i, name, t, w)
    for old, news in swap_options.items():
        for new in news:
            if (spec.get("swaps") or {}).get(old) == new:
                continue
            for t in TEMPS_SWAP:
                yield ("swap", new, old), f"{old} ⇒ {new} (T={t})", _swap(spec, old, new, t)


def greedy(spec, m_aj, m_cf, add_names, swap_options, max_rounds=8, log=print):
    """Devuelve (spec final, tabla de rondas)."""
    actual = copy.deepcopy(spec)
    f_aj, f_cf = f1(actual, m_aj), f1(actual, m_cf)
    filas = [{"ronda": 0, "movimiento": "inicio", "f1_ajuste": f_aj, "f1_confirmacion": f_cf, "aceptado": True}]
    for r in range(1, max_rounds + 1):
        # mejor configuración de cada movimiento (clave) según la semilla de ajuste
        mejores = {}
        for clave, desc, s in candidates(actual, add_names, swap_options):
            v = f1(s, m_aj)
            if clave not in mejores or v > mejores[clave][0]:
                mejores[clave] = (v, desc, s)
        orden = sorted(mejores.values(), key=lambda x: -x[0])
        aceptado = None
        for v_aj, desc, s in orden[:6]:  # se confirman los 6 mejores de ajuste, en orden
            if v_aj <= f_aj:
                break
            v_cf = f1(s, m_cf)
            filas.append({"ronda": r, "movimiento": desc, "f1_ajuste": v_aj, "f1_confirmacion": v_cf, "aceptado": v_cf > f_cf})
            if v_cf > f_cf:
                aceptado = (v_aj, v_cf, s)
                break
        if aceptado is None:
            log(f"ronda {r}: ningún movimiento mejora en ambas semillas")
            break
        f_aj, f_cf, actual = aceptado
        log(f"ronda {r}: {filas[-1]['movimiento']}  ajuste {f_aj:.4f}  confirmación {f_cf:.4f}")
    return actual, pd.DataFrame(filas)


# ---------------------------------------------------------------- alternativas probadas (no adoptadas)

import numpy as np  # noqa: E402

from tweetage import blend_opt as B  # noqa: E402


def componentes_log(spec, m) -> dict:
    """log p de cada canal de la spec, con sus temperaturas y compuertas (para la mezcla log-lineal)."""
    p = m.probs(spec["flags"]["tweet_tok"])
    sw, st = spec.get("swaps") or {}, spec.get("swap_temps") or {}
    comp = {}
    for c in spec["channels"]:
        q = p[sw.get(c, c)]
        if c in st:
            q = B.with_temperature(q, st[c])
        if c == "mentions":
            q = B.with_temperature(q, 0.35)
        g = m.masks["mention"].astype(float) if c == "mention_pos" else np.ones(len(q))
        comp[c] = g[:, None] * np.log(np.clip(q, 1e-12, 1))
    for a in spec.get("adds") or []:
        comp[a["name"]] = np.log(np.clip(B.with_temperature(p[a["name"]], a["temperature"]), 1e-12, 1))
    return comp


def loglineal_f1(spec, m_aj, m_cf, rondas=4, factores=(0, 0.25, 0.5, 0.75, 1.33, 2, 4)):
    """Mezcla log-lineal Σ w_i·log p_i con pesos buscados por coordenadas sobre el F1 de ajuste."""
    L = {k: np.stack(list(componentes_log(spec, m).values())) for k, m in (("aj", m_aj), ("cf", m_cf))}
    nombres = list(componentes_log(spec, m_aj))
    f = lambda k, w, m: E.f1_macro(m.y_idx, np.einsum("k,knc->nc", w, L[k]))
    w = np.full(len(nombres), 0.1)
    mejor = f("aj", w, m_aj)
    filas = []
    for r in range(rondas):
        for i in range(len(w)):
            for fac in factores:
                c = w.copy()
                c[i] = w[i] * fac if w[i] > 0 else 0.05 * fac
                v = f("aj", c, m_aj)
                if v > mejor + 1e-6:
                    mejor, w = v, c
        filas.append({"ronda": r + 1, "f1_ajuste": mejor, "f1_confirmacion": f("cf", w, m_cf)})
    return pd.DataFrame(filas), pd.Series(w, index=nombres)


def sesgos_cruzados(p_aj, y_aj, p_cf, y_cf) -> pd.DataFrame:
    """Sesgos por clase ajustados en una semilla y evaluados en la otra (en ambos sentidos)."""
    filas = []
    for nombre, (pa, ya, pb, yb) in {"ajuste → confirmación": (p_aj, y_aj, p_cf, y_cf),
                                    "confirmación → ajuste": (p_cf, y_cf, p_aj, y_aj)}.items():
        b = B.fit_bias(pa, ya)
        lo, hi = E.paired_bootstrap(yb, pb, B.apply_bias(pb, b))
        filas.append({"sentido": nombre, "F1 sin sesgos": E.f1_macro(yb, pb), "F1 con sesgos": E.f1_macro(yb, B.apply_bias(pb, b)),
                      "ic95 bajo": lo, "ic95 alto": hi, "sesgos": np.round(b, 3).tolist()})
    return pd.DataFrame(filas).set_index("sentido")
