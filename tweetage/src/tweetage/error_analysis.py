"""Análisis de errores (notebook 03): mezcla vigente frente a la mejor spec (greedy_ent).

No entrena nada nuevo: todas las probabilidades salen de la caché del 02.

* ``cargar``: probabilidades de validación de ambos modelos para una semilla.
* Métricas por clase, matriz de confusión y error ordinal.
* ``tipos_de_tweet``: indicadores por tweet (RT, URL, plantilla, longitud...).
* Confianza (margen top-1 − top-2) y calibración.
* ``cambios``: predicciones corregidas / estropeadas entre dos modelos.
* ``votos_canales``: cuánto empuja cada canal hacia la clase correcta en cada tweet.
* ``ablacion``: aporte de cada canal y de cada decisión del camino greedy.

Quitar un canal = reemplazar sus probabilidades por la distribución uniforme (1/6). Como
todas las mezclas son combinaciones lineales, eso suma la misma constante a las 6 clases:
equivale a sacar el canal y renormalizar los pesos de los demás (el argmax no cambia).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tweetage import channels_extra as CX
from tweetage import evaluation as E
from tweetage import experiments_v2 as XV
from tweetage import pipeline as P
from tweetage import preprocess as PP
from tweetage import validation
from tweetage.pipeline import CLASES

SEED_AJUSTE, SEED_CONFIRMACION = 20261108, 20261109
K = len(CLASES)
WORD_TT = ("word1", "position", "mention_pos", "word_seq3")
# F1 de la mezcla vigente reportado en el 02 (sección 3)
F1_VIGENTE = {SEED_AJUSTE: 0.5849, SEED_CONFIRMACION: 0.5869}

# ---------------------------------------------------------------- carga


@dataclass
class Modelos:
    seed: int
    spec: dict
    X_tr: list
    y_tr: np.ndarray
    X_va: list
    y_idx: np.ndarray
    masks: dict
    probs_base: dict  # canales con el tokenizador regex (variante base)
    probs_tt: dict  # word1/position/mention_pos/word_seq3 con TweetTokenizer
    p_vig: np.ndarray | None = None
    p_greedy: np.ndarray | None = None

    def probs(self, tokenizer: bool) -> dict:
        p = dict(self.probs_base)
        if tokenizer:
            p.update(self.probs_tt)
        return p

    def predict(self, spec) -> np.ndarray:
        """Probabilidades de una spec (sin sesgos) con las probabilidades en memoria."""
        return XV.predict_spec(spec, self.probs(spec["flags"]["tweet_tok"]), self.masks, proba=True)


def cargar(X, y, seed, spec_name="greedy_ent", n_jobs=2) -> Modelos:
    """Carga de la caché los canales de la mezcla vigente y de la spec.

    Igual que la sección 14 del 02: los canales de caracteres, forma y entidades se
    leen de la variante base (etiqueta ``seed{s}``) y los 4 canales de palabras de la
    variante A2 (``seed{s}_A2_tweettok``). Usar ``run_seed`` con los flags de A2 para todo
    cambiaría la etiqueta y obligaría a reentrenar.
    """
    spec = XV.load_spec(spec_name)
    extra = [c for c in XV.spec_channels(spec) if c not in CX.BASE_NAMES]
    extra += ["word_seq3_ent5"]  # para la ablación "sin TweetTokenizer"
    base = XV.run_seed(X, y, seed, CX.VARIANTS["base"], CX.BASE_NAMES + list(dict.fromkeys(extra)), n_jobs)
    tt = XV.run_seed(X, y, seed, CX.VARIANTS["A2_tweettok"], list(WORD_TT), n_jobs)
    Xtr, ytr, Xva, _ = validation.split(X, y, seed)
    m = Modelos(seed, spec, Xtr, ytr, Xva, base.y_idx, base.masks, base.probs, tt.probs)
    m.p_vig = P.blend(base.probs, base.masks["mention"])
    m.p_greedy = m.predict(spec)
    return m


def verificar(modelos: dict, tol=2e-4) -> pd.DataFrame:
    """Compara el F1 recalculado con el reportado en el 02 (para detectar caché equivocada)."""
    filas = []
    for seed, m in modelos.items():
        esperado_g = m.spec.get("f1_confirmacion") if seed == SEED_CONFIRMACION else None
        for nombre, p, esperado in (("vigente", m.p_vig, F1_VIGENTE.get(seed)), ("greedy_ent", m.p_greedy, esperado_g)):
            f1 = E.f1_macro(m.y_idx, p)
            filas.append({"semilla": seed, "modelo": nombre, "f1_macro": f1, "f1_reportado_02": esperado,
                          "coincide": None if esperado is None else abs(f1 - esperado) < tol})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- por clase


def confusion(y_idx, p, normalize=True) -> pd.DataFrame:
    """Filas = clase real, columnas = clase predicha. Normalizada por fila (= recall)."""
    pred = E.as_pred(p)
    cm = np.bincount(y_idx * K + pred, minlength=K * K).reshape(K, K)
    df = pd.DataFrame(cm, index=pd.Index(CLASES, name="real"), columns=pd.Index(CLASES, name="predicha"))
    return df.div(df.sum(axis=1), axis=0) if normalize else df


def _prf(y_idx, p) -> pd.DataFrame:
    pred = E.as_pred(p)
    filas = {}
    for k, c in enumerate(CLASES):
        tp = int(((pred == k) & (y_idx == k)).sum())
        n_pred, n_real = int((pred == k).sum()), int((y_idx == k).sum())
        filas[c] = {"precision": tp / max(n_pred, 1), "recall": tp / max(n_real, 1),
                    "f1": 2 * tp / max(1, n_pred + n_real), "real": n_real, "predichas": n_pred}
    df = pd.DataFrame(filas).T
    df.loc["macro"] = [df["precision"].mean(), df["recall"].mean(), df["f1"].mean(), df["real"].sum(), df["predichas"].sum()]
    return df


def reporte_clases(y_idx, preds: dict) -> pd.DataFrame:
    """Precision, recall, F1, tweets reales y predichos por clase, para cada modelo."""
    return pd.concat({nombre: _prf(y_idx, p) for nombre, p in preds.items()}, axis=1)


def errores_ordinales(y_idx, preds: dict) -> pd.DataFrame:
    """Qué tan lejos cae cada error: distancia = cuántos rangos de edad hay entre real y predicha."""
    filas = {}
    for nombre, p in preds.items():
        d = np.abs(E.as_pred(p) - y_idx)
        err = d > 0
        fila = {"% de tweets mal clasificados": 100 * err.mean()}
        for k in range(1, K):
            fila[f"% de errores a distancia {k}"] = 100 * (d[err] == k).mean()
        fila["distancia media de los errores"] = d[err].mean()
        filas[nombre] = fila
    return pd.DataFrame(filas).T


# ---------------------------------------------------------------- tipos de tweet

_RX = {
    "RT": re.compile(r"^RT @\w+"),
    "mención al inicio": re.compile(r"^@\w+"),
    "con URL": re.compile(r"https?://|www\."),
    "con hashtag": re.compile(r"#\w+"),
    "emoticono": re.compile(r"(?<!\w)[:;=][-']?[)(DPpOo3](?!\w)|<3|&lt;3"),
    "risa": re.compile(r"\b(?:j[aeiou]){2,}j?\b|\bjs(?:js)+\b|\bx+d+\b|\b(?:ha){2,}\b|\blol\b", re.I),
    "alargamiento (siii)": re.compile(r"([a-z])\1{2,}", re.I),
}
LONGITUDES = ["0-30", "31-60", "61-90", "91-120", "121-139", "140 (truncado)"]


def _sin_texto(t):
    resto = re.sub(r"https?://\S*|www\.\S+|@\w+|#\w+|\brt\b", " ", t.casefold())
    return re.search(r"[a-z]{2,}", resto) is None


def _gritado(t):
    letras = sum(ch.isalpha() for ch in t)
    return letras >= 10 and sum(ch.isupper() for ch in t) / letras >= 0.7


def tipos_de_tweet(X_va, X_tr, min_plantilla=5) -> pd.DataFrame:
    """Un indicador (True/False) por tipo de tweet, con las mismas reglas del 01.

    "plantilla frecuente en train" = su plantilla N2 (URL→URL, @x→@U, números→0...)
    aparece al menos ``min_plantilla`` veces en el train de esta semilla.
    """
    frecuentes = {p for p, n in Counter(PP.n2(t) for t in X_tr).items() if n >= min_plantilla}
    df = pd.DataFrame({
        "sin texto propio": [_sin_texto(t) for t in X_va],
        **{nombre: [rx.search(t) is not None for t in X_va] for nombre, rx in _RX.items()},
        "plantilla frecuente en train": [PP.n2(t) in frecuentes for t in X_va],
        "mayúsculas sostenidas (gritado)": [_gritado(t) for t in X_va],
        "llega a 140 caracteres": [len(t) >= 140 for t in X_va],
    })
    # orden de columnas: primero los más "pobres" en texto
    orden = ["sin texto propio", "RT", "mención al inicio", "con URL", "con hashtag", "plantilla frecuente en train",
             "llega a 140 caracteres", "mayúsculas sostenidas (gritado)", "emoticono", "risa", "alargamiento (siii)"]
    return df[orden]


def longitud(X_va) -> pd.Series:
    n = np.array([len(t) for t in X_va])
    return pd.Series(pd.cut(n, [-1, 30, 60, 90, 120, 139, 10**6], labels=LONGITUDES), name="longitud")


def grupos(tipos: pd.DataFrame, long: pd.Series | None = None) -> dict:
    out = {"todos": np.ones(len(tipos), dtype=bool)}
    out.update({c: tipos[c].to_numpy() for c in tipos.columns})
    if long is not None:
        out.update({f"longitud {lab}": (long == lab).to_numpy() for lab in long.cat.categories})
    return out


def grupos_clase(y_idx) -> dict:
    return {f"real {c}": y_idx == k for k, c in enumerate(CLASES)}


def metricas_por_grupo(y_idx, preds: dict, grupos_: dict, principal=None) -> pd.DataFrame:
    """Para cada grupo: tamaño, peso en los errores del modelo principal, accuracy y F1 de cada modelo."""
    principal = principal or list(preds)[-1]
    err_principal = E.as_pred(preds[principal]) != y_idx
    filas = {}
    for g, mask in grupos_.items():
        if mask.sum() == 0:
            continue
        fila = {"tweets": int(mask.sum()), "% de los tweets": 100 * mask.mean(),
                f"% de los errores ({principal})": 100 * err_principal[mask].sum() / err_principal.sum()}
        for nombre, p in preds.items():
            fila[f"accuracy {nombre}"] = E.accuracy(y_idx[mask], E.as_pred(p)[mask])
        for nombre, p in preds.items():
            fila[f"F1 {nombre}"] = E.f1_macro(y_idx[mask], E.as_pred(p)[mask])
        nombres = list(preds)
        if len(nombres) == 2:
            fila["Δ accuracy"] = fila[f"accuracy {nombres[1]}"] - fila[f"accuracy {nombres[0]}"]
        filas[g] = fila
    return pd.DataFrame(filas).T


# ---------------------------------------------------------------- confianza


def margen(p) -> np.ndarray:
    """Diferencia entre la probabilidad de la 1.ª y la 2.ª clase: qué tan "seguro" está el modelo."""
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def tabla_confianza(y_idx, p, n_bins=5) -> pd.DataFrame:
    """Tweets ordenados por margen y partidos en ``n_bins`` grupos del mismo tamaño."""
    m = margen(p)
    etiquetas = [f"Q{i + 1}" for i in range(n_bins)]
    etiquetas[0] += " (más dudoso)"
    etiquetas[-1] += " (más seguro)"
    q = pd.qcut(m, n_bins, labels=etiquetas)
    err = E.as_pred(p) != y_idx
    filas = {}
    for lab in etiquetas:
        mask = (q == lab)
        filas[lab] = {"tweets": int(mask.sum()), "margen mínimo": m[mask].min(), "margen máximo": m[mask].max(),
                      "accuracy": 1 - err[mask].mean(), "% de los errores": 100 * err[mask].sum() / err.sum()}
    return pd.DataFrame(filas).T


def calibracion(y_idx, p, n_bins=10):
    """Confianza (prob. máxima) agrupada en intervalos frente a la accuracy real.

    Devuelve (tabla, ECE). ECE = promedio ponderado de |accuracy − confianza|: 0 es un
    modelo perfectamente calibrado.
    """
    conf = p.max(axis=1)
    ok = E.as_pred(p) == y_idx
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bordes) - 1, 0, n_bins - 1)
    filas = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        filas.append({"intervalo": f"{bordes[b]:.1f}-{bordes[b + 1]:.1f}", "tweets": int(mask.sum()),
                      "confianza media": conf[mask].mean(), "accuracy": ok[mask].mean()})
    tabla = pd.DataFrame(filas).set_index("intervalo")
    ece = float((tabla["tweets"] / len(conf) * (tabla["accuracy"] - tabla["confianza media"]).abs()).sum())
    return tabla, ece


# ---------------------------------------------------------------- cambios entre modelos

CATEGORIAS_CAMBIO = ["corregida", "estropeada", "ambos fallan (cambia la clase)", "ambos fallan (igual)", "ambos aciertan"]


def cambios(y_idx, p_a, p_b) -> pd.Series:
    """Qué le pasó a cada tweet al pasar del modelo A al B."""
    a, b = E.as_pred(p_a), E.as_pred(p_b)
    ok_a, ok_b = a == y_idx, b == y_idx
    cat = np.select(
        [~ok_a & ok_b, ok_a & ~ok_b, ~ok_a & ~ok_b & (a != b), ~ok_a & ~ok_b],
        CATEGORIAS_CAMBIO[:4],
        default=CATEGORIAS_CAMBIO[4],
    )
    return pd.Series(pd.Categorical(cat, categories=CATEGORIAS_CAMBIO), name="cambio")


def tabla_cambios(cat: pd.Series, grupos_: dict) -> pd.DataFrame:
    filas = {}
    for g, mask in grupos_.items():
        c = cat[mask].value_counts()
        filas[g] = {"tweets": int(mask.sum()), "corregidas": int(c["corregida"]), "estropeadas": int(c["estropeada"]),
                    "neto": int(c["corregida"] - c["estropeada"]),
                    "neto por cada 1000 tweets": 1000 * (c["corregida"] - c["estropeada"]) / max(mask.sum(), 1)}
    return pd.DataFrame(filas).T


# ---------------------------------------------------------------- aporte de cada canal


def canales(spec) -> dict:
    """Etiqueta legible -> clave en el dict de probabilidades que hay que neutralizar."""
    swaps = spec.get("swaps") or {}
    tok = spec["flags"]["tweet_tok"]
    out = {}
    for c in spec["channels"]:
        real = swaps.get(c, c)
        etiqueta = real if c in swaps else (CX.TT_REPLACE.get(c, c) if tok and c in WORD_TT else c)
        out[etiqueta] = real
    for a in spec.get("adds") or []:
        out[a["name"]] = a["name"]
    return out


def sin_canal(m: Modelos, spec, clave) -> np.ndarray:
    p = m.probs(spec["flags"]["tweet_tok"])
    p[clave] = np.full_like(p[clave], 1.0 / K)
    return XV.predict_spec(spec, p, m.masks, proba=True)


def votos_canales(m: Modelos, spec=None):
    """Aporte de cada canal al margen (prob. clase real − prob. clase rival), tweet por tweet.

    Rival = la clase predicha si el modelo se equivoca, o la 2.ª más probable si acierta.
    Voto > 0: el canal empujó hacia la clase correcta; voto < 0: empujó hacia el error.
    Devuelve (DataFrame n × canales, rival).
    """
    spec = spec or m.spec
    p = m.predict(spec)
    y = m.y_idx
    orden = np.argsort(-p, axis=1)
    rival = np.where(orden[:, 0] == y, orden[:, 1], orden[:, 0])
    filas = np.arange(len(y))
    margen_total = p[filas, y] - p[filas, rival]
    votos = {}
    for etiqueta, clave in canales(spec).items():
        q = sin_canal(m, spec, clave)
        votos[etiqueta] = margen_total - (q[filas, y] - q[filas, rival])
    return pd.DataFrame(votos), rival


def resumen_votos(votos: pd.DataFrame, err: np.ndarray) -> pd.DataFrame:
    """Por canal: voto medio en aciertos y errores, y en qué % de los errores empujó hacia el error."""
    culpable = votos[err].idxmin(axis=1).value_counts()
    return pd.DataFrame({
        "voto medio en aciertos": votos[~err].mean(),
        "voto medio en errores": votos[err].mean(),
        "% de errores donde empujó al error": 100 * (votos[err] < 0).mean(),
        "% de errores donde fue el que más empujó": 100 * culpable.reindex(votos.columns, fill_value=0) / err.sum(),
    }).sort_values("% de errores donde fue el que más empujó", ascending=False)


# ---------------------------------------------------------------- ablación


def variantes_reversion(spec) -> dict:
    """Specs que deshacen, una a la vez, cada decisión del camino greedy."""
    swaps, adds, flags = dict(spec.get("swaps") or {}), list(spec.get("adds") or []), spec["flags"]
    out = {}
    cs = {k for k, v in swaps.items() if v.endswith("_cs")}
    if cs:
        out["sin mayúsculas (char*_cs → char*)"] = {**spec, "swaps": {k: v for k, v in swaps.items() if k not in cs}}
    if "word_seq3" in swaps:
        out["sin entidades (word_seq3_ent5 → word_seq3)"] = {**spec, "swaps": {k: v for k, v in swaps.items() if k != "word_seq3"}}
    if flags["tweet_tok"]:
        sin_tt = {k: (v[:-3] if v.endswith("_tt") else v) for k, v in swaps.items()}
        out["sin TweetTokenizer"] = {**spec, "flags": {**flags, "tweet_tok": False}, "swaps": sin_tt}
    for a in adds:
        out[f"sin {a['name']} (agregado)"] = {**spec, "adds": [b for b in adds if b is not a]}
    return out


def ablacion(modelos: dict, n_boot=1000) -> pd.DataFrame:
    """Aporte = F1(greedy_ent) − F1(sin el elemento), en ambas semillas.

    El intervalo bootstrap al 95 % es del aporte en la semilla de confirmación.
    """
    m0 = next(iter(modelos.values()))
    spec = m0.spec
    candidatos = {f"quitar canal: {e}": (lambda m, c=c: sin_canal(m, spec, c)) for e, c in canales(spec).items()}
    candidatos.update({f"revertir: {n}": (lambda m, s=s: m.predict(s)) for n, s in variantes_reversion(spec).items()})
    candidatos["revertir todo (= mezcla vigente)"] = lambda m: m.p_vig
    filas = {}
    for nombre, fn in candidatos.items():
        fila = {}
        for seed, m in modelos.items():
            q = fn(m)
            col = "ajuste" if seed == SEED_AJUSTE else "confirmacion"
            fila[f"F1 sin ({col})"] = E.f1_macro(m.y_idx, q)
            fila[f"aporte {col}"] = E.f1_macro(m.y_idx, m.p_greedy) - fila[f"F1 sin ({col})"]
            if seed == SEED_CONFIRMACION:
                lo, hi = E.paired_bootstrap(m.y_idx, q, m.p_greedy, n_boot=n_boot)
                fila["ic95 bajo"], fila["ic95 alto"] = lo, hi
                fila["% predicciones que cambian"] = 100 * np.mean(E.as_pred(q) != E.as_pred(m.p_greedy))
        filas[nombre] = fila
    df = pd.DataFrame(filas).T
    df["ayuda en ambas"] = (df["aporte ajuste"] > 0) & (df["aporte confirmacion"] > 0) & (df["ic95 bajo"] > 0)
    return df


# ---------------------------------------------------------------- ejemplos


def muestrear(mask, n, seed) -> np.ndarray:
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else idx


def tabla_ejemplos(m: Modelos, idx, votos: pd.DataFrame | None = None, rival=None, n_canales=2) -> pd.DataFrame:
    """Texto, clase real, predicciones de ambos modelos, probabilidades y canales que más empujaron."""
    cl = np.asarray(CLASES)
    pv, pg = E.as_pred(m.p_vig), E.as_pred(m.p_greedy)
    filas = []
    for i in idx:
        top = np.argsort(-m.p_greedy[i])[:3]
        fila = {"fila": int(i), "texto": m.X_va[i], "real": cl[m.y_idx[i]], "vigente": cl[pv[i]], "greedy_ent": cl[pg[i]],
                "top-3 greedy_ent": ", ".join(f"{cl[k]} {m.p_greedy[i, k]:.2f}" for k in top)}
        if votos is not None:
            v = votos.iloc[i].sort_values()
            fila["empujaron hacia la rival"] = ", ".join(
                f"{c} ({x:+.3f})" for c, x in v.head(n_canales).items() if x < 0) or "—"
            fila["empujaron hacia la real"] = ", ".join(f"{c} ({x:+.3f})" for c, x in v.tail(n_canales)[::-1].items() if x > 0) or "—"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("fila")


# ---------------------------------------------------------------- gráficos


def plot_confusiones(cms: dict, titulo_extra=""):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.4), squeeze=False)
    for ax, (nombre, cm) in zip(axes[0], cms.items()):
        ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=0.8)
        ax.set_xticks(range(K), CLASES, rotation=45)
        ax.set_yticks(range(K), CLASES)
        ax.set_xlabel("predicha")
        ax.set_ylabel("real")
        ax.set_title(f"{nombre}{titulo_extra}")
        for i in range(K):
            for j in range(K):
                v = cm.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.45 else "black")
    fig.tight_layout()
    return fig


def plot_diferencia_confusion(cm_a, cm_b, nombre_a, nombre_b):
    import matplotlib.pyplot as plt

    d = (cm_b - cm_a).to_numpy()
    lim = max(0.01, np.abs(d).max())
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(d, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(K), CLASES, rotation=45)
    ax.set_yticks(range(K), CLASES)
    ax.set_xlabel("predicha")
    ax.set_ylabel("real")
    ax.set_title(f"{nombre_b} − {nombre_a}\n(azul en la diagonal = acierta más)")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{100 * d[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="diferencia (fracción de la fila)")
    fig.tight_layout()
    return fig


def plot_confianza(tablas: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ancho = 0.8 / len(tablas)
    for i, (nombre, t) in enumerate(tablas.items()):
        x = np.arange(len(t)) + i * ancho
        ax.bar(x, t["accuracy"], width=ancho, label=nombre)
        for xi, v in zip(x, t["accuracy"]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(t)) + ancho * (len(tablas) - 1) / 2, t.index, fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy según qué tan seguro está el modelo (quintiles de margen)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_calibracion(tablas: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="calibración perfecta")
    for nombre, (t, ece) in tablas.items():
        ax.plot(t["confianza media"], t["accuracy"], "o-", label=f"{nombre} (ECE {ece:.3f})")
    ax.set_xlabel("confianza del modelo (prob. máxima)")
    ax.set_ylabel("accuracy real")
    ax.set_title("Diagrama de calibración")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_ablacion(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = df.sort_values("aporte confirmacion")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["aporte ajuste"], height=0.4, label="semilla de ajuste", color="#9DB4D3")
    ax.barh(y + 0.2, d["aporte confirmacion"], height=0.4, label="semilla de confirmación", color="#4C72B0")
    err = np.vstack([d["aporte confirmacion"] - d["ic95 bajo"], d["ic95 alto"] - d["aporte confirmacion"]])
    ax.errorbar(d["aporte confirmacion"], y + 0.2, xerr=np.abs(err), fmt="none", ecolor="black", lw=0.8)
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(y, d.index, fontsize=8)
    ax.set_xlabel("aporte al F1 macro (F1 con − F1 sin)")
    ax.set_title("¿Cuánto pierde greedy_ent si se quita cada pieza?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- complementos


PARES = [("18-24", "25-34"), ("25-34", "18-24"), ("13-17", "18-24"), ("18-24", "13-17"),
         ("50-64", "65-xx"), ("65-xx", "50-64"), ("35-49", "50-64"), ("50-64", "35-49")]


def mascara_par(y_idx, p, real, predicha) -> np.ndarray:
    return (y_idx == CLASES.index(real)) & (E.as_pred(p) == CLASES.index(predicha))


def votos_por_par(votos: pd.DataFrame, y_idx, p, pares=PARES) -> pd.DataFrame:
    """Voto medio de cada canal en los errores real → predicha (negativo = empujó al error)."""
    cols = {}
    for real, pred in pares:
        mask = mascara_par(y_idx, p, real, pred)
        cols[f"{real} → {pred} (n={mask.sum()})"] = votos[mask].mean()
    return pd.DataFrame(cols)


def plot_grupos(df: pd.DataFrame, modelos=("vigente", "greedy_ent")):
    """Accuracy de cada modelo por grupo, con la accuracy global como referencia."""
    import matplotlib.pyplot as plt

    d = df.drop(index="todos")
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    y = np.arange(len(d))
    for i, nombre in enumerate(modelos):
        ax.barh(y + (i - 0.5) * 0.4, d[f"accuracy {nombre}"], height=0.4, label=nombre,
                color=["#9DB4D3", "#4C72B0"][i % 2])
    ax.axvline(df.loc["todos", f"accuracy {modelos[-1]}"], color="#C44E52", ls="--", lw=1, label="accuracy global")
    ax.set_yticks(y, [f"{g} ({p:.1f} %)" for g, p in zip(d.index, d["% de los tweets"])], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy (entre paréntesis: % de los tweets de validación)")
    ax.set_title("Accuracy por tipo de tweet")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    return fig


def culpables(votos: pd.DataFrame, err: np.ndarray, mask: np.ndarray, nombres=("no", "sí")) -> pd.DataFrame:
    """% de los errores en que cada canal fue el que más empujó al error, separando por ``mask``."""
    cul = votos[err].idxmin(axis=1)
    m = np.asarray(mask)[err]
    return pd.DataFrame({nombres[0]: 100 * cul[~m].value_counts(normalize=True),
                         nombres[1]: 100 * cul[m].value_counts(normalize=True)}).fillna(0).sort_values(nombres[1], ascending=False)
