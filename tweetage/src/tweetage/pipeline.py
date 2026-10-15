"""Mezcla vigente de canales (mejor submission: 0.60021 en Kaggle).

Cada canal produce probabilidades (n_textos, 6) en el orden de ``CLASES``. La función
``channel_probs`` entrena y predice un canal; ``blend`` combina los canales con los
pesos validados. Separar ambos pasos permite cachear cada canal y re-mezclar sin
reentrenar.
"""

from __future__ import annotations

import gc

import numpy as np

from tweetage import models as M

CLASES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]

# Configuración de cada canal: (constructor, kwargs, temperatura de la softmax)
CHANNELS = {
    "word1": (M.ClassConditionalNgramLM, {"unit": "word", "orders": (1,), "alpha": 0.01}, 1.0),
    "char4": (M.ClassConditionalNgramLM, {"unit": "char", "orders": (4,), "alpha": 0.1}, 1.0),
    "char5": (M.ClassConditionalNgramLM, {"unit": "char", "orders": (5,), "alpha": 0.1}, 1.0),
    "char6": (M.ClassConditionalNgramLM, {"unit": "char", "orders": (6,), "alpha": 0.03}, 1.0),
    "mentions": (M.IdentityClassLM, {"pattern": r"@\w+", "alpha": 0.1}, 1.0),
    "hashtags": (M.IdentityClassLM, {"pattern": r"#\w+", "alpha": 0.1}, 1.0),
    "position": (M.PositionalWordLM, {"mode": "start_end", "alpha": 0.03}, 1.0),
    "mention_pos": (M.PositionAwareMentionLM, {"alpha": 0.01, "mode": "all"}, 1.0),
    "char_seq2": (M.CharacterSequenceLM, {"alpha": 0.1}, 0.05),
    "word_seq3": (M.WordSequenceLM, {"alpha": 0.1, "min_count": 3, "order": 3}, 0.2),
}


def channel_probs(name, X_train, y_train, X_eval, channels=CHANNELS):
    cls, kwargs, temperature = channels[name]
    model = cls(**kwargs).fit(X_train, y_train)
    prob = M.probabilities(model, X_eval, CLASES, temperature)
    del model
    gc.collect()
    return prob


def _channel_task(args):
    return args[0], channel_probs(*args)


def all_channel_probs(X_train, y_train, X_eval, names=None, n_jobs=1, channels=CHANNELS):
    """Calcula varios canales; con ``n_jobs > 1`` los reparte en procesos paralelos.

    Cada proceso entrena un modelo completo, así que la memoria crece con ``n_jobs``.
    Los resultados son idénticos con 1 o con varios procesos.
    """
    names = list(names or channels)
    if n_jobs == 1:
        return {n: channel_probs(n, X_train, y_train, X_eval, channels) for n in names}
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # "fork" permite usar funciones definidas en el propio notebook (Linux: Kaggle/Colab).
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    tasks = [(n, X_train, y_train, X_eval, channels) for n in names]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        return dict(ex.map(_channel_task, tasks))


def blend(p: dict, mention_mask: np.ndarray) -> np.ndarray:
    """Mezcla vigente. ``p`` mapea nombre de canal -> probabilidades."""
    char_mix = 0.216 * p["word1"] + 0.144 * p["char4"] + 0.240 * p["char5"] + 0.400 * p["char6"]
    mentions_t = M.softmax(np.log(np.clip(p["mentions"], 1e-12, 1.0)) / 0.35)
    no_mentions = (0.36 * char_mix + 0.40 * p["hashtags"]) / 0.76
    previous = 0.70 * no_mentions + 0.30 * mentions_t
    base = 0.90 * previous + 0.10 * p["position"]
    out = base.copy()
    out[mention_mask] = 0.70 * base[mention_mask] + 0.30 * p["mention_pos"][mention_mask]
    out = 0.85 * out + 0.15 * p["char_seq2"]
    return 0.80 * out + 0.20 * p["word_seq3"]


def predict(p: dict, mention_mask: np.ndarray) -> np.ndarray:
    return np.asarray(CLASES)[blend(p, mention_mask).argmax(axis=1)]
