"""Canales v2, variantes de preprocesamiento y caché paralela.

* ``CHANNELS_V2``: los 10 canales vigentes (``pipeline.CHANNELS``, sin cambios) más los
  nuevos. Los nuevos se guardan a temperatura 1; la temperatura se ajusta en la mezcla.
* ``VARIANTS``: preprocesamientos que obligan a reentrenar (texto o filas de train).
* ``cached_probs``: igual que ``cache.channel`` (mismo nombre de archivo y misma huella,
  así que reutiliza la caché del notebook 02), pero calcula en paralelo los canales que
  falten.
"""

from __future__ import annotations

import json

import numpy as np

from tweetage import cache
from tweetage import models_extra as MX
from tweetage import pipeline as P

CHANNELS_V2 = {
    **P.CHANNELS,
    # --- A2: canales de palabras vigentes con TweetTokenizer (mismas temperaturas) ---
    "word1_tt": (MX.TTWordNgramLM, {"orders": (1,), "alpha": 0.01}, 1.0),
    "position_tt": (MX.TTPositionalWordLM, {"mode": "start_end", "alpha": 0.03}, 1.0),
    "mention_pos_tt": (MX.TTPositionAwareMentionLM, {"alpha": 0.01, "mode": "all"}, 1.0),
    "word_seq3_tt": (MX.WordSeqLM, {"alpha": 0.1, "min_count": 3, "order": 3, "tokenizer": "tweet"}, 0.2),
    # --- Submission 0.60270 del equipo: LM de palabras que conserva entidades frecuentes ---
    "word_seq3_ent5": (MX.EntityWordSeqLM, {"alpha": 0.1, "min_count": 3, "entity_min_count": 5, "order": 3}, 0.2),
    "word_seq3_ent5_tt": (
        MX.EntityWordSeqLM,
        {"alpha": 0.1, "min_count": 3, "entity_min_count": 5, "order": 3, "tokenizer": "tweet"},
        0.2,
    ),
    # --- C1: interpolación Jelinek-Mercer ---
    "word_seq3_jm": (MX.InterpolatedWordSeqLM, {"alpha": 0.1, "min_count": 3, "lambda3": 0.6, "lambda2": 0.6}, 1.0),
    "word_seq3_jm_tt": (
        MX.InterpolatedWordSeqLM,
        {"alpha": 0.1, "min_count": 3, "lambda3": 0.6, "lambda2": 0.6, "tokenizer": "tweet"},
        1.0,
    ),
    # --- C2/C3: mayúsculas y forma ---
    "char4_cs": (MX.CaseSensitiveNgramLM, {"orders": (4,), "alpha": 0.1}, 1.0),
    "char5_cs": (MX.CaseSensitiveNgramLM, {"orders": (5,), "alpha": 0.1}, 1.0),
    "char6_cs": (MX.CaseSensitiveNgramLM, {"orders": (6,), "alpha": 0.03}, 1.0),
    "shape5": (MX.ShapeNgramLM, {"orders": (5,), "alpha": 0.1}, 1.0),
    "shape7": (MX.ShapeNgramLM, {"orders": (7,), "alpha": 0.1}, 1.0),
    # --- C4: Markov en reversa y de orden mayor ---
    "char_seq2_rev": (MX.CharSeqLM, {"alpha": 0.1, "order": 2, "reverse": True}, 1.0),
    "char_seq4": (MX.CharSeqLM, {"alpha": 0.1, "order": 4}, 1.0),
    "word_seq3_rev": (MX.WordSeqLM, {"alpha": 0.1, "min_count": 3, "order": 3, "reverse": True}, 1.0),
    "word_seq3_rev_tt": (
        MX.WordSeqLM,
        {"alpha": 0.1, "min_count": 3, "order": 3, "reverse": True, "tokenizer": "tweet"},
        1.0,
    ),
    # --- C5: longitud ---
    "length": (MX.LengthBinLM, {"alpha": 1.0}, 1.0),
    # --- C6: especialistas 18-24 vs 25-34 ---
    "spec_char5": (MX.PairSpecialistLM, {"unit": "char", "orders": (5,), "alpha": 0.1}, 1.0),
    "spec_word1": (MX.PairSpecialistLM, {"unit": "word", "orders": (1,), "alpha": 0.01}, 1.0),
}

BASE_NAMES = list(P.CHANNELS)
NEW_NAMES = ["word_seq3_jm", "char5_cs", "shape5", "shape7", "char_seq2_rev", "char_seq4", "word_seq3_rev", "length"]
SPECIALISTS = ["spec_char5", "spec_word1"]

# Si la variante usa TweetTokenizer, estos canales se reemplazan por su versión _tt.
TT_REPLACE = {
    "word1": "word1_tt",
    "position": "position_tt",
    "mention_pos": "mention_pos_tt",
    "word_seq3": "word_seq3_tt",
    "word_seq3_jm": "word_seq3_jm_tt",
    "word_seq3_rev": "word_seq3_rev_tt",
}

# Compuertas: el canal solo vota en los tweets que tienen su evidencia.
GATES_DEFAULT = {"mention_pos": "mention"}  # igual que la mezcla vigente
GATES_SOCIAL = {"mention_pos": "mention", "mentions": "mention", "hashtags": "hashtag"}

# Variantes de preprocesamiento (A1–A5). "base" reproduce exactamente el 02.
VARIANTS = {
    "base": {"unescape": False, "tweet_tok": False, "clean": False, "template": False},
    "A1_unescape": {"unescape": True, "tweet_tok": False, "clean": False, "template": False},
    "A2_tweettok": {"unescape": False, "tweet_tok": True, "clean": False, "template": False},
    "A3_unescape_tweettok": {"unescape": True, "tweet_tok": True, "clean": False, "template": False},
    "A4_clean": {"unescape": False, "tweet_tok": False, "clean": True, "template": False},
    "A5_template": {"unescape": False, "tweet_tok": False, "clean": False, "template": True},
}


def variant_name(flags: dict) -> str:
    for name, f in VARIANTS.items():
        if f == flags:
            return name
    on = [k for k, v in flags.items() if v]
    return "v_" + "_".join(on) if on else "base"


def resolve(names, flags: dict) -> dict:
    """Nombre lógico -> nombre real del canal según la variante (p. ej. word1 -> word1_tt)."""
    return {n: (TT_REPLACE.get(n, n) if flags["tweet_tok"] else n) for n in names}


# ---------------------------------------------------------------- caché paralela


def cache_path(name, X_train, y_train, X_eval, tag, channels=CHANNELS_V2):
    """Misma ruta que ``cache.channel`` (reutiliza los archivos ya calculados en el 02)."""
    cls, kwargs, temperature = channels[name]
    config = json.dumps({"cls": cls.__name__, "kwargs": kwargs, "t": temperature}, sort_keys=True, default=str)
    key = cache._fingerprint(config, X_train, y_train, X_eval)
    return cache.CACHE_DIR / f"{name}__{tag or 'notag'}__{key}.npy"


def cached_probs(names, X_train, y_train, X_eval, tag, channels=CHANNELS_V2, n_jobs=4, verbose=True):
    """Probabilidades de varios canales: carga las que están en caché y calcula el resto."""
    names = list(dict.fromkeys(names))
    paths = {n: cache_path(n, X_train, y_train, X_eval, tag, channels) for n in names}
    out = {n: np.load(p) for n, p in paths.items() if p.exists()}
    missing = [n for n in names if n not in out]
    if missing:
        if verbose:
            print(f"[{tag}] entrenando {len(missing)} canal(es): {', '.join(missing)}")
        new = P.all_channel_probs(X_train, y_train, X_eval, names=missing, n_jobs=min(n_jobs, len(missing)), channels=channels)
        cache.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for n, prob in new.items():
            np.save(paths[n], prob)
        out.update(new)
    return {n: out[n] for n in names}
