"""Preprocesamiento v2: entidades HTML, TweetTokenizer, limpieza de train y plantillas.

Todo es determinista y basado en reglas (sin modelos entrenados):

* ``unescape``: decodifica entidades HTML (``&lt;3`` -> ``<3``) con ``html.unescape``
  de la biblioteca estándar.
* ``tweet_tokens``: ``nltk.tokenize.TweetTokenizer`` (solo expresiones regulares; NLTK,
  licencia Apache 2.0). Mantiene emoticonos (``:)``, ``<3``), URLs, @menciones y
  #hashtags; decodifica entidades HTML y recorta alargamientos (``siiiii`` -> ``siii``).
  No usar ``word_tokenize``/``sent_tokenize``: dependen de Punkt, que es preentrenado.
* ``clean_mask`` / ``template_filter_mask``: filas de train que se conservan.
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------- texto


def unescape(texts) -> list[str]:
    """Decodifica entidades HTML. Aplicarlo dos veces no cambia el resultado útil."""
    return [html.unescape(t) for t in texts]


_TWEET_TOK = None


def _tokenizer():
    global _TWEET_TOK
    if _TWEET_TOK is None:
        from nltk.tokenize import TweetTokenizer  # import diferido: el módulo carga sin NLTK

        _TWEET_TOK = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    return _TWEET_TOK


@lru_cache(maxsize=600_000)
def tweet_tokens(text: str) -> tuple[str, ...]:
    """Tokens de TweetTokenizer en minúsculas (los emoticonos conservan su forma).

    Se tokeniza el texto original, no el casefold, para que ``:D`` siga siendo emoticono.
    Se cachea por texto: varios canales tokenizan los mismos tweets.
    """
    return tuple(_tokenizer().tokenize("" if text is None else str(text)))


def is_url(tok: str) -> bool:
    t = tok.casefold()
    return t.startswith("http") or t.startswith("www.")


# ---------------------------------------------------------------- limpieza de train


def clean_mask(X) -> np.ndarray:
    """True = conservar. Quita textos vacíos y las celdas '#¿NOMBRE?' (error de Excel)."""
    return np.array([bool(t.strip()) and "¿NOMBRE?" not in t for t in X], dtype=bool)


def n2(texto: str) -> str:
    """Plantilla N2 (igual que en 01_exploracion): URL, @, #, números y alargamientos."""
    t = re.sub(r"\s+", " ", texto.strip().casefold())
    t = re.sub(r"https?://\S*|www\.\S+", "URL", t)
    t = re.sub(r"@\w+", "@U", t)
    t = re.sub(r"#\w+", "#H", t)
    t = re.sub(r"\d+", "0", t)
    return re.sub(r"(.)\1{2,}", r"\1\1", t)


def template_filter_mask(X, y, max_purity: float = 0.5, min_n: int = 5) -> np.ndarray:
    """True = conservar. Quita los tweets cuya plantilla N2 es ruidosa en ESTE train.

    Ruidosa = aparece al menos ``min_n`` veces y su clase más frecuente cubre como
    máximo ``max_purity``. Se calcula solo con los datos recibidos (el Xtr de cada
    semilla), así que no hay fuga desde validación ni test.
    """
    plantillas = [n2(t) for t in X]
    conteo: dict = defaultdict(Counter)
    for p, c in zip(plantillas, y, strict=True):
        conteo[p][c] += 1
    ruidosas = set()
    for p, cnt in conteo.items():
        n = sum(cnt.values())
        if n >= min_n and max(cnt.values()) / n <= max_purity:
            ruidosas.add(p)
    return np.array([p not in ruidosas for p in plantillas], dtype=bool)


# ---------------------------------------------------------------- compuertas (gates)


def gate_masks(X) -> dict[str, np.ndarray]:
    """Máscaras por tweet: True si el canal tiene evidencia propia en ese tweet."""
    has_mention = np.array([bool(re.search(r"@\w+", t)) for t in X], dtype=bool)
    has_hashtag = np.array([bool(re.search(r"#\w+", t)) for t in X], dtype=bool)
    return {"mention": has_mention, "hashtag": has_hashtag}
