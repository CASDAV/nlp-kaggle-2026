"""Caché en disco de probabilidades por canal.

La clave combina el nombre del canal, su configuración (clase, kwargs, temperatura),
una huella de los datos de entrenamiento/evaluación y una etiqueta libre (p. ej. la
semilla del split). Si cambia cualquiera de esas cosas, se recalcula.

Uso:
    from tweetage import cache, pipeline as P
    p = {name: cache.channel(name, Xtr, ytr, Xva, tag="seed20261108") for name in P.CHANNELS}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tweetage import pipeline as P
from tweetage.paths import work_dir

CACHE_DIR = Path(os.environ.get("TWEETAGE_CACHE", work_dir() / "cache"))


def _fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for part in parts:
        if isinstance(part, (list, tuple, np.ndarray)):
            for item in part:
                h.update(str(item).encode("utf-8"))
                h.update(b"\x1f")
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def channel(name, X_train, y_train, X_eval, tag="", channels=None, enabled=True):
    channels = channels or P.CHANNELS
    if not enabled:
        return P.channel_probs(name, X_train, y_train, X_eval, channels)
    cls, kwargs, temperature = channels[name]
    config = json.dumps({"cls": cls.__name__, "kwargs": kwargs, "t": temperature}, sort_keys=True, default=str)
    key = _fingerprint(config, X_train, y_train, X_eval)
    path = CACHE_DIR / f"{name}__{tag or 'notag'}__{key}.npy"
    if path.exists():
        return np.load(path)
    prob = P.channel_probs(name, X_train, y_train, X_eval, channels)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, prob)
    return prob
