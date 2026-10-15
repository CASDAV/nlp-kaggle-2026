"""Registro incremental de experimentos: un CSV al que solo se agregan filas."""

from __future__ import annotations

import csv
import json
from datetime import datetime

import pandas as pd

from tweetage.paths import work_dir

FIELDS = ["id", "fecha", "nombre", "config", "seed", "f1_macro", "accuracy", "kaggle_public", "notas"]


def _path():
    return work_dir() / "experiments" / "registry.csv"


def log(nombre, config, seed, f1_macro, accuracy, notas="", kaggle_public=""):
    """Agrega un experimento y devuelve su id."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists()
    next_id = 1 if nuevo else len(pd.read_csv(path)) + 1
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if nuevo:
            w.writeheader()
        w.writerow(
            {
                "id": next_id,
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "nombre": nombre,
                "config": json.dumps(config, sort_keys=True, default=str),
                "seed": seed,
                "f1_macro": round(float(f1_macro), 5),
                "accuracy": round(float(accuracy), 5),
                "kaggle_public": kaggle_public,
                "notas": notas,
            }
        )
    return next_id


def set_kaggle_score(exp_id, score):
    """Anota el score público de Kaggle en un experimento ya registrado."""
    df = pd.read_csv(_path())
    df.loc[df["id"] == exp_id, "kaggle_public"] = score
    df.to_csv(_path(), index=False)


def show(sort="f1_macro", ascending=False):
    path = _path()
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path).sort_values(sort, ascending=ascending).reset_index(drop=True)
