"""Registro incremental de experimentos: un CSV con una fila por experimento.

Si se vuelve a registrar un experimento con el mismo nombre y semilla (por ejemplo al
re-ejecutar el notebook), se actualiza su fila en lugar de duplicarla.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from tweetage.paths import work_dir

FIELDS = ["id", "fecha", "nombre", "config", "seed", "f1_macro", "accuracy", "kaggle_public", "notas"]


def _path():
    return work_dir() / "experiments" / "registry.csv"


def _read() -> pd.DataFrame:
    path = _path()
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path, dtype={"nombre": str, "notas": str, "config": str})


def log(nombre, config, seed, f1_macro, accuracy, notas="", kaggle_public=None):
    """Agrega (o actualiza, si ya existe nombre+seed) un experimento y devuelve su id."""
    df = _read()
    row = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "nombre": nombre,
        "config": json.dumps(config, sort_keys=True, default=str, ensure_ascii=False),
        "seed": seed,
        "f1_macro": round(float(f1_macro), 5),
        "accuracy": round(float(accuracy), 5),
        "notas": notas,
    }
    same = (df["nombre"] == nombre) & (df["seed"].astype(str) == str(seed))
    if same.any():
        i = df.index[same][0]
        for key, value in row.items():
            df.loc[i, key] = value
        if kaggle_public is not None:
            df.loc[i, "kaggle_public"] = kaggle_public
        exp_id = int(df.loc[i, "id"])
    else:
        exp_id = 1 if df.empty else int(df["id"].max()) + 1
        new = pd.DataFrame([{**row, "id": exp_id, "kaggle_public": kaggle_public}])
        df = new if df.empty else pd.concat([df, new], ignore_index=True)
    _path().parent.mkdir(parents=True, exist_ok=True)
    df[FIELDS].to_csv(_path(), index=False)
    return exp_id


def set_kaggle_score(exp_id, score):
    """Anota el score público de Kaggle en un experimento ya registrado."""
    df = _read()
    df.loc[df["id"] == exp_id, "kaggle_public"] = score
    df[FIELDS].to_csv(_path(), index=False)


def show(sort="id", ascending=True):
    return _read().sort_values(sort, ascending=ascending).reset_index(drop=True)
