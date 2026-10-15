"""Rutas y carga de datos, iguales en local (uv), Kaggle y Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

COMPETITION = "tweet-age-2026-puj"
TRAIN_NAMES = ("age_train.csv", "train.csv")
TEST_NAMES = ("age_test.csv", "test.csv")


def environment() -> str:
    if Path("/kaggle/input").exists():
        return "kaggle"
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def project_root() -> Path:
    """Carpeta tweetage/. Si el código está pegado en un notebook, usa el directorio actual."""
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


def work_dir() -> Path:
    """Carpeta con permisos de escritura: registro, caché y submissions."""
    env = environment()
    if env == "kaggle":
        return Path("/kaggle/working")
    if env == "colab":
        return Path("/content")
    return project_root()


def data_dir() -> Path:
    """Carpeta de los CSV. Se puede forzar con la variable de entorno TWEETAGE_DATA."""
    if os.environ.get("TWEETAGE_DATA"):
        return Path(os.environ["TWEETAGE_DATA"])
    env = environment()
    if env == "kaggle":
        # Busca el train dentro de /kaggle/input, sin depender de la ruta exacta del montaje.
        for name in TRAIN_NAMES:
            hits = sorted(Path("/kaggle/input").rglob(name))
            if hits:
                return hits[0].parent
        raise FileNotFoundError(
            "No se encontró el train en /kaggle/input; agrega los datos de la competencia"
        )
    if env == "colab":
        return Path("/content/data")
    return project_root() / "data" / "raw"


def _find(directory: Path, names) -> Path:
    for name in names:
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"Ninguno de {names} existe en {directory}")


def load_data():
    """Devuelve (train, test) como DataFrames y valida las columnas esperadas."""
    d = data_dir()
    train = pd.read_csv(_find(d, TRAIN_NAMES), sep=";", encoding="utf-8")
    test = pd.read_csv(_find(d, TEST_NAMES), sep=";", encoding="utf-8")
    for df, cols in ((train, {"id", "text", "age_range"}), (test, {"id", "text"})):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")
    return train, test


def texts(df: pd.DataFrame) -> list[str]:
    """Textos listos para los modelos: nulos -> "", sin espacios en los extremos."""
    return df["text"].fillna("").astype(str).str.strip().tolist()
