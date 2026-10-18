"""Orquestación de la tanda v2 (la usan el 02 y el 04; los notebooks solo llaman y muestran).

``SeedRun`` agrupa, para una semilla y una variante de preprocesamiento, las
probabilidades de validación de cada canal, las compuertas y las etiquetas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from tweetage import channels_extra as CX
from tweetage import evaluation as E
from tweetage import preprocess as PP
from tweetage import validation
from tweetage.paths import work_dir


@dataclass
class SeedRun:
    seed: int
    variant: str
    probs: dict  # nombre lógico -> probabilidades (n_valid, 6)
    masks: dict  # "mention" / "hashtag" -> bool (n_valid,)
    y_idx: np.ndarray
    resolved: dict = field(default_factory=dict)  # nombre lógico -> canal real


def apply_variant_train(Xtr, ytr, flags):
    """Filtra filas de train (A4/A5) y decodifica HTML (A1/A3) según la variante."""
    keep = np.ones(len(Xtr), dtype=bool)
    if flags["clean"]:
        keep &= PP.clean_mask(Xtr)
    if flags["template"]:
        keep &= PP.template_filter_mask(Xtr, ytr)
    idx = np.flatnonzero(keep)
    Xtr, ytr = [Xtr[i] for i in idx], np.asarray(ytr)[idx]
    if flags["unescape"]:
        Xtr = PP.unescape(Xtr)
    return Xtr, ytr


def apply_variant_eval(Xev, flags):
    return PP.unescape(Xev) if flags["unescape"] else list(Xev)


def run_seed(X, y, seed, flags, names, n_jobs=4) -> SeedRun:
    """Entrena (o carga de caché) ``names`` para la semilla y variante dadas."""
    Xtr, ytr, Xva, yva = validation.split(X, y, seed)
    vname = CX.variant_name(flags)
    # La variante base usa la misma etiqueta que el 02 para reutilizar su caché.
    tag = f"seed{seed}" if vname == "base" else f"seed{seed}_{vname}"
    Xtr_v, ytr_v = apply_variant_train(Xtr, ytr, flags)
    Xva_v = apply_variant_eval(Xva, flags)
    resolved = CX.resolve(names, flags)
    probs = CX.cached_probs(list(resolved.values()), Xtr_v, ytr_v, Xva_v, tag=tag, n_jobs=n_jobs)
    return SeedRun(
        seed=seed,
        variant=vname,
        probs={logical: probs[real] for logical, real in resolved.items()},
        masks=PP.gate_masks(Xva_v),
        y_idx=E.to_idx(yva),
        resolved=resolved,
    )


def concat_runs(runs):
    """Une las validaciones de varias semillas (para ajustar los parámetros finales)."""
    names = list(runs[0].probs)
    return (
        {n: np.concatenate([r.probs[n] for r in runs]) for n in names},
        {k: np.concatenate([r.masks[k] for r in runs]) for k in runs[0].masks},
        np.concatenate([r.y_idx for r in runs]),
    )


# ---------------------------------------------------------------- especificaciones
#
# Una spec describe una submission completa:
#   flags       variante de preprocesamiento (VARIANTS)
#   channels    canales lógicos que alimentan la mezcla
#   swaps       {canal de la mezcla vigente: canal que lo reemplaza}, p. ej. char5 -> char5_cs
#   blend       {"kind": "pipeline_v1"} (pesos vigentes) o una mezcla de blend_opt (to_dict)
#   adds        [{"name", "temperature", "weight"}]: canales agregados en orden
#   specialist  {"name", "temperature", "weight"} o None
#   bias        6 sesgos o None


def spec_channels(spec) -> list[str]:
    names = list(spec.get("channels", []))
    names += list((spec.get("swaps") or {}).values())
    names += [a["name"] for a in spec.get("adds") or []]
    if spec.get("specialist"):
        names.append(spec["specialist"]["name"])
    return list(dict.fromkeys(names))


def predict_spec(spec, probs, masks, proba=False):
    """Puntajes (n, 6) de una spec: log p + sesgos (para argmax).

    Con ``proba=True`` devuelve las probabilidades de la mezcla antes de los sesgos
    (necesario para seguir agregando canales encima).
    """
    from tweetage import blend_opt as B
    from tweetage import pipeline as P

    p = dict(probs)
    for old, new in (spec.get("swaps") or {}).items():
        p[old] = probs[new]
    if spec["blend"]["kind"] == "pipeline_v1":
        out = P.blend(p, masks["mention"])
    else:
        out = B.blend_from_dict(spec["blend"]).predict_proba(p, masks)
    for a in spec.get("adds") or []:
        out = B.add_channel(out, probs[a["name"]], a["temperature"], a["weight"])
    if spec.get("specialist"):
        sp = spec["specialist"]
        out = B.specialist_apply(out, probs[sp["name"]], sp["temperature"], sp["weight"])
    return out if proba else B.apply_bias(out, spec.get("bias"))



def spec_path(nombre):
    return work_dir() / "experiments" / f"spec_{nombre}.json"


def save_spec(nombre, spec: dict):
    path = spec_path(nombre)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False, default=float))
    return path


def load_spec(nombre) -> dict:
    return json.loads(spec_path(nombre).read_text())
