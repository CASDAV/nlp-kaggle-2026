"""Modelos de lenguaje condicionales a la clase para TweetAge (versión optimizada).

Misma matemática y misma API (``fit`` / ``decision_function`` / ``classes_``) que las
clases de los notebooks originales; cambia solo cómo se calcula:

* En ``fit`` cada evento (n-grama, mención, transición...) recibe un id entero y los
  conteos por clase se guardan en una matriz ``(V, n_clases)``.
* Los logaritmos se precalculan una sola vez (``log(conteo + alpha)``), con una fila
  extra para eventos no vistos (``log(alpha)``).
* En ``decision_function`` cada texto se convierte en una lista de ids y el puntaje
  de las 6 clases se obtiene con una suma vectorizada de numpy: una búsqueda en un
  diccionario por evento, en lugar de 6 búsquedas y 6 ``math.log``.

No hay vectorización del texto en el sentido de la regla de TweetAge: no se construye
ninguna representación vectorial de los documentos (ni BoW, ni TF-IDF, ni embeddings).
Son los mismos modelos probabilísticos de n-gramas; los arreglos de numpy solo guardan
los parámetros del modelo (conteos por clase) y acumulan log-verosimilitudes.
"""

from __future__ import annotations

import math
import re
from array import array

import numpy as np

TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|@\w+|#\w+|\w+|[^\w\s]", flags=re.UNICODE | re.IGNORECASE)
CHUNK = 20_000  # textos por bloque al predecir (limita memoria)


def _norm(text) -> str:
    return "" if text is None else str(text).casefold()


def _segment_sums(table: np.ndarray, ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Suma, por texto, las filas ``table[ids]`` de cada segmento (textos vacíos -> 0)."""
    out = np.zeros((len(lengths), table.shape[1]), dtype=np.float64)
    nonempty = lengths > 0
    if ids.size:
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        out[nonempty] = np.add.reduceat(table[ids], starts[nonempty], axis=0)
    return out


class _EventLM:
    """Base: LM multinomial/categórico sobre eventos discretos con suavizado aditivo.

    score_c = log P(c) - n·log(N_c + α·V) + Σ_e log(conteo_c(e) + α)
    """

    def __init__(self, alpha: float):
        if alpha <= 0:
            raise ValueError("alpha debe ser positivo")
        self.alpha = float(alpha)

    # --- a implementar por cada modelo ---
    def _events(self, text) -> list:
        raise NotImplementedError

    def _extra_vocab(self) -> int:
        return 0

    # --- comunes ---
    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, sizes = np.unique(y, return_counts=True)
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        k = len(self.classes_)
        self.index_: dict = {}
        index = self.index_
        flat = array("q")
        for text, label in zip(X, y, strict=True):
            c = class_idx[label]
            setdef = index.setdefault
            flat.extend(setdef(e, len(index)) * k + c for e in self._events(text))
        n_events = len(index)
        counts = np.bincount(np.frombuffer(flat, dtype=np.int64), minlength=n_events * k).reshape(n_events, k)
        del flat
        self.totals_ = counts.sum(axis=0)
        self.vocab_size_ = n_events + self._extra_vocab()
        self.log_priors_ = np.log(sizes / len(y))
        # fila extra al final = evento no visto
        self.log_table_ = np.vstack([np.log(counts + self.alpha), np.full((1, k), math.log(self.alpha))])
        self.log_denom_ = np.log(self.totals_ + self.alpha * self.vocab_size_)
        return self

    def decision_function(self, X):
        X = list(X)
        unk = len(self.index_)
        get = self.index_.get
        out = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        for s in range(0, len(X), CHUNK):
            block = X[s : s + CHUNK]
            ids, lengths = array("q"), np.empty(len(block), dtype=np.int64)
            for i, text in enumerate(block):
                ev = [get(e, unk) for e in self._events(text)]
                lengths[i] = len(ev)
                ids.extend(ev)
            ids = np.frombuffer(ids, dtype=np.int64)
            sums = _segment_sums(self.log_table_, ids, lengths)
            out[s : s + len(block)] = self.log_priors_ - lengths[:, None] * self.log_denom_ + sums
        return out


class ClassConditionalNgramLM(_EventLM):
    """Naive Bayes multinomial sobre n-gramas de caracteres o palabras."""

    def __init__(self, unit="char", orders=(3,), alpha=0.1):
        if unit not in {"char", "word"} or not orders:
            raise ValueError("Configuración inválida")
        super().__init__(alpha)
        self.unit, self.orders = unit, tuple(orders)

    def _events(self, text):
        text = _norm(text)
        if self.unit == "char":
            # Un substring de longitud n equivale a la tupla de sus n caracteres.
            return [text[i : i + n] for n in self.orders for i in range(len(text) - n + 1)]
        toks = TOKEN_RE.findall(text)
        return [tuple(toks[i : i + n]) for n in self.orders for i in range(len(toks) - n + 1)]

    def _extra_vocab(self):
        return len(self.orders)


class IdentityClassLM(_EventLM):
    """Modelo categórico sobre el conjunto de ítems únicos (menciones, hashtags...)."""

    def __init__(self, pattern, alpha=0.1):
        super().__init__(alpha)
        self.pattern = re.compile(pattern, flags=re.UNICODE)

    def _events(self, text):
        return list(set(self.pattern.findall(_norm(text))))

    def _extra_vocab(self):
        return 1


class PositionalWordLM(_EventLM):
    """LM multinomial sobre eventos (zona relativa, token)."""

    def __init__(self, mode="start_end", alpha=0.1):
        if mode not in {"start_end", "three_zones"}:
            raise ValueError("mode must be start_end or three_zones")
        super().__init__(alpha)
        self.mode = mode

    def _events(self, text):
        tokens = TOKEN_RE.findall(_norm(text))
        n = len(tokens)
        if not n:
            return []
        if self.mode == "start_end":
            k = max(1, n // 4)
            k = min(k, n // 2) if n > 1 else 1
            return [("start", t) for t in tokens[:k]] + [("end", t) for t in tokens[n - k :]]
        return [(("start", "middle", "end")[min(2, (3 * i) // n)], t) for i, t in enumerate(tokens)]

    def _extra_vocab(self):
        return len({e[0] for e in self.index_})


class PositionAwareMentionLM(_EventLM):
    """Menciones distinguiendo la inicial (respuesta) de las que van dentro del texto."""

    def __init__(self, alpha=0.01, mode="all"):
        if mode not in {"all", "leading", "embedded"}:
            raise ValueError("modo inválido")
        super().__init__(alpha)
        self.mode = mode

    def _events(self, text):
        events = set()
        for i, tok in enumerate(TOKEN_RE.findall(_norm(text))):
            if not tok.startswith("@"):
                continue
            zone = "leading" if i == 0 else "in_text"
            if (
                self.mode == "all"
                or (self.mode == "leading" and zone == "leading")
                or (self.mode == "embedded" and zone == "in_text")
            ):
                events.add((zone, tok))
        return list(events)

    def _extra_vocab(self):
        return 2

    @staticmethod
    def masks(texts):
        has = np.asarray([bool(re.search(r"@\w+", "" if t is None else str(t))) for t in texts], dtype=bool)
        leading = np.asarray([bool(re.match(r"^\s*@\w+", "" if t is None else str(t))) for t in texts], dtype=bool)
        return {"all": has, "leading": leading, "embedded": has & ~leading}


class _SequenceLM:
    """Base: LM de Markov condicional a la clase, P(x_t | contexto), normalizado por longitud.

    score_c = (Σ_t log[(trans_c(ctx,x)+α) / (ctx_c(ctx)+α·V)] + log P(c)) / n
    """

    def __init__(self, alpha: float):
        if alpha <= 0:
            raise ValueError("alpha debe ser positivo")
        self.alpha = float(alpha)

    def _prepare_vocab(self, texts):
        raise NotImplementedError

    def _pairs(self, text):
        """Devuelve (lista de claves (ctx, x), lista de claves ctx) del texto."""
        raise NotImplementedError

    def fit(self, texts, y):
        texts, y = list(texts), np.asarray(y)
        self.classes_, sizes = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(sizes / sizes.sum())
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        k = len(self.classes_)
        self._prepare_vocab(texts)
        self.pair_index_, self.ctx_index_ = {}, {}
        p_idx, c_idx = self.pair_index_, self.ctx_index_
        p_flat, c_flat = array("q"), array("q")
        for text, label in zip(texts, y, strict=True):
            c = class_idx[label]
            pairs, ctxs = self._pairs(text)
            p_flat.extend(p_idx.setdefault(p, len(p_idx)) * k + c for p in pairs)
            c_flat.extend(c_idx.setdefault(x, len(c_idx)) * k + c for x in ctxs)
        trans = np.bincount(np.frombuffer(p_flat, dtype=np.int64), minlength=len(p_idx) * k).reshape(-1, k)
        ctx = np.bincount(np.frombuffer(c_flat, dtype=np.int64), minlength=len(c_idx) * k).reshape(-1, k)
        a, V = self.alpha, self.vocab_size_
        self.log_trans_ = np.vstack([np.log(trans + a), np.full((1, k), math.log(a))])
        self.log_ctx_ = np.vstack([np.log(ctx + a * V), np.full((1, k), math.log(a * V))])
        return self

    def decision_function(self, texts):
        texts = list(texts)
        p_get, c_get = self.pair_index_.get, self.ctx_index_.get
        p_unk, c_unk = len(self.pair_index_), len(self.ctx_index_)
        out = np.empty((len(texts), len(self.classes_)), dtype=np.float64)
        for s in range(0, len(texts), CHUNK):
            block = texts[s : s + CHUNK]
            p_ids, c_ids = array("q"), array("q")
            lengths = np.empty(len(block), dtype=np.int64)
            for i, text in enumerate(block):
                pairs, ctxs = self._pairs(text)
                lengths[i] = len(pairs)
                p_ids.extend(p_get(p, p_unk) for p in pairs)
                c_ids.extend(c_get(x, c_unk) for x in ctxs)
            p_ids = np.frombuffer(p_ids, dtype=np.int64)
            c_ids = np.frombuffer(c_ids, dtype=np.int64)
            ll = _segment_sums(self.log_trans_, p_ids, lengths) - _segment_sums(self.log_ctx_, c_ids, lengths)
            n = np.maximum(1, lengths)[:, None]
            out[s : s + len(block)] = ll / n + self.log_priors_ / n
        return out


class CharacterSequenceLM(_SequenceLM):
    """LM de caracteres con contexto de 2 caracteres (trigramas de caracteres)."""

    START, END, UNK = "\u0002", "\u0003", "\u0004"
    ORDER = 2

    def __init__(self, alpha=0.1):
        super().__init__(alpha)

    def _prepare_vocab(self, texts):
        self.alphabet_ = {ch for t in texts for ch in _norm(t)}
        self.alphabet_.update((self.START, self.END, self.UNK))
        self.vocab_size_ = len(self.alphabet_)

    def _pairs(self, text):
        alphabet, unk, o = self.alphabet_, self.UNK, self.ORDER
        body = "".join(ch if ch in alphabet else unk for ch in _norm(text))
        seq = self.START * o + body + self.END
        rng = range(o, len(seq))
        return [seq[p - o : p + 1] for p in rng], [seq[p - o : p] for p in rng]


class WordSequenceLM(_SequenceLM):
    """LM de palabras P(token_t | tokens anteriores), orden 2 o 3."""

    START, END, UNK = "<s>", "</s>", "<unk>"

    def __init__(self, alpha=0.1, min_count=3, order=2):
        if min_count < 1 or order not in (2, 3):
            raise ValueError("min_count u order inválido")
        super().__init__(alpha)
        self.min_count, self.order = int(min_count), int(order)
        self.context_size = self.order - 1

    @staticmethod
    def _raw_tokens(text):
        out = []
        for tok in TOKEN_RE.findall(_norm(text)):
            if tok.startswith("http") or tok.startswith("www."):
                out.append("URL")
            elif tok.startswith("@"):
                out.append("@USER")
            elif tok.startswith("#"):
                out.append("#TAG")
            else:
                out.append(tok)
        return out

    def _prepare_vocab(self, texts):
        counts: dict = {}
        for t in texts:
            for tok in self._raw_tokens(t):
                counts[tok] = counts.get(tok, 0) + 1
        self.vocab_ = {tok for tok, n in counts.items() if n >= self.min_count}
        self.vocab_.update((self.START, self.END, self.UNK))
        self.vocab_size_ = len(self.vocab_)

    def _pairs(self, text):
        vocab, unk, k = self.vocab_, self.UNK, self.context_size
        seq = [self.START] * k + [t if t in vocab else unk for t in self._raw_tokens(text)] + [self.END]
        rng = range(k, len(seq))
        return [tuple(seq[p - k : p + 1]) for p in rng], [tuple(seq[p - k : p]) for p in rng]


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def probabilities(model, X, classes, temperature: float = 1.0) -> np.ndarray:
    """Probabilidades por clase en el orden de ``classes``, con temperatura opcional."""
    scores = model.decision_function(X)
    order = [int(np.flatnonzero(model.classes_ == c)[0]) for c in classes]
    return softmax(scores[:, order] / temperature)
