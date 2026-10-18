"""Modelos de lenguaje nuevos (v2). Extienden las bases de ``models.py`` sin modificarlas.

Todos siguen siendo modelos probabilísticos de n-gramas condicionales a la clase con la
misma API (``fit`` / ``decision_function`` / ``classes_``); no construyen ninguna
representación vectorial de los documentos.

* ``TT*``: mismos modelos de palabras pero con ``TweetTokenizer`` (emoticonos completos,
  entidades HTML decodificadas, alargamientos recortados).
* ``CaseSensitiveNgramLM``: n-gramas de caracteres sin casefold (mayúsculas).
* ``ShapeNgramLM``: n-gramas sobre la "forma" del texto (A/a/9/puntuación).
* ``LengthBinLM``: longitud del tweet como evento categórico.
* ``CharSeqLM`` / ``WordSeqLM``: Markov de caracteres/palabras con orden y sentido
  (hacia adelante o en reversa) configurables.
* ``InterpolatedWordSeqLM``: Markov de palabras de orden 3 con interpolación
  Jelinek-Mercer recursiva (3 -> 2 -> 1) en lugar de solo suavizado aditivo.
* ``PairSpecialistLM``: NB entrenado solo con dos clases (18-24 frente a 25-34).
"""

from __future__ import annotations

import re
from array import array

import numpy as np

from tweetage import models as M
from tweetage.preprocess import is_url, tweet_tokens

# ---------------------------------------------------------------- TweetTokenizer


def _tt_ngrams(text, orders):
    toks = tweet_tokens(text)
    return [toks[i : i + n] for n in orders for i in range(len(toks) - n + 1)]


class TTWordNgramLM(M.ClassConditionalNgramLM):
    """NB multinomial sobre n-gramas de palabras de TweetTokenizer."""

    def __init__(self, orders=(1,), alpha=0.01):
        super().__init__(unit="word", orders=orders, alpha=alpha)

    def _events(self, text):
        return _tt_ngrams(text, self.orders)


class TTPositionalWordLM(M.PositionalWordLM):
    def _events(self, text):
        tokens = tweet_tokens(text)
        n = len(tokens)
        if not n:
            return []
        if self.mode == "start_end":
            k = max(1, n // 4)
            k = min(k, n // 2) if n > 1 else 1
            return [("start", t) for t in tokens[:k]] + [("end", t) for t in tokens[n - k :]]
        return [(("start", "middle", "end")[min(2, (3 * i) // n)], t) for i, t in enumerate(tokens)]


class TTPositionAwareMentionLM(M.PositionAwareMentionLM):
    def _events(self, text):
        events = set()
        for i, tok in enumerate(tweet_tokens(text)):
            if not (tok.startswith("@") and len(tok) > 1):
                continue
            zone = "leading" if i == 0 else "in_text"
            if (
                self.mode == "all"
                or (self.mode == "leading" and zone == "leading")
                or (self.mode == "embedded" and zone == "in_text")
            ):
                events.add((zone, tok))
        return list(events)


def _word_units(text, tokenizer: str) -> list[str]:
    """Tokens con URL/@USER/#TAG como en WordSequenceLM, con el tokenizador elegido."""
    if tokenizer == "regex":
        return M.WordSequenceLM._raw_tokens(text)
    out = []
    for tok in tweet_tokens(text):
        if is_url(tok):
            out.append("URL")
        elif tok.startswith("@") and len(tok) > 1:
            out.append("@USER")
        elif tok.startswith("#") and len(tok) > 1:
            out.append("#TAG")
        else:
            out.append(tok)
    return out


class WordSeqLM(M.WordSequenceLM):
    """WordSequenceLM con tokenizador ("regex" | "tweet") y sentido configurables."""

    def __init__(self, alpha=0.1, min_count=3, order=3, tokenizer="regex", reverse=False):
        if tokenizer not in {"regex", "tweet"}:
            raise ValueError("tokenizer debe ser 'regex' o 'tweet'")
        super().__init__(alpha=alpha, min_count=min_count, order=order)
        self.tokenizer, self.reverse = tokenizer, bool(reverse)

    def _raw_tokens(self, text):  # la base lo llama como self._raw_tokens(...)
        toks = _word_units(text, self.tokenizer)
        return toks[::-1] if self.reverse else toks


class EntityWordSeqLM(WordSeqLM):
    """WordSeqLM que conserva menciones y hashtags frecuentes (submission 0.60270 del equipo).

    Las URLs se reducen a ``URL``. Una mención/hashtag entra al vocabulario si aparece al
    menos ``entity_min_count`` veces en train; las más raras se agrupan en ``@USER`` /
    ``#TAG``. Las palabras normales siguen con ``min_count``.
    """

    def __init__(self, alpha=0.1, min_count=3, entity_min_count=5, order=3, tokenizer="regex", reverse=False):
        super().__init__(alpha=alpha, min_count=min_count, order=order, tokenizer=tokenizer, reverse=reverse)
        self.entity_min_count = int(entity_min_count)

    @staticmethod
    def _is_entity(tok):
        return tok[:1] in ("@", "#") and len(tok) > 1

    def _raw_tokens(self, text):
        if self.tokenizer == "regex":
            toks = M.TOKEN_RE.findall(M._norm(text))
        else:
            toks = list(tweet_tokens(text))
        toks = ["URL" if is_url(t) else t for t in toks]
        return toks[::-1] if self.reverse else toks

    def _prepare_vocab(self, texts):
        counts: dict = {}
        for t in texts:
            for tok in self._raw_tokens(t):
                counts[tok] = counts.get(tok, 0) + 1
        self.vocab_ = {
            tok for tok, n in counts.items()
            if n >= (self.entity_min_count if self._is_entity(tok) else self.min_count)
        }
        self.vocab_.update((self.START, self.END, self.UNK, "@USER", "#TAG"))
        self.vocab_size_ = len(self.vocab_)

    def _encode(self, tok):
        if tok in self.vocab_:
            return tok
        if self._is_entity(tok):
            return "@USER" if tok[0] == "@" else "#TAG"
        return self.UNK

    def _pairs(self, text):
        k = self.context_size
        seq = [self.START] * k + [self._encode(t) for t in self._raw_tokens(text)] + [self.END]
        rng = range(k, len(seq))
        return [tuple(seq[p - k : p + 1]) for p in rng], [tuple(seq[p - k : p]) for p in rng]


# ---------------------------------------------------------------- mayúsculas y forma


class CaseSensitiveNgramLM(M.ClassConditionalNgramLM):
    """NB sobre n-gramas de caracteres SIN casefold (recupera el uso de mayúsculas)."""

    def __init__(self, orders=(5,), alpha=0.1):
        super().__init__(unit="char", orders=orders, alpha=alpha)

    def _events(self, text):
        text = "" if text is None else str(text)
        return [text[i : i + n] for n in self.orders for i in range(len(text) - n + 1)]


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")


def shape(text, max_run: int = 6) -> str:
    """Forma del texto: mayúscula->A, minúscula->a, dígito->9, espacio->' ', puntuación igual.

    URLs, menciones y hashtags se reducen a un símbolo (\\x01, @, #). Las rachas de un
    mismo símbolo se cortan en ``max_run`` para que las palabras largas no dominen.
    """
    t = "" if text is None else str(text)
    t = _URL_RE.sub("\x01", t)
    t = _MENTION_RE.sub("@", t)
    t = _HASHTAG_RE.sub("#", t)
    out, prev, run = [], None, 0
    for ch in t:
        if ch.isupper():
            s = "A"
        elif ch.islower():
            s = "a"
        elif ch.isdigit():
            s = "9"
        elif ch.isspace():
            s = " "
        else:
            s = ch
        run = run + 1 if s == prev else 1
        prev = s
        if run <= max_run:
            out.append(s)
    return "".join(out)


class ShapeNgramLM(M.ClassConditionalNgramLM):
    """NB sobre n-gramas de caracteres de ``shape(text)``."""

    def __init__(self, orders=(5,), alpha=0.1, max_run=6):
        super().__init__(unit="char", orders=orders, alpha=alpha)
        self.max_run = int(max_run)

    def _events(self, text):
        s = shape(text, self.max_run)
        return [s[i : i + n] for n in self.orders for i in range(len(s) - n + 1)]


class LengthBinLM(M._EventLM):
    """Longitud como evento categórico: bin de caracteres (de 10 en 10) y de tokens (de 3 en 3).

    Es un histograma por clase (Naive Bayes con dos eventos por tweet); no es una
    representación vectorial del texto.
    """

    def __init__(self, alpha=1.0, char_bin=10, token_bin=3):
        super().__init__(alpha)
        self.char_bin, self.token_bin = int(char_bin), int(token_bin)

    def _events(self, text):
        t = "" if text is None else str(text)
        n_tok = len(M.TOKEN_RE.findall(t))
        return [("c", min(len(t) // self.char_bin, 14)), ("t", min(n_tok // self.token_bin, 12))]


# ---------------------------------------------------------------- Markov configurables


class CharSeqLM(M.CharacterSequenceLM):
    """Markov de caracteres con contexto de ``order`` caracteres, opcionalmente en reversa."""

    def __init__(self, alpha=0.1, order=2, reverse=False):
        super().__init__(alpha=alpha)
        self.ORDER = int(order)  # la base usa self.ORDER
        self.reverse = bool(reverse)

    def _pairs(self, text):
        t = "" if text is None else str(text)
        return super()._pairs(t[::-1] if self.reverse else t)


class InterpolatedWordSeqLM:
    """Markov de palabras de orden 3 con interpolación Jelinek-Mercer recursiva por clase.

    P1(w)        = (c(w) + α) / (N + α·V)
    P2(w|u)      = λ2·c(u,w)/c(u·) + (1-λ2)·P1(w)          si c(u·) > 0, si no P1(w)
    P3(w|v,u)    = λ3·c(v,u,w)/c(v,u·) + (1-λ3)·P2(w|u)    si c(v,u·) > 0, si no P2(w|u)

    Puntaje = (Σ log P3 + log P(c)) / n, igual que WordSequenceLM.
    """

    START, END, UNK = "<s>", "</s>", "<unk>"

    def __init__(self, alpha=0.1, min_count=3, lambda3=0.6, lambda2=0.6, tokenizer="regex", reverse=False):
        if alpha <= 0 or not (0 < lambda3 < 1) or not (0 < lambda2 < 1):
            raise ValueError("parámetros inválidos")
        self.alpha, self.min_count = float(alpha), int(min_count)
        self.lambda3, self.lambda2 = float(lambda3), float(lambda2)
        self.tokenizer, self.reverse = tokenizer, bool(reverse)

    def _tokens(self, text):
        toks = _word_units(text, self.tokenizer)
        return toks[::-1] if self.reverse else toks

    def _keys(self, text):
        vocab, unk, S = self.vocab_, self.UNK, self.START
        seq = [S, S] + [t if t in vocab else unk for t in self._tokens(text)] + [self.END]
        rng = range(2, len(seq))
        tri = [(seq[p - 2], seq[p - 1], seq[p]) for p in rng]
        ctx2 = [(seq[p - 2], seq[p - 1]) for p in rng]
        bi = [(seq[p - 1], seq[p]) for p in rng]
        ctx1 = [seq[p - 1] for p in rng]
        uni = [seq[p] for p in rng]
        return tri, ctx2, bi, ctx1, uni

    def fit(self, texts, y):
        texts, y = list(texts), np.asarray(y)
        self.classes_, sizes = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(sizes / sizes.sum())
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        k = len(self.classes_)
        counts: dict = {}
        for t in texts:
            for tok in self._tokens(t):
                counts[tok] = counts.get(tok, 0) + 1
        self.vocab_ = {tok for tok, n in counts.items() if n >= self.min_count}
        self.vocab_.update((self.START, self.END, self.UNK))
        self.vocab_size_ = len(self.vocab_)
        self.indexes_ = [{} for _ in range(5)]
        flats = [array("q") for _ in range(5)]
        for text, label in zip(texts, y, strict=True):
            c = class_idx[label]
            for idx, flat, keys in zip(self.indexes_, flats, self._keys(text), strict=True):
                setdef = idx.setdefault
                flat.extend(setdef(key, len(idx)) * k + c for key in keys)
        self.tables_ = []
        for idx, flat in zip(self.indexes_, flats, strict=True):
            cnt = np.bincount(np.frombuffer(flat, dtype=np.int64), minlength=len(idx) * k).reshape(-1, k)
            self.tables_.append(np.vstack([cnt, np.zeros((1, k), dtype=cnt.dtype)]).astype(np.float64))
        self.total_ = self.tables_[4].sum(axis=0)
        return self

    def decision_function(self, texts):
        texts = list(texts)
        k = len(self.classes_)
        a, V, l3, l2 = self.alpha, self.vocab_size_, self.lambda3, self.lambda2
        gets = [idx.get for idx in self.indexes_]
        unks = [len(idx) for idx in self.indexes_]
        out = np.empty((len(texts), k), dtype=np.float64)
        for s in range(0, len(texts), M.CHUNK):
            block = texts[s : s + M.CHUNK]
            ids = [array("q") for _ in range(5)]
            lengths = np.empty(len(block), dtype=np.int64)
            for i, text in enumerate(block):
                keys = self._keys(text)
                lengths[i] = len(keys[0])
                for arr, get, unk, kk in zip(ids, gets, unks, keys, strict=True):
                    arr.extend(get(key, unk) for key in kk)
            c3, n3, c2, n2, c1 = (t[np.frombuffer(arr, dtype=np.int64)] for t, arr in zip(self.tables_, ids, strict=True))
            p1 = (c1 + a) / (self.total_ + a * V)
            p2 = np.where(n2 > 0, l2 * c2 / np.maximum(n2, 1) + (1 - l2) * p1, p1)
            p3 = np.where(n3 > 0, l3 * c3 / np.maximum(n3, 1) + (1 - l3) * p2, p2)
            ll = M._segment_sums(np.log(p3), np.arange(len(p3)), lengths)  # suma por tweet
            n = np.maximum(1, lengths)[:, None]
            out[s : s + len(block)] = ll / n + self.log_priors_ / n
        return out


# ---------------------------------------------------------------- especialista


class PairSpecialistLM:
    """NB entrenado solo con las filas de dos clases; el resto de clases queda en -inf.

    Con ``M.probabilities`` se obtiene P(18-24 | x) y P(25-34 | x) que suman 1, y 0 para
    las demás. Se usa para redistribuir la masa entre esas dos clases cuando la mezcla
    duda entre ellas.
    """

    def __init__(self, pair=("18-24", "25-34"), unit="char", orders=(5,), alpha=0.1):
        self.pair, self.unit, self.orders, self.alpha = tuple(pair), unit, tuple(orders), float(alpha)

    def fit(self, X, y):
        X, y = list(X), np.asarray(y)
        self.classes_ = np.unique(y)
        keep = np.flatnonzero(np.isin(y, self.pair))
        self.inner_ = M.ClassConditionalNgramLM(self.unit, self.orders, self.alpha).fit(
            [X[i] for i in keep], y[keep]
        )
        return self

    def decision_function(self, X):
        inner = self.inner_.decision_function(X)
        out = np.full((inner.shape[0], len(self.classes_)), -1e9)
        for j, c in enumerate(self.inner_.classes_):
            out[:, int(np.flatnonzero(self.classes_ == c)[0])] = inner[:, j]
        return out


__all__ = [
    "TTWordNgramLM",
    "TTPositionalWordLM",
    "TTPositionAwareMentionLM",
    "WordSeqLM",
    "CaseSensitiveNgramLM",
    "ShapeNgramLM",
    "LengthBinLM",
    "CharSeqLM",
    "InterpolatedWordSeqLM",
    "PairSpecialistLM",
    "shape",
]

