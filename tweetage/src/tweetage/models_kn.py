"""LMs de n-gramas con mejor suavizado y NB binarizado (tanda v3).

Siguen siendo modelos de lenguaje probabilísticos condicionales a la clase, entrenados
solo con los datos del reto y con la misma API (``fit`` / ``decision_function`` /
``classes_``). No construyen ninguna representación vectorial de los documentos.

* ``KNSeqLM``: LM de Markov de caracteres o palabras con **Kneser-Ney interpolado
  modificado** (Chen y Goodman, 1998), uno por clase. En vez de sumar α a todos los
  conteos (add-α), descuenta una cantidad fija D a cada n-grama visto y reparte esa masa
  según el modelo de orden menor, que usa *conteos de continuación* (en cuántos contextos
  distintos aparece un n-grama, no cuántas veces). Es el suavizado estándar para LMs de
  n-gramas y trata mucho mejor los n-gramas no vistos.

      P(w|h) = [max(c(h,w) − D(c), 0) + γ(h)·P_KN(w|h')] / c(h)
      γ(h)   = D1·N1(h·) + D2·N2(h·) + D3·N3+(h·)

  Los descuentos D1, D2, D3+ se estiman por orden y por clase con las fórmulas de Chen y
  Goodman a partir de cuántos n-gramas aparecen 1, 2, 3 y 4 veces.

* ``BinaryNgramLM``: Naive Bayes multinomial sobre el **conjunto** de n-gramas del tweet
  (cada n-grama cuenta una vez por tweet) y con varios órdenes en el mismo modelo.
"""

from __future__ import annotations

import math
from array import array

import numpy as np

from tweetage import models as M
from tweetage import models_extra as MX

# ---------------------------------------------------------------- Kneser-Ney


def _discounts(T: np.ndarray) -> np.ndarray:
    """Descuentos (3, k) de Chen y Goodman a partir de los conteos-de-conteos de cada clase."""
    n = [np.maximum((T == i).sum(axis=0), 1).astype(float) for i in (1, 2, 3, 4)]
    Y = n[0] / (n[0] + 2 * n[1])
    D = np.vstack([1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])
    return np.clip(D, 0.05, np.array([[1.0], [2.0], [3.0]]))


class KNSeqLM:
    """LM de Markov con Kneser-Ney interpolado modificado, condicional a la clase.

    unit="char": caracteres (``case_sensitive`` controla el casefold).
    unit="word": tokens de ``EntityWordSeqLM`` (URL → URL, menciones/hashtags frecuentes
    se conservan, el resto → @USER/#TAG, palabras raras → <unk>).
    ``order`` = tamaño del n-grama (contexto de order − 1 unidades).
    Puntaje = (Σ log P(x_t | contexto) + log P(c)) / n, como los demás LMs secuenciales.
    """

    START_C, END_C = "\x02", "\x03"

    def __init__(self, unit="char", order=6, case_sensitive=True, min_count=3, entity_min_count=5, tokenizer="tweet"):
        if unit not in {"char", "word"} or order < 2:
            raise ValueError("configuración inválida")
        self.unit, self.order, self.case_sensitive = unit, int(order), bool(case_sensitive)
        self.min_count, self.entity_min_count, self.tokenizer = int(min_count), int(entity_min_count), tokenizer

    # ---- unidades
    def _seq(self, text):
        k = self.order - 1
        if self.unit == "char":
            t = "" if text is None else str(text)
            if not self.case_sensitive:
                t = t.casefold()
            return self.START_C * k + t + self.END_C
        enc = self.enc_
        return ("<s>",) * k + tuple(enc._encode(w) for w in enc._raw_tokens(text)) + ("</s>",)

    def _grams(self, seq, n):
        """n-gramas que terminan en cada posición predicha (una por unidad + fin)."""
        k = self.order - 1
        return [seq[p - n + 1 : p + 1] for p in range(k, len(seq))]

    # ---- entrenamiento
    def fit(self, texts, y):
        texts, y = list(texts), np.asarray(y)
        self.classes_, sizes = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(sizes / sizes.sum())
        cidx = {c: i for i, c in enumerate(self.classes_)}
        k = len(self.classes_)
        if self.unit == "word":
            self.enc_ = MX.EntityWordSeqLM(min_count=self.min_count, entity_min_count=self.entity_min_count,
                                           order=3, tokenizer=self.tokenizer)
            self.enc_._prepare_vocab(texts)
        N = self.order
        # conteos crudos del orden mayor
        index, flat = {}, array("q")
        for text, label in zip(texts, y, strict=True):
            c = cidx[label]
            setdef = index.setdefault
            flat.extend(setdef(g, len(index)) * k + c for g in self._grams(self._seq(text), N))
        keys = list(index)
        T = np.bincount(np.frombuffer(flat, dtype=np.int64), minlength=len(keys) * k).reshape(-1, k).astype(np.float32)
        del flat
        self.index_, self.T_, self.hist_, self.D_ = {N: index}, {N: T}, {}, {}
        # órdenes menores: conteos de continuación
        for n in range(N, 1, -1):
            sub_index = {}
            m = np.fromiter((sub_index.setdefault(g[1:], len(sub_index)) for g in keys), dtype=np.int64, count=len(keys))
            pos = (self.T_[n] > 0).astype(np.float32)
            Tn1 = np.stack([np.bincount(m, weights=pos[:, c], minlength=len(sub_index)) for c in range(k)], axis=1)
            del m, pos
            # las historias (g[:-1]) se indexan en el mismo dict del orden n − 1 (ahorra memoria);
            # las que no aparecen como sufijo se agregan con conteo de continuación 0
            self._build_hist(n, keys, sub_index)
            if len(sub_index) > len(Tn1):
                Tn1 = np.vstack([Tn1, np.zeros((len(sub_index) - len(Tn1), k))])
            self.index_[n - 1], self.T_[n - 1] = sub_index, Tn1.astype(np.float32)
            keys = list(sub_index)
        # orden 1 (sin historia): interpolado con la uniforme
        T1 = self.T_[1]
        D1 = _discounts(T1)
        self.D_[1] = D1
        tot = T1.sum(axis=0)
        gam = (D1[0] * (T1 == 1).sum(0) + D1[1] * (T1 == 2).sum(0) + D1[2] * (T1 >= 3).sum(0)) / tot
        V = len(self.index_[1]) + 1
        disc = self._disc(T1, D1)
        self.P1_ = np.vstack([(np.maximum(T1 - disc, 0) / tot) + gam / V, gam / V])  # última fila = no visto
        return self

    def _build_hist(self, n, keys, hidx):
        """Para el orden n: índice de historias (g[:-1]), total, γ-numerador y descuentos."""
        T = self.T_[n]
        k = T.shape[1]
        D = _discounts(T)
        h = np.fromiter((hidx.setdefault(g[:-1], len(hidx)) for g in keys), dtype=np.int64, count=len(keys))
        H = len(hidx)
        tot = np.stack([np.bincount(h, weights=T[:, c], minlength=H) for c in range(k)], axis=1)
        gnum = np.zeros((H, k))
        for j, mask in enumerate((T == 1, T == 2, T >= 3)):
            gnum += D[j] * np.stack([np.bincount(h, weights=mask[:, c].astype(float), minlength=H) for c in range(k)], 1)
        # fila extra = historia no vista (total 0)
        self.hist_[n] = (hidx, np.vstack([tot, np.zeros((1, k))]).astype(np.float32), np.vstack([gnum, np.zeros((1, k))]).astype(np.float32))
        self.D_[n] = D
        # tabla de conteos con fila extra = n-grama no visto
        self.T_[n] = np.vstack([T, np.zeros((1, k), dtype=np.float32)])

    @staticmethod
    def _disc(c, D):
        return np.where(c >= 3, D[2], np.where(c == 2, D[1], np.where(c == 1, D[0], 0.0)))

    # ---- predicción
    def decision_function(self, texts):
        texts = list(texts)
        N, k = self.order, len(self.classes_)
        out = np.empty((len(texts), k))
        for s in range(0, len(texts), M.CHUNK):
            block = texts[s : s + M.CHUNK]
            seqs = [self._seq(t) for t in block]
            lengths = np.array([len(sq) - (N - 1) for sq in seqs], dtype=np.int64)
            idx1 = self.index_[1]
            u1 = len(idx1)
            ids = np.fromiter((idx1.get(g, u1) for sq in seqs for g in self._grams(sq, 1)), dtype=np.int64)
            p = self.P1_[ids]
            for n in range(2, N + 1):
                gi, gu = self.index_[n], len(self.index_[n])
                hidx, tot, gnum = self.hist_[n]
                hu = len(hidx)
                grams = [g for sq in seqs for g in self._grams(sq, n)]
                gid = np.fromiter((gi.get(g, gu) for g in grams), dtype=np.int64, count=len(grams))
                hid = np.fromiter((hidx.get(g[:-1], hu) for g in grams), dtype=np.int64, count=len(grams))
                c, t, gn = self.T_[n][gid], tot[hid], gnum[hid]
                seen = t > 0
                num = np.maximum(c - self._disc(c, self.D_[n]), 0) + gn * p
                p = np.where(seen, num / np.where(seen, t, 1.0), p)
            ll = M._segment_sums(np.log(np.clip(p, 1e-300, None)), np.arange(len(p)), lengths)
            nn = np.maximum(1, lengths)[:, None]
            out[s : s + len(block)] = ll / nn + self.log_priors_ / nn
        return out


# ---------------------------------------------------------------- NB binarizado


class BinaryNgramLM(M.ClassConditionalNgramLM):
    """NB multinomial sobre el conjunto de n-gramas de caracteres del tweet (presencia, no frecuencia).

    ``orders`` puede tener varios órdenes: todos van al mismo modelo.
    """

    def __init__(self, orders=(4, 5, 6), alpha=0.1, case_sensitive=True):
        super().__init__(unit="char", orders=orders, alpha=alpha)
        self.case_sensitive = bool(case_sensitive)

    def _events(self, text):
        t = "" if text is None else str(text)
        if not self.case_sensitive:
            t = t.casefold()
        return list({t[i : i + n] for n in self.orders for i in range(len(t) - n + 1)})
