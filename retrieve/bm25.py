"""Real BM25 ranking — stdlib only, no embeddings.

Classic Okapi BM25 (IDF + length normalization) over a corpus of documents.
This is the launch-time ranker for Remembrancer retrieval: an upgrade from
naive keyword overlap, with no vector DB and no LLM call on the read path.

    from retrieve.bm25 import BM25, rank
    bm = BM25(docs)          # docs: list[str]
    scores = bm.score(query) # list[float], aligned to docs
    ranked = rank(query, docs)  # list[(idx, score)] sorted desc

Parameters: k1=1.5 (term-frequency saturation), b=0.75 (length normalization).
These are the standard Okapi defaults.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Okapi BM25 defaults.
K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"\W+")

# CJK scripts are written without spaces, and Python counts their characters as
# word characters, so ``\W+`` alone left "Acme是什么" as ONE token that
# matched neither "acme" nor "是什么": a question of that shape returned 0
# candidates while the spaced form returned plenty. Each ``\W+`` token is
# therefore cut again at every CJK /
# non-CJK boundary. A pure CJK run stays one token (no segmentation, no
# bigrams) and a token with no CJK in it is untouched. Only word characters
# reach this split, so the punctuation in U+3000-U+303F is already a separator;
# that block is listed for 々 〆 〇 and the kana iteration marks.
_CJK = (
    "\u1100-\u11ff"          # Hangul Jamo
    "\u3000-\u303f"          # CJK Symbols and Punctuation
    "\u3040-\u309f"          # Hiragana
    "\u30a0-\u30ff"          # Katakana
    "\u3100-\u312f"          # Bopomofo
    "\u3130-\u318f"          # Hangul Compatibility Jamo
    "\u31a0-\u31bf"          # Bopomofo Extended
    "\u31f0-\u31ff"          # Katakana Phonetic Extensions
    "\u3400-\u4dbf"          # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"          # CJK Unified Ideographs
    "\ua960-\ua97f"          # Hangul Jamo Extended-A
    "\uac00-\ud7ff"          # Hangul Syllables + Hangul Jamo Extended-B
    "\uf900-\ufaff"          # CJK Compatibility Ideographs
    "\uff66-\uffdc"          # Halfwidth Katakana and Hangul
    "\U00020000-\U0003ffff"  # planes 2-3: Extensions B-H + Compatibility Supplement
)
_CJK_SPLIT_RE = re.compile(f"[{_CJK}]+|[^{_CJK}]+")
_WORD_RE = re.compile(r"\w+")

# A number glued to a CJK date unit ("2026年9月", "8月5日") splits into the
# number and a one-character unit. As standalone tokens those units are rare,
# so each earned a high IDF and pulled any
# date-bearing document toward any date-bearing query: a remembrancer_dig target for
# "2026年9月 Acme 报告" fell from rank 1 to 10. The unit is dropped
# when it directly follows a number; the number is kept.
_CJK_DATE_UNITS = frozenset("年月日号")

# Bump whenever tokenize() can return different tokens for the same text, so a
# cache of token-derived features (the artifact graph's) is rebuilt
# instead of mixing two tokenizers. 2 = the CJK boundary split above.
TOKENIZER_VERSION = 2

# Suffix-strip stemmer. Cheap (regex + tuple of common English suffixes),
# no external dep, no Porter stepper. Slightly more recall than raw
# tokens at the cost of some false-positive matches; tunable by
# editing _SUFFIXES. Length floor (4) keeps "it" / "is" / "as" from
# collapsing to noise.
_SUFFIXES: tuple[str, ...] = ("ational", "tional", "ization", "iveness",
                            "fulness", "ousness", "aliti", "biliti",
                            "iviti", "ment", "ness", "ation", "ator",
                            "izer", "ical", "ful", "ous", "ive", "ing",
                            "ed", "es", "ly", "er", "est", "s")
_STEM_FLOOR = 4  # don't shorten below this many characters


def _strip_suffix(token: str) -> str:
    """Strip the longest matching English suffix, leaving the stem.

    A 4-character floor avoids over-stemming short words (``it`` -> ``i``).
    Floor is checked as ``remaining >= floor`` so ``pricing`` -> ``pric``
    (4 chars) but ``is`` -> ``i`` (1 char) is rejected.
    """
    for suf in _SUFFIXES:
        if len(token) - len(suf) >= _STEM_FLOOR and token.endswith(suf):
            return token[: -len(suf)]
    return token


def word_spans(text: str) -> list[tuple[int, str]]:
    """``(offset, word)`` for every unstemmed word ``tokenize`` keeps, in order.

    Words are the ``\\w+`` runs of the lowercased text, cut again at CJK
    boundaries, minus a date unit that follows a number. Offsets index the
    lowercased text, which matches the original except where lowercasing
    changes length (e.g. "İ"); callers that only place a window can ignore
    that. An all-ASCII run cannot hold CJK, so it skips the second regex;
    that keeps the cost of the CJK split off the English bulk of the corpus.
    """
    out: list[tuple[int, str]] = []
    for m in _WORD_RE.finditer(text.lower()):
        token = m.group()
        if token.isascii():
            out.append((m.start(), token))
            continue
        prev = ""
        for piece in _CJK_SPLIT_RE.finditer(token):
            word = piece.group()
            if not (word in _CJK_DATE_UNITS and prev.isdigit()):
                out.append((m.start() + piece.start(), word))
            prev = word
    return out


def _words(text: str) -> list[str]:
    """Lowercased, unstemmed tokens (see ``word_spans``)."""
    return [word for _, word in word_spans(text)]


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word runs and CJK boundaries, stem each token.

    Stemming broadens recall: ``pricing`` matches ``priced`` matches
    ``price``. The cost is some false-positive matches (``university`` ->
    ``univers``); acceptable for an index-first ranker where the next
    stage re-ranks using a richer context. The suffixes are ASCII, so a
    CJK token never changes under the stemmer.
    """
    return [_strip_suffix(t) for t in _words(text)]


# --- query-side stopwords (gated by retrieval.ranking.query_stopwords) -------
#
# NLTK's published English stopword list, verbatim (contraction variants folded
# in).  It is deliberately a *standard* list rather than one tuned against an
# evaluation set: the failure it addresses is structural, not lexical.  Remembrancer's BM25
# documents are the terse `name + description + tags + index_keys` strings, so a
# function word such as "our" occurs in only a few percent of them and earns an
# IDF on par with a real content term; a short note that repeats it then outranks
# the note that states the fact.  Pruning happens on the QUERY only — documents
# are never re-tokenized and the stored index is untouched.
_STOPWORDS_RAW = """
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were be been being have
has had having do does did doing a an the and but if or because as until while
of at by for with about against between into through during before after above
below to from up down in out on off over under again further then once here
there when where why how all any both each few more most other some such no nor
not only own same so than too very s t can will just don should now d ll m o re
ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
shouldn wasn weren won wouldn
"""
STOPWORDS: frozenset[str] = frozenset(
    _strip_suffix(word) for word in _STOPWORDS_RAW.split()
)


def strip_stopwords(query: str) -> str:
    """``query`` with English function words removed, matched in stem space.

    Kept terms are returned in their ORIGINAL form, never their stems, so the
    caller's BM25 applies :func:`_strip_suffix` exactly once (stemming is not
    idempotent: ``processes`` -> ``process`` -> ``proc``).

    A query made entirely of stopwords is returned unchanged — pruning must
    never empty the query and zero out an otherwise-good result set.
    """
    raw = _words(query)
    kept = [token for token in raw if _strip_suffix(token) not in STOPWORDS]
    return " ".join(kept) if kept else query


class BM25:
    """Okapi BM25 over a fixed corpus.

    IDF uses the standard BM25 probabilistic form with a +1 shift to keep weights
    non-negative even for terms appearing in more than half the corpus.
    """

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens: list[list[str]] = [tokenize(d) for d in docs]
        self.doc_freqs: list[Counter] = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens: list[int] = [len(toks) for toks in self.doc_tokens]
        self.N = len(docs)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0

        # Document frequency per term, then IDF.
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] += 1
        self.idf: dict[str, float] = {}
        for term, n_q in df.items():
            # BM25 IDF with +1 shift: log(1 + (N - n + 0.5)/(n + 0.5)) >= 0.
            self.idf[term] = math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5))

    def score(self, query: str) -> list[float]:
        """Score every doc against the query. Returns a list aligned to `docs`."""
        q_terms = tokenize(query)
        scores = [0.0] * self.N
        if not q_terms or self.N == 0:
            return scores
        for i in range(self.N):
            freqs = self.doc_freqs[i]
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl else self.k1
            s = 0.0
            for term in q_terms:
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
            scores[i] = s
        return scores


def rank(query: str, docs: list[str], k1: float = K1, b: float = B) -> list[tuple[int, float]]:
    """Rank docs against the query. Returns (idx, score) pairs sorted desc by score.

    Ties broken by original index for deterministic output.
    """
    bm = BM25(docs, k1=k1, b=b)
    scored = list(enumerate(bm.score(query)))
    scored.sort(key=lambda p: (-p[1], p[0]))
    return scored
