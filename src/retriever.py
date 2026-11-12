"""Policy retrieval (TF-IDF over markdown sections).

The index is built in memory from ``data/hr_documents/*.md`` the first time it is
needed, so there are no generated artifacts to commit or keep in sync, and no
pickle files to load. Every hit carries its source file so answers can cite it.
"""

import re
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.chunker import chunk_text
from src.loader import load_document

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Tiny stemmer for plurals and a few suffixes: leaves -> leave, policies -> policy."""
    if word.endswith("ibility") or word.endswith("ability"):
        return word[:-7] + ("ible" if word.endswith("ibility") else "able")  # eligibility -> eligible
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _analyze(text: str) -> list[str]:
    tokens = [
        _stem(w) for w in _WORD.findall(text.lower())
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS   # len > 1 drops the stray "s" in "what's"
    ]
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _title_of(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _section_heading(chunk: str) -> str | None:
    for line in chunk.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return None


class PolicyIndex:
    def __init__(self, documents_dir: Path):
        self.chunks: list[dict] = []
        for path in sorted(Path(documents_dir).glob("*.md")):
            text = load_document(path)
            title = _title_of(text, path.stem)
            for chunk in chunk_text(text):
                self.chunks.append({
                    "source": path.name,
                    "title": title,
                    "section": _section_heading(chunk),
                    "text": chunk,
                })

        self._vectorizer = None
        self._matrix = None
        if self.chunks:
            self._vectorizer = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
            # The document title is part of the indexed text, so "leave policy" finds leave_policy.md
            self._matrix = self._vectorizer.fit_transform(
                [f"{c['title']}\n{c['text']}" for c in self.chunks]
            )

    def search(self, query: str, top_k: int = 3, threshold: float = 0.08) -> list[dict]:
        if not self.chunks or not query or not query.strip():
            return []
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            {**chunk, "score": round(float(score), 3)}
            for chunk, score in ranked[:top_k]
            if score >= threshold
        ]


@lru_cache(maxsize=4)
def _index_for(directory: str) -> PolicyIndex:
    return PolicyIndex(Path(directory))


def get_index() -> PolicyIndex:
    return _index_for(str(config.documents_dir()))


def reload_index() -> None:
    """Call after editing the markdown files while the app is running."""
    _index_for.cache_clear()


def retrieve(query: str, top_k: int = 3, threshold: float = 0.08) -> list[dict]:
    return get_index().search(query, top_k=top_k, threshold=threshold)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How do I claim health insurance?"
    print(f"Query: {question}\n")
    hits = retrieve(question)
    if not hits:
        print("No relevant policy text found.")
    for hit in hits:
        print(f"[{hit['score']}] {hit['source']} :: {hit['section'] or hit['title']}")
        print(hit["text"], "\n" + "-" * 50)
