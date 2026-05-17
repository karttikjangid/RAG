import re
from typing import Iterable, List, Set

WORD_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "about",
    "from",
    "by",
    "as",
    "at",
    "be",
    "this",
    "that",
    "it",
    "or",
    "if",
    "into",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "will",
    "may",
    "might",
    "your",
    "you",
    "i",
    "we",
    "they",
    "them",
    "their",
    "our",
    "us",
}


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def keyword_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def token_set(text: str) -> Set[str]:
    return set(keyword_tokens(text))


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def split_sentences(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    parts = re.split(r"[.!?]+\s*", normalized)
    return [part.strip() for part in parts if part.strip()]
