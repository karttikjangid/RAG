from typing import List

from .config import CRAGConfig
from .text_utils import keyword_tokens, normalize_whitespace


def is_weak_query(query: str, config: CRAGConfig) -> bool:
    keywords = keyword_tokens(query)
    if len(keywords) < config.min_query_tokens:
        return True
    return len(query.strip()) < 10


def rewrite_query(query: str, config: CRAGConfig) -> List[str]:
    cleaned = normalize_whitespace(query)
    rewrites = [cleaned]

    keywords = keyword_tokens(cleaned)
    if not keywords:
        return rewrites

    keyword_phrase = " ".join(keywords)
    if keyword_phrase and keyword_phrase != cleaned:
        rewrites.append(keyword_phrase)

    rewrites.append(f"information about {keyword_phrase}")

    lowered = cleaned.lower()
    if lowered.startswith("what is") or lowered.startswith("define"):
        rewrites.append(f"definition of {keyword_phrase}")

    if cleaned.endswith("?"):
        rewrites.append(cleaned.rstrip("?"))

    deduped = []
    seen = set()
    for candidate in rewrites:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped
