"""Matching logic for LLM topic answers against the 19-topic CSO vocabulary.

Implements brainstorm.md idea 1: strip quotes/punctuation before matching,
then fall back to a word-containment fuzzy match for near-misses. This is
deliberately not a raw string-similarity match, see the design notes below.
"""

import re

# The paper's own most common hallucination. It is the ontology's parent
# category, not one of the 19 topics. Resolving it via fuzzy match would
# invent an answer the model never actually gave, so it always stays a
# hallucination.
EXCLUDED_FROM_FUZZY_MATCH = {"computer science"}


def canonicalize(text):
    """Lowercase, strip whitespace, and strip quotes/punctuation from a
    single candidate answer string."""

    text = text.lower().strip()
    text = text.strip("'\"` .,;:!?")
    return text


def words_of(text):
    return set(re.findall(r"[a-z]+", text))


def fuzzy_match(candidate, vocabulary):
    """Return the single vocabulary term whose words fully contain the
    candidate's words, or None if there is no match or more than one.

    Example: "computer vision" -> {computer, vision}, which is a subset of
    "computer imaging and vision" -> {computer, imaging, and, vision}, so
    it matches. It is not a subset of "computer aided design"'s words, so
    that case is correctly avoided.
    """

    if candidate in EXCLUDED_FROM_FUZZY_MATCH:
        return None

    candidate_words = words_of(candidate)
    matches = [
        target for target in vocabulary
        if candidate_words and candidate_words.issubset(words_of(target))
    ]

    if len(matches) == 1:
        return matches[0]
    return None


def match_topic(raw_answer, vocabulary):
    """Match a single raw LLM answer string against the vocabulary.

    :param raw_answer: One candidate string extracted from the LLM response
        (already split on commas by the caller if needed).
    :param vocabulary: The list of 19 valid topic strings, canonical form
        (lowercase, no extra whitespace).
    :return: The matched vocabulary term, or None if it is a hallucination.
    """

    candidate = canonicalize(raw_answer)

    if candidate in vocabulary:
        return candidate

    return fuzzy_match(candidate, vocabulary)
