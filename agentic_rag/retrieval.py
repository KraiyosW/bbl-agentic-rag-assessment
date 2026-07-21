"""Deterministic and explainable retrieval over a paragraph-based text file."""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from agentic_rag.models import KnowledgeChunk, RetrievalBundle, ScoredChunk

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?")
_HEADING_PATTERN = re.compile(r"^\[(?P<title>[^\]]+)]$")
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "company",
    "do",
    "does",
    "employee",
    "employees",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "must",
    "of",
    "on",
    "our",
    "should",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_MUTUALLY_EXCLUSIVE_QUALIFIERS = (frozenset({"domestic", "international"}),)
_NON_SEARCHABLE_SECTIONS = frozenset({"Document Notice"})


class RetrievalError(RuntimeError):
    """Base error for safe retrieval failures."""


class KnowledgeBaseNotFoundError(RetrievalError):
    """Raised when the configured knowledge-base file cannot be found."""


class EmptyKnowledgeBaseError(RetrievalError):
    """Raised when the knowledge-base file has no usable paragraphs."""


def _normalize_token(token: str) -> str:
    token = token.casefold()
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ied"):
        token = f"{token[:-3]}y"
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize(text: str) -> tuple[str, ...]:
    """Return normalized, meaningful English tokens in their original order."""
    normalized = (_normalize_token(match.group()) for match in _TOKEN_PATTERN.finditer(text))
    return tuple(token for token in normalized if token and token not in _STOP_WORDS)


def parse_knowledge_base(text: str) -> tuple[KnowledgeChunk, ...]:
    """Split bracket-headed plain text into stable paragraph chunks."""
    chunks: list[KnowledgeChunk] = []
    current_title = "General"
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if paragraph:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"KB-{len(chunks) + 1:03d}",
                    title=current_title,
                    text=paragraph,
                )
            )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _HEADING_PATTERN.fullmatch(line)
        if heading:
            flush_paragraph()
            current_title = heading.group("title").strip()
        elif not line:
            flush_paragraph()
        else:
            paragraph_lines.append(line)
    flush_paragraph()
    return tuple(chunks)


class BM25Retriever:
    """Read the source file per search and rank every paragraph transparently."""

    def __init__(
        self,
        knowledge_base_path: str | Path,
        *,
        max_results: int = 5,
        min_score: float = 0.15,
        min_query_coverage: float = 0.40,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if min_score < 0:
            raise ValueError("min_score cannot be negative")
        if not 0 <= min_query_coverage <= 1:
            raise ValueError("min_query_coverage must be between 0 and 1")
        self.knowledge_base_path = Path(knowledge_base_path)
        self.max_results = max_results
        self.min_score = min_score
        self.min_query_coverage = min_query_coverage
        self.k1 = k1
        self.b = b

    def load_chunks(self) -> tuple[KnowledgeChunk, ...]:
        """Load fresh content so the tool never relies on hidden cached state."""
        if not self.knowledge_base_path.is_file():
            raise KnowledgeBaseNotFoundError(
                f"Knowledge base not found: {self.knowledge_base_path}"
            )
        text = self.knowledge_base_path.read_text(encoding="utf-8")
        chunks = parse_knowledge_base(text)
        if not chunks:
            raise EmptyKnowledgeBaseError("Knowledge base contains no usable paragraphs")
        return chunks

    def search(self, query: str) -> RetrievalBundle:
        query_terms = tuple(dict.fromkeys(tokenize(query)))
        if not query_terms:
            raise ValueError("query must contain at least one meaningful term")

        loaded_chunks = self.load_chunks()
        chunks = tuple(
            chunk for chunk in loaded_chunks if chunk.title not in _NON_SEARCHABLE_SECTIONS
        )
        tokenized_documents = [tokenize(f"{chunk.title} {chunk.text}") for chunk in chunks]
        tokenized_titles = [set(tokenize(chunk.title)) for chunk in chunks]
        document_count = len(chunks)
        average_length = sum(map(len, tokenized_documents)) / document_count
        document_frequency = Counter(
            term for document in tokenized_documents for term in set(document)
        )
        title_frequency = Counter(term for title in tokenized_titles for term in title)
        title_candidates = [term for term in query_terms if title_frequency[term]]
        rarest_title_frequency = min(
            (title_frequency[term] for term in title_candidates),
            default=0,
        )
        intent_anchor_terms = tuple(
            term for term in title_candidates if title_frequency[term] == rarest_title_frequency
        )
        query_bigrams = set(pairwise(query_terms))

        scored: list[ScoredChunk] = []
        for chunk, document_terms in zip(chunks, tokenized_documents, strict=True):
            document_term_set = set(document_terms)
            contradicts_query = any(
                (query_qualifiers := set(query_terms) & qualifier_group)
                and (document_qualifiers := document_term_set & qualifier_group)
                and query_qualifiers.isdisjoint(document_qualifiers)
                for qualifier_group in _MUTUALLY_EXCLUSIVE_QUALIFIERS
            )
            if contradicts_query:
                continue
            if intent_anchor_terms and document_term_set.isdisjoint(intent_anchor_terms):
                continue
            frequencies = Counter(document_terms)
            contributions: list[tuple[str, float]] = []
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(document_terms) / average_length
                )
                contribution = inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )
                contributions.append((term, contribution))

            matched_terms = tuple(term for term, _ in contributions)
            coverage = len(matched_terms) / len(query_terms)
            document_bigrams = set(pairwise(document_terms))
            phrase_bonus = 0.35 * len(query_bigrams & document_bigrams)
            score = sum(value for _, value in contributions) * (0.7 + 0.3 * coverage)
            score += phrase_bonus

            if matched_terms and score >= self.min_score and coverage >= self.min_query_coverage:
                scored.append(
                    ScoredChunk(
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        text=chunk.text,
                        score=score,
                        matched_terms=matched_terms,
                        term_contributions=tuple(contributions),
                        coverage=coverage,
                        phrase_bonus=phrase_bonus,
                    )
                )

        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return RetrievalBundle(
            query=query,
            chunks=tuple(scored[: self.max_results]),
            total_chunks=document_count,
            intent_anchor_terms=intent_anchor_terms,
        )
