"""Typed contracts crossing retrieval, agent, and presentation boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

_SOURCE_ID_PATTERN = re.compile(r"^KB-\d{3}$")


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A stable paragraph-sized unit loaded from the local knowledge base."""

    chunk_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk plus the evidence needed to explain its ranking."""

    chunk_id: str
    title: str
    text: str
    score: float
    matched_terms: tuple[str, ...]
    term_contributions: tuple[tuple[str, float], ...]
    coverage: float
    phrase_bonus: float

    def contribution_dict(self) -> dict[str, float]:
        return dict(self.term_contributions)


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    """The complete, auditable output of one retrieval operation."""

    query: str
    chunks: tuple[ScoredChunk, ...]
    total_chunks: int
    intent_anchor_terms: tuple[str, ...] = ()
    algorithm: str = "BM25 + query coverage + phrase bonus"

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)

    def to_agent_text(self) -> str:
        """Render raw snippets with provenance, without synthesizing an answer."""
        if not self.chunks:
            return "NO_RELEVANT_EVIDENCE"

        rendered: list[str] = []
        for chunk in self.chunks:
            matched = ", ".join(chunk.matched_terms)
            rendered.append(
                f"[{chunk.chunk_id}] {chunk.title}\n"
                f"retrieval_score={chunk.score:.4f}; matched_terms={matched}\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(rendered)


class GroundedReport(BaseModel):
    """Structured answer produced by the Report Generator Agent."""

    answer_markdown: str = Field(
        min_length=1,
        description="A concise answer with inline citations such as [KB-003].",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Only knowledge-base IDs actually used in the answer.",
    )
    insufficient_evidence: bool = False

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("source_ids must not contain duplicates")
        invalid = [value for value in values if not _SOURCE_ID_PATTERN.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid source ID format: {invalid}")
        return values


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Final response plus timings and evidence for audit and display."""

    query: str
    retrieval: RetrievalBundle
    report: GroundedReport
    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds
