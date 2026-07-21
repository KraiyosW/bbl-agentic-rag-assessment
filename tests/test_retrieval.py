from pathlib import Path

import pytest

from agentic_rag.retrieval import (
    BM25Retriever,
    EmptyKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    parse_knowledge_base,
    tokenize,
)


def test_parse_assigns_stable_ids_and_titles() -> None:
    chunks = parse_knowledge_base("[One]\nFirst paragraph.\n\n[Two]\nSecond paragraph.")
    assert [(chunk.chunk_id, chunk.title) for chunk in chunks] == [
        ("KB-001", "One"),
        ("KB-002", "Two"),
    ]


def test_tokenize_normalizes_common_inflections() -> None:
    assert tokenize("Employees submitted expenses and approvals") == (
        "submitt",
        "expense",
        "approval",
    )


def test_search_ranks_relevant_chunks_and_exposes_contributions(
    sample_knowledge_base: Path,
) -> None:
    result = BM25Retriever(sample_knowledge_base).search(
        "What approvals are needed for international travel?"
    )
    assert result.chunks[0].title == "International Travel"
    assert "international" in result.chunks[0].matched_terms
    assert result.chunks[0].contribution_dict()["international"] > 0
    assert result.chunks[0].coverage > 0
    assert "international" in result.intent_anchor_terms


def test_search_returns_all_relevant_chunks_under_cap(sample_knowledge_base: Path) -> None:
    result = BM25Retriever(sample_knowledge_base, max_results=5).search(
        "international travel security insurance approval"
    )
    assert [chunk.chunk_id for chunk in result.chunks] == ["KB-002", "KB-001"]


def test_international_query_excludes_domestic_qualifier(
    sample_knowledge_base: Path,
) -> None:
    result = BM25Retriever(sample_knowledge_base).search("international travel approval")
    assert all("domestic" not in chunk.text.casefold() for chunk in result.chunks)


def test_rarest_title_term_anchors_topic(sample_knowledge_base: Path) -> None:
    result = BM25Retriever(sample_knowledge_base).search(
        "When must travel expenses be submitted and what documents are required?"
    )
    assert result.intent_anchor_terms == ("expense",)
    assert [chunk.title for chunk in result.chunks] == ["Travel Expenses"]


def test_search_is_deterministic(sample_knowledge_base: Path) -> None:
    retriever = BM25Retriever(sample_knowledge_base)
    first = retriever.search("expense receipts")
    second = retriever.search("expense receipts")
    assert first == second


def test_unknown_topic_returns_no_evidence(sample_knowledge_base: Path) -> None:
    result = BM25Retriever(sample_knowledge_base).search("dress code attire")
    assert result.chunks == ()
    assert result.to_agent_text() == "NO_RELEVANT_EVIDENCE"


def test_missing_knowledge_base_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseNotFoundError):
        BM25Retriever(tmp_path / "missing.txt").search("travel")


def test_empty_knowledge_base_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(EmptyKnowledgeBaseError):
        BM25Retriever(path).search("travel")


@pytest.mark.parametrize("max_results", [0, -1])
def test_invalid_max_results_is_rejected(sample_knowledge_base: Path, max_results: int) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(sample_knowledge_base, max_results=max_results)


def test_empty_query_is_rejected(sample_knowledge_base: Path) -> None:
    with pytest.raises(ValueError, match="meaningful term"):
        BM25Retriever(sample_knowledge_base).search("the and what")
