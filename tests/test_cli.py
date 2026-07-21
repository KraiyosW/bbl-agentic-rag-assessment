from io import StringIO
from pathlib import Path

from rich.console import Console

from agentic_rag.cli import DEMO_QUERIES, main, render_result
from agentic_rag.models import GroundedReport, RetrievalBundle, WorkflowResult
from agentic_rag.retrieval import BM25Retriever


def test_demo_queries_cover_five_explainable_scenarios() -> None:
    assert len(DEMO_QUERIES) == 5

    retriever = BM25Retriever(Path(__file__).parents[1] / "knowledge_base.txt")
    source_ids = [retriever.search(query).source_ids for query in DEMO_QUERIES]

    assert source_ids == [
        ("KB-003", "KB-004"),  # assignment example: international policy
        ("KB-002",),  # near-neighbor disambiguation: domestic, not international
        ("KB-006", "KB-007"),  # multi-paragraph expense evidence
        ("KB-008",),  # separate policy domain and customer-information constraint
        (),  # unsupported question must fail closed
    ]


def test_fail_closed_result_is_labeled_as_workflow_guardrail() -> None:
    console = Console(record=True, width=112, file=StringIO())
    result = WorkflowResult(
        query="What is the company dress code?",
        retrieval=RetrievalBundle(
            query="What is the company dress code?", chunks=(), total_chunks=9
        ),
        report=GroundedReport(
            answer_markdown="No relevant information was found.",
            source_ids=[],
            insufficient_evidence=True,
        ),
        retrieval_seconds=0.1,
        generation_seconds=0.0,
    )

    render_result(console, result)
    rendered = console.export_text()

    assert "Workflow Guardrail · Safe Response" in rendered
    assert "Report Generator · Final Answer" not in rendered


def test_retrieval_only_cli_needs_no_api_key(
    sample_knowledge_base: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(sample_knowledge_base))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    exit_code = main(["international travel approval", "--retrieval-only"])
    assert exit_code == 0


def test_cli_reports_missing_query_without_traceback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    exit_code = main(["--retrieval-only"])
    assert exit_code == 2
