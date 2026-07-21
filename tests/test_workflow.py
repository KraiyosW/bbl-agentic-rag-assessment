from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.agents import AgentSet, RetrievalContext, build_agents
from agentic_rag.config import LLMProvider, Settings
from agentic_rag.models import GroundedReport
from agentic_rag.retrieval import BM25Retriever
from agentic_rag.workflow import (
    AgenticRAGWorkflow,
    GroundingValidationError,
    WorkflowExecutionError,
)


class FakeRunner:
    def __init__(
        self,
        reports: list[Any] | None = None,
        *,
        retriever_output: str | None = None,
        retriever_error: Exception | None = None,
        skip_retrieval_tool: bool = False,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.reports = reports or []
        self.retriever_output = retriever_output
        self.retriever_error = retriever_error
        self.skip_retrieval_tool = skip_retrieval_tool

    async def run(
        self,
        agent: Any,
        input_text: str,
        *,
        context: RetrievalContext | None = None,
        max_turns: int = 3,
    ) -> Any:
        del max_turns
        self.calls.append((agent.name, input_text))
        if agent.name == "Data Retriever":
            if self.retriever_error is not None:
                raise self.retriever_error
            assert context is not None
            if self.skip_retrieval_tool:
                return SimpleNamespace(final_output="no tool call")
            context.last_bundle = context.retriever.search(context.original_query)
            final_output = self.retriever_output or context.last_bundle.to_agent_text()
            return SimpleNamespace(final_output=final_output)
        report = self.reports.pop(0)
        if isinstance(report, Exception):
            raise report
        return SimpleNamespace(final_output=report)


def _agent_set() -> AgentSet:
    return build_agents(
        Settings(
            provider=LLMProvider.OPENAI,
            knowledge_base_path=Path("unused.txt"),
            api_key="test-only",
        )
    )


@pytest.mark.asyncio
async def test_workflow_passes_retrieved_evidence_to_reporter(
    sample_knowledge_base: Path,
) -> None:
    runner = FakeRunner(
        reports=[
            GroundedReport(
                answer_markdown="Approval is required at least 14 days ahead [KB-001].",
                source_ids=["KB-001"],
            )
        ]
    )
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )
    result = await workflow.run("international travel approval")

    assert [name for name, _ in runner.calls] == ["Data Retriever", "Report Generator"]
    assert result.retrieval.source_ids
    assert result.retrieval.query == "international travel approval"
    reporter_input = runner.calls[1][1]
    assert result.retrieval.to_agent_text() in reporter_input
    assert result.report.source_ids == ["KB-001"]


@pytest.mark.asyncio
async def test_no_evidence_fails_closed_without_reporter(sample_knowledge_base: Path) -> None:
    runner = FakeRunner()
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )
    result = await workflow.run("dress code attire")

    assert [name for name, _ in runner.calls] == ["Data Retriever"]
    assert result.report.insufficient_evidence is True
    assert result.report.source_ids == []


@pytest.mark.asyncio
async def test_grounding_failure_is_retried_then_rejected(sample_knowledge_base: Path) -> None:
    invalid = GroundedReport(
        answer_markdown="Unsupported statement [KB-999].",
        source_ids=["KB-999"],
    )
    runner = FakeRunner(reports=[invalid, invalid])
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )
    with pytest.raises(GroundingValidationError, match="not retrieved"):
        await workflow.run("international travel")
    assert [name for name, _ in runner.calls] == [
        "Data Retriever",
        "Report Generator",
        "Report Generator",
    ]
    assert "CORRECTION REQUIRED" in runner.calls[-1][1]


@pytest.mark.asyncio
async def test_insufficient_flag_cannot_bypass_citations(
    sample_knowledge_base: Path,
) -> None:
    invalid = GroundedReport(
        answer_markdown="I cannot answer from the evidence.",
        source_ids=[],
        insufficient_evidence=True,
    )
    runner = FakeRunner(reports=[invalid, invalid])
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(GroundingValidationError, match="declared and inline citations"):
        await workflow.run("international travel approval")


@pytest.mark.asyncio
async def test_retriever_output_tampering_is_rejected(
    sample_knowledge_base: Path,
) -> None:
    runner = FakeRunner(retriever_output="altered output")
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(WorkflowExecutionError, match="altered the raw retrieval tool output"):
        await workflow.run("international travel approval")
    assert [name for name, _ in runner.calls] == ["Data Retriever"]


@pytest.mark.asyncio
async def test_json_report_output_is_validated(sample_knowledge_base: Path) -> None:
    report_json = GroundedReport(
        answer_markdown="Approval is required [KB-001].",
        source_ids=["KB-001"],
    ).model_dump_json()
    runner = FakeRunner(reports=[report_json])
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    result = await workflow.run("international travel approval")
    assert result.report.source_ids == ["KB-001"]


@pytest.mark.asyncio
async def test_reporter_failure_is_sanitized(sample_knowledge_base: Path) -> None:
    runner = FakeRunner(reports=[RuntimeError("sensitive request body")])
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await workflow.run("international travel approval")
    assert "Report Generator failed (RuntimeError)" in str(caught.value)
    assert "sensitive request body" not in str(caught.value)


@pytest.mark.asyncio
async def test_retriever_failure_is_sanitized(sample_knowledge_base: Path) -> None:
    runner = FakeRunner(retriever_error=RuntimeError("sensitive request body"))
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await workflow.run("international travel approval")
    assert "Data Retriever failed (RuntimeError)" in str(caught.value)
    assert "sensitive request body" not in str(caught.value)


@pytest.mark.asyncio
async def test_retriever_must_call_its_tool(sample_knowledge_base: Path) -> None:
    runner = FakeRunner(skip_retrieval_tool=True)
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(WorkflowExecutionError, match="without calling its required tool"):
        await workflow.run("international travel approval")


@pytest.mark.asyncio
async def test_inline_and_declared_citations_must_match(
    sample_knowledge_base: Path,
) -> None:
    invalid = GroundedReport(
        answer_markdown="Approval is required [KB-001].",
        source_ids=["KB-002"],
    )
    runner = FakeRunner(reports=[invalid, invalid])
    workflow = AgenticRAGWorkflow(
        retriever=BM25Retriever(sample_knowledge_base),
        agents=_agent_set(),
        runner=runner,
    )

    with pytest.raises(GroundingValidationError, match="must contain the same IDs"):
        await workflow.run("international travel approval")
