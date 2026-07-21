"""Polished command-line interface for reviewers and screenshot capture."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agentic_rag.agents import build_agents
from agentic_rag.config import ConfigurationError, Settings
from agentic_rag.models import RetrievalBundle, WorkflowResult
from agentic_rag.retrieval import BM25Retriever, RetrievalError
from agentic_rag.workflow import AgenticRAGWorkflow, WorkflowError

DEMO_QUERIES = (
    "What is the policy on international business travel?",
    "What approval is required before booking domestic business travel?",
    "When must travel expenses be submitted, and what documents are required?",
    "Can employees work remotely, and how must customer information be handled?",
    "What is the company dress code?",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable two-agent RAG over a local policy knowledge base."
    )
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--query", dest="query_option", help="Question to answer")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run five purpose-built evaluation queries",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Inspect BM25 evidence without an API key or LLM call",
    )
    parser.add_argument("--max-results", type=int, help="Maximum evidence chunks")
    parser.add_argument("--min-score", type=float, help="Minimum retrieval score")
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Minimum fraction of meaningful query terms a chunk must match",
    )
    parser.add_argument("--export-svg", type=Path, help="Save the actual Rich console transcript")
    return parser


def _make_retriever(settings: Settings) -> BM25Retriever:
    return BM25Retriever(
        settings.knowledge_base_path,
        max_results=settings.max_results,
        min_score=settings.min_score,
        min_query_coverage=settings.min_query_coverage,
    )


def render_query_heading(console: Console, index: int, query: str) -> None:
    """Render a readable query heading with no decorative rule."""
    if index > 1:
        console.print()
    console.print(f"[bold blue]Query {index}[/bold blue] · [blue]{query}[/blue]")


def render_retrieval(console: Console, bundle: RetrievalBundle) -> None:
    if not bundle.chunks:
        console.print("[bold yellow]Data Retriever · No Evidence[/bold yellow]")
        console.print(Panel("No relevant evidence found.", border_style="yellow"))
        return

    anchors = ", ".join(bundle.intent_anchor_terms) or "none"
    table = Table(
        title=f"Retrieved Evidence · {bundle.algorithm} · intent anchors: {anchors}",
        show_lines=True,
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Section", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Matched terms")
    table.add_column("Evidence", overflow="fold")
    for chunk in bundle.chunks:
        excerpt = chunk.text if len(chunk.text) <= 180 else f"{chunk.text[:177]}..."
        table.add_row(
            chunk.chunk_id,
            chunk.title,
            f"{chunk.score:.3f}",
            ", ".join(chunk.matched_terms),
            excerpt,
        )
    console.print(table)


def render_result(console: Console, result: WorkflowResult) -> None:
    render_retrieval(console, result.retrieval)
    insufficient_evidence = result.report.insufficient_evidence
    border_style = "yellow" if insufficient_evidence else "green"
    stage_label = (
        "Workflow Guardrail · Safe Response"
        if insufficient_evidence
        else "Report Generator · Final Answer"
    )
    console.print(f"[bold {border_style}]{stage_label}[/bold {border_style}]")
    console.print(
        Panel(
            Markdown(result.report.answer_markdown),
            border_style=border_style,
        )
    )
    sources = ", ".join(result.report.source_ids) or "none"
    console.print(
        f"[dim]Sources: {sources} · retrieval={result.retrieval_seconds:.2f}s · "
        f"generation={result.generation_seconds:.2f}s · total={result.total_seconds:.2f}s[/dim]"
    )


async def _run(args: argparse.Namespace, console: Console) -> int:
    require_credentials = not args.retrieval_only
    settings = Settings.from_env(require_credentials=require_credentials)
    if args.max_results is not None:
        settings = replace(settings, max_results=args.max_results)
    if args.min_score is not None:
        settings = replace(settings, min_score=args.min_score)
    if args.min_coverage is not None:
        settings = replace(settings, min_query_coverage=args.min_coverage)

    queries = list(DEMO_QUERIES) if args.demo else [args.query_option or args.query]
    if not queries[0]:
        raise ConfigurationError("Provide a query or use --demo")

    retriever = _make_retriever(settings)
    workflow = None
    if not args.retrieval_only:
        workflow = AgenticRAGWorkflow(retriever=retriever, agents=build_agents(settings))

    for index, query in enumerate(queries, start=1):
        render_query_heading(console, index, query)
        if args.retrieval_only:
            render_retrieval(console, retriever.search(query))
        else:
            assert workflow is not None
            render_result(console, await workflow.run(query))
    if args.export_svg:
        # Keep the final content row clear of Rich's terminal-wide SVG clip.
        # Two spacer rows are required because the terminal group is vertically
        # translated after Rich calculates the clipping rectangle.
        console.print()
        console.print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query and args.query_option:
        parser.error("Use either the positional query or --query, not both")
    if args.demo and (args.query or args.query_option):
        parser.error("Use either --demo or one query, not both")

    console = Console(record=bool(args.export_svg), width=112)
    try:
        exit_code = asyncio.run(_run(args, console))
    except (ConfigurationError, RetrievalError, WorkflowError, ValueError) as exc:
        console.print(Panel(str(exc), title="Unable to run", border_style="red"))
        exit_code = 2

    if args.export_svg:
        args.export_svg.parent.mkdir(parents=True, exist_ok=True)
        console.save_svg(str(args.export_svg), title="BBL Agentic RAG Demo")
    return exit_code
