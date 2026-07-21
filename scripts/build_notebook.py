"""Build the explainability companion notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Explainable Agentic RAG Walkthrough

## Goal

This companion notebook explains *why* each knowledge-base paragraph is retrieved.
The production implementation remains in the `agentic_rag` package and CLI;
this notebook imports that code rather than duplicating it."""
        ),
        nbf.v4.new_markdown_cell(
            """## Setup

Run from the repository root after `pip install -e \".[notebook]\"`.
Retrieval cells require no API key. The optional final cell runs the two-agent
workflow only when `OPENAI_API_KEY` is already present in the environment."""
        ),
        nbf.v4.new_code_cell(
            """import os
import sys
from pathlib import Path

repo_root = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "knowledge_base.txt").is_file()
)
sys.path.insert(0, str(repo_root))

from agentic_rag.retrieval import BM25Retriever  # noqa: E402

knowledge_base_path = repo_root / "knowledge_base.txt"
retriever = BM25Retriever(
    knowledge_base_path,
    max_results=5,
    min_score=0.15,
    min_query_coverage=0.40,
)
chunks = retriever.load_chunks()
print(f"Loaded {len(chunks)} stable paragraph chunks from {knowledge_base_path.name}")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Steps

### 1. Retrieve evidence and inspect ranking factors

The ranking is deterministic. Each result exposes its matched query terms,
per-term BM25 contribution, query coverage, and consecutive-phrase bonus.
A 40% coverage gate and mutually exclusive domestic/international qualifier
check suppress chunks that match only generic policy vocabulary. The rarest
query term appearing in section titles becomes an auditable intent anchor."""
        ),
        nbf.v4.new_code_cell(
            """query = (
    "What approvals and preparations are required "
    "for international business travel?"
)
bundle = retriever.search(query)
print(f"Intent anchor terms: {bundle.intent_anchor_terms}\\n")

for chunk in bundle.chunks:
    print(f"{chunk.chunk_id} | {chunk.title} | score={chunk.score:.4f}")
    print(f"  matched_terms={chunk.matched_terms}")
    print(f"  term_contributions={chunk.contribution_dict()}")
    print(f"  coverage={chunk.coverage:.2%}; phrase_bonus={chunk.phrase_bonus:.2f}")
    print(f"  evidence={chunk.text}\\n")"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Verify the no-hallucination retrieval gate

An unrelated query should return no evidence. The workflow then stops before
the Report Generator and returns a deterministic insufficient-information response."""
        ),
        nbf.v4.new_code_cell(
            """unknown_bundle = retriever.search("What is the company dress code?")
print(unknown_bundle.to_agent_text())
assert unknown_bundle.chunks == ()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Checks

These checks make the core retrieval expectations executable rather than
relying on screenshots or prose."""
        ),
        nbf.v4.new_code_cell(
            """assert bundle.has_evidence
assert bundle.chunks[0].title == "International Business Travel"
assert all(chunk.score >= retriever.min_score for chunk in bundle.chunks)
assert len(bundle.source_ids) == len(set(bundle.source_ids))
print("Explainability checks passed.")"""
        ),
        nbf.v4.new_markdown_cell(
            """### Optional: run the complete two-agent workflow

This cell is credential-aware so the notebook remains reproducible in CI.
It never prompts for or displays a secret."""
        ),
        nbf.v4.new_code_cell(
            """if os.getenv("OPENAI_API_KEY"):
    from agentic_rag.agents import build_agents
    from agentic_rag.config import Settings
    from agentic_rag.workflow import AgenticRAGWorkflow

    settings = Settings.from_env()
    workflow = AgenticRAGWorkflow(retriever=retriever, agents=build_agents(settings))
    live_result = await workflow.run(query)
    print(live_result.report.answer_markdown)
else:
    print("Skipped live agent call: OPENAI_API_KEY is not set.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Next Steps

For the submission demo, run `python main.py --demo` with a locally exported
API key. The CLI displays the same evidence IDs and scores before the grounded
final answer, making the end-to-end decision path auditable."""
        ),
    ]
    output = Path("notebooks/explainable_rag_walkthrough.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
