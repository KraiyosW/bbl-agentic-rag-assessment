from pathlib import Path

import pytest
from agents import OpenAIChatCompletionsModel
from agents.tool_context import ToolContext
from agents.usage import Usage

from agentic_rag.agents import RetrievalContext, build_agents, search_knowledge_base
from agentic_rag.config import LLMProvider, Settings
from agentic_rag.retrieval import BM25Retriever


def test_agent_responsibilities_are_unambiguous() -> None:
    settings = Settings(
        provider=LLMProvider.OPENAI,
        knowledge_base_path=Path("knowledge_base.txt"),
        api_key="test-only",
    )
    agents = build_agents(settings)

    assert agents.retriever.name == "Data Retriever"
    assert [tool.name for tool in agents.retriever.tools] == ["search_knowledge_base"]
    assert agents.retriever.model_settings.tool_choice == "search_knowledge_base"
    assert agents.retriever.tool_use_behavior == "stop_on_first_tool"

    assert agents.reporter.name == "Report Generator"
    assert agents.reporter.tools == []
    assert agents.reporter.output_type is not None


def test_azure_configuration_builds_chat_completions_models() -> None:
    settings = Settings(
        provider=LLMProvider.AZURE,
        knowledge_base_path=Path("knowledge_base.txt"),
        api_key="test-only",
        azure_endpoint="https://example.openai.azure.com/",
        azure_deployment="gpt-5-mini",
        azure_api_version="2024-10-21",
    )
    agents = build_agents(settings)

    assert isinstance(agents.retriever.model, OpenAIChatCompletionsModel)
    assert agents.retriever.model is agents.reporter.model
    assert agents.retriever.model.model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_sdk_function_tool_preserves_raw_retrieval_output(
    sample_knowledge_base: Path,
) -> None:
    original_query = "international travel approvals"
    retrieval_context = RetrievalContext(
        BM25Retriever(sample_knowledge_base),
        original_query=original_query,
    )
    tool_context = ToolContext(
        context=retrieval_context,
        usage=Usage(),
        tool_name="search_knowledge_base",
        tool_call_id="offline-test",
        tool_arguments="{}",
    )
    output = await search_knowledge_base.on_invoke_tool(
        tool_context,
        "{}",
    )

    assert retrieval_context.last_bundle is not None
    assert retrieval_context.last_bundle.query == original_query
    assert output == retrieval_context.last_bundle.to_agent_text()
