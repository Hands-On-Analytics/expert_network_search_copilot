from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from app.core.config import settings
from app.models.schemas import ExtractedFilters, ResearchPlanner,ReflectorOutput
from app.services.document_grader import grade_document
from app.services.filter_extractor import extract_filters
from app.services.qdrant_service import QdrantService


logger = logging.getLogger(__name__)

# ── state shared across nodes ──────────────────────────────────────────
class ResearchState(TypedDict):
    query: str
    queries_used: str
    search_query: str
    documents: list[Document]
    context: str
    answer: str
    reasoning_history: str

llm = ChatOpenAI()


planner_prompt = """You are a Research Strategy Planner. Your goal is to find experts using a funnel approach.

Current State:
- Original Goal: {query}
- Previous Queries: {queries_used}

Strategy Instructions:
1. NARROW: Use strict, specific terms. Use quotes for must-have skills. Focus on exact technical niches (e.g., "PostgreSQL indexing expert" instead of "Database expert").
2. MID: Relax one or two specific constraints. Use synonyms and related technologies (e.g., "Relational Database performance" or "SQL optimization").
3. BROAD: Focus on the general domain and high-level problem solving (e.g., "Backend Architecture" or "Data Engineering").

Task:
Generate a JSON response with:
- "search_query": The optimized query string for the VectorDB.
- "reasoning": Why this query is appropriate for the {specificity} stage.
"""

def planner_node(state: ResearchState):
    
    # Call LLM with the prompt above
    response = llm.with_structured_output(ResearchPlanner).invoke(planner_prompt.format(**state))
    
    return {
        "queries_used": state["queries_used"] + [response.search_query],
        "reasoning_history": state["reasoning_history"] + [response.reasoning]
    }



# search_node


def search_node(qdrant_service: QdrantService) -> StateGraph:
    """Return a compiled LangGraph for Corrective-RAG over the candidate collection."""

    vectorstore = qdrant_service.vectorstore()

    llm = ChatOpenAI(
        model=settings.OPENROUTER_CHAT_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )

    # ── node: retrieve ─────────────────────────────────────────────
    def retrieve(state: ResearchState) -> dict:
        # qdrant_filter = _build_qdrant_filter(state.get("filters"))
        search_query = state.get("search_query") or state["query"]

        docs = vectorstore.similarity_search(
            query=search_query,
            k=settings.RAG_TOP_K,
            # filter=qdrant_filter,
        )

        if not docs:
            # Build a helpful "no records" message that mentions the filters
            filters = state.get("filters")
            parts = []
            if filters:
                if filters.country_name:
                    parts.append(f"country = {filters.country_name}")
                if filters.city_name:
                    parts.append(f"city = {filters.city_name}")
                if filters.gender:
                    parts.append(f"gender = {filters.gender}")
                if filters.min_years_of_experience is not None:
                    parts.append(f"min experience = {filters.min_years_of_experience} years")
                if filters.max_years_of_experience is not None:
                    parts.append(f"max experience = {filters.max_years_of_experience} years")
            filter_desc = ", ".join(parts) if parts else "your query"
            return {
                "documents": [],
                "context": "",
                "answer": f"No records found matching the given filters ({filter_desc}).",
            }

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return {"documents": docs, "context": context}


# reflector_node----------------------------------------------------------------

reflector_prompt ="""You are a Senior Research Auditor. Review the results found so far.

User Query: {query}
Last Reasoning: {reasoning_history}


Assessment:
- Did the last search provide new, high-quality experts?
- Is the current list comprehensive enough to answer the user's request?
- Have we hit the maximum iteration limit?

Decision:
If you have enough quality results or are seeing diminishing returns, set "is_finished" to true.
If we need more variety or broader matches, set "is_finished" to false and suggest the next stage.

Return JSON: {"is_finished": boolean, "critique": string}"""

def reflector_node(state: ResearchState):
        # Call LLM with the prompt above
    response = llm.with_structured_output(ReflectorOutput).invoke(reflector_prompt.format(**state))
    
    return {
        "is_finished": state["queries_used"] + [response.search_query],
        "critique": state["reasoning_history"] + [response.reasoning]
    }



def should_continue(state: ResearchState): # A function that checks is_finished
    if not state.get("is_finished"):
        return "end"
    return "planner_node"


# ____________graph
workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("search", search_node)
workflow.add_node("reflector", reflector_node)

# Define Edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "search")
workflow.add_edge("search", "reflector")

# Conditional Logic: Loop back to planner or go to END
workflow.add_conditional_edges(
    "reflector",
    should_continue, # A function that checks is_finished
    {
        "continue": "planner",
        "end": END
    }
)

app = workflow.compile()