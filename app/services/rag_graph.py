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
from app.models.schemas import ExtractedFilters
from app.services.document_grader import grade_document
from app.services.filter_extractor import extract_filters
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


# ── state shared across nodes ──────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    search_query: str
    filters: ExtractedFilters | None
    documents: list[Document]
    context: str
    answer: str


# ── prompt ─────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = (
    "You are an expert-network copilot. "
    "Use ONLY the context below to answer the user's question. "
    "If the context does not contain enough information, say so.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)


# ── helpers ────────────────────────────────────────────────────────────
def _build_qdrant_filter(filters: ExtractedFilters | None) -> Filter | None:
    """Convert ExtractedFilters into a Qdrant Filter object (or None)."""
    if filters is None:
        return None

    conditions: list[FieldCondition] = []

    if filters.country_name or filters.country_code:
        country_conditions = []
        if filters.country_name:
            country_conditions.append(
                FieldCondition(key="metadata.country_name", match=MatchValue(value=filters.country_name))
            )
        if filters.country_code:
            country_conditions.append(
                FieldCondition(key="metadata.country_code", match=MatchValue(value=filters.country_code))
            )
        # OR: match on either country_name or country_code
        conditions.append(Filter(should=country_conditions))
    if filters.city_name:
        conditions.append(
            FieldCondition(key="metadata.city_name", match=MatchValue(value=filters.city_name))
        )
    if filters.gender:
        conditions.append(
            FieldCondition(key="metadata.gender", match=MatchValue(value=filters.gender))
        )
    if filters.min_years_of_experience is not None or filters.max_years_of_experience is not None:
        range_kwargs = {}
        if filters.min_years_of_experience is not None:
            range_kwargs["gte"] = filters.min_years_of_experience
        if filters.max_years_of_experience is not None:
            range_kwargs["lte"] = filters.max_years_of_experience
        conditions.append(
            FieldCondition(key="metadata.years_of_experience", range=Range(**range_kwargs))
        )

    return Filter(must=conditions) if conditions else None


# ── graph builder ──────────────────────────────────────────────────────
def build_rag_graph(qdrant_service: QdrantService) -> StateGraph:
    """Return a compiled LangGraph for Corrective-RAG over the candidate collection."""

    vectorstore = qdrant_service.vectorstore()

    llm = ChatOpenAI(
        model=settings.OPENROUTER_CHAT_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )

    # ── node: extract_filters ──────────────────────────────────────
    def extract_filters_node(state: RAGState) -> dict:
        filters = extract_filters(state["query"])
        search_query = filters.keywords or state["query"]
        logger.info("Extracted filters: %s | search_query: %s", filters, search_query)
        return {"filters": filters, "search_query": search_query}

    # ── node: retrieve ─────────────────────────────────────────────
    def retrieve(state: RAGState) -> dict:
        qdrant_filter = _build_qdrant_filter(state.get("filters"))
        search_query = state.get("search_query") or state["query"]

        docs = vectorstore.similarity_search(
            query=search_query,
            k=settings.RAG_TOP_K,
            filter=qdrant_filter,
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

    # ── node: grade_documents (Corrective RAG) ─────────────────────
    def grade_documents(state: RAGState) -> dict:
        query = state["query"]
        docs = state.get("documents", [])

        relevant_docs: list[Document] = []
        for doc in docs:
            if grade_document(query, doc):
                relevant_docs.append(doc)
            else:
                logger.info("Grader marked document as irrelevant: %s", doc.metadata.get("candidate_id"))

        if not relevant_docs:
            return {
                "documents": [],
                "context": "",
                "answer": "No relevant candidate records found for your query after relevance check.",
            }

        context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
        return {"documents": relevant_docs, "context": context}

    # ── node: generate ─────────────────────────────────────────────
    def generate(state: RAGState) -> dict:
        chain = RAG_PROMPT | llm
        response = chain.invoke(
            {"context": state["context"], "query": state["query"]}
        )
        return {"answer": response.content}

    # ── routing functions ──────────────────────────────────────────
    def after_retrieve(state: RAGState) -> str:
        if not state.get("documents"):
            return "end"
        return "grade_documents"

    def after_grade(state: RAGState) -> str:
        if not state.get("documents"):
            return "end"
        return "generate"

    # ── assemble graph ─────────────────────────────────────────────
    graph = StateGraph(RAGState)
    graph.add_node("extract_filters", extract_filters_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    graph.set_entry_point("extract_filters")
    graph.add_edge("extract_filters", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"grade_documents": "grade_documents", "end": END})
    graph.add_conditional_edges("grade_documents", after_grade, {"generate": "generate", "end": END})
    graph.add_edge("generate", END)

    return graph.compile()
