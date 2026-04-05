from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from candidate_vector_api.embeddings import OpenRouterEmbeddingClient
from candidate_vector_api.settings import Settings
from candidate_vector_api.vector_store import ChromaVectorStore

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "these",
    "those",
    "to",
    "with",
}

_REFERENCE_TERMS = (
    "those",
    "them",
    "these",
    "previous",
    "above",
    "that list",
    "filter",
    "narrow",
)

_MIDDLE_EAST_LOCATIONS = {
    "saudi arabia",
    "united arab emirates",
    "uae",
    "qatar",
    "kuwait",
    "bahrain",
    "oman",
    "jordan",
    "lebanon",
    "egypt",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        marker = value.strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        deduped.append(value.strip())
    return deduped


def _term_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-/]+", text.lower())
    return {token for token in tokens if token not in _STOP_WORDS and len(token) > 2}


@dataclass(slots=True)
class ExpertMatch:
    candidate_id: str
    full_name: str
    headline: str
    city: str
    country: str
    nationality: str
    years_of_experience: int
    match_score: float
    why_match: list[str]
    key_highlights: list[str]
    matched_chunk_types: list[str]


@dataclass(slots=True)
class SearchExecutionResult:
    session_id: str
    original_query: str
    transformed_query: str
    query_variants: list[str]
    follow_up_detected: bool
    applied_filters: dict[str, str | bool]
    experts: list[ExpertMatch]


@dataclass(slots=True)
class SessionState:
    session_id: str
    last_standalone_query: str | None = None
    last_result_candidate_ids: list[str] = field(default_factory=list)
    last_experts: list[ExpertMatch] = field(default_factory=list)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str | None) -> SessionState:
        resolved = session_id or str(uuid4())
        with self._lock:
            if resolved not in self._sessions:
                self._sessions[resolved] = SessionState(session_id=resolved)
            return self._sessions[resolved]

    def save(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state


@dataclass(slots=True)
class QueryPlan:
    original_query: str
    standalone_query: str
    variants: list[str]
    follow_up_detected: bool
    reference_follow_up: bool
    location_filter: str | None
    restricted_candidate_ids: set[str] | None


@dataclass(slots=True)
class _CandidateAggregate:
    candidate_id: str
    full_name: str = "Unknown Candidate"
    headline: str = ""
    city: str = ""
    country: str = ""
    nationality: str = ""
    years_of_experience: int = 0
    best_score: float = 0.0
    weighted_score_sum: float = 0.0
    hit_count: int = 0
    chunk_types: set[str] = field(default_factory=set)
    highlights: list[str] = field(default_factory=list)
    overlap_terms: set[str] = field(default_factory=set)


class CandidateSearchService:
    def __init__(
        self,
        embedding_client: OpenRouterEmbeddingClient,
        vector_store: ChromaVectorStore,
        session_store: InMemorySessionStore,
        candidate_pool_size: int,
        max_query_variants: int,
        highlight_char_limit: int,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._session_store = session_store
        self._candidate_pool_size = candidate_pool_size
        self._max_query_variants = max_query_variants
        self._highlight_char_limit = highlight_char_limit

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        session_store: InMemorySessionStore | None = None,
    ) -> CandidateSearchService:
        embedding_client = OpenRouterEmbeddingClient(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
        vector_store = ChromaVectorStore(
            persist_path=settings.resolved_chroma_path,
            collection_name=settings.chroma_collection_name,
        )
        return cls(
            embedding_client=embedding_client,
            vector_store=vector_store,
            session_store=session_store or InMemorySessionStore(),
            candidate_pool_size=settings.search_candidate_pool_size,
            max_query_variants=settings.search_max_query_variants,
            highlight_char_limit=settings.search_highlight_char_limit,
        )

    def search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> SearchExecutionResult:
        cleaned_query = _normalize_text(query)
        if not cleaned_query:
            raise ValueError("query must not be empty")

        session = self._session_store.get_or_create(session_id)
        plan = self._build_query_plan(cleaned_query, session)
        experts = self._retrieve_and_rank(plan, top_k)

        if plan.reference_follow_up and not experts:
            experts = self._fallback_filter_from_session(session, plan, top_k)

        session.last_standalone_query = plan.standalone_query
        session.last_result_candidate_ids = [expert.candidate_id for expert in experts]
        session.last_experts = experts
        self._session_store.save(session)

        applied_filters: dict[str, str | bool] = {
            "restricted_to_previous_results": bool(plan.reference_follow_up),
        }
        if plan.location_filter:
            applied_filters["location"] = plan.location_filter

        return SearchExecutionResult(
            session_id=session.session_id,
            original_query=plan.original_query,
            transformed_query=plan.standalone_query,
            query_variants=plan.variants,
            follow_up_detected=plan.follow_up_detected,
            applied_filters=applied_filters,
            experts=experts,
        )

    def _build_query_plan(self, query: str, session: SessionState) -> QueryPlan:
        lower_query = query.lower()
        follow_up_detected = any(term in lower_query for term in _REFERENCE_TERMS)
        reference_follow_up = follow_up_detected and bool(session.last_result_candidate_ids)

        standalone_query = query
        if reference_follow_up and session.last_standalone_query:
            standalone_query = (
                f"{session.last_standalone_query}. "
                f"Additional user constraint: {query}"
            )

        variants = self._build_query_variants(standalone_query, query)
        location_filter = self._extract_location_filter(query)

        restricted: set[str] | None = None
        if reference_follow_up and session.last_result_candidate_ids:
            restricted = set(session.last_result_candidate_ids)

        return QueryPlan(
            original_query=query,
            standalone_query=standalone_query,
            variants=variants[: self._max_query_variants],
            follow_up_detected=follow_up_detected,
            reference_follow_up=reference_follow_up,
            location_filter=location_filter,
            restricted_candidate_ids=restricted,
        )

    def _build_query_variants(self, standalone_query: str, original_query: str) -> list[str]:
        variants: list[str] = [standalone_query]
        lower = standalone_query.lower()

        experience_match = re.search(
            r"(?:experience|background|expertise)\s+in\s+([^.,;]+)",
            standalone_query,
            re.IGNORECASE,
        )
        if experience_match:
            variants.append(f"Experts with experience in {experience_match.group(1).strip()}")

        keywords = sorted(_term_tokens(standalone_query))
        if keywords:
            variants.append(" ".join(keywords[:12]))

        if "middle east" in lower:
            variants.append(
                "Middle East experts in Saudi Arabia UAE Qatar Kuwait Bahrain Oman Jordan Lebanon Egypt"
            )

        location_filter = self._extract_location_filter(original_query)
        if location_filter:
            variants.append(f"Experts based in {location_filter}")

        return _dedupe_preserving_order(variants)

    def _extract_location_filter(self, query: str) -> str | None:
        lower = query.lower()
        if "middle east" in lower:
            return "Middle East"

        patterns = [
            r"(?:based|located)\s+in\s+([a-zA-Z][a-zA-Z\s\-]{1,40})",
            r"only\s+(?:people|experts|candidates)\s+(?:based\s+)?in\s+([a-zA-Z][a-zA-Z\s\-]{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip(" .,;")
                if value:
                    return value.title()
        return None

    def _retrieve_and_rank(self, plan: QueryPlan, top_k: int) -> list[ExpertMatch]:
        query_embeddings = self._embedding_client.embed_texts(plan.variants)
        pool_size = max(self._candidate_pool_size, top_k * 20)
        raw = self._vector_store.query(query_embeddings=query_embeddings, n_results=pool_size)

        aggregates: dict[str, _CandidateAggregate] = {}
        query_terms = _term_tokens(plan.standalone_query)

        ids_rows = raw.get("ids") or []
        docs_rows = raw.get("documents") or []
        metas_rows = raw.get("metadatas") or []
        distance_rows = raw.get("distances") or []

        query_count = len(plan.variants)
        for query_index in range(query_count):
            ids = ids_rows[query_index] if query_index < len(ids_rows) and ids_rows[query_index] else []
            docs = docs_rows[query_index] if query_index < len(docs_rows) and docs_rows[query_index] else []
            metas = metas_rows[query_index] if query_index < len(metas_rows) and metas_rows[query_index] else []
            distances = (
                distance_rows[query_index]
                if query_index < len(distance_rows) and distance_rows[query_index]
                else []
            )

            for hit_index, chunk_id in enumerate(ids):
                metadata = metas[hit_index] if hit_index < len(metas) and metas[hit_index] else {}
                document = docs[hit_index] if hit_index < len(docs) and docs[hit_index] else ""
                distance = distances[hit_index] if hit_index < len(distances) else 1.0

                candidate_id = str(metadata.get("candidate_id") or str(chunk_id).split(":")[0])
                if plan.restricted_candidate_ids and candidate_id not in plan.restricted_candidate_ids:
                    continue
                if not self._matches_location_filter(metadata, plan.location_filter):
                    continue

                aggregate = aggregates.get(candidate_id)
                if aggregate is None:
                    aggregate = _CandidateAggregate(candidate_id=candidate_id)
                    aggregates[candidate_id] = aggregate

                aggregate.full_name = str(metadata.get("full_name") or aggregate.full_name)
                aggregate.headline = str(metadata.get("headline") or aggregate.headline)
                aggregate.city = str(metadata.get("city") or aggregate.city)
                aggregate.country = str(metadata.get("country") or aggregate.country)
                aggregate.nationality = str(metadata.get("nationality") or aggregate.nationality)
                aggregate.years_of_experience = int(
                    metadata.get("years_of_experience") or aggregate.years_of_experience or 0
                )

                similarity = self._distance_to_similarity(distance)
                query_weight = max(0.55, 1.0 - (query_index * 0.12))
                weighted_similarity = similarity * query_weight
                aggregate.best_score = max(aggregate.best_score, weighted_similarity)
                aggregate.weighted_score_sum += weighted_similarity
                aggregate.hit_count += 1

                chunk_type = str(metadata.get("chunk_type") or "profile_summary")
                aggregate.chunk_types.add(chunk_type)
                if document:
                    aggregate.highlights.append(self._truncate_highlight(document))
                    aggregate.overlap_terms.update(
                        query_terms.intersection(_term_tokens(document + " " + aggregate.headline))
                    )

        experts: list[ExpertMatch] = []
        for aggregate in aggregates.values():
            average_score = aggregate.weighted_score_sum / aggregate.hit_count if aggregate.hit_count else 0.0
            match_score = (0.65 * aggregate.best_score) + (0.35 * average_score)
            why_match = self._build_why_match(
                aggregate=aggregate,
                location_filter=plan.location_filter,
            )
            highlights = _dedupe_preserving_order(aggregate.highlights)[:3]

            experts.append(
                ExpertMatch(
                    candidate_id=aggregate.candidate_id,
                    full_name=aggregate.full_name,
                    headline=aggregate.headline,
                    city=aggregate.city,
                    country=aggregate.country,
                    nationality=aggregate.nationality,
                    years_of_experience=aggregate.years_of_experience,
                    match_score=round(match_score, 4),
                    why_match=why_match,
                    key_highlights=highlights,
                    matched_chunk_types=sorted(aggregate.chunk_types),
                )
            )

        experts.sort(key=lambda item: item.match_score, reverse=True)
        return experts[:top_k]

    def _fallback_filter_from_session(
        self,
        session: SessionState,
        plan: QueryPlan,
        top_k: int,
    ) -> list[ExpertMatch]:
        filtered: list[ExpertMatch] = []
        for expert in session.last_experts:
            metadata = {
                "city": expert.city,
                "country": expert.country,
                "nationality": expert.nationality,
            }
            if not self._matches_location_filter(metadata, plan.location_filter):
                continue
            adjusted = ExpertMatch(
                candidate_id=expert.candidate_id,
                full_name=expert.full_name,
                headline=expert.headline,
                city=expert.city,
                country=expert.country,
                nationality=expert.nationality,
                years_of_experience=expert.years_of_experience,
                match_score=round(expert.match_score * 0.95, 4),
                why_match=expert.why_match + ["Retained from prior session results after follow-up filtering."],
                key_highlights=expert.key_highlights,
                matched_chunk_types=expert.matched_chunk_types,
            )
            filtered.append(adjusted)

        filtered.sort(key=lambda item: item.match_score, reverse=True)
        return filtered[:top_k]

    def _build_why_match(
        self,
        aggregate: _CandidateAggregate,
        location_filter: str | None,
    ) -> list[str]:
        reasons: list[str] = []
        if "skills" in aggregate.chunk_types:
            reasons.append("Strong alignment on skill-related profile chunks.")
        if "work_experience" in aggregate.chunk_types:
            reasons.append("Relevant work-experience history matched the query intent.")
        if "profile_summary" in aggregate.chunk_types and aggregate.headline:
            reasons.append("Profile summary/headline closely aligns with the requested profile.")
        if "languages" in aggregate.chunk_types:
            reasons.append("Language capabilities contributed to the match.")
        if "education" in aggregate.chunk_types:
            reasons.append("Educational background contributed to the match.")
        if aggregate.overlap_terms:
            overlap = ", ".join(sorted(aggregate.overlap_terms)[:4])
            reasons.append(f"Query term overlap found in matched chunks: {overlap}.")
        if location_filter:
            reasons.append(f"Location filter satisfied: {location_filter}.")
        if not reasons:
            reasons.append("High vector similarity against multiple profile chunks.")
        return reasons[:4]

    def _distance_to_similarity(self, distance: float | int | None) -> float:
        if distance is None:
            return 0.0
        safe = max(float(distance), 0.0)
        return 1.0 / (1.0 + safe)

    def _truncate_highlight(self, text: str) -> str:
        cleaned = _normalize_text(text).replace("\n", " ")
        if len(cleaned) <= self._highlight_char_limit:
            return cleaned
        return cleaned[: self._highlight_char_limit - 3].rstrip() + "..."

    def _matches_location_filter(self, metadata: dict[str, object], location_filter: str | None) -> bool:
        if not location_filter:
            return True

        city = str(metadata.get("city") or "").lower()
        country = str(metadata.get("country") or "").lower()
        nationality = str(metadata.get("nationality") or "").lower()
        haystack = f"{city} {country} {nationality}"

        normalized_filter = location_filter.lower()
        if normalized_filter == "middle east":
            return any(location in haystack for location in _MIDDLE_EAST_LOCATIONS)
        return normalized_filter in haystack
