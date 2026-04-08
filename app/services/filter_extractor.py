from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.models.schemas import ExtractedFilters


FILTER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query-understanding assistant for an expert-network candidate database.\n"
            "Given a user query, extract any structured filters that apply.\n\n"
            "Available filter fields:\n"
            "- country_name: the FULL official country name (e.g. 'United States', 'United Kingdom', 'India', 'Germany'). "
            "NEVER use abbreviations like 'USA', 'UK', 'UAE'. Always expand to the full name.\n"
            "- country_code: the ISO 3166-1 alpha-2 code for the country (e.g. 'US', 'GB', 'IN', 'DE'). "
            "Always provide this alongside country_name when a country is mentioned.\n"
            "- city_name: the city the candidate is in\n"
            "- gender: Male or Female\n"
            "- min_years_of_experience: minimum years of professional experience\n"
            "- max_years_of_experience: maximum years of professional experience\n\n"
            "Also produce a 'keywords' field that contains the remaining semantic part of "
            "the query after stripping out the filter terms. If the entire query is just a "
            "filter (e.g. 'Uganda'), set keywords to null.\n\n"
            "Return null for any filter that is not mentioned in the query.",
        ),
        ("human", "{query}"),
    ]
)


def build_filter_extractor() -> ChatOpenAI:
    """Return an LLM configured with structured output for filter extraction."""
    llm = ChatOpenAI(
        model=settings.OPENROUTER_CHAT_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )
    return llm.with_structured_output(ExtractedFilters)


def extract_filters(query: str) -> ExtractedFilters:
    """Parse a natural-language query and return structured filters."""
    chain = FILTER_EXTRACTION_PROMPT | build_filter_extractor()
    return chain.invoke({"query": query})
