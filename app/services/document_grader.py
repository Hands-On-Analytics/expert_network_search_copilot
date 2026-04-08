from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.models.schemas import GradeResult


GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader for a candidate search system.\n"
            "Given a user query and a retrieved candidate document, decide whether "
            "the document is relevant to the query.\n\n"
            "A document is relevant if it meaningfully matches the intent of the query "
            "(e.g. the candidate's country, skills, experience, or other attributes "
            "align with what the user asked for).\n\n"
            "Return relevant=true if the document is relevant, relevant=false otherwise.",
        ),
        (
            "human",
            "User query: {query}\n\nRetrieved document:\n{document}",
        ),
    ]
)


def build_document_grader() -> ChatOpenAI:
    """Return an LLM configured with structured output for grading."""
    llm = ChatOpenAI(
        model=settings.OPENROUTER_CHAT_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )
    return llm.with_structured_output(GradeResult)


def grade_document(query: str, document: Document) -> bool:
    """Return True if the document is relevant to the query."""
    chain = GRADER_PROMPT | build_document_grader()
    result: GradeResult = chain.invoke(
        {"query": query, "document": document.page_content}
    )
    return result.relevant
