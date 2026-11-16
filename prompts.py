"""LangChain ChatPromptTemplates and summary styles for PDF summarization."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = (
    "You are an expert research analyst and technical writer. "
    "Summarize and analyze PDF documents accurately using only the provided text. "
    "Do not invent facts, citations, or numbers not present in the source. "
    "Write clearly using clean Markdown formatting."
)

STYLES: dict[str, str] = {
    "brief": (
        "Write a concise 2-3 sentence overview capturing the document's core "
        "subject, main argument, and primary conclusion."
    ),
    "detailed": (
        "Write a comprehensive, well-structured summary covering all key points, "
        "supporting arguments, methodology or evidence, and conclusions."
    ),
    "bullet_points": (
        "Write a scannable bullet-point summary organized by major themes, "
        "highlighting the most important facts, findings, and takeaways."
    ),
    "executive": (
        "Write a business-focused executive summary highlighting strategic insights, "
        "key metrics or outcomes, implications, risks, and actionable recommendations."
    ),
}

STYLE_LABELS: dict[str, str] = {
    "brief": "Brief (2-3 sentences)",
    "detailed": "Detailed",
    "bullet_points": "Bullet Points",
    "executive": "Executive",
}


def _template(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            ("human", human),
        ]
    )


CHUNK_SUMMARY = _template(
    "Summarize the following document excerpt.\n\n"
    "Style instructions:\n{style}\n\n"
    "Document excerpt:\n"
    "\"\"\"\n{text}\n\"\"\""
)

COMBINE = _template(
    "You are given summaries of consecutive sections from a single PDF document. "
    "Synthesize them into one cohesive, non-repetitive final summary that reads as "
    "a unified whole.\n\n"
    "Style instructions:\n{style}\n\n"
    "Section summaries:\n"
    "\"\"\"\n{summaries}\n\"\"\""
)

ANALYSIS = _template(
    "Analyze the structure and composition of the following document text. "
    "Provide a clear breakdown with Markdown headings for:\n"
    "1. Document Type & Purpose\n"
    "2. Main Sections / Thematic Flow\n"
    "3. Target Audience & Tone\n"
    "4. Key Themes & Takeaways\n\n"
    "Document text:\n"
    "\"\"\"\n{text}\n\"\"\""
)

QUOTES = _template(
    "Extract 5 to 8 of the most impactful, representative verbatim quotes from "
    "the following document text. For each quote, include the exact quote in "
    "quotation marks followed by a one-sentence explanation of its significance.\n\n"
    "Document text:\n"
    "\"\"\"\n{text}\n\"\"\""
)
