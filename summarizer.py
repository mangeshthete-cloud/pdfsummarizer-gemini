"""LangChain + Google Gemini summarization, analysis, and quote extraction."""

from __future__ import annotations

import time
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from pdf_processor import count_tokens
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

MAX_DIRECT_ANALYSIS_TOKENS = 24000


def _create_llm(
    api_key: str,
    model_name: str,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    if not api_key or not api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is missing. Set it in .env or enter it in the sidebar."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key.strip(),
        temperature=temperature,
    )


def _format_api_error(exc: Exception, model_name: str) -> RuntimeError:
    msg = str(exc)
    lower = msg.lower()
    if "429" in msg or "rate limit" in lower or "quota" in lower or "resource_exhausted" in lower:
        return RuntimeError(
            f"Rate limit or quota exceeded for '{model_name}'. "
            "Wait a moment and retry, use a smaller document, or switch models."
        )
    if "404" in msg or "not found" in lower or "not supported" in lower:
        return RuntimeError(
            f"Model '{model_name}' is unavailable for your API key (404). "
            "Update GEMINI_MODEL in .env or choose another model in the sidebar."
        )
    if "401" in msg or "403" in msg or "api_key" in lower or "permission" in lower or "unauthenticated" in lower:
        return RuntimeError(
            "Authentication failed with Google Gemini. Verify GEMINI_API_KEY in .env or the sidebar."
        )
    return RuntimeError(f"Gemini API error ({type(exc).__name__}): {msg}")


def _sample_text_for_prompt(text: str, max_tokens: int = MAX_DIRECT_ANALYSIS_TOKENS) -> str:
    """Return *text* if within *max_tokens*, otherwise sample head/middle/tail."""
    if count_tokens(text) <= max_tokens:
        return text
    char_limit = max_tokens * 3
    part = char_limit // 3
    head = text[:part]
    mid_start = max((len(text) - part) // 2, part)
    middle = text[mid_start : mid_start + part]
    tail = text[-part:]
    return (
        f"{head}\n\n[... middle excerpt ...]\n\n"
        f"{middle}\n\n[... ending excerpt ...]\n\n{tail}"
    )


def summarize_chunks(
    chunks: list[dict],
    style_key: str,
    llm: ChatGoogleGenerativeAI,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Summarize each chunk (Map step) and collect per-chunk metrics."""
    style_instruction = STYLES[style_key]
    chain = CHUNK_SUMMARY | llm | StrOutputParser()
    results: list[dict] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(idx, total, f"Summarizing chunk {idx} of {total}...")

        start_time = time.perf_counter()
        try:
            summary_text = chain.invoke(
                {"style": style_instruction, "text": chunk["text"]}
            ).strip()
        except Exception as exc:
            raise _format_api_error(exc, llm.model) from exc

        elapsed = round(time.perf_counter() - start_time, 2)
        results.append(
            {
                "chunk_index": chunk["index"],
                "input_tokens": chunk["tokens"],
                "summary": summary_text,
                "summary_tokens": count_tokens(summary_text),
                "elapsed_seconds": elapsed,
            }
        )

    return results


def combine_summaries(
    chunk_results: list[dict],
    style_key: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Combine chunk summaries into one final summary (Reduce step)."""
    if not chunk_results:
        return ""
    if len(chunk_results) == 1:
        return chunk_results[0]["summary"]

    style_instruction = STYLES[style_key]
    formatted_parts = [
        f"--- Section {item['chunk_index']} ---\n{item['summary']}"
        for item in chunk_results
    ]
    combined_input = "\n\n".join(formatted_parts)
    chain = COMBINE | llm | StrOutputParser()

    try:
        return chain.invoke(
            {"style": style_instruction, "summaries": combined_input}
        ).strip()
    except Exception as exc:
        raise _format_api_error(exc, llm.model) from exc


def analyze_structure(text: str, llm: ChatGoogleGenerativeAI) -> str:
    """Analyze the document's structure, purpose, audience, and themes."""
    bounded_text = _sample_text_for_prompt(text)
    chain = ANALYSIS | llm | StrOutputParser()
    try:
        return chain.invoke({"text": bounded_text}).strip()
    except Exception as exc:
        raise _format_api_error(exc, llm.model) from exc


def extract_quotes(text: str, llm: ChatGoogleGenerativeAI) -> str:
    """Extract notable verbatim quotes and their significance."""
    bounded_text = _sample_text_for_prompt(text)
    chain = QUOTES | llm | StrOutputParser()
    try:
        return chain.invoke({"text": bounded_text}).strip()
    except Exception as exc:
        raise _format_api_error(exc, llm.model) from exc


def summarize_document(
    cleaned_text: str,
    chunks: list[dict],
    style_key: str,
    api_key: str,
    model_name: str,
    include_analysis: bool = True,
    include_quotes: bool = False,
    temperature: float = 0.2,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run the full Map-Reduce summarization workflow with optional analysis & quotes."""
    if style_key not in STYLES:
        raise ValueError(f"Unsupported summary style: {style_key}")
    if not chunks:
        raise ValueError("No text chunks available to summarize.")

    total_start = time.perf_counter()
    llm = _create_llm(api_key=api_key, model_name=model_name, temperature=temperature)

    chunk_results = summarize_chunks(
        chunks=chunks,
        style_key=style_key,
        llm=llm,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback(len(chunks), len(chunks), "Combining section summaries...")
    final_summary = combine_summaries(
        chunk_results=chunk_results,
        style_key=style_key,
        llm=llm,
    )

    structure_analysis = ""
    if include_analysis:
        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Analyzing document structure...")
        structure_analysis = analyze_structure(cleaned_text, llm=llm)

    key_quotes = ""
    if include_quotes:
        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Extracting key quotes...")
        key_quotes = extract_quotes(cleaned_text, llm=llm)

    total_elapsed = round(time.perf_counter() - total_start, 2)

    return {
        "final_summary": final_summary,
        "chunk_results": chunk_results,
        "structure_analysis": structure_analysis,
        "key_quotes": key_quotes,
        "total_time_seconds": total_elapsed,
        "model_name": model_name,
        "style": style_key,
    }
