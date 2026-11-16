"""Configuration, statistics, and export helpers for the PDF Summarizer."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from pdf_processor import count_tokens
from prompts import STYLE_LABELS

DEFAULT_MODELS: list[str] = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]


def load_environment() -> None:
    """Load variables from .env if present."""
    load_dotenv(override=False)


def get_api_key(override_key: str | None = None) -> str:
    """Return API key from explicit override or environment."""
    load_environment()
    if override_key and override_key.strip():
        return override_key.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_default_model() -> str:
    """Return configured GEMINI_MODEL or fallback default."""
    load_environment()
    model = os.getenv("GEMINI_MODEL", "").strip()
    return model or "gemini-2.5-flash"


def get_model_options() -> list[str]:
    """Return selectable Gemini models, ensuring the configured default comes first."""
    default_model = get_default_model()
    options = [default_model]
    for model in DEFAULT_MODELS:
        if model not in options:
            options.append(model)
    return options


def compute_statistics(pdf_data: dict, summary_result: dict) -> dict:
    """Compute document and summary statistics."""
    original_tokens = pdf_data["total_tokens"]
    summary_tokens = count_tokens(summary_result["final_summary"])
    reduction_pct = (
        round((1 - (summary_tokens / original_tokens)) * 100, 1)
        if original_tokens > 0
        else 0.0
    )
    compression_ratio = (
        round(original_tokens / summary_tokens, 1)
        if summary_tokens > 0
        else 0.0
    )

    return {
        "page_count": pdf_data["page_count"],
        "original_tokens": original_tokens,
        "chunk_count": len(pdf_data["chunks"]),
        "summary_tokens": summary_tokens,
        "reduction_pct": max(reduction_pct, 0.0),
        "compression_ratio": compression_ratio,
        "total_time_seconds": summary_result["total_time_seconds"],
    }


def build_chunk_stats_dataframe(chunk_results: list[dict]) -> pd.DataFrame:
    """Create a DataFrame with per-chunk token and timing statistics."""
    rows = []
    for item in chunk_results:
        inp = item["input_tokens"]
        out = item["summary_tokens"]
        reduction = round((1 - (out / inp)) * 100, 1) if inp > 0 else 0.0
        rows.append(
            {
                "Chunk": item["chunk_index"],
                "Input Tokens": inp,
                "Summary Tokens": out,
                "Reduction (%)": max(reduction, 0.0),
                "Processing Time (s)": item["elapsed_seconds"],
            }
        )
    return pd.DataFrame(rows)


def export_summary_text(
    filename: str,
    summary_result: dict,
    stats: dict,
) -> str:
    """Format the summary, analysis, quotes, and metrics as a downloadable text report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    style_label = STYLE_LABELS.get(summary_result["style"], summary_result["style"])

    sections = [
        "=" * 60,
        "PDF AI SUMMARIZER REPORT",
        "=" * 60,
        f"Source File:      {filename}",
        f"Generated:        {timestamp}",
        f"Model:            {summary_result['model_name']}",
        f"Summary Style:    {style_label}",
        f"Pages:            {stats['page_count']}",
        f"Original Tokens:  {stats['original_tokens']:,}",
        f"Summary Tokens:   {stats['summary_tokens']:,}",
        f"Chunks Processed: {stats['chunk_count']}",
        f"Compression:      {stats['reduction_pct']}% ({stats['compression_ratio']}x)",
        f"Total Time:       {stats['total_time_seconds']}s",
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
        summary_result["final_summary"],
    ]

    if summary_result.get("structure_analysis"):
        sections.extend(
            [
                "",
                "-" * 60,
                "STRUCTURE ANALYSIS",
                "-" * 60,
                summary_result["structure_analysis"],
            ]
        )

    if summary_result.get("key_quotes"):
        sections.extend(
            [
                "",
                "-" * 60,
                "KEY QUOTES",
                "-" * 60,
                summary_result["key_quotes"],
            ]
        )

    return "\n".join(sections).strip() + "\n"


def export_chunk_stats_csv(df: pd.DataFrame) -> bytes:
    """Serialize chunk statistics DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
