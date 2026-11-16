"""PDF text extraction, cleaning, token counting, and chunking utilities."""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader
from pypdf.errors import PdfReadError


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in *text* using tiktoken."""
    if not text:
        return 0
    encoding = _get_encoding(encoding_name)
    return len(encoding.encode(text, disallowed_special=()))


def clean_text(text: str) -> str:
    """Normalize extracted PDF text by removing control chars and excess whitespace."""
    if not text:
        return ""

    # Normalize line endings and remove non-printable control characters (keep \n and \t)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Fix hyphenated line breaks common in PDFs (e.g., "summariza-\ntion")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse horizontal whitespace on each line and strip trailing spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse 3+ consecutive newlines into double newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(pdf_source: str | Path | BinaryIO | bytes) -> tuple[str, int]:
    """Extract text and page count from a PDF path, bytes, or binary stream."""
    try:
        if isinstance(pdf_source, (str, Path)):
            reader = PdfReader(str(pdf_source))
        elif isinstance(pdf_source, bytes):
            reader = PdfReader(io.BytesIO(pdf_source))
        else:
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            reader = PdfReader(pdf_source)
    except PdfReadError as exc:
        raise ValueError(f"Unable to read PDF file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to open PDF file: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise ValueError("PDF is password-protected and cannot be read.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF is encrypted and cannot be decrypted.") from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF has no pages.")

    page_texts: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            page_texts.append(extracted)

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may consist of scanned images—please run OCR on it first."
        )

    return full_text, page_count


def chunk_text(
    text: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
    encoding_name: str = "cl100k_base",
) -> list[dict]:
    """Split *text* into token-bounded chunks with configurable overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []

    encoding = _get_encoding(encoding_name)
    tokens = encoding.encode(cleaned, disallowed_special=())
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        return [
            {
                "index": 1,
                "text": cleaned,
                "tokens": total_tokens,
                "char_count": len(cleaned),
            }
        ]

    chunks: list[dict] = []
    start = 0
    chunk_idx = 1

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        window_tokens = tokens[start:end]
        chunk_str = encoding.decode(window_tokens)

        # Snap to a natural boundary near the end of the chunk if not at EOF
        if end < total_tokens:
            snap_min = int(len(chunk_str) * 0.8)
            split_pos = -1
            for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
                pos = chunk_str.rfind(sep, snap_min)
                if pos != -1:
                    split_pos = pos + len(sep)
                    break
            if split_pos > 0:
                snapped_str = chunk_str[:split_pos].strip()
                if snapped_str:
                    snapped_tokens = encoding.encode(snapped_str, disallowed_special=())
                    if snapped_tokens:
                        chunk_str = snapped_str
                        end = start + len(snapped_tokens)
                        window_tokens = snapped_tokens

        chunk_str = chunk_str.strip()
        if chunk_str:
            chunks.append(
                {
                    "index": chunk_idx,
                    "text": chunk_str,
                    "tokens": len(window_tokens),
                    "char_count": len(chunk_str),
                }
            )
            chunk_idx += 1

        if end >= total_tokens:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def process_pdf(
    pdf_source: str | Path | BinaryIO | bytes,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
) -> dict:
    """Extract, clean, count tokens, and chunk a PDF document."""
    raw_text, page_count = extract_text_from_pdf(pdf_source)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise ValueError(
            "No extractable text remained after cleaning this PDF."
        )

    total_tokens = count_tokens(cleaned_text)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "page_count": page_count,
        "total_tokens": total_tokens,
        "chunks": chunks,
    }
