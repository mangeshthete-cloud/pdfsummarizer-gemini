"""Unit tests for the PDF AI Summarizer pipeline (no live API calls required)."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from pdf_processor import chunk_text, clean_text, count_tokens, process_pdf
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES
from utils import (
    build_chunk_stats_dataframe,
    compute_statistics,
    export_chunk_stats_csv,
    export_summary_text,
)


def _make_sample_pdf(pages_text: list[str]) -> bytes:
    objects, kids_refs, page_objs = [], [], []
    base_id = 4
    for idx, txt in enumerate(pages_text):
        page_id = base_id + idx * 2
        content_id = page_id + 1
        kids_refs.append(f"{page_id} 0 R")
        safe = txt.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin1")
        page_objs.append(
            (
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode(),
            )
        )
        page_objs.append(
            (
                content_id,
                b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            )
        )
    objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    objects.append((2, f"<< /Type /Pages /Kids [{' '.join(kids_refs)}] /Count {len(pages_text)} >>".encode()))
    objects.append((3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.extend(page_objs)

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in objects:
        offsets.append(len(buf))
        buf.extend(f"{obj_id} 0 obj\n".encode())
        buf.extend(body)
        buf.extend(b"\nendobj\n")
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode())
    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(buf)


class TestPDFProcessor(unittest.TestCase):
    def test_clean_text_fixes_hyphenation_and_whitespace(self) -> None:
        raw = "Docu-\nment   summary\r\n\r\n\r\nNext   line\x00"
        self.assertEqual(clean_text(raw), "Document summary\n\nNext line")

    def test_chunk_text_respects_max_tokens(self) -> None:
        text = "LangChain and Gemini summarize PDF documents reliably. " * 60
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["tokens"], 100)
            self.assertGreater(chunk["tokens"], 0)

    def test_process_pdf_and_empty_error(self) -> None:
        pdf_bytes = _make_sample_pdf(
            ["Page one discusses LangChain pipelines. " * 15, "Page two discusses Gemini. " * 15]
        )
        data = process_pdf(pdf_bytes, chunk_size=80, chunk_overlap=10)
        self.assertEqual(data["page_count"], 2)
        self.assertGreater(data["total_tokens"], 50)
        self.assertGreaterEqual(len(data["chunks"]), 2)

        empty_pdf = _make_sample_pdf(["   "])
        with self.assertRaises(ValueError):
            process_pdf(empty_pdf)


class TestPromptsAndUtils(unittest.TestCase):
    def test_prompt_templates_format(self) -> None:
        for style_text in STYLES.values():
            msgs = CHUNK_SUMMARY.format_messages(style=style_text, text="Sample text")
            self.assertEqual(len(msgs), 2)
        self.assertEqual(len(COMBINE.format_messages(style=STYLES["brief"], summaries="A\nB")), 2)
        self.assertEqual(len(ANALYSIS.format_messages(text="Sample")), 2)
        self.assertEqual(len(QUOTES.format_messages(text="Sample")), 2)

    def test_stats_and_exports(self) -> None:
        pdf_data = {"page_count": 3, "total_tokens": 1000, "chunks": [{"index": 1}, {"index": 2}]}
        summary_result = {
            "final_summary": "Concise summary text.",
            "chunk_results": [
                {"chunk_index": 1, "input_tokens": 500, "summary": "S1", "summary_tokens": 50, "elapsed_seconds": 1.1},
                {"chunk_index": 2, "input_tokens": 500, "summary": "S2", "summary_tokens": 40, "elapsed_seconds": 0.9},
            ],
            "structure_analysis": "Structure details",
            "key_quotes": '"Quote one"',
            "total_time_seconds": 2.5,
            "model_name": "gemini-3.1-flash-lite",
            "style": "brief",
        }
        stats = compute_statistics(pdf_data, summary_result)
        self.assertEqual(stats["page_count"], 3)
        self.assertEqual(stats["summary_tokens"], count_tokens("Concise summary text."))

        df = build_chunk_stats_dataframe(summary_result["chunk_results"])
        self.assertEqual(len(df), 2)
        txt = export_summary_text("report.pdf", summary_result, stats)
        self.assertIn("Concise summary text.", txt)
        self.assertIn("STRUCTURE ANALYSIS", txt)
        self.assertIn("KEY QUOTES", txt)
        csv_bytes = export_chunk_stats_csv(df)
        self.assertIn(b"Chunk,Input Tokens,Summary Tokens", csv_bytes)


class TestStreamlitApp(unittest.TestCase):
    def test_app_boots_without_exception(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=15)
        at.run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
