"""Streamlit UI for the LangChain + Google Gemini PDF Summarizer."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pdf_processor import process_pdf
from prompts import STYLE_LABELS
from summarizer import summarize_document
from utils import (
    build_chunk_stats_dataframe,
    compute_statistics,
    export_chunk_stats_csv,
    export_summary_text,
    get_api_key,
    get_model_options,
)

st.set_page_config(
    page_title="PDF AI Summarizer",
    page_icon="📄",
    layout="wide",
)


def _render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")

        model_options = get_model_options()
        model_name = st.selectbox(
            "Gemini Model",
            options=model_options,
            index=0,
            help="Defaults to GEMINI_MODEL in .env",
        )

        api_key_override = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            placeholder="Uses GEMINI_API_KEY from .env",
            help="Leave blank to use the key configured in .env.",
        )

        resolved_key = get_api_key(api_key_override)
        if resolved_key:
            source = "sidebar input" if api_key_override.strip() else ".env"
            st.caption(f"API key loaded from `{source}`.")
        else:
            st.warning("No GEMINI_API_KEY found. Add it to .env or enter it above.")

        st.divider()

        style_keys = list(STYLE_LABELS.keys())
        style_key = st.selectbox(
            "Summary Type",
            options=style_keys,
            index=1,
            format_func=lambda k: STYLE_LABELS[k],
        )

        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )

        st.divider()
        st.subheader("Chunking")
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=500,
            max_value=6000,
            value=3000,
            step=250,
        )
        max_overlap = min(500, chunk_size - 100)
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=max_overlap,
            value=min(200, max_overlap),
            step=50,
        )

        st.divider()
        st.subheader("Additional Analysis")
        include_analysis = st.checkbox("Document Structure Analysis", value=True)
        include_quotes = st.checkbox("Extract Key Quotes", value=False)

    return {
        "model_name": model_name,
        "api_key": resolved_key,
        "style_key": style_key,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "include_analysis": include_analysis,
        "include_quotes": include_quotes,
    }


def _render_results(filename: str, pdf_data: dict, result: dict, stats: dict) -> None:
    st.subheader("Results")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pages", stats["page_count"])
    c2.metric("Original Tokens", f"{stats['original_tokens']:,}")
    c3.metric("Summary Tokens", f"{stats['summary_tokens']:,}")
    c4.metric("Reduction", f"{stats['reduction_pct']}%")
    c5.metric("Time", f"{stats['total_time_seconds']}s")

    tab_names = ["Summary"]
    if result.get("structure_analysis"):
        tab_names.append("Structure Analysis")
    if result.get("key_quotes"):
        tab_names.append("Key Quotes")
    tab_names.append("Chunk Stats")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    with tabs[tab_idx]:
        st.markdown(result["final_summary"])
    tab_idx += 1

    if result.get("structure_analysis"):
        with tabs[tab_idx]:
            st.markdown(result["structure_analysis"])
        tab_idx += 1

    if result.get("key_quotes"):
        with tabs[tab_idx]:
            st.markdown(result["key_quotes"])
        tab_idx += 1

    with tabs[tab_idx]:
        df = build_chunk_stats_dataframe(result["chunk_results"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("View per-chunk summaries"):
            for item in result["chunk_results"]:
                st.markdown(
                    f"**Chunk {item['chunk_index']}** "
                    f"({item['input_tokens']} → {item['summary_tokens']} tokens, "
                    f"{item['elapsed_seconds']}s)"
                )
                st.markdown(item["summary"])
                st.divider()

    st.divider()
    stem = Path(filename).stem
    report_txt = export_summary_text(filename, result, stats)
    stats_csv = export_chunk_stats_csv(build_chunk_stats_dataframe(result["chunk_results"]))

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download Summary (.txt)",
            data=report_txt,
            file_name=f"{stem}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Download Chunk Stats (.csv)",
            data=stats_csv,
            file_name=f"{stem}_chunk_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.title("📄 PDF AI Summarizer")
    st.caption(
        "Powered by LangChain and Google Gemini — extract, chunk, summarize, "
        "analyze structure, and export insights."
    )

    cfg = _render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned PDFs require OCR first.",
    )

    if uploaded_file is None:
        st.info("Upload a PDF file to begin.")
        return

    pdf_bytes = uploaded_file.getvalue()
    try:
        pdf_data = process_pdf(
            pdf_bytes,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(
        f"Loaded **{uploaded_file.name}** — {pdf_data['page_count']} pages, "
        f"{pdf_data['total_tokens']:,} tokens split into {len(pdf_data['chunks'])} chunk(s)."
    )

    run_clicked = st.button(
        "Generate Summary",
        type="primary",
        disabled=not bool(cfg["api_key"]),
        use_container_width=True,
    )

    state_key = (
        uploaded_file.name,
        len(pdf_bytes),
        cfg["model_name"],
        cfg["style_key"],
        cfg["temperature"],
        cfg["chunk_size"],
        cfg["chunk_overlap"],
        cfg["include_analysis"],
        cfg["include_quotes"],
    )

    if run_clicked:
        progress_bar = st.progress(0.0, text="Starting summarization...")
        status_text = st.empty()

        def on_progress(current: int, total: int, message: str) -> None:
            ratio = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(ratio, text=message)
            status_text.caption(message)

        try:
            result = summarize_document(
                cleaned_text=pdf_data["cleaned_text"],
                chunks=pdf_data["chunks"],
                style_key=cfg["style_key"],
                api_key=cfg["api_key"],
                model_name=cfg["model_name"],
                include_analysis=cfg["include_analysis"],
                include_quotes=cfg["include_quotes"],
                temperature=cfg["temperature"],
                progress_callback=on_progress,
            )
            stats = compute_statistics(pdf_data, result)
            st.session_state["last_run_key"] = state_key
            st.session_state["last_result"] = result
            st.session_state["last_stats"] = stats
            progress_bar.progress(1.0, text="Done!")
            status_text.empty()
        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(str(exc))
            return

    if st.session_state.get("last_run_key") == state_key:
        _render_results(
            filename=uploaded_file.name,
            pdf_data=pdf_data,
            result=st.session_state["last_result"],
            stats=st.session_state["last_stats"],
        )


if __name__ == "__main__":
    main()
