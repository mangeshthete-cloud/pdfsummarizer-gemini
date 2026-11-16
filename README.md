# 📄 PDF AI Summarizer

A Streamlit web app that summarizes PDF documents with **Google Gemini** via **LangChain**, featuring map-reduce chunking, clean prompt templates, structure analysis, key quote extraction, and text/CSV exports.

## Features

- Extracts and cleans text from multi-page PDFs (`pypdf`), with clear error handling for scanned/empty or encrypted PDFs
- Token-accurate chunking with configurable chunk size and overlap (`tiktoken`)
- Map-reduce summarization pipeline built with LangChain (`langchain-core` + `langchain-google-genai`)
- Four summary styles:
  - **brief**: 2–3 sentence concise overview
  - **detailed**: comprehensive points, evidence, and conclusions
  - **bullet_points**: scannable bulleted breakdown by theme
  - **executive**: business insights, risks, and recommendations
- Optional **Document Structure Analysis** and **Key Quotes** extraction
- Downloadable summary report (`.txt`) and per-chunk token/timing statistics (`.csv`)

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and configure `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

## Run

```bash
streamlit run app.py
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```text
app.py              Streamlit UI (sidebar settings, upload, tabbed results, exports)
pdf_processor.py    PDF text extraction, cleaning, token counting, and chunking
prompts.py          LangChain ChatPromptTemplates and summary style definitions
summarizer.py       Gemini map-reduce summarization, structure analysis, and quotes
utils.py            Environment loading, statistics calculation, and .txt/.csv exports
tests/              Unit and Streamlit AppTest suite
requirements.txt    Python dependencies
.env.example        Environment variable template
```

## How It Works

```text
PDF -> extract text -> clean -> chunk (by tokens)
                                    |
                    +---------------+---------------+
                    v               v               v
               summary 1       summary 2   ...  summary N   (Map)
                    +---------------+---------------+
                                    v
                          combine -> final summary          (Reduce)
```

## Troubleshooting

- **"No extractable text found"**: The PDF likely contains scanned images. Run OCR on it before uploading.
- **429 / rate limit errors**: Wait a minute and retry, increase chunk size, or switch models in the sidebar.
- **Model not available (404)**: Update `GEMINI_MODEL` in `.env` or choose another model in the sidebar.
- **API key missing**: Ensure `.env` sits next to `app.py` with `GEMINI_API_KEY` set, or enter it in the sidebar.
