# Receipt & Invoice Sorter

Local-first tool that turns a folder of receipt or invoice **images and PDFs** into a CSV (`vendor`, `date`, `amount`, `currency`, `category`, `source_filename`) using a vision-language model that runs in **Ollama on your machine**. Documents are never sent to a cloud API.

## Prerequisites

1. **Python 3.10+** with the project virtualenv (already created in `venv/` for this repo).
2. **[Ollama](https://ollama.com/)** installed and running locally.
3. A vision model pulled **before** you extract:

```bash
ollama pull qwen2.5vl:7b
```

If `qwen2.5vl:7b` is unavailable, pull the fallback:

```bash
ollama pull llama3.2-vision
```

The app picks `qwen2.5vl:7b` (or any installed `qwen2.5vl` tag) first, then `llama3.2-vision`. If Ollama is down or neither model is present, the UI shows those exact `ollama pull …` commands instead of a stack trace.

Qwen2.5-VL needs a recent Ollama (0.7+).

## Setup

```bash
cd ollama-tax-man
source venv/bin/activate
```

Packages already in the environment: `streamlit`, `pandas`, `pydantic`, `pillow`, `pymupdf`, `requests` (used to call the local Ollama HTTP API). No extra install step is required for this repo.

## How to run

```bash
source venv/bin/activate
streamlit run app.py
```

1. Confirm the sidebar shows a detected local vision model.
2. Point **Folder path** at `sample_data/` (default) or drag files into the uploader.
3. Choose **overwrite** (replace rows for the same filename) or **append** (skip filenames already in the table).
4. Click **Extract receipts**. CPU-only machines can take a long time per page — there is a progress bar.
5. Fix mistakes in the editable table. Rows with parse failures (`needs_review`) or **low** confidence are flagged in the first column.
6. **Download CSV**, or write it to the path in the sidebar.

Synthetic placeholders live in `sample_data/` so you can try the pipeline without real financial documents. Regenerate them with:

```bash
python sample_data/generate.py
```




<img width="705" height="541" alt="image" src="https://github.com/user-attachments/assets/a1e9f79a-f894-48ca-ad44-b1679f8ee9d8" />




## Known limitations

- **Handwriting**, **faded thermal receipts**, and **creased photos** often come back low-confidence or `needs_review`.
- **Non-English** documents may need a different (or larger) vision model than `qwen2.5vl:7b`.
- Dates like `03/04/2026` are **ambiguous** (US vs rest-of-world). The original string is kept and `date_ambiguous` is set.
- Multi-page PDFs produce **one row per page**. Totals that span pages may be incomplete.
- Extraction quality is bounded by the local model; always spot-check the table before using the CSV for bookkeeping.
- Very large images or long PDFs will be slow on CPU.

## Design decisions

- **No `requirements.txt`**: the environment is pre-provisioned; versions are not pinned here.
- **PyMuPDF (`fitz`)**, not pdf2image/poppler, rasterizes PDF pages at ~150 DPI.
- **Ollama is called over `http://127.0.0.1:11434`** with `requests` (`/api/tags`, `/api/chat`). That is the same local API the `ollama` Python package wraps. Images are sent as base64 in the chat payload and never leave the host.
- **JSON-only prompt** plus Ollama `format: json` and `temperature: 0`. One retry appends “return ONLY valid JSON, no markdown fences.” A second failure writes `status=needs_review` and stores `raw_response` instead of dropping the file.
- **Dedupe key is `source_filename`** (not page). Overwrite replaces every row for that name; append skips the whole file if any row already exists.
- Extra CSV columns (`page`, `confidence`, `status`, `date_normalized`, `date_ambiguous`, `notes`, `raw_response`) are kept so review is possible; the download still includes the requested fields.
- Sample receipts are **synthetic PIL/PyMuPDF drawings**, not scans of real purchases.

## Assumptions

- Ollama listens on `127.0.0.1:11434` (the default).
- Folder scans are non-recursive unless you enable **Scan subfolders**.
- “Amount” means the **grand total**, not line items or tax-only.
- Category set is fixed: Meals, Travel, Software, Office Supplies, Professional Services, Utilities, Other.
