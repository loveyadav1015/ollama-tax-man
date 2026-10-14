# Receipt & Invoice Sorter

Local-first tool that takes a folder of receipt/invoice images or PDFs and
outputs a clean CSV with vendor, date, amount, currency, category, and source
filename — using a local vision-language model via Ollama, so no financial
document ever leaves the user's machine.

## Tech stack

- Python 3.10+
- Ollama running locally, model: qwen2.5vl:7b (fallback to llama3.2-vision)
- PyMuPDF (`fitz`) to rasterize PDF pages to images
- pandas for CSV output
- Streamlit for the UI
- No cloud API calls. No telemetry.

## Core behavior

1. User points the app at a folder (or drags files into Streamlit).
2. For each file:
   - If PDF, convert each page to an image.
   - Send image to Ollama vision model with a strict JSON-only extraction prompt.
   - Parse the JSON response. If parsing fails, retry once with a
     "return ONLY valid JSON, no markdown fences" correction appended.
   - If it still fails, record the row with status="needs_review" and
     raw_response saved for debugging, rather than crashing or skipping.
3. Normalize dates to YYYY-MM-DD where possible; leave as-is with a flag
   if ambiguous.
4. Append all results to a pandas DataFrame, show as an editable table in
   Streamlit (`st.data_editor`) so the user can fix mistakes before export.
5. Download button exports CSV.
6. Running the tool on the same folder twice should not duplicate rows —
   dedupe by filename, or let the user choose "overwrite" vs "append" in
   the UI.

## Extraction JSON shape

```json
{
  "vendor": "string",
  "date": "string",
  "amount": 0,
  "currency": "string",
  "category": "Meals | Travel | Software | Office Supplies | Professional Services | Utilities | Other",
  "confidence": "high | medium | low",
  "notes": "string"
}
```
