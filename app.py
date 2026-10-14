from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.csv_writer import frame_to_csv_bytes, merge_frames, rows_to_frame, write_csv
from core.extractor import OllamaUnavailable, PULL_HINT, check_ollama, extract_page
from core.file_loader import (
    SUPPORTED_SUFFIXES,
    describe_batch,
    iter_folder,
    load_bytes,
    load_path,
)
from core.schema import CATEGORIES, CSV_COLUMNS, ReceiptRow

st.set_page_config(
    page_title="Receipt & Invoice Sorter",
    page_icon="🧾",
    layout="wide",
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

if "results" not in st.session_state:
    st.session_state.results = rows_to_frame([])
if "editor_nonce" not in st.session_state:
    st.session_state.editor_nonce = 0


def main() -> None:
    st.title("Receipt & Invoice Sorter")
    st.caption(
        "Local-first extraction via Ollama. Receipts and invoices stay on this machine — "
        "nothing is sent to a cloud API."
    )
    st.info(
        "Vision models can take **several seconds to minutes per page** on CPU-only "
        "machines. Leave this tab open while a batch runs."
    )

    status = check_ollama()
    with st.sidebar:
        st.header("Ollama")
        if status.ok:
            if status.vision:
                st.success(status.message)
            else:
                st.warning(status.message)
            if status.available:
                st.caption("Installed: " + ", ".join(status.available))
        else:
            st.error(status.message)

        st.header("Input")
        default_folder = str(SAMPLE_DIR) if SAMPLE_DIR.is_dir() else ""
        folder = st.text_input("Folder path", value=default_folder)
        recursive = st.checkbox("Scan subfolders", value=False)
        uploaded = st.file_uploader(
            "Or drag files here",
            type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
            accept_multiple_files=True,
        )

        st.header("Duplicates")
        mode = st.radio(
            "If a filename was already processed",
            options=("overwrite", "append"),
            format_func=lambda value: (
                "Overwrite rows for that filename"
                if value == "overwrite"
                else "Skip filenames already in the table"
            ),
            index=0,
        )

        st.header("Save")
        save_path = st.text_input(
            "Optional CSV path on disk",
            value=str(Path(__file__).resolve().parent / "output" / "receipts.csv"),
        )

    if not status.ok:
        st.error(status.message)
        st.code("ollama pull qwen2.5vl:7b")
        st.code("ollama pull llama3.2-vision")
        st.stop()

    run = st.button("Extract receipts", type="primary", disabled=not status.ok)

    if run:
        try:
            pages = _collect_pages(folder, recursive, uploaded)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        if not pages:
            st.warning("No supported images or PDFs found. Add files or use sample_data/.")
        else:
            st.write(f"Processing {describe_batch(pages)} with `{status.model}`.")
            if not status.vision:
                st.warning(
                    "Text-only test model: images are not sent. Pull `qwen2.5vl:7b` for real OCR."
                )
            rows: list[ReceiptRow] = []
            bar = st.progress(0.0, text="Starting…")
            try:
                for index, page in enumerate(pages, start=1):
                    bar.progress(
                        (index - 1) / len(pages),
                        text=f"Page {index}/{len(pages)} — {page.source_filename} (p.{page.page})",
                    )
                    rows.append(
                        extract_page(page, model=status.model, vision=status.vision)
                    )
                bar.progress(1.0, text="Done.")
            except OllamaUnavailable as exc:
                st.error(str(exc))
                st.code("ollama pull qwen2.5vl:7b")
                st.stop()

            incoming = rows_to_frame(rows)
            merged, counted = merge_frames(st.session_state.results, incoming, mode)
            st.session_state.results = merged
            st.session_state.editor_nonce += 1
            if mode == "append" and counted:
                st.warning(
                    f"Skipped {counted} page(s) whose filename was already in the table."
                )
            elif mode == "overwrite" and counted:
                st.info(f"Replaced {counted} existing row(s) for overlapping filenames.")

    _render_table(save_path)


def _collect_pages(folder: str, recursive: bool, uploaded) -> list:
    pages = []
    seen_names: set[str] = set()

    if uploaded:
        for item in uploaded:
            pages.extend(load_bytes(item.name, item.getvalue()))
            seen_names.add(item.name)

    folder_path = Path(folder).expanduser() if folder.strip() else None
    if folder_path and folder_path.is_dir():
        for path in iter_folder(folder_path, recursive=recursive):
            if path.name in seen_names:
                continue
            pages.extend(load_path(path))
    elif folder.strip() and not uploaded:
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    return pages


def _render_table(save_path: str) -> None:
    frame: pd.DataFrame = st.session_state.results
    if frame.empty:
        st.write("No rows yet. Point at a folder (sample_data/ is filled in) and extract.")
        return

    needs = int((frame["status"] == "needs_review").sum()) if "status" in frame.columns else 0
    low = int((frame["confidence"] == "low").sum()) if "confidence" in frame.columns else 0
    if needs or low:
        st.warning(
            f"{needs} row(s) marked **needs_review** and {low} with **low** confidence. "
            "Those rows are flagged in the first column — edit them before export."
        )

    st.subheader("Results")
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "flag": st.column_config.TextColumn("flag", disabled=True, width="medium"),
            "source_filename": st.column_config.TextColumn("source_filename", disabled=True),
            "page": st.column_config.NumberColumn("page", min_value=1, step=1),
            "amount": st.column_config.NumberColumn("amount", format="%.2f"),
            "category": st.column_config.SelectboxColumn("category", options=list(CATEGORIES)),
            "confidence": st.column_config.SelectboxColumn(
                "confidence", options=["high", "medium", "low"]
            ),
            "status": st.column_config.SelectboxColumn(
                "status", options=["ok", "needs_review"]
            ),
            "date_ambiguous": st.column_config.CheckboxColumn("date_ambiguous"),
            "raw_response": st.column_config.TextColumn("raw_response", width="large"),
        },
        key=f"results_editor_{st.session_state.editor_nonce}",
    )
    st.session_state.results = edited.reindex(columns=CSV_COLUMNS)

    csv_bytes = frame_to_csv_bytes(st.session_state.results)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="receipts.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("Write CSV to disk"):
            path = Path(save_path).expanduser()
            write_csv(st.session_state.results, path)
            st.success(f"Wrote {path}")

    if st.button("Clear table"):
        st.session_state.results = rows_to_frame([])
        st.session_state.editor_nonce += 1
        st.rerun()


if __name__ == "__main__":
    main()
