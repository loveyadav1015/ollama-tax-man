from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from core.schema import CSV_COLUMNS, ReceiptRow

MergeMode = Literal["overwrite", "append"]


def rows_to_frame(rows: Iterable[ReceiptRow]) -> pd.DataFrame:
    records = [row.to_table_dict() for row in rows]
    if not records:
        return pd.DataFrame(columns=CSV_COLUMNS)
    frame = pd.DataFrame.from_records(records)
    for column in CSV_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CSV_COLUMNS]


def merge_frames(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    mode: MergeMode,
) -> tuple[pd.DataFrame, int]:
    """Merge by source_filename. Returns (result, skipped_or_replaced_count)."""
    if incoming is None or incoming.empty:
        return _ensure_columns(existing), 0
    incoming = _ensure_columns(incoming)
    if existing is None or existing.empty:
        return incoming.reset_index(drop=True), 0

    existing = _ensure_columns(existing)
    names = set(incoming["source_filename"].astype(str))
    if mode == "overwrite":
        kept = existing[~existing["source_filename"].astype(str).isin(names)]
        replaced = len(existing) - len(kept)
        return pd.concat([kept, incoming], ignore_index=True), replaced

    existing_names = set(existing["source_filename"].astype(str))
    added = incoming[~incoming["source_filename"].astype(str).isin(existing_names)]
    skipped = len(incoming) - len(added)
    return pd.concat([existing, added], ignore_index=True), skipped


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    export = _export_columns(frame)
    return export.to_csv(index=False).encode("utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _export_columns(frame).to_csv(path, index=False)


def _export_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_columns(frame)
    return frame.drop(columns=["flag"], errors="ignore")


def _ensure_columns(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)
    out = frame.copy()
    for column in CSV_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[CSV_COLUMNS]
