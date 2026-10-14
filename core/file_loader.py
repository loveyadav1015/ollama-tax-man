from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import pymupdf as fitz
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIX = ".pdf"
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {PDF_SUFFIX}

# 150 DPI is a good tradeoff for VLM OCR vs. payload size.
PDF_ZOOM = 150 / 72


@dataclass(frozen=True)
class DocumentPage:
    source_filename: str
    source_path: Path | None
    page: int
    image: Image.Image

    def png_bytes(self) -> bytes:
        image = self.image
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_folder(folder: Path, recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(folder.glob(pattern))
        if path.is_file() and is_supported(path)
    ]
    return files


def load_path(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        return _load_pdf(path, source_filename=path.name)
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return [
                DocumentPage(
                    source_filename=path.name,
                    source_path=path,
                    page=1,
                    image=rgb.copy(),
                )
            ]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_bytes(filename: str, data: bytes) -> list[DocumentPage]:
    suffix = Path(filename).suffix.lower()
    if suffix == PDF_SUFFIX:
        return _pdf_pages_from_bytes(data, source_filename=filename)
    if suffix in IMAGE_SUFFIXES or not suffix:
        with Image.open(BytesIO(data)) as image:
            rgb = image.convert("RGB")
            return [
                DocumentPage(
                    source_filename=filename,
                    source_path=None,
                    page=1,
                    image=rgb.copy(),
                )
            ]
    raise ValueError(f"Unsupported file type: {filename}")


def load_many(paths: Sequence[Path]) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for path in paths:
        pages.extend(load_path(path))
    return pages


def _load_pdf(path: Path, source_filename: str) -> list[DocumentPage]:
    document = fitz.open(path)
    try:
        return _pages_from_pdf(document, source_filename=source_filename, source_path=path)
    finally:
        document.close()


def _pdf_pages_from_bytes(data: bytes, source_filename: str) -> list[DocumentPage]:
    document = fitz.open(stream=data, filetype="pdf")
    try:
        return _pages_from_pdf(document, source_filename=source_filename, source_path=None)
    finally:
        document.close()


def _pages_from_pdf(
    document: fitz.Document,
    source_filename: str,
    source_path: Path | None,
) -> list[DocumentPage]:
    matrix = fitz.Matrix(PDF_ZOOM, PDF_ZOOM)
    pages: list[DocumentPage] = []
    for index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        pages.append(
            DocumentPage(
                source_filename=source_filename,
                source_path=source_path,
                page=index,
                image=image,
            )
        )
    if not pages:
        raise ValueError(f"PDF has no pages: {source_filename}")
    return pages


def describe_batch(pages: Iterable[DocumentPage]) -> str:
    page_list = list(pages)
    files = {page.source_filename for page in page_list}
    return f"{len(files)} file(s), {len(page_list)} page(s)"
