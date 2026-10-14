"""Generate synthetic placeholder receipts (not real financial documents)."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SIZE = (700, 900)
BG = (252, 250, 244)
INK = (32, 32, 32)
MUTED = (90, 90, 90)
RULE = (180, 170, 150)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_receipt(lines: list[tuple[str, int, bool, tuple[int, int, int]]]) -> Image.Image:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, SIZE[0] - 24, SIZE[1] - 24), outline=RULE, width=2)
    y = 56
    for text, size, bold, color in lines:
        if text == "---":
            draw.line((60, y, SIZE[0] - 60, y), fill=RULE, width=1)
            y += 24
            continue
        font = _font(size, bold=bold)
        draw.text((60, y), text, fill=color, font=font)
        y += size + 14
    return image


def generate() -> list[Path]:
    ROOT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    coffee = _draw_receipt(
        [
            ("NORTH BEAN COFFEE", 36, True, INK),
            ("123 Harbor Street", 20, False, MUTED),
            ("Receipt #NB-1042", 20, False, MUTED),
            ("---", 16, False, RULE),
            ("Date: 2026-03-12", 22, False, INK),
            ("Item: Oat latte x2", 22, False, INK),
            ("Item: Blueberry muffin", 22, False, INK),
            ("---", 16, False, RULE),
            ("Subtotal: 14.10", 22, False, INK),
            ("Tax: 1.13", 22, False, INK),
            ("TOTAL: 15.23 USD", 28, True, INK),
            ("Thank you — dine in", 18, False, MUTED),
        ]
    )
    coffee_path = ROOT / "north_bean_coffee.png"
    coffee.save(coffee_path)
    written.append(coffee_path)

    software = _draw_receipt(
        [
            ("PIXELFORGE APPS", 34, True, INK),
            ("INVOICE", 28, True, INK),
            ("Invoice #: PF-8891", 20, False, MUTED),
            ("Bill to: Local Workshop LLC", 20, False, MUTED),
            ("---", 16, False, RULE),
            ("Date: 04/18/2026", 22, False, INK),
            ("Subscription: Design Suite Pro", 22, False, INK),
            ("Period: Apr 2026", 22, False, INK),
            ("---", 16, False, RULE),
            ("Amount due: 49.00", 26, True, INK),
            ("Currency: USD", 22, False, INK),
            ("Category hint: Software", 18, False, MUTED),
        ]
    )
    software_path = ROOT / "pixelforge_invoice.png"
    software.save(software_path)
    written.append(software_path)

    pdf_path = ROOT / "willow_travel_invoice.pdf"
    document = fitz.open()
    page = document.new_page(width=420, height=594)
    blocks = [
        (50, 50, "WILLOW TRAVEL CO.", 18),
        (50, 78, "Invoice", 14),
        (50, 110, "Invoice #: WT-2207", 11),
        (50, 130, "Date: 12 May 2026", 11),
        (50, 150, "Passenger: A. Example", 11),
        (50, 190, "Route: LIS -> AMS  (train)", 11),
        (50, 210, "Seat: Coach 4", 11),
        (50, 250, "Fare: 86.40 EUR", 12),
        (50, 272, "Booking fee: 2.00 EUR", 11),
        (50, 310, "TOTAL: 88.40 EUR", 14),
        (50, 350, "Placeholder synthetic document — not a real ticket.", 9),
    ]
    for x, y, text, size in blocks:
        page.insert_text((x, y), text, fontsize=size, color=(0.12, 0.12, 0.12))
    document.save(pdf_path)
    document.close()
    written.append(pdf_path)
    return written


if __name__ == "__main__":
    for path in generate():
        print(path)
