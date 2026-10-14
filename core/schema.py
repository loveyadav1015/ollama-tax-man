from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

CATEGORIES = (
    "Meals",
    "Travel",
    "Software",
    "Office Supplies",
    "Professional Services",
    "Utilities",
    "Other",
)

CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}
CONFIDENCE_VALUES = ("high", "medium", "low")


class ExtractionPayload(BaseModel):
    """JSON object the vision model is asked to return."""

    vendor: str = ""
    date: str = ""
    amount: Optional[float] = None
    currency: str = ""
    category: str = "Other"
    confidence: Literal["high", "medium", "low"] = "low"
    notes: str = ""

    @field_validator("vendor", "date", "currency", "notes", mode="before")
    @classmethod
    def stringify(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("amount", mode="before")
    @classmethod
    def parse_amount(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        text = re.sub(r"[^\d.\-]", "", text)
        if not text or text in {".", "-", "-."}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            return "Other"
        text = str(value).strip()
        mapped = CATEGORY_LOOKUP.get(text.lower())
        if mapped:
            return mapped
        lowered = text.lower()
        for category in CATEGORIES:
            if category.lower() in lowered or lowered in category.lower():
                return category
        return "Other"

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        text = str(value or "low").strip().lower()
        if text in CONFIDENCE_VALUES:
            return text
        return "low"

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z]", "", value).upper()
        if len(cleaned) == 3:
            return cleaned
        symbols = {
            "$": "USD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
        }
        for symbol, code in symbols.items():
            if symbol in value:
                return code
        return cleaned or value.upper()


class ReceiptRow(BaseModel):
    source_filename: str
    page: int = 1
    vendor: str = ""
    date: str = ""
    date_normalized: str = ""
    date_ambiguous: bool = False
    amount: Optional[float] = None
    currency: str = ""
    category: str = "Other"
    confidence: str = "low"
    status: str = "ok"
    notes: str = ""
    raw_response: str = ""
    flag: str = ""

    @model_validator(mode="after")
    def set_flag(self) -> "ReceiptRow":
        if self.status == "needs_review":
            self.flag = "🔴 needs review"
        elif self.confidence == "low":
            self.flag = "🟡 low confidence"
        elif self.confidence == "medium":
            self.flag = "🟠 medium"
        else:
            self.flag = "🟢 ok"
        return self

    def to_table_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "source_filename": self.source_filename,
            "page": self.page,
            "vendor": self.vendor,
            "date": self.date,
            "date_normalized": self.date_normalized,
            "date_ambiguous": self.date_ambiguous,
            "amount": self.amount,
            "currency": self.currency,
            "category": self.category,
            "confidence": self.confidence,
            "status": self.status,
            "notes": self.notes,
            "raw_response": self.raw_response,
        }


CSV_COLUMNS = [
    "flag",
    "source_filename",
    "page",
    "vendor",
    "date",
    "date_normalized",
    "date_ambiguous",
    "amount",
    "currency",
    "category",
    "confidence",
    "status",
    "notes",
    "raw_response",
]
