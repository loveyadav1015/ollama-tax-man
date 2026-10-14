from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests
from dateutil import parser as date_parser

from core.file_loader import DocumentPage
from core.schema import ExtractionPayload, ReceiptRow

OLLAMA_HOST = "http://127.0.0.1:11434"
VISION_MODELS = ("qwen2.5vl:7b", "qwen2.5vl", "llama3.2-vision")
# Text-only models allowed so the pipeline can be exercised without a VLM pulled.
TESTING_MODELS = ("qwen2.5-coder",)
PULL_HINT = (
    "Start Ollama (`ollama serve` if it is not already running), then pull a vision model:\n\n"
    "`ollama pull qwen2.5vl:7b`\n\n"
    "Fallback if that model is unavailable:\n\n"
    "`ollama pull llama3.2-vision`"
)

EXTRACT_PROMPT = """You are extracting structured fields from a receipt or invoice image.
Return ONLY a single JSON object with this exact shape and no other text, no markdown, no code fences:

{
  "vendor": string,
  "date": string,
  "amount": number,
  "currency": string,
  "category": string,
  "confidence": "high" | "medium" | "low",
  "notes": string
}

Rules:
- vendor: merchant or issuer name as printed.
- date: the document date exactly as seen.
- amount: the total amount due / grand total as a number (no currency symbol).
- currency: ISO 4217 code if determinable (USD, EUR, INR, ...), otherwise your best guess.
- category: exactly one of: Meals, Travel, Software, Office Supplies, Professional Services, Utilities, Other.
- confidence: high if the key fields are clearly readable, medium if partly inferred, low if guessed.
- notes: anything unclear, e.g. "amount partially obscured". Empty string if nothing to note.
- If a field cannot be read, use "" for strings, null for amount, confidence "low", and explain in notes.
"""

RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. "
    "Return ONLY valid JSON, no markdown fences, no commentary."
)

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OllamaUnavailable(Exception):
    """Ollama is not reachable or no suitable vision model is pulled."""


@dataclass
class OllamaStatus:
    ok: bool
    model: Optional[str]
    available: list[str]
    message: str
    vision: bool = False


def check_ollama(host: str = OLLAMA_HOST) -> OllamaStatus:
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return OllamaStatus(
            ok=False,
            model=None,
            available=[],
            message=(
                "Ollama is not running or is not reachable at "
                f"{host}. {PULL_HINT}\n\nDetails: {exc}"
            ),
        )

    names, capabilities = _model_listing(payload)
    chosen = pick_model(names)
    if not chosen:
        listed = ", ".join(names) if names else "(none)"
        return OllamaStatus(
            ok=False,
            model=None,
            available=names,
            message=(
                "Ollama is running, but no supported model is available. "
                f"Installed models: {listed}.\n\n{PULL_HINT}"
            ),
        )
    vision = _is_vision_model(chosen, capabilities.get(chosen, set()))
    if vision:
        message = f"Using local vision model `{chosen}`."
    else:
        message = (
            f"Using text-only model `{chosen}` for testing. "
            "It cannot read receipt images — rows will be low-confidence guesses "
            "from the filename. Pull `qwen2.5vl:7b` for real extraction."
        )
    return OllamaStatus(
        ok=True,
        model=chosen,
        available=names,
        message=message,
        vision=vision,
    )


def pick_model(available: list[str]) -> Optional[str]:
    return _match(available, *VISION_MODELS) or _match(available, *TESTING_MODELS)


def _match(available: list[str], *prefixes: str) -> Optional[str]:
    for prefix in prefixes:
        needle = prefix.lower()
        for name in available:
            low = name.lower()
            if low == needle or low.startswith(needle + ":"):
                return name
    return None


def _is_vision_model(name: str, caps: set[str]) -> bool:
    if "vision" in caps:
        return True
    low = name.lower()
    return any(low == p or low.startswith(p + ":") for p in VISION_MODELS)


def extract_page(
    page: DocumentPage,
    model: str,
    host: str = OLLAMA_HOST,
    timeout: int = 600,
    vision: bool = True,
) -> ReceiptRow:
    raw, parsed = _chat_for_payload(
        page, model, host=host, timeout=timeout, extra="", vision=vision
    )
    if parsed is None:
        raw_retry, parsed = _chat_for_payload(
            page,
            model,
            host=host,
            timeout=timeout,
            extra=RETRY_SUFFIX,
            vision=vision,
        )
        raw = raw_retry or raw

    if parsed is None:
        normalized, ambiguous = normalize_date("")
        return ReceiptRow(
            source_filename=page.source_filename,
            page=page.page,
            date_normalized=normalized,
            date_ambiguous=True if not normalized else ambiguous,
            status="needs_review",
            notes="Model response was not valid JSON after one retry.",
            raw_response=raw,
            confidence="low",
        )

    normalized, ambiguous = normalize_date(parsed.date)
    status = "ok"
    notes = parsed.notes
    if not parsed.vendor and parsed.amount is None:
        status = "needs_review"
        if notes:
            notes = f"{notes} | Missing vendor and amount."
        else:
            notes = "Missing vendor and amount."

    return ReceiptRow(
        source_filename=page.source_filename,
        page=page.page,
        vendor=parsed.vendor,
        date=parsed.date,
        date_normalized=normalized,
        date_ambiguous=ambiguous or not normalized,
        amount=parsed.amount,
        currency=parsed.currency,
        category=parsed.category,
        confidence=parsed.confidence,
        status=status,
        notes=notes,
        raw_response=raw,
    )


def normalize_date(raw: str) -> tuple[str, bool]:
    text = (raw or "").strip()
    if not text:
        return "", True
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        return text, False
    try:
        dayfirst = date_parser.parse(text, dayfirst=True, fuzzy=True, yearfirst=False)
        monthfirst = date_parser.parse(text, dayfirst=False, fuzzy=True, yearfirst=False)
    except (ValueError, OverflowError, TypeError):
        return "", True
    if dayfirst.date() != monthfirst.date():
        return "", True
    return dayfirst.strftime("%Y-%m-%d"), False


def _chat_for_payload(
    page: DocumentPage,
    model: str,
    host: str,
    timeout: int,
    extra: str,
    vision: bool,
) -> tuple[str, Optional[ExtractionPayload]]:
    prompt = EXTRACT_PROMPT
    if not vision:
        prompt += (
            "\n\nTESTING MODE: this model cannot view images. "
            f"Source filename: {page.source_filename}. Page: {page.page}. "
            "Do not claim you read the document. You may guess only from the filename. "
            'Set confidence to "low". notes must include "text-only test model; image not read".'
        )
    prompt += extra
    try:
        content = _chat(page, model, prompt, host=host, timeout=timeout, vision=vision)
    except OllamaUnavailable:
        raise
    except requests.RequestException as exc:
        raise OllamaUnavailable(
            f"Failed to call Ollama at {host}. {PULL_HINT}\n\nDetails: {exc}"
        ) from exc
    return content, parse_payload(content)


def _chat(
    page: DocumentPage,
    model: str,
    prompt: str,
    host: str,
    timeout: int,
    vision: bool,
) -> str:
    import base64

    message: dict[str, Any] = {"role": "user", "content": prompt}
    if vision:
        message["images"] = [base64.b64encode(page.png_bytes()).decode("ascii")]
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [message],
    }
    try:
        response = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    except requests.ConnectionError as exc:
        raise OllamaUnavailable(
            f"Ollama is not running or is not reachable at {host}. {PULL_HINT}"
        ) from exc

    if response.status_code == 404:
        raise OllamaUnavailable(
            f"Model `{model}` was not found. {PULL_HINT}"
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise OllamaUnavailable(
            f"Ollama returned HTTP {response.status_code}. {PULL_HINT}\n\n"
            f"{response.text[:500]}"
        ) from exc

    payload = response.json()
    error = payload.get("error")
    if error:
        raise OllamaUnavailable(f"Ollama error: {error}\n\n{PULL_HINT}")
    message = payload.get("message") or {}
    return str(message.get("content") or "")


def parse_payload(text: str) -> Optional[ExtractionPayload]:
    blob = _extract_json_object(text)
    if blob is None:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ExtractionPayload.model_validate(data)
    except Exception:
        return None


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = JSON_FENCE_RE.sub("", text.strip()).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return cleaned[start : end + 1]


def _model_listing(payload: Any) -> tuple[list[str], dict[str, set[str]]]:
    models = []
    if isinstance(payload, dict):
        models = payload.get("models") or []
    names: list[str] = []
    capabilities: dict[str, set[str]] = {}
    for item in models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model") or ""
            caps = item.get("capabilities") or []
            details = item.get("details") or {}
            if not caps and isinstance(details, dict):
                caps = details.get("capabilities") or []
        else:
            name = str(item)
            caps = []
        if name:
            names.append(name)
            capabilities[name] = {str(c).lower() for c in caps}
    return names, capabilities
