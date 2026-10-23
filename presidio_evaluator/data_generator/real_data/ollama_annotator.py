"""Pre-annotate PII in text with a local LLM served by Ollama (e.g. Gemma).

The LLM is only asked for entity *values*; character offsets are computed by ``locate_spans``,
so a hallucinated value never becomes a span. The output is a draft for human review, not gold.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import requests

from presidio_evaluator.data_objects import Span

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"

# Entity type -> description given to the LLM. Labels match the synthetic datasets.
DEFAULT_ENTITIES = {
    "PERSON": "Full or partial name of a person (customer or agent)",
    "TITLE": "Honorific title such as Mr, Mrs, Ms, Dr",
    "EMAIL_ADDRESS": "Email address",
    "PHONE_NUMBER": "Phone number",
    "STREET_ADDRESS": "Street address, house number or flat",
    "ZIP_CODE": "Postcode",
    "GPE": "City, town, county or country",
    "DATE_TIME": "Date of birth or other specific date or time",
    "AGE": "Age of a person",
    "ORGANIZATION": "Company or organisation name, e.g. employer or bank",
    "CREDIT_CARD": "Payment card number",
    "IBAN_CODE": "IBAN",
    "ACCOUNT_NUMBER": "Bank account number, sort code or loan/agreement reference",
    "UK_DRIVING_LICENCE": "UK driving licence number",
    "UK_NINO": "UK National Insurance number",
    "UK_VEHICLE_REGISTRATION": "Vehicle registration plate, e.g. AB12 CDE",
}

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """You annotate personally identifiable information (PII) in customer \
support chat transcripts of a UK car finance company.
Return every PII value in the text with one of these entity types:
{entity_descriptions}

Rules:
- "text" must be copied verbatim from the input, character for character.
- Annotate the value only, without surrounding words or punctuation.
- Lines start with a speaker label such as "Speaker 1:" or "agent:"; never annotate it.
- Do not annotate generic words (e.g. "car", "loan", "today") or company products.
- If there is no PII, return an empty list."""


def _response_schema(entity_types: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string", "enum": entity_types},
                        "text": {"type": "string"},
                    },
                    "required": ["entity_type", "text"],
                },
            },
        },
        "required": ["entities"],
    }


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """True if text[start:end] is not glued to letters/digits on either side."""
    before_ok = start == 0 or not (text[start - 1].isalnum() and text[start].isalnum())
    after_ok = end == len(text) or not (text[end - 1].isalnum() and text[end].isalnum())
    return before_ok and after_ok


def locate_spans(
    text: str,
    llm_entities: list[dict],
    entity_types: list[str] | None = None,
) -> list[Span]:
    """Turn LLM ``{"entity_type", "text"}`` items into Spans with character offsets.

    Every whole-word occurrence of a value is annotated. Values not found in the text and
    unknown entity types are dropped. Overlapping candidates are resolved by keeping the longest.
    """
    candidates = set()
    for entity in llm_entities:
        entity_type = entity.get("entity_type")
        value = (entity.get("text") or "").strip()
        if not value or (entity_types is not None and entity_type not in entity_types):
            logger.debug("Dropping entity %r", entity)
            continue
        found = False
        for match in re.finditer(re.escape(value), text):
            if _is_word_boundary(text, match.start(), match.end()):
                candidates.add((match.start(), match.end(), entity_type))
                found = True
        if not found:
            logger.debug("Dropping entity not found in text: %r", entity)

    accepted = []
    for start, end, entity_type in sorted(
        candidates,
        key=lambda c: (-(c[1] - c[0]), c[0], c[2]),
    ):
        if all(end <= a_start or start >= a_end for a_start, a_end, _ in accepted):
            accepted.append((start, end, entity_type))
        else:
            logger.debug("Dropping overlapping %s %r", entity_type, text[start:end])

    return [
        Span(
            entity_type=entity_type,
            entity_value=text[start:end],
            start_position=start,
            end_position=end,
        )
        for start, end, entity_type in sorted(accepted)
    ]


class OllamaAnnotator:
    """Annotate PII spans with an Ollama-served LLM.

    :param model: Ollama model name
    :param host: Ollama server URL
    :param entities: Entity type -> description given to the LLM
    :param cache_dir: If set, raw LLM answers are cached here by model, prompt and text
    :param max_retries: Extra attempts on a malformed answer; after that no spans are returned
    :param timeout: Request timeout in seconds
    :param think: Ollama ``think`` flag, None to omit it. Off by default: with thinking on,
    gemma4 sometimes answers a JSON-schema request by echoing the input
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        entities: dict[str, str] | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 2,
        timeout: float = 300,
        think: bool | None = False,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.entities = entities if entities is not None else DEFAULT_ENTITIES
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.timeout = timeout
        self.think = think
        self.system_prompt = SYSTEM_PROMPT.format(
            entity_descriptions="\n".join(
                f"- {entity_type}: {description}"
                for entity_type, description in self.entities.items()
            ),
        )

    def annotate(self, text: str) -> list[Span]:
        """Return the PII spans the LLM finds in ``text``."""
        return locate_spans(text, self.extract_entities(text), list(self.entities))

    def extract_entities(self, text: str) -> list[dict]:
        """Return the raw ``{"entity_type", "text"}`` items from the LLM (cached)."""
        cache_file = self._cache_file(text)
        if cache_file is not None and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        for attempt in range(self.max_retries + 1):
            try:
                entities = self._parse(self._chat(text))
                break
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(
                    "Malformed answer from %s (attempt %d): %s",
                    self.model,
                    attempt + 1,
                    e,
                )
        else:
            # Not cached, so the text is retried on the next run
            return []

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(entities), encoding="utf-8")
        return entities

    def _chat(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "format": _response_schema(list(self.entities)),
            "stream": False,
            "options": {"temperature": 0},
        }
        if self.think is not None:
            payload["think"] = self.think
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @staticmethod
    def _parse(content: str) -> list[dict]:
        entities = json.loads(content)["entities"]
        if not isinstance(entities, list) or not all(
            isinstance(entity, dict) for entity in entities
        ):
            raise ValueError(f"Expected a list of objects, got {entities!r}")
        return entities

    def _cache_file(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            "\0".join([self.model, PROMPT_VERSION, self.system_prompt, text]).encode(),
        ).hexdigest()
        return self.cache_dir / f"{key}.json"
