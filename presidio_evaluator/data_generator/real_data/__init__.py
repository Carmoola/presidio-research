"""Build evaluation datasets from real data: LLM pre-annotation followed by human review."""

from .label_studio import (
    from_label_studio_export,
    label_config,
    mask_text,
    prediction_agreement,
    to_input_samples,
    to_label_studio_tasks,
)
from .ollama_annotator import DEFAULT_ENTITIES, OllamaAnnotator, locate_spans
from .snowflake_chats import (
    ANONYMISE_SPEAKERS,
    CHAT_QUERIES,
    chat_messages_to_windows,
    load_snowflake_chat_messages,
)

__all__ = [
    "ANONYMISE_SPEAKERS",
    "CHAT_QUERIES",
    "DEFAULT_ENTITIES",
    "OllamaAnnotator",
    "chat_messages_to_windows",
    "from_label_studio_export",
    "label_config",
    "load_snowflake_chat_messages",
    "locate_spans",
    "mask_text",
    "prediction_agreement",
    "to_input_samples",
    "to_label_studio_tasks",
]
