"""Load real chat transcripts from the Snowflake DWH and split them into annotation windows.

The output contains real customer PII: keep it (and anything derived from it) in the gitignored
``/data`` folder and do not share it outside approved environments.

Requires the optional ``snowflake`` extra: ``uv sync --extra snowflake``.
"""

from typing import Any

import pandas as pd
from pandas import DataFrame

from presidio_evaluator.data_generator.faker_extensions.snowflake_records import (
    connect_to_snowflake,
)

# Every chat query returns these columns, one row per message
CHAT_COLUMNS = ["conversation_id", "message_index", "speaker", "text"]

# Samples whole tickets, then returns all their messages
ZENDESK_CHAT_QUERY = """
WITH sampled AS (
    SELECT zendesk_platform_ticket_id
    FROM source_zendesk.zendesk_platform_ticket_messages_pii
    WHERE message IS NOT NULL
    GROUP BY zendesk_platform_ticket_id
    ORDER BY RANDOM()
    LIMIT %(limit)s
)

SELECT
    m.zendesk_platform_ticket_id AS conversation_id
    , m.message_sequence_number AS message_index
    , m.message_username AS speaker
    , m.message AS text
FROM source_zendesk.zendesk_platform_ticket_messages_pii m
INNER JOIN sampled s
    ON m.zendesk_platform_ticket_id = s.zendesk_platform_ticket_id
WHERE m.message IS NOT NULL
"""

# Samples whole interactions, then returns all their messages
TALKDESK_CHAT_QUERY = """
WITH sampled AS (
    SELECT interaction_id
    FROM source_talkdesk.talkdesk_interaction_messages_pii
    WHERE text IS NOT NULL
    GROUP BY interaction_id
    ORDER BY RANDOM()
    LIMIT %(limit)s
)

SELECT
    m.interaction_id AS conversation_id
    , ROW_NUMBER() OVER (
        PARTITION BY m.interaction_id
        ORDER BY m.message_start_at, m.recording_index, m.message_index
    ) AS message_index
    , m.participant AS speaker
    , m.text AS text
FROM source_talkdesk.talkdesk_interaction_messages_pii m
INNER JOIN sampled s
    ON m.interaction_id = s.interaction_id
WHERE m.text IS NOT NULL
"""

CHAT_QUERIES = {"zendesk": ZENDESK_CHAT_QUERY, "talkdesk": TALKDESK_CHAT_QUERY}

# Talkdesk speakers are roles (agent/customer); Zendesk speakers are usernames, i.e. PII
ANONYMISE_SPEAKERS = {"zendesk": True, "talkdesk": False}


def load_snowflake_chat_messages(
    query: str = ZENDESK_CHAT_QUERY,
    limit: int = 100,
    connection=None,
    **connect_kwargs: Any,
) -> DataFrame:
    """Sample chat conversations from the Snowflake DWH.

    :param query: SQL returning ``CHAT_COLUMNS``; may reference ``%(limit)s``
    :param limit: Number of conversations to sample, passed to the query as ``%(limit)s``
    :param connection: An open Snowflake connection. If None, one is opened via
    ``connect_to_snowflake(**connect_kwargs)`` and closed afterwards
    :return: A DataFrame with one row per message and ``CHAT_COLUMNS`` columns
    """
    owns_connection = connection is None
    if owns_connection:
        connection = connect_to_snowflake(**connect_kwargs)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(query, {"limit": limit})
            columns = [column[0].lower() for column in cursor.description]
            messages_df = pd.DataFrame(cursor.fetchall(), columns=columns)
        finally:
            cursor.close()
    finally:
        if owns_connection:
            connection.close()

    missing = set(CHAT_COLUMNS) - set(messages_df.columns)
    if missing:
        raise ValueError(
            f"Chat query must return columns {CHAT_COLUMNS}, missing {missing}"
        )
    return messages_df[CHAT_COLUMNS]


def chat_messages_to_windows(
    messages_df: DataFrame,
    window_size: int = 6,
    source: str = "zendesk",
    anonymise_speakers: bool = True,
) -> list[dict]:
    """Group chat messages into windows of consecutive messages from one conversation.

    Each window is rendered as ``speaker: text`` lines. With ``anonymise_speakers``, speakers
    are replaced by ``Speaker 1``, ``Speaker 2``, ... in order of first appearance in the
    conversation, so usernames are not part of the text to annotate.

    :return: A list of ``{"text", "conversation_id", "window_index", "source"}`` dicts
    """
    windows = []
    messages_df = messages_df.sort_values(["conversation_id", "message_index"])
    for conversation_id, conversation in messages_df.groupby(
        "conversation_id",
        sort=False,
    ):
        speakers = {}
        lines = []
        for raw_speaker, text in zip(conversation["speaker"], conversation["text"]):
            if not isinstance(text, str) or not text.strip():
                continue
            speaker = raw_speaker
            if anonymise_speakers:
                speaker = speakers.setdefault(
                    raw_speaker, f"Speaker {len(speakers) + 1}"
                )
            lines.append(f"{speaker}: {' '.join(text.split())}")

        for window_index, start in enumerate(range(0, len(lines), window_size)):
            windows.append(
                {
                    "text": "\n".join(lines[start : start + window_size]),
                    "conversation_id": str(conversation_id),
                    "window_index": window_index,
                    "source": source,
                },
            )
    return windows
