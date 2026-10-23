import json

import pandas as pd
import pytest

from presidio_evaluator.data_generator.real_data import (
    OllamaAnnotator,
    chat_messages_to_windows,
    from_label_studio_export,
    label_config,
    load_snowflake_chat_messages,
    locate_spans,
    mask_text,
    ollama_annotator,
    prediction_agreement,
    to_input_samples,
    to_label_studio_tasks,
)
from presidio_evaluator.data_objects import InputSample, Span


class FakeCursor:
    def __init__(self, rows, columns):
        self._rows = rows
        self.description = [(column,) for column in columns]
        self.executed = None

    def execute(self, query, params):
        self.executed = (query, params)

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def spans_as_tuples(spans):
    return [
        (s.entity_type, s.entity_value, s.start_position, s.end_position) for s in spans
    ]


def test_load_snowflake_chat_messages_lowercases_columns():
    cursor = FakeCursor(
        [(1, 1, "jo", "hi")],
        ["CONVERSATION_ID", "MESSAGE_INDEX", "SPEAKER", "TEXT"],
    )

    messages_df = load_snowflake_chat_messages(
        query="q",
        limit=3,
        connection=FakeConnection(cursor),
    )

    assert cursor.executed == ("q", {"limit": 3})
    assert list(messages_df.columns) == [
        "conversation_id",
        "message_index",
        "speaker",
        "text",
    ]


def test_load_snowflake_chat_messages_requires_chat_columns():
    cursor = FakeCursor([(1, "hi")], ["CONVERSATION_ID", "TEXT"])

    with pytest.raises(ValueError, match="missing"):
        load_snowflake_chat_messages(query="q", connection=FakeConnection(cursor))


def test_chat_messages_to_windows_orders_groups_and_anonymises_speakers():
    messages_df = pd.DataFrame(
        {
            "conversation_id": [7, 7, 7, 8, 7],
            "message_index": [3, 1, 2, 1, 4],
            "speaker": ["Jane Doe", "Jane Doe", "Agent Bob", "x", "Jane Doe"],
            "text": ["third", "first  line\nwrapped", "second", "other", None],
        },
    )

    windows = chat_messages_to_windows(messages_df, window_size=2, source="zendesk")

    assert windows == [
        {
            "text": "Speaker 1: first line wrapped\nSpeaker 2: second",
            "conversation_id": "7",
            "window_index": 0,
            "source": "zendesk",
        },
        {
            "text": "Speaker 1: third",
            "conversation_id": "7",
            "window_index": 1,
            "source": "zendesk",
        },
        {
            "text": "Speaker 1: other",
            "conversation_id": "8",
            "window_index": 0,
            "source": "zendesk",
        },
    ]


def test_chat_messages_to_windows_keeps_role_speakers():
    messages_df = pd.DataFrame(
        {
            "conversation_id": [1],
            "message_index": [1],
            "speaker": ["agent"],
            "text": ["hi"],
        },
    )

    windows = chat_messages_to_windows(messages_df, anonymise_speakers=False)

    assert windows[0]["text"] == "agent: hi"


def test_locate_spans_finds_all_occurrences_and_drops_bad_entities():
    text = "Speaker 1: I am Al, email al@x.com. Also Al again."

    spans = locate_spans(
        text,
        [
            {"entity_type": "PERSON", "text": "Al"},
            {"entity_type": "EMAIL_ADDRESS", "text": " al@x.com "},
            {"entity_type": "PERSON", "text": "Bob"},  # hallucinated
            {"entity_type": "PET", "text": "email"},  # unknown label
            {"entity_type": "PERSON", "text": ""},
        ],
        entity_types=["PERSON", "EMAIL_ADDRESS"],
    )

    # "Al" inside "Also" is not a whole word
    assert spans_as_tuples(spans) == [
        ("PERSON", "Al", 16, 18),
        ("EMAIL_ADDRESS", "al@x.com", 26, 34),
        ("PERSON", "Al", 41, 43),
    ]


def test_locate_spans_resolves_overlaps_by_longest():
    text = "Call Jane Doe on 07700 900123"

    spans = locate_spans(
        text,
        [
            {"entity_type": "PERSON", "text": "Jane"},
            {"entity_type": "PERSON", "text": "Jane Doe"},
            {"entity_type": "PHONE_NUMBER", "text": "07700 900123"},
            {"entity_type": "DATE_TIME", "text": "900123"},
        ],
    )

    assert spans_as_tuples(spans) == [
        ("PERSON", "Jane Doe", 5, 13),
        ("PHONE_NUMBER", "07700 900123", 17, 29),
    ]


def test_ollama_annotator_parses_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return FakeResponse('{"entities": [{"entity_type": "PERSON", "text": "Jane"}]}')

    monkeypatch.setattr(ollama_annotator.requests, "post", fake_post)
    annotator = OllamaAnnotator(model="gemma4:latest", cache_dir=tmp_path)

    first = annotator.annotate("Hi Jane")
    second = annotator.annotate("Hi Jane")

    assert (
        spans_as_tuples(first) == spans_as_tuples(second) == [("PERSON", "Jane", 3, 7)]
    )
    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "gemma4:latest"
    assert payload["options"]["temperature"] == 0
    assert payload["think"] is False
    assert (
        "PERSON"
        in payload["format"]["properties"]["entities"]["items"]["properties"][
            "entity_type"
        ]["enum"]
    )


def test_ollama_annotator_retries_then_gives_up_without_caching(monkeypatch, tmp_path):
    answers = iter(["not json", '{"entities": "nope"}', '{"wrong": []}'])
    monkeypatch.setattr(
        ollama_annotator.requests,
        "post",
        lambda url, json, timeout: FakeResponse(next(answers)),
    )
    annotator = OllamaAnnotator(cache_dir=tmp_path, max_retries=2)

    assert annotator.annotate("Hi Jane") == []
    assert list(tmp_path.iterdir()) == []


def test_label_config_lists_entities():
    config = label_config(["PERSON", "ZIP_CODE"])

    assert '<Label value="PERSON"/>' in config
    assert '<Label value="ZIP_CODE"/>' in config
    assert 'value="$text"' in config


def test_label_studio_round_trip(tmp_path):
    windows = [
        {"text": "Speaker 1: I am Jane Doe", "conversation_id": "7", "window_index": 0},
        {"text": "Speaker 1: not reviewed", "conversation_id": "8", "window_index": 0},
        {"text": "Speaker 1: cancelled", "conversation_id": "9", "window_index": 0},
    ]
    predicted = [
        [Span("PERSON", "Jane", 16, 20)],
        [],
        [],
    ]
    tasks = to_label_studio_tasks(windows, predicted, model_version="gemma4:latest")
    assert tasks[0]["predictions"][0]["result"][0]["value"] == {
        "start": 16,
        "end": 20,
        "text": "Jane",
        "labels": ["PERSON"],
    }

    # Simulate Label Studio's export: the reviewer widened the span in a second annotation
    reviewed = json.loads(json.dumps(tasks[0]["predictions"][0]["result"]))
    reviewed[0]["value"].update({"end": 24, "text": "Jane Doe"})
    for task_id, task in enumerate(tasks, start=1):
        task["id"] = task_id
        task["annotations"] = []
    tasks[0]["annotations"] = [
        {
            "id": 1,
            "completed_by": 1,
            "updated_at": "2026-10-23T10:00:00Z",
            "result": [],
        },
        {
            "id": 2,
            "completed_by": 2,
            "updated_at": "2026-10-23T11:00:00Z",
            "result": reviewed,
        },
    ]
    tasks[2]["annotations"] = [{"id": 3, "was_cancelled": True, "result": []}]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(tasks))

    samples = from_label_studio_export(export_path)

    assert len(samples) == 1
    (sample,) = samples
    assert sample.full_text == "Speaker 1: I am Jane Doe"
    assert sample.masked == "Speaker 1: I am {{PERSON}}"
    assert sample.template_id == 1
    assert spans_as_tuples(sample.spans) == [("PERSON", "Jane Doe", 16, 24)]
    assert sample.metadata["conversation_id"] == "7"
    assert sample.metadata["reviewer"] == 2
    assert sample.metadata["predicted_by"] == "gemma4:latest"
    assert "PERSON" in sample.tags[-1]

    assert prediction_agreement(samples) == {
        "samples": 1,
        "predicted_spans": 1,
        "reviewed_spans": 1,
        "precision": 0.0,
        "recall": 0.0,
    }

    # The dataset loads like the synthetic ones
    dataset_path = tmp_path / "gold.json"
    InputSample.to_json(samples, dataset_path)
    (loaded,) = InputSample.read_dataset_json(dataset_path)
    assert loaded.masked == sample.masked
    assert loaded.template_id == 1
    assert spans_as_tuples(loaded.spans) == [("PERSON", "Jane Doe", 16, 24)]


def test_mask_text_matches_sentence_faker_placeholders():
    text = "I am Jane Doe, jane@x.com"
    spans = [
        Span("EMAIL_ADDRESS", "jane@x.com", 15, 25),
        Span("PERSON", "Jane Doe", 5, 13),
        Span("PERSON", "Doe", 10, 13),  # overlapping, skipped
    ]

    assert mask_text(text, spans) == "I am {{PERSON}}, {{EMAIL_ADDRESS}}"
    assert mask_text(text, []) == text


def test_windows_of_one_conversation_share_template_id(tmp_path):
    def task(conversation_id, window_index):
        return {
            "id": window_index,
            "data": {
                "text": "hi",
                "conversation_id": conversation_id,
                "window_index": window_index,
                "source": "zendesk",
            },
            "annotations": [{"id": 1, "result": []}],
        }

    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps([task("a", 0), task("b", 0), task("a", 1)]))

    samples = from_label_studio_export(export_path)

    assert [sample.template_id for sample in samples] == [1, 2, 1]


def test_to_input_samples_uses_sentence_faker_format(tmp_path):
    windows = [
        {"text": "Speaker 1: I am Jane", "conversation_id": "7", "source": "zendesk"},
        {"text": "Speaker 1: bye", "conversation_id": "7", "source": "zendesk"},
        {"text": "Speaker 1: hi", "conversation_id": "8", "source": "zendesk"},
    ]
    spans = [[Span("PERSON", "Jane", 16, 20)], [], []]

    samples = to_input_samples(windows, spans, model_version="gemma4:latest")
    path = tmp_path / "annotated.json"
    InputSample.to_json(samples, path)

    first = json.loads(path.read_text())[0]
    assert set(first) == {"full_text", "masked", "spans", "template_id", "metadata"}
    assert first["masked"] == "Speaker 1: I am {{PERSON}}"
    assert first["metadata"] == {
        "conversation_id": "7",
        "source": "zendesk",
        "predicted_by": "gemma4:latest",
    }
    assert [sample.template_id for sample in samples] == [1, 1, 2]
    loaded = InputSample.read_dataset_json(path)
    assert spans_as_tuples(loaded[0].spans) == [("PERSON", "Jane", 16, 20)]
