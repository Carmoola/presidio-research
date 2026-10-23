"""Turn pre-annotated windows into InputSamples and round-trip them through Label Studio.

``to_input_samples`` builds samples in PresidioSentenceFaker's format from the LLM spans;
``to_label_studio_tasks`` writes tasks with the LLM spans as ``predictions`` (shown to the
reviewer as editable pre-annotations); ``from_label_studio_export`` reads the reviewed
Label Studio JSON export back into ``InputSample`` objects.
"""

import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

from presidio_evaluator.data_objects import InputSample, Span

FROM_NAME = "label"
TO_NAME = "text"


def label_config(entity_types: list[str]) -> str:
    """Label Studio labeling config for span annotation with the given entity types."""
    labels = "\n".join(
        f"    <Label value={quoteattr(entity_type)}/>" for entity_type in entity_types
    )
    return (
        "<View>\n"
        f'  <Labels name="{FROM_NAME}" toName="{TO_NAME}">\n'
        f"{labels}\n"
        "  </Labels>\n"
        f'  <Text name="{TO_NAME}" value="$text"/>\n'
        "</View>\n"
    )


def _span_to_result(span: Span, index: int) -> dict:
    return {
        "id": f"s{index}",
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "type": "labels",
        "value": {
            "start": span.start_position,
            "end": span.end_position,
            "text": span.entity_value,
            "labels": [span.entity_type],
        },
    }


def _result_to_spans(text: str, results: list[dict]) -> list[Span]:
    spans = []
    for result in results:
        if result.get("type") != "labels":
            continue
        value = result["value"]
        start, end = value["start"], value["end"]
        spans.extend(
            Span(
                entity_type=entity_type,
                entity_value=text[start:end],
                start_position=start,
                end_position=end,
            )
            for entity_type in value["labels"]
        )
    return sorted(spans, key=lambda span: (span.start_position, span.end_position))


def to_label_studio_tasks(
    windows: list[dict],
    spans_per_window: list[list[Span]],
    model_version: str,
) -> list[dict]:
    """Build Label Studio import tasks with pre-annotations.

    :param windows: Dicts with a ``text`` key; all keys are kept in the task ``data``
    :param spans_per_window: Predicted spans for each window
    :param model_version: Shown in Label Studio as the prediction source
    """
    return [
        {
            "data": dict(window),
            "predictions": [
                {
                    "model_version": model_version,
                    "result": [
                        _span_to_result(span, index) for index, span in enumerate(spans)
                    ],
                },
            ],
        }
        for window, spans in zip(windows, spans_per_window, strict=True)
    ]


def _latest_annotation(task: dict) -> dict | None:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled")
    ]
    if not annotations:
        return None
    return max(
        annotations,
        key=lambda a: (
            a.get("updated_at") or a.get("created_at") or "",
            a.get("id", 0),
        ),
    )


def mask_text(text: str, spans: list[Span]) -> str:
    """Replace spans with ``{{ENTITY_TYPE}}``, like PresidioSentenceFaker's ``masked`` field.

    Of overlapping spans, only the first (by start position) is masked.
    """
    parts = []
    position = 0
    for span in sorted(spans, key=lambda s: (s.start_position, -s.end_position)):
        if span.start_position < position:
            continue
        parts.append(text[position : span.start_position])
        parts.append(f"{{{{{span.entity_type}}}}}")
        position = span.end_position
    parts.append(text[position:])
    return "".join(parts)


def _input_sample(
    text: str,
    spans: list[Span],
    metadata: dict,
    sample_id: int,
    template_ids: dict,
    create_tags_from_span: bool,
) -> InputSample:
    """Build an InputSample in PresidioSentenceFaker's format.

    ``full_text``, ``masked`` (``{{ENTITY_TYPE}}`` placeholders), ``spans`` and
    ``template_id``. A conversation plays the role of a template: its windows share a
    ``template_id`` (from 1; InputSample treats 0 as unset when metadata is given), so
    splitting by template (``presidio_evaluator.validation``) never puts one conversation
    in both train and test. ``template_ids`` maps conversations to ids across calls.
    """
    conversation_key = (metadata.get("source"), metadata.get("conversation_id"))
    return InputSample(
        full_text=text,
        masked=mask_text(text, spans),
        spans=spans,
        metadata=metadata,
        sample_id=sample_id,
        template_id=template_ids.setdefault(conversation_key, len(template_ids) + 1),
        create_tags_from_span=create_tags_from_span,
    )


def to_input_samples(
    windows: list[dict],
    spans_per_window: list[list[Span]],
    model_version: str | None = None,
    create_tags_from_span: bool = False,
) -> list[InputSample]:
    """Build InputSamples (PresidioSentenceFaker's format) from windows and their spans.

    :param windows: Dicts with a ``text`` key; the other keys go into ``metadata``
    :param spans_per_window: Spans for each window, e.g. from ``OllamaAnnotator.annotate``
    :param model_version: Stored as ``metadata["predicted_by"]`` if given
    """
    samples = []
    template_ids = {}
    for window, spans in zip(windows, spans_per_window, strict=True):
        metadata = {key: value for key, value in window.items() if key != "text"}
        if model_version is not None:
            metadata["predicted_by"] = model_version
        samples.append(
            _input_sample(
                window["text"],
                spans,
                metadata,
                len(samples),
                template_ids,
                create_tags_from_span,
            ),
        )
    return samples


def from_label_studio_export(
    export_path: str | Path,
    create_tags_from_span: bool = True,
) -> list[InputSample]:
    """Read a Label Studio JSON export into InputSamples, using reviewed annotations only.

    The samples are in PresidioSentenceFaker's format, like ``to_input_samples``.
    Tasks without a submitted (non-cancelled) annotation are skipped. The latest annotation
    is the gold; the first prediction is kept in ``metadata["predicted_spans"]``.
    """
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    samples = []
    template_ids = {}
    for task in tasks:
        annotation = _latest_annotation(task)
        if annotation is None:
            continue
        data = dict(task["data"])
        text = data.pop("text")
        predictions = task.get("predictions") or [{}]
        metadata = {
            **data,
            "label_studio_task_id": task.get("id"),
            "reviewer": annotation.get("completed_by"),
            "predicted_by": predictions[0].get("model_version"),
            "predicted_spans": [
                span.__dict__
                for span in _result_to_spans(text, predictions[0].get("result", []))
            ],
        }
        samples.append(
            _input_sample(
                text,
                _result_to_spans(text, annotation["result"]),
                metadata,
                len(samples),
                template_ids,
                create_tags_from_span,
            ),
        )
    return samples


def prediction_agreement(samples: list[InputSample]) -> dict:
    """Exact-match precision/recall of the LLM pre-annotations against the reviewed spans."""

    def keys(spans: list[dict]) -> set:
        return {
            (s["start_position"], s["end_position"], s["entity_type"]) for s in spans
        }

    true_positives = predicted = gold = 0
    for sample in samples:
        predicted_keys = keys(sample.metadata.get("predicted_spans", []))
        gold_keys = keys([span.__dict__ for span in sample.spans])
        true_positives += len(predicted_keys & gold_keys)
        predicted += len(predicted_keys)
        gold += len(gold_keys)
    return {
        "samples": len(samples),
        "predicted_spans": predicted,
        "reviewed_spans": gold,
        "precision": true_positives / predicted if predicted else None,
        "recall": true_positives / gold if gold else None,
    }
