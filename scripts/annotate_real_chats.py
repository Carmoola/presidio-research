#!/usr/bin/env python3
"""Build a Presidio evaluation dataset from real chat transcripts.

1. ``prepare``: sample chats from Snowflake, pre-annotate PII with a local Ollama model
   (Gemma) and write the draft dataset in PresidioSentenceFaker's InputSample format, plus
   Label Studio tasks and the labeling config for review.
2. Review the tasks in a local Label Studio (``uvx label-studio start``) and export them as JSON.
3. ``import``: convert the reviewed export into an InputSample JSON dataset for NB4/NB5.

Outputs contain real customer PII: keep them under the gitignored ``data/`` folder.

Usage:
    uv run python scripts/annotate_real_chats.py prepare --source zendesk --limit 50
    uv run python scripts/annotate_real_chats.py import --export data/ls_export.json
"""

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from presidio_evaluator.data_generator.real_data import (
    ANONYMISE_SPEAKERS,
    CHAT_QUERIES,
    DEFAULT_ENTITIES,
    OllamaAnnotator,
    chat_messages_to_windows,
    from_label_studio_export,
    label_config,
    load_snowflake_chat_messages,
    prediction_agreement,
    to_input_samples,
    to_label_studio_tasks,
)
from presidio_evaluator.data_generator.real_data.ollama_annotator import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
)
from presidio_evaluator.data_objects import InputSample

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def prepare(args: argparse.Namespace) -> None:
    query = (
        args.query_file.read_text() if args.query_file else CHAT_QUERIES[args.source]
    )
    messages_df = load_snowflake_chat_messages(query=query, limit=args.limit)
    windows = chat_messages_to_windows(
        messages_df,
        window_size=args.window,
        source=args.source,
        anonymise_speakers=ANONYMISE_SPEAKERS[args.source],
    )
    print(f"{len(messages_df)} messages -> {len(windows)} windows")

    annotator = OllamaAnnotator(
        model=args.model,
        host=args.host,
        cache_dir=args.cache_dir,
    )
    spans_per_window = [
        annotator.annotate(window["text"])
        for window in tqdm(windows, desc=f"Annotating with {args.model}")
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = to_input_samples(windows, spans_per_window, model_version=args.model)
    InputSample.to_json(samples, args.out)
    print(f"Wrote {len(samples)} pre-annotated (unreviewed) samples to {args.out}")

    tasks = to_label_studio_tasks(windows, spans_per_window, model_version=args.model)
    args.tasks_out.parent.mkdir(parents=True, exist_ok=True)
    args.tasks_out.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_path = args.tasks_out.with_suffix(".label_config.xml")
    config_path.write_text(label_config(list(DEFAULT_ENTITIES)), encoding="utf-8")
    print(f"Wrote {len(tasks)} Label Studio tasks for review to {args.tasks_out}")
    print(f"Labeling config for the Label Studio project: {config_path}")


def import_reviewed(args: argparse.Namespace) -> None:
    samples = from_label_studio_export(args.export)
    InputSample.to_json(samples, args.out)
    print(f"Wrote {len(samples)} reviewed samples to {args.out}")
    print("Pre-annotation agreement:", prediction_agreement(samples))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Fetch and pre-annotate chats"
    )
    prepare_parser.add_argument(
        "--source", choices=sorted(CHAT_QUERIES), default="zendesk"
    )
    prepare_parser.add_argument(
        "--query-file",
        type=Path,
        help="SQL file overriding the source's default query (same output columns)",
    )
    prepare_parser.add_argument("--limit", type=int, default=50, help="Conversations")
    prepare_parser.add_argument(
        "--window", type=int, default=6, help="Messages per sample"
    )
    prepare_parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
    )
    prepare_parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST,
    )
    prepare_parser.add_argument(
        "--cache-dir", type=Path, default=DATA_DIR / ".ollama_cache"
    )
    prepare_parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / f"real_chats_annotated_{date.today():%Y%m%d}.json",
        help="Pre-annotated dataset in PresidioSentenceFaker's format",
    )
    prepare_parser.add_argument(
        "--tasks-out",
        type=Path,
        default=DATA_DIR / "real_chats_tasks.json",
        help="Label Studio tasks for human review",
    )
    prepare_parser.set_defaults(func=prepare)

    import_parser = subparsers.add_parser("import", help="Convert a reviewed export")
    import_parser.add_argument("--export", type=Path, required=True)
    import_parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / f"real_chats_gold_{date.today():%Y%m%d}.json",
    )
    import_parser.set_defaults(func=import_reviewed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
