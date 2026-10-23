# Data Generation

📖 The data generation documentation now lives in the Presidio-Research docs:

- **Online:** <https://presidio-research.dataprivacystack.org/data_generation/>
- **Source:** [`docs/data_generation.md`](../../docs/data_generation.md)

It covers the `PresidioSentenceFaker` scenarios, the generation process, and a
full end-to-end example.

## Real chat transcripts (LLM pre-annotation + human review)

`real_data/` builds an evaluation set from real chats in the Snowflake DWH. A local Ollama model
(`gemma4:latest`) drafts the PII spans and a human reviews them in a local Label Studio.
Everything stays on your machine. The outputs contain real customer PII, so keep them in the
gitignored `data/` folder.

1. Prerequisites: `uv sync --extra snowflake`, `.env` filled from `.env.template`, `ollama serve`.
2. Pre-annotate:
   `uv run python scripts/annotate_real_chats.py prepare --source zendesk --limit 50 --window 6`.
   This writes Gemma's draft dataset, `data/real_chats_annotated_<date>.json`, in the same format as
   PresidioSentenceFaker's output (`full_text`, `masked`, `spans`, `template_id`; one
   `template_id` per conversation). For review it also writes `data/real_chats_tasks.json` and
   `data/real_chats_tasks.label_config.xml`. Use
   `--source talkdesk` or `--query-file my.sql` for other sources; the query must return
   `conversation_id, message_index, speaker, text`.
3. Review: `uvx label-studio start`. Create a project, paste the label config under
   *Settings → Labeling Interface → Code*, import the tasks and correct every task; Gemma's spans
   appear as predictions. Export the project as **JSON**.
4. Import: `uv run python scripts/annotate_real_chats.py import --export path/to/export.json`.
   This writes `data/real_chats_gold_<date>.json` and prints how well Gemma agreed with the review.
   Only submitted annotations are used.
5. Point NB4/NB5 at the gold file with `InputSample.read_dataset_json`.

Label Studio keeps a copy of the data in `~/.local/share/label-studio`. Delete the project when
the review is done.
