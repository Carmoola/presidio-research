import re

import pandas as pd

from presidio_evaluator.data_generator import PresidioSentenceFaker
from presidio_evaluator.data_generator.faker_extensions.snowflake_records import (
    DWH_COLUMN_ALIASES,
    dwh_rows_to_person_df,
    load_snowflake_person_df,
)
from presidio_evaluator.data_generator.presidio_sentence_faker import (
    presidio_templates_file_path,
)


class FakeCursor:
    def __init__(self, rows, columns):
        self._rows = rows
        self.description = [(column,) for column in columns]
        self.executed = None
        self.closed = False

    def execute(self, query, params):
        self.executed = (query, params)

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_dwh_rows_to_person_df_renames_and_splits_names():
    dwh_df = pd.DataFrame(
        {
            "FULL_NAME": ["Jane Mary Doe", "Prince"],
            "PHONE": ["07700 900123", None],
            "FULL_ADDRESS": ["1 High Street, London, SW1A 1AA", "  "],
            "POST_CODE": ["SW1A 1AA", float("nan")],
        },
    )

    person_df = dwh_rows_to_person_df(dwh_df)

    first, second = person_df.to_dict(orient="records")
    assert first["person"] == first["name"] == "Jane Mary Doe"
    assert first["first_name"] == "Jane"
    assert first["last_name"] == "Doe"
    assert first["phone_number"] == "07700 900123"
    assert first["address"] == "1 High Street, London, SW1A 1AA"
    assert first["postcode"] == "SW1A 1AA"

    assert second["first_name"] == "Prince"
    assert second["last_name"] is None
    assert second["phone_number"] is None
    assert second["address"] is None
    assert second["postcode"] is None


def test_load_snowflake_person_df_uses_given_connection():
    cursor = FakeCursor(
        rows=[("John Smith", "+447700900123")],
        columns=["FULL_NAME", "PHONE"],
    )
    connection = FakeConnection(cursor)

    person_df = load_snowflake_person_df(limit=5, connection=connection)

    assert cursor.executed[1] == {"limit": 5}
    assert cursor.closed
    assert not connection.closed  # caller owns the connection
    assert person_df.loc[0, "person"] == "John Smith"
    assert person_df.loc[0, "phone_number"] == "+447700900123"


def test_dwh_records_feed_sentence_faker_without_nan_values():
    person_df = dwh_rows_to_person_df(
        pd.DataFrame({"FULL_NAME": ["Jane Doe"], "PHONE": [None]}),
    )
    sentence_faker = PresidioSentenceFaker(
        "en_GB",
        lower_case_ratio=0,
        sentence_templates=["Call {{name}} on {{phone_number}}"],
        base_records=person_df,
        random_seed=1,
    )

    (sample,) = sentence_faker.generate_new_fake_sentences(1)

    values = {span.entity_type: span.entity_value for span in sample.spans}
    assert values["PERSON"] == "Jane Doe"
    # Missing DWH values fall back to Faker instead of rendering as "None"/"nan"
    assert values["PHONE_NUMBER"] not in ("None", "nan")


def test_dwh_rows_to_person_df_adds_aliases():
    person_df = dwh_rows_to_person_df(
        pd.DataFrame({"FULL_NAME": ["Jane Doe"], "POST_CODE": ["SW1A 1AA"]}),
    )

    (record,) = person_df.to_dict(orient="records")
    for alias, column in DWH_COLUMN_ALIASES.items():
        assert record[alias] == record[column]


def test_all_dwh_entities_come_from_dwh_records_with_fill_from_other_records():
    dwh_df = pd.DataFrame(
        {
            "FULL_NAME": [f"Zzfirst{i} Zzlast{i}" for i in range(5)] + ["Zzmononym"],
            "PHONE": ["+447000000001", None, "+447000000003", None, None, None],
            "EMAIL": ["zz1@zz.com"] + [None] * 5,
            "FULL_ADDRESS": ["1 Zz Road, Zztown, ZZ1 1ZZ"] + [None] * 5,
            "POST_CODE": ["ZZ1 1ZZ"] + [None] * 5,
            "CITY": ["Zztown"] + [None] * 5,
            "COUNTRY": ["Zzland"] + [None] * 5,
        },
    )
    person_df = dwh_rows_to_person_df(dwh_df)
    dwh_values = {
        column: set(person_df[column].dropna()) for column in person_df.columns
    }
    templates = [
        line.strip() for line in presidio_templates_file_path.read_text().splitlines()
    ]
    # Identity mapping keeps template entity names on the spans
    entity_type_mapping = {
        token.strip(): token.strip()
        for template in templates
        for token in re.findall(
            r"\{\{(.*?)\}\}", PresidioSentenceFaker._preprocess_template(template)
        )
    }
    sentence_faker = PresidioSentenceFaker(
        "en_GB",
        lower_case_ratio=0,
        sentence_templates=templates,
        entity_type_mapping=entity_type_mapping,
        base_records=person_df,
        random_seed=1,
        fill_from_other_records=True,
    )

    samples = sentence_faker.generate_new_fake_sentences(2000)

    checked = 0
    for sample in samples:
        for span in sample.spans:
            if span.entity_type in dwh_values:
                assert span.entity_value in dwh_values[span.entity_type], span
                checked += 1
    assert checked > 1000
