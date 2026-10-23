"""Load real person records (name, address, phone, ...) from the Snowflake DWH.

The returned DataFrame uses the same column names as ``load_fake_person_df``, so it can be
passed directly as ``base_records`` to ``PresidioSentenceFaker``.

The output contains real customer PII: keep generated datasets out of version control
(``/data`` is gitignored) and do not share them outside approved environments.

Requires the optional ``snowflake`` extra: ``uv sync --extra snowflake``.
Connection parameters default to these environment variables (a ``.env`` file is honoured):
``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_USER``, ``SNOWFLAKE_ROLE``, ``SNOWFLAKE_WAREHOUSE``,
``SNOWFLAKE_DATABASE`` (default ``analytics``) and ``SNOWFLAKE_AUTHENTICATOR``
(default ``externalbrowser``).
"""

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

DEFAULT_PERSON_QUERY = """
SELECT
    full_name
    , phone
    , email
    , full_address
    , post_code
    , city
    , country
FROM source_user.users_pii
WHERE
    NOT is_test_user
    AND NOT is_deleted
    AND NOT is_anonymized
    AND full_name IS NOT NULL
ORDER BY RANDOM()
LIMIT %(limit)s
"""

# DWH column -> faker/PresidioSentenceFaker record key
DWH_COLUMN_MAPPING = {
    "full_name": "person",
    "phone": "phone_number",
    "email": "email",
    "full_address": "address",
    "post_code": "postcode",
    "city": "city",
    "country": "country",
}

# Template entity -> record key it is copied from, so these entities are also drawn from the DWH.
# The DWH has no gender, so gendered name entities get the customer's name regardless.
DWH_COLUMN_ALIASES = {
    "name": "person",
    "name_female": "person",
    "name_male": "person",
    "first_name_female": "first_name",
    "first_name_male": "first_name",
    "first_name_nonbinary": "first_name",
    "last_name_female": "last_name",
    "last_name_male": "last_name",
    "zipcode": "postcode",
}


def connect_to_snowflake(**connect_kwargs: Any):
    """Open a Snowflake connection, filling missing parameters from environment variables."""
    try:
        import snowflake.connector
    except ImportError as e:
        raise ImportError(
            "snowflake-connector-python is required to load DWH records. "
            "Install it with `uv sync --extra snowflake`.",
        ) from e

    load_dotenv()
    # `or` so that empty values (e.g. copied from .env.template) count as unset
    params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE") or "prod",
        "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR") or "externalbrowser",
    }
    params.update(connect_kwargs)
    params = {key: value for key, value in params.items() if value}
    connection = snowflake.connector.connect(**params)

    # Snowflake silently ignores a database/warehouse the role cannot use, so fail early and clearly
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_WAREHOUSE()")
        role, database, warehouse = cursor.fetchone() or (None, None, None)
    finally:
        cursor.close()
    if database is None or warehouse is None:
        connection.close()
        raise RuntimeError(
            f"Snowflake session has role={role}, database={database}, warehouse={warehouse}. "
            f"Role {role} probably cannot use database {params['database']!r} or "
            f"warehouse {params.get('warehouse')!r}: set SNOWFLAKE_ROLE / SNOWFLAKE_WAREHOUSE "
            "in .env to ones with access.",
        )
    return connection


def dwh_rows_to_person_df(dwh_df: DataFrame) -> DataFrame:
    """Convert DWH person rows into PresidioSentenceFaker records.

    Columns are renamed according to ``DWH_COLUMN_MAPPING`` (unknown columns are kept as-is),
    ``first_name`` and ``last_name`` are derived from ``person``, ``DWH_COLUMN_ALIASES`` columns
    are added, and missing or blank values become ``None`` so the sentence faker does not use them
    (pass ``fill_from_other_records=True`` to ``PresidioSentenceFaker`` to draw them from other
    records instead of Faker providers).
    """
    person_df = dwh_df.rename(columns=lambda col: col.lower()).rename(
        columns=DWH_COLUMN_MAPPING,
    )
    person_df = person_df.astype(object).map(
        lambda value: value.strip() if isinstance(value, str) else value,
    )

    if "person" in person_df.columns:
        name_parts = person_df["person"].map(
            lambda name: name.split() if isinstance(name, str) else [],
        )
        person_df["first_name"] = name_parts.map(
            lambda parts: parts[0] if parts else None
        )
        person_df["last_name"] = name_parts.map(
            lambda parts: parts[-1] if len(parts) > 1 else None,
        )

    for alias, column in DWH_COLUMN_ALIASES.items():
        if column in person_df.columns:
            person_df[alias] = person_df[column]

    # Normalise last: pandas may turn None into NaN, which is truthy and would be rendered as "nan"
    person_df = person_df.astype(object).replace("", None)
    return person_df.where(person_df.notna(), None)


def load_snowflake_person_df(
    limit: int = 3000,
    query: str = DEFAULT_PERSON_QUERY,
    connection=None,
    **connect_kwargs: Any,
) -> DataFrame:
    """Sample person records (name, phone, email, address) from the Snowflake DWH.

    :param limit: Maximum number of records to sample, passed to the query as ``%(limit)s``
    :param query: SQL returning DWH columns (see ``DWH_COLUMN_MAPPING``); may reference ``%(limit)s``
    :param connection: An open Snowflake connection. If None, one is opened via
    ``connect_to_snowflake(**connect_kwargs)`` and closed afterwards
    :return: A DataFrame usable as ``base_records`` for ``PresidioSentenceFaker``
    """
    owns_connection = connection is None
    if owns_connection:
        connection = connect_to_snowflake(**connect_kwargs)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(query, {"limit": limit})
            columns = [column[0] for column in cursor.description]
            dwh_df = pd.DataFrame(cursor.fetchall(), columns=columns)
        finally:
            cursor.close()
    finally:
        if owns_connection:
            connection.close()
    return dwh_rows_to_person_df(dwh_df)
