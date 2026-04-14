"""Unified integration test fixtures for multiple database backends."""

import os

import ibis
import pandas as pd
import pytest
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)


def _load_snowflake_private_key() -> bytes:
    """Load Snowflake private key from PEM file and return DER-encoded bytes.

    Returns:
        bytes: DER-encoded private key bytes suitable for Snowflake authentication.
    """
    key_path = os.environ["SNOWFLAKE_CI_PRIVATE_KEY_PATH"]
    with open(key_path, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)
    return private_key.private_bytes(
        encoding=Encoding.DER,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )


@pytest.fixture(
    params=["bigquery", "pyspark", "snowflake"],
    ids=lambda backend: f"backend={backend}",
)
def transactions_table(request):
    """Parameterized fixture that provides transactions table from different backends."""
    if request.param == "bigquery":
        connection = ibis.bigquery.connect(
            project_id=os.environ["GCP_PROJECT_ID"],
        )
        return connection.table("test_data.transactions")
    if request.param == "pyspark":
        connection = ibis.pyspark.connect()
        # Use pandas to read the parquet file first, then convert to Spark
        # This handles timestamp compatibility issues automatically
        df = pd.read_parquet("data/transactions.parquet")
        # Pyspark has no time column so we have to convert it to a datetime
        df["transaction_time"] = pd.to_datetime(
            df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str),
        )
        spark_df = connection._session.createDataFrame(df)
        # Create a temporary view and read it back as an ibis table
        spark_df.createOrReplaceTempView("transactions")
        return connection.table("transactions")
    if request.param == "snowflake":
        connection = ibis.snowflake.connect(
            account=os.environ["SNOWFLAKE_CI_ACCOUNT"],
            user=os.environ["SNOWFLAKE_CI_USER"],
            private_key=_load_snowflake_private_key(),
            database=os.environ["SNOWFLAKE_CI_DATABASE"],
            schema=os.environ["SNOWFLAKE_CI_SCHEMA"],
            warehouse=os.environ["SNOWFLAKE_CI_WAREHOUSE"],
        )
        table = connection.table("TRANSACTIONS")
        return table.rename({col.lower(): col for col in table.columns})
    error_msg = f"Unknown backend: {request.param}"
    raise ValueError(error_msg)
