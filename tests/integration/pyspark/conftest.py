"""PySpark integration test fixtures."""

import ibis
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def pyspark_connection():
    """Connect to PySpark for integration tests."""
    return ibis.pyspark.connect()


@pytest.fixture(scope="session")
def transactions_table(pyspark_connection):
    """Get the transactions table for testing."""
    # Use pandas to read the parquet file first, then convert to Spark
    # This handles timestamp compatibility issues automatically
    df = pd.read_parquet("data/transactions.parquet")
    # # Pyspark has no time column so we have to convert it to a datetime
    df["transaction_time"] = pd.to_datetime(
        df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str),
    )
    spark_df = pyspark_connection._session.createDataFrame(df)
    # Create a temporary view and read it back as an ibis table
    spark_df.createOrReplaceTempView("transactions")
    return pyspark_connection.table("transactions")
