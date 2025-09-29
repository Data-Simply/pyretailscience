"""PySpark integration test fixtures."""

import atexit
import tempfile
from pathlib import Path

import ibis
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def pyspark_connection():
    """Connect to PySpark for integration tests."""
    try:
        conn = ibis.pyspark.connect()
    except Exception as e:
        msg = f"Failed to connect to PySpark: {e}"
        raise RuntimeError(msg) from e
    else:
        return conn


@pytest.fixture(scope="session")
def transactions_table(pyspark_connection):
    """Get the transactions table for testing."""
    # Determine data path (Docker vs local)
    data_path = "/app/data/transactions.parquet" if Path("/app/data").exists() else "data/transactions.parquet"

    # Read parquet with pandas first to handle Arrow time types
    df = pd.read_parquet(data_path)

    # Convert time columns to string format, then back to date format for PySpark
    # PySpark can handle date strings in YYYY-MM-DD format
    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
    df["transaction_time"] = df["transaction_time"].astype(str)

    # Save to temporary parquet file that PySpark can read
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, engine="pyarrow")
        temp_path = tmp.name

    # Register cleanup function
    def cleanup():
        temp_file = Path(temp_path)
        if temp_file.exists():
            temp_file.unlink()

    atexit.register(cleanup)

    # Read the processed parquet file with PySpark
    return pyspark_connection.read_parquet(temp_path)
