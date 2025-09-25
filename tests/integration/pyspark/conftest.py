"""PySpark integration test fixtures."""

import ibis
import pytest
from loguru import logger


@pytest.fixture(scope="session")
def pyspark_connection():
    """Connect to PySpark for integration tests."""
    try:
        conn = ibis.pyspark.connect()
        logger.info("Connected to PySpark")
    except Exception as e:
        logger.error(f"Failed to connect to PySpark: {e}")
        raise
    else:
        return conn


@pytest.fixture(scope="session")
def transactions_table(pyspark_connection):
    """Get the transactions table for testing."""
    return pyspark_connection.read_parquet("/app/data/transactions.parquet")
