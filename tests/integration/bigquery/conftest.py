"""BigQuery integration test fixtures."""

import os

import ibis
import pytest
from dotenv import load_dotenv
from google.cloud import bigquery
from loguru import logger

load_dotenv()
client = bigquery.Client(project="pyretailscience-infra")


@pytest.fixture(scope="session")
def bigquery_connection():
    """Connect to BigQuery for integration tests."""
    try:
        conn = ibis.bigquery.connect(
            project_id=os.environ.get("GCP_PROJECT_ID"),
        )
        logger.info("Connected to BigQuery")
    except Exception as e:
        logger.error(f"Failed to connect to BigQuery: {e}")
        raise
    else:
        return conn


@pytest.fixture(scope="session")
def transactions_table(bigquery_connection):
    """Get the transactions table for testing."""
    return bigquery_connection.table("test_data.transactions")
