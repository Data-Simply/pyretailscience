"""BigQuery integration test fixtures."""

import os

import ibis
import pytest
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()
client = bigquery.Client(project="pyretailscience-infra")


@pytest.fixture(scope="session")
def bigquery_connection():
    """Connect to BigQuery for integration tests."""
    return ibis.bigquery.connect(
        project_id=os.environ.get("GCP_PROJECT_ID"),
    )


@pytest.fixture(scope="session")
def transactions_table(bigquery_connection):
    """Get the transactions table for testing."""
    return bigquery_connection.table("test_data.transactions")
