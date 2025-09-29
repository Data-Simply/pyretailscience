FROM apache/spark-py:latest

# Switch to root to install packages
USER root

# Install py4j and uv for fast dependency management
RUN pip install py4j==0.10.9.7 uv==0.5.30

# Set working directory
WORKDIR /app/work

# Copy project files
COPY pyproject.toml ./
COPY uv.lock ./
COPY data/ /app/data/

# Install dependencies
RUN uv sync --group dev --link-mode=copy

# Set Spark configurations for resource constraints
ENV SPARK_DRIVER_MEMORY=512m
ENV SPARK_EXECUTOR_MEMORY=512m
ENV SPARK_LOCAL_DIRS=/tmp/spark
ENV PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
ENV SPARK_HOME=/opt/spark
ENV UV_LINK_MODE=copy

# Run tests by default
CMD ["uv", "run", "pytest", "tests/integration/pyspark", "-v", "--tb=short"]
