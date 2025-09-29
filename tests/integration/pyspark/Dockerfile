FROM apache/spark-py:v3.5.0

# Switch to root to install packages
USER root

# Install uv for fast dependency management
RUN pip install uv==0.5.30

# Set working directory
WORKDIR /app/work

# Copy project files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --group dev

# Set Spark configurations for resource constraints
ENV SPARK_DRIVER_MEMORY=512m
ENV SPARK_EXECUTOR_MEMORY=512m
ENV SPARK_LOCAL_DIRS=/tmp/spark

# Keep container running
CMD ["tail", "-f", "/dev/null"]
