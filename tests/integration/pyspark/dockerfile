# Dockerfile
FROM python:3.11

# Install OpenJDK 21
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jdk procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    SPARK_DRIVER_MEMORY=1g \
    SPARK_EXECUTOR_MEMORY=1g

WORKDIR /app

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy all files needed for installation
COPY . .

# Install deps with uv
RUN uv sync --locked

CMD ["uv", "run", "pytest", "-v", "tests/integration/pyspark"]
