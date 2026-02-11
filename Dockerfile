FROM python:3.11-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System dependencies for pysam (htslib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libz-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry
# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false


WORKDIR /opt

# Install dependencies first (cache layer)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

# Copy source and install package
COPY . .
RUN poetry install

CMD ["bash"]
