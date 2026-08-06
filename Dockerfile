ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

WORKDIR /app

# The LM backend is required and is compiled whenever the container starts.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# An editable install keeps the installed CLI and package source in one place.
RUN pip install --no-cache-dir --no-deps --editable .

# The runner uses the host user so generated files retain host ownership.
RUN chmod a+w /app/src/lm_idnet/algorithms/native

ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["--help"]
