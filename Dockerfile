FROM python:3.11-slim

WORKDIR /app

# The LM backend is required and is compiled whenever the container starts.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install the authoritative dependency set from the package metadata.
RUN pip install --no-cache-dir --editable '.[test]'

# Keep alternate entrypoints, such as pytest, on the image-built native library.
RUN /bin/sh /app/scripts/docker-entrypoint.sh --help >/dev/null

# The runner uses the host user so generated files retain host ownership.
RUN chmod a+w /app/src/lm_idnet/algorithms/native

ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["--help"]
