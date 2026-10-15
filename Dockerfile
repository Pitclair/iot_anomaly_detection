ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS build
WORKDIR /build
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN gcc -O3 -fPIC -shared src/lm_idnet/algorithms/native/LM_time_lib.c \
      -lm -o src/lm_idnet/algorithms/native/LM_time_lib.so \
    && pip install --no-cache-dir --prefix=/install .

FROM python:${PYTHON_VERSION}
ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision=${VCS_REF}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LM_IDNET_REVISION=${VCS_REF} \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY --from=build /install /usr/local
COPY configs ./configs
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["--help"]
