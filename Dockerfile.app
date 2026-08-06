ARG PYTHON_VERSION=3.11-slim

FROM python:${PYTHON_VERSION}

# Create app directory
WORKDIR /srv/app

# The LM backend is required, so every runtime image includes its compiler.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /srv/app/requirements.txt

# Install dependencies in the image (one time at build)
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

# Copy the rest of the project
COPY . /srv/app

# Install the source-layout package and its lm-idnet console command.
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["/bin/sh", "/srv/app/scripts/docker-entrypoint.sh"]
CMD ["lm-idnet"]
