ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

# Create app directory
WORKDIR /srv/app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /srv/app/requirements.txt

# Install dependencies in the image (one time at build)
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

# Copy the rest of the project
COPY . /srv/app

# Install the source-layout package and its lm-idnet console command.
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["lm-idnet"]

