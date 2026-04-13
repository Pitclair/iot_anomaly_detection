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

# Ensure the entrypoint runs python module; CMD can be overridden by compose
# ENTRYPOINT ["python", "-u", "main.py"]
# CMD ["model", "--config", "/configs/config.json"]


