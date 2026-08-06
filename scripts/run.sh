#!/bin/sh

# Simple runner for the build-based Docker Compose application.
# - ensures a .env exists (copies from .env.LOCAL if available)
# - builds the app image, brings compose down, starts it, then brings it down after exit

# set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  if [ -f .env.LOCAL ]; then
    cp .env.LOCAL .env
    echo "Created .env from .env.LOCAL"
  else
    echo ".env not found and .env.LOCAL missing; creating empty .env"
    touch .env
  fi
fi

# Build the image that compiles the native LM library during startup.
echo "Building app image..."
docker compose -f docker-compose.yml build app

# Ensure any previous run is stopped and orphan containers removed
echo "Tearing down previous compose state..."
docker compose -f docker-compose.yml down --remove-orphans

# Run compose in foreground so the user sees logs
echo "Starting app (foreground). Use Ctrl+C to stop."
docker compose -f docker-compose.yml up --remove-orphans

# After exit, bring everything down
echo "Bringing compose down..."
docker compose -f docker-compose.yml down

echo "Run finished."
