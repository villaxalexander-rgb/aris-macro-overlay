FROM python:3.11-slim

WORKDIR /app

# System deps: netcat for gateway health check, cron for scheduling
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p logs data/signals data/cache docs/fund_notes state

# Make scripts executable
RUN chmod +x docker/scripts/*.sh

ENTRYPOINT ["/app/docker/scripts/entrypoint.sh"]
